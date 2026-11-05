from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from .classification import classify_candidate, confusion_summary, persistent_threshold
from .config import normalize_scenario_name
from .configuration import load_all_scenario_records, load_workflow_config


FEATURE_COLUMNS = [
    "commanded_kla_scaled",
    "load_index_scaled",
    "temperature_scaled",
    "oxygen_transfer_factor",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Cannot parse Boolean surrogate classification values.")
    return normalized.isin(["true", "1"])


@dataclass
class MonotonicLogisticClassifier:
    l2_penalty: float = 1e-3
    coefficients_: np.ndarray | None = None
    intercept_: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "MonotonicLogisticClassifier":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4 or len(y) != len(x):
            raise ValueError("The monotonic logistic baseline expects four features and one target per row.")

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = parameters[0]
            beta = parameters[1:]
            probability = _sigmoid(intercept + x @ beta)
            eps = 1e-12
            loss = -np.sum(y * np.log(probability + eps) + (1.0 - y) * np.log(1.0 - probability + eps))
            loss += self.l2_penalty * float(beta @ beta)
            residual = probability - y
            gradient = np.concatenate(
                [[np.sum(residual)], x.T @ residual + 2.0 * self.l2_penalty * beta]
            )
            return float(loss), gradient

        # KLa, temperature and oxygen-transfer effects must be non-negative;
        # the load effect must be non-positive. Constant calibration features
        # remain identifiable only through regularization and are flagged later.
        bounds = [(None, None), (0.0, None), (None, 0.0), (0.0, None), (0.0, None)]
        result = minimize(
            fun=lambda value: objective(value)[0],
            x0=np.zeros(5, dtype=float),
            jac=lambda value: objective(value)[1],
            bounds=bounds,
            method="L-BFGS-B",
        )
        if not result.success:
            raise RuntimeError(f"Monotonic logistic optimization failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.coefficients_ = np.asarray(result.x[1:], dtype=float)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("Fit the monotonic logistic classifier before prediction.")
        return _sigmoid(self.intercept_ + np.asarray(features, dtype=float) @ self.coefficients_)


def add_scenario_features(frame: pd.DataFrame, workflow: Mapping | None = None) -> pd.DataFrame:
    active = dict(workflow or load_workflow_config())
    scenario_records = {record["name"]: record for record in load_all_scenario_records(active)}
    out = frame.copy()
    out["scenario"] = out["scenario"].map(normalize_scenario_name)
    out["commanded_kla_d-1"] = pd.to_numeric(
        out["commanded_kla_d-1"] if "commanded_kla_d-1" in out else out["kla"], errors="raise"
    )
    out["load_index"] = [
        float(scenario_records[name]["nh4_scale"]) * float(scenario_records[name]["flow_scale"])
        for name in out["scenario"]
    ]
    out["temperature_c"] = [float(scenario_records[name]["temperature_c"]) for name in out["scenario"]]
    out["oxygen_transfer_factor"] = [float(scenario_records[name]["kla_scale"]) for name in out["scenario"]]
    out["commanded_kla_scaled"] = out["commanded_kla_d-1"] / 240.0
    out["load_index_scaled"] = out["load_index"]
    out["temperature_scaled"] = out["temperature_c"] / 20.0
    return out


def _monotonicity_violations(group: pd.DataFrame, column: str, increasing: bool = True) -> int:
    values = pd.to_numeric(group.sort_values("commanded_kla_d-1")[column], errors="coerce").to_numpy(dtype=float)
    differences = np.diff(values)
    return int(np.sum(differences < -1e-10)) if increasing else int(np.sum(differences > 1e-10))


def compare_simple_baselines(
    bsm1_results: pd.DataFrame,
    surrogate_classification: pd.DataFrame,
    *,
    workflow: Mapping | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active = dict(workflow or load_workflow_config())
    benchmark = active["benchmark"]
    persistent_points = int(active["candidate_grid"]["persistent_compliance_points"])
    frame = add_scenario_features(bsm1_results, workflow=active)
    frame["bsm1_satisfies_study_benchmark"] = (
        pd.to_numeric(frame["SNH_violation_hours_gt_limit"], errors="raise") <= 1e-12
    )
    calibration_names = set(active["calibration_scenarios"])
    calibration = frame[frame["scenario"].isin(calibration_names)].copy()
    if calibration.empty:
        raise ValueError("No calibration rows are available for the simple baselines.")

    baseline_threshold = persistent_threshold(
        calibration,
        kla_col="commanded_kla_d-1",
        compliant_col="bsm1_satisfies_study_benchmark",
        persistent_points=persistent_points,
    )
    if baseline_threshold.first_compliant_grid_point is None:
        raise RuntimeError("Threshold lookup cannot be fit because the calibration transition is unresolved.")

    x_train = calibration[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = calibration["bsm1_satisfies_study_benchmark"].astype(float).to_numpy()
    logistic = MonotonicLogisticClassifier()
    tic = time.perf_counter()
    logistic.fit(x_train, y_train)
    logistic_fit_seconds = time.perf_counter() - tic

    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    tic = time.perf_counter()
    isotonic.fit(calibration["commanded_kla_scaled"], y_train)
    isotonic_fit_seconds = time.perf_counter() - tic

    prediction_rows: list[pd.DataFrame] = []
    definitions = []
    for model_name in ["threshold_lookup", "monotonic_logistic", "isotonic_regression"]:
        prediction = frame[["scenario", "commanded_kla_d-1", "bsm1_satisfies_study_benchmark"]].copy()
        tic = time.perf_counter()
        if model_name == "threshold_lookup":
            prediction["predicted_probability"] = (
                frame["commanded_kla_d-1"] >= float(baseline_threshold.first_compliant_grid_point)
            ).astype(float)
            fit_seconds = 0.0
            interpretation = "One baseline-scenario commanded-KLa threshold applied unchanged to all scenarios."
        elif model_name == "monotonic_logistic":
            prediction["predicted_probability"] = logistic.predict_proba(frame[FEATURE_COLUMNS].to_numpy(dtype=float))
            fit_seconds = logistic_fit_seconds
            interpretation = "Sign-constrained logistic model with KLa, load, temperature, and oxygen-transfer inputs."
        else:
            prediction["predicted_probability"] = isotonic.predict(frame["commanded_kla_scaled"])
            fit_seconds = isotonic_fit_seconds
            interpretation = "One-dimensional monotonic isotonic compliance curve fit only on the calibration scenario."
        predict_seconds = time.perf_counter() - tic
        prediction["predicted_satisfies_study_benchmark"] = prediction["predicted_probability"] >= 0.5
        prediction["classification"] = [
            classify_candidate(bool(predicted), bool(actual))
            for predicted, actual in zip(
                prediction["predicted_satisfies_study_benchmark"],
                prediction["bsm1_satisfies_study_benchmark"],
            )
        ]
        prediction["model"] = model_name
        prediction_rows.append(prediction)
        definitions.append(
            {
                "model": model_name,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "training_scenarios": ",".join(sorted(calibration_names)),
                "validation_scenarios": ",".join(active["validation_scenarios"]),
                "interpretation": interpretation,
                "calibration_covariate_identifiability_limit": (
                    "Load, temperature and oxygen-transfer coefficients cannot be strongly identified from baseline-only calibration."
                    if model_name == "monotonic_logistic" else "not_applicable"
                ),
            }
        )

    surrogate = surrogate_classification[
        ["scenario", "commanded_kla_d-1", "bsm1_satisfies_study_benchmark", "classification"]
    ].copy()
    surrogate["predicted_satisfies_study_benchmark"] = _boolean_series(
        surrogate_classification["surrogate_satisfies_study_benchmark"]
    )
    surrogate["predicted_probability"] = surrogate["predicted_satisfies_study_benchmark"].astype(float)
    surrogate["model"] = "six_state_mechanistic_surrogate"
    prediction_rows.append(surrogate)
    definitions.append(
        {
            "model": "six_state_mechanistic_surrogate",
            "fit_seconds": np.nan,
            "predict_seconds": np.nan,
            "training_scenarios": ",".join(sorted(calibration_names)),
            "validation_scenarios": ",".join(active["validation_scenarios"]),
            "interpretation": "Six-state mechanistic low-order surrogate evaluated with the same benchmark rule.",
            "calibration_covariate_identifiability_limit": "See frozen surrogate parameter record.",
        }
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)

    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    definition_map = {row["model"]: row for row in definitions}
    for model_name, model_frame in predictions.groupby("model", sort=False):
        summary = confusion_summary(model_frame)
        for _, row in summary.iterrows():
            record = row.to_dict()
            record.update(definition_map[model_name])
            metric_rows.append(record)
        for scenario, group in model_frame.groupby("scenario", sort=False):
            predicted_threshold = persistent_threshold(
                group,
                kla_col="commanded_kla_d-1",
                compliant_col="predicted_satisfies_study_benchmark",
                persistent_points=persistent_points,
            )
            actual_threshold = persistent_threshold(
                group,
                kla_col="commanded_kla_d-1",
                compliant_col="bsm1_satisfies_study_benchmark",
                persistent_points=persistent_points,
            )
            threshold_rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "predicted_threshold_status": predicted_threshold.status,
                    "predicted_first_compliant_grid_point_d-1": predicted_threshold.first_compliant_grid_point,
                    "bsm1_threshold_status": actual_threshold.status,
                    "bsm1_first_compliant_grid_point_d-1": actual_threshold.first_compliant_grid_point,
                    "absolute_threshold_error_d-1": (
                        abs(float(predicted_threshold.first_compliant_grid_point) - float(actual_threshold.first_compliant_grid_point))
                        if predicted_threshold.first_compliant_grid_point is not None and actual_threshold.first_compliant_grid_point is not None
                        else np.nan
                    ),
                    "relative_threshold_error": (
                        abs(float(predicted_threshold.first_compliant_grid_point) - float(actual_threshold.first_compliant_grid_point))
                        / float(actual_threshold.first_compliant_grid_point)
                        if predicted_threshold.first_compliant_grid_point is not None
                        and actual_threshold.first_compliant_grid_point is not None
                        and float(actual_threshold.first_compliant_grid_point) != 0.0
                        else np.nan
                    ),
                    "grid_step_d-1": actual_threshold.grid_step,
                    "persistent_compliance_points": persistent_points,
                    "monotonicity_violations": _monotonicity_violations(group, "predicted_probability", increasing=True),
                    "is_independent_validation_scenario": scenario in active["validation_scenarios"],
                }
            )
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(threshold_rows)
