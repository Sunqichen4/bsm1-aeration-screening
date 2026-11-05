"""Continuous NH4 baselines fitted using development scenarios only.

The public prediction contract uses only L = load_coordinate and
z = oxygen_transfer_factor * kla.  No simulation outputs, compliance labels,
scenario names, or evaluation data enter the feature map.  Hyperparameters are
selected by an unweighted mean of three leave-one-scenario-out MSEs in raw
SNH_max_eval units.  All transformations are learned inside each training fold.

These are *continuous-supervision* baselines.  The historical M1 logistic
classifier used binary labels and must remain identified separately.
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


FEATURE_COLUMNS = ("load_coordinate", "oxygen_transfer_factor", "kla")
TARGET_COLUMN = "SNH_max_eval"
MODEL_NAMES = ("poly_ridge", "rbf_kernel_ridge")
POLY_GRID = tuple(
    {"degree": degree, "penalty": penalty}
    for degree, penalty in product((1, 2, 3), (1e-6, 1e-3, 1.0, 100.0))
)
RBF_GRID = tuple(
    {"length_scale": length, "penalty": penalty}
    for length, penalty in product((0.3, 1.0, 3.0), (1e-6, 1e-3, 1.0))
)


def _features(frame: pd.DataFrame) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not supported")
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    try:
        inputs = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All three input columns must be numeric") from exc
    if not np.isfinite(inputs).all():
        raise ValueError("Input columns must contain finite values")
    if np.any(inputs[:, 1] <= 0):
        raise ValueError("oxygen_transfer_factor must be strictly positive")
    if np.any(inputs[:, 2] < 0):
        raise ValueError("kla must be nonnegative")
    with np.errstate(over="ignore"):
        result = np.column_stack((inputs[:, 0], inputs[:, 1] * inputs[:, 2]))
    if not np.isfinite(result).all():
        raise ValueError("Effective oxygen transfer z must be finite")
    return result


def _scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    # A held-out load scenario can leave constant L in the training fold.
    # Unit scale preserves the coordinate without learning from validation.
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def _powers(degree: int) -> list[list[int]]:
    # No constant column: an unpenalized intercept is fitted by centering.
    return [[total - b, b] for total in range(1, degree + 1) for b in range(total + 1)]


def _polynomial(x: np.ndarray, powers: list[list[int]]) -> np.ndarray:
    return np.column_stack([x[:, 0] ** a * x[:, 1] ** b for a, b in powers])


def _fit_model(
    x: np.ndarray, y: np.ndarray, family: str, hyperparameters: dict[str, Any]
) -> dict[str, Any]:
    mean, scale = _scale_fit(x)
    standardized = (x - mean) / scale
    model: dict[str, Any] = {
        "schema_version": 1,
        "model_name": family,
        "feature_columns": list(FEATURE_COLUMNS),
        "derived_features": ["L = load_coordinate", "z = oxygen_transfer_factor * kla"],
        "target_column": TARGET_COLUMN,
        "target_units": "mg N/L",
        "supervision": "continuous raw NH4 peak; historical M1 logistic uses binary labels",
        "nonnegative_prediction_clip": True,
        "hyperparameters": dict(hyperparameters),
        "input_mean": mean.tolist(),
        "input_scale": scale.tolist(),
        "input_constant_columns": np.flatnonzero(np.std(x, axis=0) == 0.0).tolist(),
        "target_mean": float(np.mean(y)),
        "training_rows": len(x),
    }
    centered_y = y - model["target_mean"]
    if family == "poly_ridge":
        powers = _powers(int(hyperparameters["degree"]))
        design = _polynomial(standardized, powers)
        feature_mean, feature_scale = _scale_fit(design)
        design = (design - feature_mean) / feature_scale
        system = design.T @ design
        system.flat[:: system.shape[0] + 1] += hyperparameters["penalty"]
        weights = np.linalg.solve(system, design.T @ centered_y)
        model.update(
            polynomial_powers=powers,
            polynomial_mean=feature_mean.tolist(),
            polynomial_scale=feature_scale.tolist(),
            weights=weights.tolist(),
        )
    elif family == "rbf_kernel_ridge":
        kernel = np.exp(
            -cdist(standardized, standardized, "sqeuclidean")
            / (2.0 * hyperparameters["length_scale"] ** 2)
        )
        kernel.flat[:: len(kernel) + 1] += hyperparameters["penalty"]
        weights = cho_solve(cho_factor(kernel, lower=True, check_finite=False), centered_y)
        model.update(training_features_standardized=standardized.tolist(), weights=weights.tolist())
    else:
        raise ValueError(f"Unknown model family: {family}")
    return model


def _predict_raw(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    if model.get("schema_version") != 1 or model.get("model_name") not in MODEL_NAMES:
        raise ValueError("Unsupported statistical model schema or family")
    try:
        mean = np.asarray(model["input_mean"], dtype=float)
        scale = np.asarray(model["input_scale"], dtype=float)
        if mean.shape != (2,) or scale.shape != (2,) or np.any(scale <= 0):
            raise ValueError("Invalid input scaling parameters")
        standardized = (x - mean) / scale
        weights = np.asarray(model["weights"], dtype=float)
        if model["model_name"] == "poly_ridge":
            design = _polynomial(standardized, model["polynomial_powers"])
            design = (design - np.asarray(model["polynomial_mean"])) / np.asarray(
                model["polynomial_scale"]
            )
        else:
            training = np.asarray(model["training_features_standardized"], dtype=float)
            length = float(model["hyperparameters"]["length_scale"])
            if length <= 0:
                raise ValueError("length_scale must be positive")
            design = np.exp(-cdist(standardized, training, "sqeuclidean") / (2.0 * length**2))
        predicted = float(model["target_mean"]) + design @ weights
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid serialized statistical model: {exc}") from exc
    if predicted.shape != (len(x),) or not np.isfinite(predicted).all():
        raise ValueError("Serialized model produced nonfinite or malformed predictions")
    return predicted


def predict_baseline(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    """Predict NH4 peak from the three allowed inputs; clip at zero mg N/L."""
    return np.maximum(_predict_raw(_features(frame), model), 0.0)


def _write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_baselines(
    frame: pd.DataFrame, output_dir: Path, seed: int = 20261016
) -> dict[str, dict[str, Any]]:
    """Select and refit two continuous baselines on development data only.

    Exactly three named development scenarios are required.  The deterministic
    grid uses no randomness; seed is retained solely as experiment metadata.
    This function never opens a data file or consults an evaluation dataset.
    """
    started = perf_counter()
    x = _features(frame)
    if len(frame) == 0:
        raise ValueError("Development data must not be empty")
    if TARGET_COLUMN not in frame or "scenario" not in frame:
        raise ValueError("Development data require SNH_max_eval and scenario columns")
    try:
        y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("SNH_max_eval must be numeric") from exc
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("SNH_max_eval must be finite and nonnegative")
    if frame["scenario"].isna().any():
        raise ValueError("scenario must not contain missing values")
    groups = frame["scenario"].astype(str).to_numpy()
    scenarios = sorted(set(groups))
    if len(scenarios) != 3 or any(not group.strip() for group in scenarios):
        raise ValueError("Exactly three nonempty development scenarios are required")

    folds: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {}
    for family, grid in (("poly_ridge", POLY_GRID), ("rbf_kernel_ridge", RBF_GRID)):
        family_started = perf_counter()
        candidates = []
        for index, parameters in enumerate(grid):
            candidate_mses = []
            for scenario in scenarios:
                train = groups != scenario
                validation = ~train
                fit_started = perf_counter()
                fold_model = _fit_model(x[train], y[train], family, parameters)
                fit_seconds = perf_counter() - fit_started
                predict_started = perf_counter()
                raw = _predict_raw(x[validation], fold_model)
                prediction_seconds = perf_counter() - predict_started
                prediction = np.maximum(raw, 0.0)
                mse = float(np.mean((prediction - y[validation]) ** 2))
                candidate_mses.append(mse)
                folds.append({
                    "model_name": family,
                    "candidate_index": index,
                    "hyperparameters": json.dumps(parameters, sort_keys=True),
                    "held_out_scenario": scenario,
                    "training_scenarios": json.dumps([s for s in scenarios if s != scenario]),
                    "training_rows": int(train.sum()),
                    "validation_rows": int(validation.sum()),
                    "training_input_mean": json.dumps(fold_model["input_mean"]),
                    "training_input_scale": json.dumps(fold_model["input_scale"]),
                    "training_constant_columns": json.dumps(fold_model["input_constant_columns"]),
                    "training_target_mean": fold_model["target_mean"],
                    "mse_mgN_per_L_squared": mse,
                    "rmse_mgN_per_L": float(np.sqrt(mse)),
                    "mae_mgN_per_L": float(np.mean(np.abs(prediction - y[validation]))),
                    "clipped_prediction_count": int(np.sum(raw < 0)),
                    "minimum_raw_prediction": float(np.min(raw)),
                    "fit_seconds": fit_seconds,
                    "prediction_seconds": prediction_seconds,
                })
            candidates.append({
                "model_name": family,
                "candidate_index": index,
                "hyperparameters": dict(parameters),
                "mean_scenario_mse_mgN_per_L_squared": float(np.mean(candidate_mses)),
                "scenario_mses": dict(zip(scenarios, candidate_mses)),
            })
        # Grid order is the predefined deterministic tie breaker.
        best = min(candidates, key=lambda candidate: candidate["mean_scenario_mse_mgN_per_L_squared"])
        for candidate in candidates:
            candidate["selected"] = candidate["candidate_index"] == best["candidate_index"]
            selections.append(candidate)
        refit_started = perf_counter()
        model = _fit_model(x, y, family, best["hyperparameters"])
        model.update(
            seed=int(seed),
            training_scenarios=scenarios,
            selection_method="leave-one-development-scenario-out; unweighted mean scenario MSE after zero clip",
            selected_cv_mean_scenario_mse=best["mean_scenario_mse_mgN_per_L_squared"],
            selected_cv_scenario_mses=best["scenario_mses"],
            refit_seconds=perf_counter() - refit_started,
            selection_and_refit_seconds=perf_counter() - family_started,
        )
        train_raw = _predict_raw(x, model)
        model["training_clipped_prediction_count"] = int(np.sum(train_raw < 0))
        models[family] = model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(output_dir / "statistical_cv_folds.csv", index=False)
    _write_json(output_dir / "statistical_model_selection.json", selections)
    _write_json(output_dir / "statistical_models.json", models)
    _write_json(output_dir / "statistical_protocol.json", {
        "allowed_input_columns": list(FEATURE_COLUMNS),
        "derived_features": ["load_coordinate", "oxygen_transfer_factor * kla"],
        "continuous_target": TARGET_COLUMN,
        "target_transform": "none; raw mg N/L",
        "selection_metric": "unweighted mean of three scenario MSEs after nonnegative clipping",
        "scaling": "population mean/std fitted on training fold only; zero std replaced with 1",
        "polynomial_scaling": "standardize inputs, then standardize polynomial terms within training fold",
        "intercept": "polynomial: unpenalized using target/term centering; RBF: fixed training-target mean plus centered-target KRR",
        "kernel": "exp(-squared_standardized_distance / (2 * length_scale**2))",
        "ridge_objectives": {
            "poly_ridge": "sum squared residuals + penalty * squared coefficient norm",
            "rbf_kernel_ridge": "sum squared centered-target residuals + penalty * squared RKHS norm",
        },
        "polynomial_grid": list(POLY_GRID),
        "rbf_grid": list(RBF_GRID),
        "tie_break": "first in predefined grid",
        "nonnegative_prediction_clip": True,
        "clip_counts_recorded": "each validation fold and final training predictions",
        "development_scenarios": scenarios,
        "development_rows": len(frame),
        "evaluation_data_access": "none; function only consumes supplied development frame",
        "seed": int(seed),
        "randomness": "none",
        "historical_M1_logistic": "binary-supervision historical baseline, not equal continuous supervision",
        "total_seconds": perf_counter() - started,
    })
    return models
