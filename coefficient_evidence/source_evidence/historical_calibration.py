from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerbench.classification import persistent_threshold
from aerbench.configuration import CONFIG_DIR, load_json, load_surrogate_params, load_workflow_config
from aerbench.fixed_aeration import run_surrogate_fixed_aeration
from aerbench.provenance import save_run_provenance


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_calibration_rows(bsm1: pd.DataFrame, n_points: int) -> pd.DataFrame:
    frame = bsm1[bsm1["scenario"].eq("baseline")].sort_values("kla").reset_index(drop=True)
    if len(frame) < n_points:
        raise ValueError(f"Baseline calibration requested {n_points} points but only {len(frame)} are available.")
    positions = np.unique(np.round(np.linspace(0, len(frame) - 1, n_points)).astype(int))
    selected = frame.iloc[positions].copy()
    if len(selected) != n_points:
        raise RuntimeError("Calibration-point selection did not produce the requested unique count.")
    return selected


def evaluate_parameter_vector(
    vector: np.ndarray,
    *,
    parameter_names: list[str],
    base_params,
    calibration: pd.DataFrame,
    workflow: dict,
) -> pd.DataFrame:
    params = replace(base_params, **{name: float(value) for name, value in zip(parameter_names, vector)})
    simulation = workflow["simulation"]
    threshold = float(workflow["benchmark"]["nh4_threshold_mgN_L"])
    rows = []
    for row in calibration.itertuples(index=False):
        result = run_surrogate_fixed_aeration(
            scenario_name="baseline",
            kla=float(row.kla),
            warmup_days=float(simulation["warmup_days"]),
            eval_days=float(simulation["evaluation_days"]),
            dt_minutes=float(simulation["surrogate_timestep_min"]),
            nh4_limit=threshold,
            oxygen_transfer_factor=0.70,
            params=params,
        )
        rows.append(
            {
                "scenario": "baseline",
                "kla": float(row.kla),
                "surrogate_SNH_mean_eval": result["SNH_mean_eval"],
                "surrogate_SNH_violation_hours_gt_limit": result["SNH_violation_hours_gt_limit"],
                "surrogate_zero_violation": result["SNH_zero_violation"],
                "bsm1_SNH_mean_eval": float(row.SNH_mean_eval),
                "bsm1_SNH_violation_hours_gt_limit": float(row.SNH_violation_hours_gt_limit),
                "bsm1_zero_violation": float(row.SNH_violation_hours_gt_limit) <= 1e-12,
            }
        )
    return pd.DataFrame(rows)


def calibration_residuals(
    comparison: pd.DataFrame,
    persistent_points: int,
    *,
    mean_snh_scale_mgN_L: float,
    violation_scale_h: float,
    threshold_scale_d_1: float,
) -> tuple[np.ndarray, dict]:
    mean_scale = float(mean_snh_scale_mgN_L)
    violation_scale = float(violation_scale_h)
    threshold_scale = float(threshold_scale_d_1)
    if min(mean_scale, violation_scale, threshold_scale) <= 0.0:
        raise ValueError("All calibration objective scales must be positive.")
    mean_residual = (
        comparison["surrogate_SNH_mean_eval"].to_numpy(float)
        - comparison["bsm1_SNH_mean_eval"].to_numpy(float)
    ) / mean_scale
    violation_residual = (
        comparison["surrogate_SNH_violation_hours_gt_limit"].to_numpy(float)
        - comparison["bsm1_SNH_violation_hours_gt_limit"].to_numpy(float)
    ) / violation_scale
    surrogate_threshold = persistent_threshold(
        comparison,
        kla_col="kla",
        compliant_col="surrogate_zero_violation",
        persistent_points=persistent_points,
    )
    bsm1_threshold = persistent_threshold(
        comparison,
        kla_col="kla",
        compliant_col="bsm1_zero_violation",
        persistent_points=persistent_points,
    )
    if surrogate_threshold.first_compliant_grid_point is None or bsm1_threshold.first_compliant_grid_point is None:
        threshold_residual = 4.0
        threshold_error = np.nan
    else:
        threshold_error = float(
            surrogate_threshold.first_compliant_grid_point - bsm1_threshold.first_compliant_grid_point
        )
        threshold_residual = threshold_error / threshold_scale
    # Divide pointwise residuals by sqrt(n), so their squared sums equal the
    # normalized MSE for each response family. The three objective families
    # therefore have equal preregistered weights despite different point counts.
    root_n = np.sqrt(len(comparison))
    residuals = np.concatenate(
        [mean_residual / root_n, violation_residual / root_n, [threshold_residual]]
    )
    metrics = {
        "rmse_mean_snh_mgN_L": float(np.sqrt(np.mean((mean_residual * mean_scale) ** 2))),
        "rmse_violation_h": float(np.sqrt(np.mean((violation_residual * violation_scale) ** 2))),
        "threshold_error_d-1": threshold_error,
        "scaled_sum_of_squares": float(np.sum(residuals * residuals)),
        "surrogate_threshold_status": surrogate_threshold.status,
        "bsm1_threshold_status": bsm1_threshold.status,
    }
    return residuals, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate selected surrogate parameters using baseline BSM1 data only.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--workflow-config", type=Path, default=CONFIG_DIR / "eti_workflow.json")
    parser.add_argument("--parameter-config", type=Path, default=CONFIG_DIR / "surrogate_parameters.json")
    parser.add_argument("--n-calibration-points", type=int, default=None)
    parser.add_argument("--multistart", type=int, default=None)
    parser.add_argument("--max-nfev", type=int, default=None)
    parser.add_argument("--activate", action="store_true", help="Replace the active parameter JSON after saving a traceable fitted copy.")
    args = parser.parse_args()

    workflow = load_workflow_config(args.workflow_config)
    record = load_json(args.parameter_config)
    protocol = record["calibration_protocol"]
    parameter_names = list(protocol["parameter_names"])
    n_points = int(args.n_calibration_points or protocol["n_calibration_points"])
    n_starts = int(args.multistart or protocol["multistart"])
    max_nfev = int(args.max_nfev or protocol["max_nfev_per_start"])
    if n_starts < 1 or max_nfev < 1:
        raise ValueError("multistart and max_nfev must be positive.")

    bsm1_path = args.results_dir / "full_grid_bsm1_results.csv"
    if not bsm1_path.exists():
        raise FileNotFoundError("Run the BSM1 full-grid stage before baseline-only calibration.")
    bsm1 = pd.read_csv(bsm1_path)
    required = {"scenario", "kla", "SNH_mean_eval", "SNH_violation_hours_gt_limit"}
    missing = sorted(required.difference(bsm1.columns))
    if missing:
        raise ValueError(f"Full-grid BSM1 table lacks calibration columns: {missing}")
    unexpected_scenarios = sorted(set(bsm1["scenario"]) - {"baseline"})
    if unexpected_scenarios:
        raise RuntimeError(
            "Baseline calibration must occur before any independent-validation BSM1 results are generated. "
            f"Found validation scenarios in the source file: {unexpected_scenarios}. Use a new results directory "
            "or retain the already-frozen parameters rather than recalibrating on a completed validation run."
        )
    calibration = select_calibration_rows(bsm1, n_points=n_points)
    if not calibration["scenario"].eq("baseline").all():
        raise RuntimeError("Validation scenarios must never enter the calibration dataset.")

    base_params = load_surrogate_params(args.parameter_config)
    initial = np.asarray([float(record["parameters"][name]["value"]) for name in parameter_names])
    lower = np.asarray([float(record["parameters"][name]["calibration_bounds"][0]) for name in parameter_names])
    upper = np.asarray([float(record["parameters"][name]["calibration_bounds"][1]) for name in parameter_names])
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("The active initial parameter vector lies outside its preregistered bounds.")

    seed = int(workflow["random_seed"])
    rng = np.random.default_rng(seed)
    starts = [initial]
    for _ in range(1, n_starts):
        jitter = rng.normal(0.0, 0.10, size=len(initial))
        starts.append(np.clip(initial * (1.0 + jitter), lower, upper))
    cache: dict[tuple[float, ...], tuple[np.ndarray, pd.DataFrame, dict]] = {}

    def objective(vector: np.ndarray) -> np.ndarray:
        key = tuple(np.round(vector, 12))
        if key not in cache:
            comparison = evaluate_parameter_vector(
                vector,
                parameter_names=parameter_names,
                base_params=base_params,
                calibration=calibration,
                workflow=workflow,
            )
            residuals, metrics = calibration_residuals(
                comparison,
                persistent_points=int(workflow["candidate_grid"]["persistent_compliance_points"]),
                mean_snh_scale_mgN_L=float(protocol["objective_scales"]["mean_snh_mgN_L"]),
                violation_scale_h=float(protocol["objective_scales"]["violation_h"]),
                threshold_scale_d_1=float(protocol["objective_scales"]["threshold_kla_d-1"]),
            )
            cache[key] = residuals, comparison, metrics
        return cache[key][0]

    optimizer_rows = []
    fitted = []
    for start_index, start in enumerate(starts, start=1):
        result = least_squares(
            objective,
            x0=start,
            bounds=(lower, upper),
            method="trf",
            ftol=float(protocol["ftol"]),
            xtol=float(protocol["xtol"]),
            gtol=float(protocol["gtol"]),
            max_nfev=max_nfev,
            x_scale="jac",
        )
        objective(result.x)
        _, comparison, metrics = cache[tuple(np.round(result.x, 12))]
        fitted.append((float(np.sum(result.fun * result.fun)), result, comparison, metrics))
        optimizer_rows.append(
            {
                "start_index": start_index,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "cost": float(result.cost),
                **metrics,
                **{f"initial_{name}": float(value) for name, value in zip(parameter_names, start)},
                **{f"fitted_{name}": float(value) for name, value in zip(parameter_names, result.x)},
            }
        )
    fitted.sort(key=lambda item: item[0])
    _, best, comparison, best_metrics = fitted[0]

    output_dir = args.results_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_dir / "baseline_calibration_bsm1_points.csv", index=False)
    comparison.to_csv(output_dir / "baseline_calibration_fit_comparison.csv", index=False)
    pd.DataFrame(optimizer_rows).to_csv(output_dir / "calibration_optimizer_runs.csv", index=False)
    fitted_record = json.loads(json.dumps(record))
    for name, value in zip(parameter_names, best.x):
        fitted_record["parameters"][name]["value"] = float(value)
        fitted_record["parameters"][name]["calibrated_or_fixed"] = "calibrated"
        fitted_record["parameters"][name]["calibration_dataset"] = "baseline BSM1 only"
    # Keep the top-level provenance consistent with the frozen fitted values.
    fitted_record["calibration_dataset"] = "baseline BSM1 only (25-point calibration set)"
    fitted_record["parameter_set_id"] = "eti_baseline_frozen_after_reproducible_calibration"
    fitted_record["frozen_before_validation"] = True
    fitted_record["calibration_run"] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "scipy.optimize.least_squares trust-region reflective",
        "objective_components": ["scaled RMSE mean SNH", "scaled RMSE violation duration", "scaled absolute threshold error"],
        "weights": [1.0, 1.0, 1.0],
        "objective_scales": protocol["objective_scales"],
        "parameter_names": parameter_names,
        "n_calibration_points": n_points,
        "calibration_scenarios": ["baseline"],
        "excluded_validation_scenarios": workflow["validation_scenarios"],
        "multistart": n_starts,
        "max_nfev_per_start": max_nfev,
        "random_seed": seed,
        "ftol": float(protocol["ftol"]),
        "xtol": float(protocol["xtol"]),
        "gtol": float(protocol["gtol"]),
        "selected_start_index": int(pd.DataFrame(optimizer_rows)["scaled_sum_of_squares"].idxmin() + 1),
        "best_metrics": best_metrics,
        "source_bsm1_file": str(bsm1_path),
        "source_bsm1_sha256": file_hash(bsm1_path),
        "validation_results_used_for_tuning": False,
    }
    fitted_path = output_dir / "surrogate_parameters_fitted_frozen.json"
    fitted_path.write_text(json.dumps(fitted_record, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.activate:
        args.parameter_config.write_text(json.dumps(fitted_record, ensure_ascii=False, indent=2), encoding="utf-8")

    save_run_provenance(
        project_root=ROOT,
        output_dir=output_dir / "manifest",
        command=sys.argv,
        config_files=[args.workflow_config, args.parameter_config],
        output_files=[
            output_dir / "baseline_calibration_bsm1_points.csv",
            output_dir / "baseline_calibration_fit_comparison.csv",
            output_dir / "calibration_optimizer_runs.csv",
            fitted_path,
        ],
    )


if __name__ == "__main__":
    main()
