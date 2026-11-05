from __future__ import annotations

from dataclasses import asdict, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .classification import persistent_threshold, wilson_interval
from .configuration import load_surrogate_parameter_record, load_surrogate_params
from .fixed_aeration import run_surrogate_fixed_aeration


def latin_hypercube_parameter_samples(
    *,
    parameter_names: Sequence[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    if n_samples < 2:
        raise ValueError("Use at least two uncertainty samples.")
    record = load_surrogate_parameter_record()
    metadata = record["parameters"]
    lows = []
    highs = []
    for name in parameter_names:
        if name not in metadata:
            raise ValueError(f"Unknown uncertainty parameter: {name}")
        bounds = metadata[name].get("sensitivity_range")
        if not bounds or len(bounds) != 2:
            raise ValueError(f"Parameter {name} has no two-value sensitivity range.")
        low, high = map(float, bounds)
        if high <= low:
            raise ValueError(f"Invalid uncertainty range for {name}: {bounds}")
        lows.append(low)
        highs.append(high)
    sampler = qmc.LatinHypercube(d=len(parameter_names), seed=int(seed))
    unit = sampler.random(n=int(n_samples))
    values = qmc.scale(unit, np.asarray(lows), np.asarray(highs))
    frame = pd.DataFrame(values, columns=parameter_names)
    frame.insert(0, "sample_id", [f"U{index:05d}" for index in range(1, len(frame) + 1)])
    frame.insert(1, "random_seed", int(seed))
    frame.insert(2, "sampling_method", "latin_hypercube")
    return frame


def run_uncertainty_propagation(
    *,
    samples: pd.DataFrame,
    scenarios: Sequence[str],
    kla_values: Sequence[float],
    warmup_days: float,
    evaluation_days: float,
    timestep_min: float,
    nh4_threshold_mgN_L: float,
    oxygen_transfer_factor: float,
    persistent_points: int,
    bsm1_full_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = load_surrogate_params()
    parameter_names = [
        column for column in samples.columns if column not in {"sample_id", "random_seed", "sampling_method"}
    ]
    run_rows: list[dict] = []
    threshold_rows: list[dict] = []
    sample_records = samples.to_dict(orient="records")
    total_samples = len(sample_records)
    for sample_index, sample in enumerate(sample_records, start=1):
        updates = {name: float(sample[name]) for name in parameter_names}
        params = replace(base, **updates)
        for scenario in scenarios:
            scenario_rows = []
            for kla in kla_values:
                row = run_surrogate_fixed_aeration(
                    scenario_name=scenario,
                    kla=float(kla),
                    warmup_days=warmup_days,
                    eval_days=evaluation_days,
                    dt_minutes=timestep_min,
                    nh4_limit=nh4_threshold_mgN_L,
                    oxygen_transfer_factor=oxygen_transfer_factor,
                    params=params,
                )
                result = {
                    "sample_id": sample["sample_id"],
                    "scenario": scenario,
                    "commanded_kla_d-1": float(kla),
                    "SNH_mean_eval": row["SNH_mean_eval"],
                    "SNH_p95_eval": row["SNH_p95_eval"],
                    "SNH_violation_hours_gt_limit": row["SNH_violation_hours_gt_limit"],
                    "surrogate_satisfies_study_benchmark": bool(row["SNH_zero_violation"]),
                }
                run_rows.append(result)
                scenario_rows.append(result)
            threshold = persistent_threshold(
                pd.DataFrame(scenario_rows),
                kla_col="commanded_kla_d-1",
                compliant_col="surrogate_satisfies_study_benchmark",
                persistent_points=persistent_points,
            )
            threshold_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "scenario": scenario,
                    "threshold_status": threshold.status,
                    "first_compliant_grid_point_d-1": threshold.first_compliant_grid_point,
                    "threshold_interval_lower_d-1": threshold.interval_lower,
                    "threshold_interval_upper_d-1": threshold.interval_upper,
                    "grid_step_d-1": threshold.grid_step,
                }
            )
        print(
            f"Uncertainty sample {sample_index}/{total_samples} completed "
            f"({sample['sample_id']}).",
            flush=True,
        )
    runs = pd.DataFrame(run_rows)
    thresholds = pd.DataFrame(threshold_rows)

    bsm1 = bsm1_full_grid.copy()
    bsm1["commanded_kla_d-1"] = pd.to_numeric(
        bsm1["commanded_kla_d-1"] if "commanded_kla_d-1" in bsm1 else bsm1["kla"], errors="raise"
    )
    bsm1["bsm1_satisfies_study_benchmark"] = (
        pd.to_numeric(bsm1["SNH_violation_hours_gt_limit"], errors="raise") <= 1e-12
    )
    # Match KLa values after numeric normalization. Values written by
    # different CSV/NumPy paths can differ by floating-point round-off even
    # though they represent the same grid candidate.
    merge_runs = runs.copy()
    merge_runs["_kla_match_key"] = pd.to_numeric(
        merge_runs["commanded_kla_d-1"], errors="raise"
    ).round(9)

    bsm1_match = bsm1[
        ["scenario", "commanded_kla_d-1", "bsm1_satisfies_study_benchmark"]
    ].copy()
    bsm1_match["_kla_match_key"] = pd.to_numeric(
        bsm1_match["commanded_kla_d-1"], errors="raise"
    ).round(9)
    if bsm1_match.duplicated(["scenario", "_kla_match_key"]).any():
        raise ValueError(
            "The BSM1 full grid contains duplicate scenario/KLa candidates "
            "after numeric normalization."
        )

    merged = merge_runs.merge(
        bsm1_match[
            ["scenario", "_kla_match_key", "bsm1_satisfies_study_benchmark"]
        ],
        on=["scenario", "_kla_match_key"],
        how="left",
        validate="many_to_one",
    ).drop(columns="_kla_match_key")
    if merged["bsm1_satisfies_study_benchmark"].isna().any():
        raise ValueError("The BSM1 full grid does not cover every uncertainty candidate.")
    merged["false_safe"] = (
        merged["surrogate_satisfies_study_benchmark"]
        & ~merged["bsm1_satisfies_study_benchmark"].astype(bool)
    )
    probability_rows = []
    for (scenario, kla), group in merged.groupby(["scenario", "commanded_kla_d-1"], sort=False):
        failures = int(group["false_safe"].sum())
        n = int(len(group))
        low, high = wilson_interval(failures, n)
        probability_rows.append(
            {
                "scenario": scenario,
                "commanded_kla_d-1": kla,
                "n_samples": n,
                "false_safe_count": failures,
                "false_safe_probability": failures / n,
                "false_safe_probability_wilson_low": low,
                "false_safe_probability_wilson_high": high,
            }
        )

    bsm1_threshold_rows = []
    for scenario, group in bsm1.groupby("scenario", sort=False):
        result = persistent_threshold(
            group,
            kla_col="commanded_kla_d-1",
            compliant_col="bsm1_satisfies_study_benchmark",
            persistent_points=persistent_points,
        )
        bsm1_threshold_rows.append(
            {"scenario": scenario, "bsm1_first_compliant_grid_point_d-1": result.first_compliant_grid_point}
        )
    bsm1_thresholds = pd.DataFrame(bsm1_threshold_rows)
    margin_source = thresholds.merge(bsm1_thresholds, on="scenario", how="left")
    margin_source["threshold_underprediction_d-1"] = (
        margin_source["bsm1_first_compliant_grid_point_d-1"]
        - margin_source["first_compliant_grid_point_d-1"]
    )
    margin_rows = []
    for scenario, group in margin_source.groupby("scenario", sort=False):
        resolved = pd.to_numeric(group["threshold_underprediction_d-1"], errors="coerce").dropna()
        unresolved_fraction = float(group["first_compliant_grid_point_d-1"].isna().mean())
        margin_rows.append(
            {
                "scenario": scenario,
                "n_uncertainty_samples": int(len(group)),
                "unresolved_transition_fraction": unresolved_fraction,
                "q95_positive_threshold_underprediction_d-1": (
                    max(0.0, float(np.quantile(resolved, 0.95))) if len(resolved) else np.nan
                ),
                "maximum_positive_threshold_underprediction_d-1": (
                    max(0.0, float(np.max(resolved))) if len(resolved) else np.nan
                ),
                "recommendation": (
                    "mandatory_bsm1_recheck" if unresolved_fraction > 0.0 else "use_pre_registered_or_q95_margin_then_revalidate"
                ),
            }
        )
    return thresholds, pd.DataFrame(probability_rows), pd.DataFrame(margin_rows), runs
