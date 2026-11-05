from __future__ import annotations

import math
import numpy as np
import pandas as pd


CORE_COLUMNS = [
    "energy_kwh",
    "nh4_mean",
    "nh4_p95",
    "nh4_max",
    "violation_hours",
    "violation_ratio",
    "iae_do",
    "hard_safety_rate",
    "actuator_intervention_rate",
]


def mean_ci95(x: pd.Series) -> tuple[float, float, float, float]:
    """Return mean, std, 95% CI half-width, and n for a numeric series.

    The CI uses a normal 1.96 multiplier. With small n this is a transparent
    descriptive interval rather than a formal inferential test.
    """
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci, float(n)


def compute_metrics(df: pd.DataFrame, nh4_limit: float, do_target: float, main_only: bool = True) -> dict:
    sub = df[df["phase"].eq("main")].copy() if main_only else df.copy()
    if sub.empty:
        raise ValueError("No rows available for metric calculation.")
    dt_h = float(sub["dt_h"].iloc[0])
    nh4 = sub["true_SNH_pre"].to_numpy(dtype=float)
    do = sub["true_SO_pre"].to_numpy(dtype=float)
    exceed = nh4 > nh4_limit
    return {
        "energy_kwh": float(sub["energy_kwh"].sum()),
        "nh4_mean": float(np.mean(nh4)),
        "nh4_p95": float(np.quantile(nh4, 0.95)),
        "nh4_max": float(np.max(nh4)),
        "violation_hours": float(np.sum(exceed) * dt_h),
        "violation_ratio": float(np.mean(exceed)),
        "iae_do": float(np.sum(np.abs(do - do_target)) * dt_h),
        "hard_safety_rate": float(sub["hard_safety"].mean()),
        "actuator_intervention_rate": float(sub["actuator_intervention"].mean()),
        "n_main_steps": int(len(sub)),
    }


def summarize_runs(run_rows: list[dict]) -> pd.DataFrame:
    """Summarize run-level metrics by scenario and controller.

    The output includes mean/std and 95% confidence-interval half-width columns.
    These CI columns are useful for paper figures and make the multi-seed
    uncertainty explicit instead of only reporting single deterministic values.
    """
    df = pd.DataFrame(run_rows)
    group_cols = ["scenario", "controller"]
    metric_cols = [c for c in CORE_COLUMNS if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols):
        row = {"scenario": keys[0], "controller": keys[1], "n_runs": int(len(g))}
        for c in metric_cols:
            mean, std, ci, _ = mean_ci95(g[c])
            row[f"{c}_mean"] = mean
            row[f"{c}_std"] = std
            row[f"{c}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)
