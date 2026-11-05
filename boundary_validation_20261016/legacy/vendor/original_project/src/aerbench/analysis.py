from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import mean_ci95


def finite_nanmean(series: pd.Series) -> float:
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def equal_energy_table(run_df: pd.DataFrame, reference_controller: str = "ABAC-PID", tolerance: float = 0.03) -> pd.DataFrame:
    """Compare risk at similar energy levels within each scenario and seed.

    A row is kept when a controller's energy is within ±tolerance of the reference
    controller for the same scenario and seed. This table prevents the paper from
    claiming a controller is better simply because it used much more aeration.
    """
    rows = []
    for (scenario, seed), g in run_df.groupby(["scenario", "seed"]):
        ref = g[g["controller"].eq(reference_controller)]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        ref_energy = float(ref["energy_kwh"])
        lo, hi = ref_energy * (1 - tolerance), ref_energy * (1 + tolerance)
        for _, row in g.iterrows():
            if lo <= row["energy_kwh"] <= hi:
                rows.append({
                    "scenario": scenario,
                    "seed": int(seed),
                    "reference_controller": reference_controller,
                    "reference_energy_kwh": ref_energy,
                    "reference_violation_hours": float(ref["violation_hours"]),
                    "reference_nh4_p95": float(ref["nh4_p95"]),
                    "tolerance_fraction": tolerance,
                    "controller": row["controller"],
                    "energy_kwh": float(row["energy_kwh"]),
                    "violation_hours": float(row["violation_hours"]),
                    "nh4_p95": float(row["nh4_p95"]),
                    "delta_energy_kwh_vs_reference": float(row["energy_kwh"] - ref_energy),
                    "delta_violation_hours_vs_reference": float(row["violation_hours"] - ref["violation_hours"]),
                    "delta_nh4_p95_vs_reference": float(row["nh4_p95"] - ref["nh4_p95"]),
                })
    return pd.DataFrame(rows)


def equal_energy_summary(equal_energy_df: pd.DataFrame) -> pd.DataFrame:
    if equal_energy_df.empty:
        return equal_energy_df
    rows = []
    for (scenario, controller), g in equal_energy_df.groupby(["scenario", "controller"]):
        row = {"scenario": scenario, "controller": controller, "n_matched_runs": int(len(g))}
        for col in ["delta_energy_kwh_vs_reference", "delta_violation_hours_vs_reference", "delta_nh4_p95_vs_reference"]:
            mean, std, ci, _ = mean_ci95(g[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "controller"])


def equal_risk_table(run_df: pd.DataFrame, reference_controller: str = "ABAC-PID", slack_hours: float = 0.25) -> pd.DataFrame:
    """Find the lowest-energy controller meeting the same risk as a reference.

    For each scenario and seed, the risk target is the reference controller's
    violation duration plus a small slack. This gives a fairer answer than a
    single arbitrary global risk threshold.
    """
    rows = []
    for (scenario, seed), g in run_df.groupby(["scenario", "seed"]):
        ref = g[g["controller"].eq(reference_controller)]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        target = float(ref["violation_hours"] + slack_hours)
        feasible = g[g["violation_hours"] <= target].copy()
        if feasible.empty:
            rows.append({
                "scenario": scenario,
                "seed": int(seed),
                "reference_controller": reference_controller,
                "risk_target_hours": target,
                "best_controller": None,
                "best_energy_kwh": np.nan,
                "best_violation_hours": np.nan,
                "reference_energy_kwh": float(ref["energy_kwh"]),
                "reference_violation_hours": float(ref["violation_hours"]),
                "delta_energy_kwh_vs_reference": np.nan,
            })
            continue
        best = feasible.sort_values(["energy_kwh", "violation_hours"]).iloc[0]
        rows.append({
            "scenario": scenario,
            "seed": int(seed),
            "reference_controller": reference_controller,
            "risk_target_hours": target,
            "best_controller": best["controller"],
            "best_energy_kwh": float(best["energy_kwh"]),
            "best_violation_hours": float(best["violation_hours"]),
            "reference_energy_kwh": float(ref["energy_kwh"]),
            "reference_violation_hours": float(ref["violation_hours"]),
            "delta_energy_kwh_vs_reference": float(best["energy_kwh"] - ref["energy_kwh"]),
        })
    return pd.DataFrame(rows)


def equal_risk_summary(equal_risk_df: pd.DataFrame) -> pd.DataFrame:
    if equal_risk_df.empty:
        return equal_risk_df
    rows = []
    for (scenario, best_controller), g in equal_risk_df.groupby(["scenario", "best_controller"], dropna=False):
        row = {"scenario": scenario, "best_controller": best_controller, "n_runs": int(len(g))}
        for col in ["best_energy_kwh", "best_violation_hours", "delta_energy_kwh_vs_reference"]:
            mean, std, ci, _ = mean_ci95(g[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "best_controller"])


def riskcue_marginal_table(run_df: pd.DataFrame, base: str = "MPC", cue: str = "Risk-cue-MPC") -> pd.DataFrame:
    rows = []
    for (scenario, seed), g in run_df.groupby(["scenario", "seed"]):
        b = g[g["controller"].eq(base)]
        c = g[g["controller"].eq(cue)]
        if b.empty or c.empty:
            continue
        b = b.iloc[0]
        c = c.iloc[0]
        d_energy = float(c["energy_kwh"] - b["energy_kwh"])
        d_v = float(c["violation_hours"] - b["violation_hours"])
        rows.append({
            "scenario": scenario,
            "seed": int(seed),
            "base_controller": base,
            "cue_controller": cue,
            "delta_energy_kwh": d_energy,
            "delta_violation_hours": d_v,
            "delta_nh4_p95": float(c["nh4_p95"] - b["nh4_p95"]),
            "energy_per_violation_hour_reduced": d_energy / abs(d_v) if d_v < 0 else np.inf,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out_rows = []
    for scenario, g in df.groupby("scenario"):
        row = {"scenario": scenario, "n_runs": int(len(g))}
        for col in ["delta_energy_kwh", "delta_violation_hours", "delta_nh4_p95"]:
            mean, std, ci, _ = mean_ci95(g[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci
        row["energy_per_violation_hour_reduced_mean"] = finite_nanmean(g["energy_per_violation_hour_reduced"])
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("scenario")


def controller_ranking_table(run_df: pd.DataFrame) -> pd.DataFrame:
    """Rank controllers within each scenario/seed and summarize rank positions."""
    tmp = run_df.copy()
    tmp["risk_rank"] = tmp.groupby(["scenario", "seed"])["violation_hours"].rank(method="min")
    tmp["energy_rank"] = tmp.groupby(["scenario", "seed"])["energy_kwh"].rank(method="min")
    tmp["balanced_rank"] = tmp["risk_rank"] + tmp["energy_rank"]
    rows = []
    for (scenario, controller), g in tmp.groupby(["scenario", "controller"]):
        rows.append({
            "scenario": scenario,
            "controller": controller,
            "risk_rank_mean": float(g["risk_rank"].mean()),
            "energy_rank_mean": float(g["energy_rank"].mean()),
            "balanced_rank_mean": float(g["balanced_rank"].mean()),
            "n_runs": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values(["scenario", "balanced_rank_mean"])


def write_analysis_tables(output_dir: Path = Path("outputs")) -> None:
    table_dir = output_dir / "tables"
    run_path = table_dir / "run_level_metrics.csv"
    if not run_path.exists():
        raise FileNotFoundError(f"Missing {run_path}. Run the benchmark first.")
    run_df = pd.read_csv(run_path)
    ee = equal_energy_table(run_df)
    er = equal_risk_table(run_df)
    ee.to_csv(table_dir / "equal_energy_comparison.csv", index=False)
    equal_energy_summary(ee).to_csv(table_dir / "equal_energy_summary.csv", index=False)
    er.to_csv(table_dir / "equal_risk_comparison.csv", index=False)
    equal_risk_summary(er).to_csv(table_dir / "equal_risk_summary.csv", index=False)
    riskcue_marginal_table(run_df).to_csv(table_dir / "riskcue_marginal_value.csv", index=False)
    controller_ranking_table(run_df).to_csv(table_dir / "controller_ranking.csv", index=False)
    warmup = run_df.groupby(["scenario", "controller"], as_index=False).agg(
        warmup_bias_energy_kwh=("warmup_bias_energy_kwh", "mean"),
        warmup_bias_violation_hours=("warmup_bias_violation_hours", "mean"),
    )
    warmup.to_csv(table_dir / "warmup_contamination_bias.csv", index=False)
