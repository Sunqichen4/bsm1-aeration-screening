from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .config import BenchmarkConfig
from .controllers import (
    ABACConfig,
    ABACPIDController,
    BaseController,
    DOPIDController,
    HybridRiskPIDController,
    PIDConfig,
    default_controllers,
)
from .simulation import run_single
from .utils import ensure_dirs, read_json, write_json


def is_pareto_efficient(df: pd.DataFrame, energy_col: str = "energy_kwh_mean", risk_col: str = "violation_hours_mean") -> pd.Series:
    efficient = []
    for _, row in df.iterrows():
        dominated = (
            (df[energy_col] <= row[energy_col])
            & (df[risk_col] <= row[risk_col])
            & ((df[energy_col] < row[energy_col]) | (df[risk_col] < row[risk_col]))
        ).any()
        efficient.append(not bool(dominated))
    return pd.Series(efficient, index=df.index)


def _tuning_scenarios_for_preset(preset: str) -> list[str]:
    if preset == "quick":
        return ["baseline"]
    return ["baseline", "high_load", "low_temperature"]


def _seeds_and_days_for_preset(preset: str) -> tuple[list[int], float]:
    if preset == "quick":
        return [0], 6.0
    return [0, 1, 2], 10.0


def _pid_grid(preset: str) -> Iterable[tuple[float, float, float]]:
    if preset == "quick":
        return product([45.0, 75.0], [3.0, 6.0], [2.0, 2.5])
    return product([35.0, 55.0, 75.0, 95.0], [3.0, 6.0], [1.5, 2.0, 2.5])


def _abac_grid(preset: str) -> Iterable[tuple[float, float, float, float, float]]:
    if preset == "quick":
        return product([55.0, 85.0], [4.0, 8.0], [1.8], [0.18, 0.30], [3.2])
    return product([45.0, 65.0, 85.0, 110.0], [3.0, 6.0, 10.0], [1.6, 2.0], [0.18, 0.30], [3.2])


def _score_and_select(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pareto_efficient"] = is_pareto_efficient(out)
    for fam, idx in out.groupby("controller_family").groups.items():
        fam_idx = list(idx)
        fam_df = out.loc[fam_idx]
        pareto = fam_df[fam_df["pareto_efficient"]]
        base = pareto if not pareto.empty else fam_df
        for col in ["energy_kwh_mean", "violation_hours_mean", "nh4_p95_mean"]:
            span = base[col].max() - base[col].min()
            if span == 0:
                out.loc[fam_idx, f"{col}_norm"] = 0.0
            else:
                out.loc[fam_idx, f"{col}_norm"] = (out.loc[fam_idx, col] - base[col].min()) / span
    out["selection_score"] = out["energy_kwh_mean_norm"] + 1.5 * out["violation_hours_mean_norm"] + 0.5 * out["nh4_p95_mean_norm"]
    out = out.sort_values(["controller_family", "pareto_efficient", "selection_score"], ascending=[True, False, True])
    return out


def tune_conventional_baselines(
    output_dir: Path = Path("outputs"),
    preset: str = "quick",
    tuning_scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Parameter sweep for stronger DO-PID and ABAC-PID baselines.

    This addresses the editor-style criticism that a weak PID/ABAC result may be
    a tuning artifact. The paper preset evaluates candidates on baseline,
    high-load, and low-temperature conditions, then saves a transparent selected
    parameter file that can be loaded by the benchmark script.
    """
    ensure_dirs(output_dir)
    seeds, days = _seeds_and_days_for_preset(preset)
    tuning_scenarios = tuning_scenarios or _tuning_scenarios_for_preset(preset)
    cfg = BenchmarkConfig(
        days=days,
        seeds=seeds,
        scenarios=tuning_scenarios,
        output_dir=output_dir,
        save_timeseries=False,
        use_risk_cue=False,
    )

    combo_rows: List[dict] = []
    scenario_rows: List[dict] = []

    combo_id = 0
    for kp, ki, do_sp in _pid_grid(preset):
        combo_id += 1
        ctrl = DOPIDController(PIDConfig(kp=kp, ki=ki, do_sp=do_sp))
        run_rows = []
        for scenario in tuning_scenarios:
            for seed in seeds:
                _, m = run_single("DO-PID-grid", ctrl, scenario, seed, cfg)
                m.update({"combo_id": combo_id})
                run_rows.append(m)
        tmp = pd.DataFrame(run_rows)
        for scenario, g in tmp.groupby("scenario"):
            scenario_rows.append({
                "combo_id": combo_id,
                "controller_family": "DO-PID",
                "scenario": scenario,
                "kp": kp,
                "ki": ki,
                "kd": 0.0,
                "do_sp": do_sp,
                "do_base": None,
                "do_sp_max": None,
                "abac_gain": None,
                "nh4_ref": None,
                "energy_kwh_mean": g["energy_kwh"].mean(),
                "violation_hours_mean": g["violation_hours"].mean(),
                "nh4_p95_mean": g["nh4_p95"].mean(),
                "hard_safety_rate_mean": g["hard_safety_rate"].mean(),
            })
        combo_rows.append({
            "combo_id": combo_id,
            "controller_family": "DO-PID",
            "kp": kp,
            "ki": ki,
            "kd": 0.0,
            "do_sp": do_sp,
            "do_base": None,
            "do_sp_max": None,
            "abac_gain": None,
            "nh4_ref": None,
            "n_runs": len(tmp),
            "tuning_scenarios": ";".join(tuning_scenarios),
            "energy_kwh_mean": tmp["energy_kwh"].mean(),
            "violation_hours_mean": tmp["violation_hours"].mean(),
            "nh4_p95_mean": tmp["nh4_p95"].mean(),
            "hard_safety_rate_mean": tmp["hard_safety_rate"].mean(),
        })

    for kp, ki, do_base, gain, do_sp_max in _abac_grid(preset):
        combo_id += 1
        ctrl = ABACPIDController(ABACConfig(kp=kp, ki=ki, do_base=do_base, abac_gain=gain, do_sp_max=do_sp_max, nh4_ref=cfg.nh4_limit))
        run_rows = []
        for scenario in tuning_scenarios:
            for seed in seeds:
                _, m = run_single("ABAC-PID-grid", ctrl, scenario, seed, cfg)
                m.update({"combo_id": combo_id})
                run_rows.append(m)
        tmp = pd.DataFrame(run_rows)
        for scenario, g in tmp.groupby("scenario"):
            scenario_rows.append({
                "combo_id": combo_id,
                "controller_family": "ABAC-PID",
                "scenario": scenario,
                "kp": kp,
                "ki": ki,
                "kd": 0.0,
                "do_sp": None,
                "do_base": do_base,
                "do_sp_max": do_sp_max,
                "abac_gain": gain,
                "nh4_ref": cfg.nh4_limit,
                "energy_kwh_mean": g["energy_kwh"].mean(),
                "violation_hours_mean": g["violation_hours"].mean(),
                "nh4_p95_mean": g["nh4_p95"].mean(),
                "hard_safety_rate_mean": g["hard_safety_rate"].mean(),
            })
        combo_rows.append({
            "combo_id": combo_id,
            "controller_family": "ABAC-PID",
            "kp": kp,
            "ki": ki,
            "kd": 0.0,
            "do_sp": None,
            "do_base": do_base,
            "do_sp_max": do_sp_max,
            "abac_gain": gain,
            "nh4_ref": cfg.nh4_limit,
            "n_runs": len(tmp),
            "tuning_scenarios": ";".join(tuning_scenarios),
            "energy_kwh_mean": tmp["energy_kwh"].mean(),
            "violation_hours_mean": tmp["violation_hours"].mean(),
            "nh4_p95_mean": tmp["nh4_p95"].mean(),
            "hard_safety_rate_mean": tmp["hard_safety_rate"].mean(),
        })

    df = _score_and_select(pd.DataFrame(combo_rows))
    scenario_df = pd.DataFrame(scenario_rows)
    df.to_csv(output_dir / "tables" / "baseline_tuning_results.csv", index=False)
    scenario_df.to_csv(output_dir / "tables" / "baseline_tuning_by_scenario.csv", index=False)
    selected = recommended_from_tuning(df)
    write_json(output_dir / "manifests" / "selected_baseline_params.json", selected)
    return df


def recommended_from_tuning(df: pd.DataFrame) -> dict:
    """Return one balanced DO-PID and one balanced ABAC-PID setting.

    Selection rule: within each controller family, select a Pareto-efficient row
    with the lowest transparent weighted score. This keeps the conventional
    baselines strong without cherry-picking by final paper scenarios.
    """
    out: dict[str, dict] = {}
    for fam, g in df.groupby("controller_family"):
        gg = g[g["pareto_efficient"]].copy()
        if gg.empty:
            gg = g.copy()
        row = gg.sort_values("selection_score").iloc[0].to_dict()
        cleaned = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        out[fam] = cleaned
    return out


def load_selected_baseline_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned baseline parameter file: {path}. Run scripts/tune_baselines.py first.")
    return read_json(path)


def controllers_from_selected_baselines(selected: dict, nh4_limit: float = 4.0) -> Dict[str, BaseController]:
    """Build benchmark controllers using tuned DO-PID and tuned ABAC-PID settings."""
    ctrls = default_controllers(nh4_limit)
    if "DO-PID" in selected:
        row = selected["DO-PID"]
        ctrls["DO-PID"] = DOPIDController(PIDConfig(kp=float(row["kp"]), ki=float(row["ki"]), kd=float(row.get("kd") or 0.0), do_sp=float(row["do_sp"])))
    if "ABAC-PID" in selected:
        row = selected["ABAC-PID"]
        abac_cfg = ABACConfig(
            kp=float(row["kp"]),
            ki=float(row["ki"]),
            kd=float(row.get("kd") or 0.0),
            do_base=float(row["do_base"]),
            do_sp_max=float(row.get("do_sp_max") or 3.2),
            abac_gain=float(row["abac_gain"]),
            nh4_ref=float(row.get("nh4_ref") or nh4_limit),
        )
        ctrls["ABAC-PID"] = ABACPIDController(abac_cfg)
        ctrls["Hybrid-Risk-PID"] = HybridRiskPIDController(abac_cfg)
    return ctrls
