from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .config import BenchmarkConfig, EnergyConfig, IDX, SCENARIOS, SafetyConfig
from .controllers import BaseController, ControllerContext, apply_safety_and_actuator, default_controllers
from .metrics import compute_metrics, summarize_runs
from .risk_cue import OnlineRiskCue, RiskCueConfig
from .surrogate import InfluentGenerator, ReducedASM1Surrogate
from .utils import ensure_dirs, write_json


def energy_power(u: float, cfg: EnergyConfig) -> float:
    return float(cfg.alpha * (u / cfg.u_ref) ** cfg.beta)


def prbs_warmup_u(seed: int, block_idx: int) -> float:
    # Reproducible PRBS without relying on Python's process-randomized hash.
    key = f"{int(seed)}-{int(block_idx)}-warmup_prbs".encode("utf-8")
    val = int(hashlib.sha256(key).hexdigest()[:8], 16) % 10000
    return 120.0 if val < 5000 else 300.0


def update_state_estimate(model: ReducedASM1Surrogate, x_est: np.ndarray, meas: Dict[str, float], gain_do: float = 0.65, gain_nh4: float = 0.55) -> np.ndarray:
    x = x_est.copy()
    x[IDX["SO"]] += gain_do * (meas["meas_DO"] - x[IDX["SO"]])
    x[IDX["SNH"]] += gain_nh4 * (meas["meas_NH4"] - x[IDX["SNH"]])
    return np.clip(x, np.array(model.params.min_vals), np.array(model.params.max_vals))


def run_single(
    controller_name: str,
    controller: BaseController,
    scenario_name: str,
    seed: int,
    cfg: BenchmarkConfig,
    energy_cfg: EnergyConfig | None = None,
    safety_cfg: SafetyConfig | None = None,
) -> Tuple[pd.DataFrame, dict]:
    energy_cfg = energy_cfg or EnergyConfig()
    safety_cfg = safety_cfg or SafetyConfig(nh4_limit=cfg.nh4_limit)
    scenario = SCENARIOS[scenario_name]
    rng = np.random.default_rng(seed)
    model = ReducedASM1Surrogate()
    influent = InfluentGenerator(scenario, seed=seed)
    x = model.initial_state()
    x_est = x.copy()
    prev_u = 180.0
    risk_cue = OnlineRiskCue(RiskCueConfig(random_state=seed)) if cfg.use_risk_cue else None
    controller.reset()
    records = []

    for k in range(cfg.n_steps):
        t_days = k * cfg.dt_days
        phase = "warmup" if k < cfg.warmup_steps else "main"
        infl = influent.at(k, t_days)
        x_pre = x.copy()
        meas = model.measure(x_pre, scenario, rng)

        if phase == "warmup":
            u_raw = prbs_warmup_u(seed, k // 4)
            info = {"do_sp": np.nan, "controller_mode": -99.0, "nh4_future": np.nan, "mpc_success": np.nan}
        else:
            x_est = update_state_estimate(model, x_est, meas)
            ctx = ControllerContext(
                step=k,
                dt_h=cfg.dt_hours,
                meas=meas,
                influent=infl,
                scenario=scenario,
                model=model,
                x_est=x_est,
                prev_u=prev_u,
                risk_cue=risk_cue,
            )
            u_raw, info = controller.compute_raw(ctx)

        u, flags = apply_safety_and_actuator(u_raw, prev_u, meas, safety_cfg)
        x_next = model.step(x_pre, u, infl, scenario, cfg.dt_hours)
        x_est = model.step(x_est, u, infl, scenario, cfg.dt_hours)
        power = energy_power(u, energy_cfg)
        row = {
            "time_d": t_days,
            "step": k,
            "dt_h": cfg.dt_hours,
            "phase": phase,
            "scenario": scenario_name,
            "controller": controller_name,
            "seed": seed,
            "in_COD": infl.cod,
            "in_NH4": infl.nh4,
            "q_scale": infl.q_scale,
            **meas,
            "u_raw": float(u_raw),
            "u": float(u),
            "energy_kwh": float(power * cfg.dt_hours if phase == "main" else 0.0),
            "energy_kwh_all": float(power * cfg.dt_hours),
            **flags,
            "do_sp": float(info.get("do_sp", np.nan)) if info.get("do_sp", np.nan) is not None else np.nan,
            "nh4_future": float(info.get("nh4_future", np.nan)) if info.get("nh4_future", np.nan) is not None else np.nan,
            "controller_mode": float(info.get("controller_mode", np.nan)),
            "mpc_success": float(info.get("mpc_success", np.nan)) if info.get("mpc_success", np.nan) is not None else np.nan,
        }
        for name, idx in IDX.items():
            row[f"true_{name}_pre"] = float(x_pre[idx])
            row[f"true_{name}_post"] = float(x_next[idx])
            row[f"xest_{name}"] = float(x_est[idx])
        records.append(row)

        if risk_cue is not None:
            risk_cue.update_history({
                "meas_NH4": row["meas_NH4"],
                "meas_DO": row["meas_DO"],
                "u": row["u"],
                "in_COD": row["in_COD"],
                "in_NH4": row["in_NH4"],
            })
        x = x_next
        prev_u = u

    df = pd.DataFrame(records)
    metrics = compute_metrics(df, nh4_limit=cfg.nh4_limit, do_target=cfg.do_target, main_only=True)
    all_metrics = compute_metrics(df.assign(energy_kwh=df["energy_kwh_all"]), nh4_limit=cfg.nh4_limit, do_target=cfg.do_target, main_only=False)
    out = {
        "scenario": scenario_name,
        "controller": controller_name,
        "seed": seed,
        **metrics,
        "warmup_mixed_energy_kwh": all_metrics["energy_kwh"],
        "warmup_mixed_violation_hours": all_metrics["violation_hours"],
        "warmup_bias_energy_kwh": all_metrics["energy_kwh"] - metrics["energy_kwh"],
        "warmup_bias_violation_hours": all_metrics["violation_hours"] - metrics["violation_hours"],
    }
    return df, out


def run_benchmark(
    cfg: BenchmarkConfig,
    controllers: Dict[str, BaseController] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(cfg.output_dir)
    controllers = controllers or default_controllers(cfg.nh4_limit)
    run_rows = []
    for scenario_name in cfg.scenarios:
        for ctrl_name, ctrl in controllers.items():
            for seed in cfg.seeds:
                print(f"Running scenario={scenario_name} controller={ctrl_name} seed={seed}")
                df_ts, row = run_single(ctrl_name, ctrl, scenario_name, seed, cfg)
                run_rows.append(row)
                if cfg.save_timeseries:
                    ts_path = cfg.output_dir / "timeseries" / f"{scenario_name}__{ctrl_name.replace('/', '_')}__seed{seed}.csv"
                    df_ts.to_csv(ts_path, index=False)
    run_df = pd.DataFrame(run_rows)
    summary_df = summarize_runs(run_rows)
    run_df.to_csv(cfg.output_dir / "tables" / "run_level_metrics.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "tables" / "summary_metrics.csv", index=False)
    write_json(
        cfg.output_dir / "manifests" / "benchmark_manifest.json",
        {
            "benchmark_config": asdict(cfg),
            "controllers": list(controllers.keys()),
            "scenarios": cfg.scenarios,
            "seeds": cfg.seeds,
        },
    )
    return run_df, summary_df
