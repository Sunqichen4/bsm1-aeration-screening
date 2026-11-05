from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd

from .bsm1_validation import (
    Q,
    SNH,
    SS,
    XS,
    build_bsm1_data,
    run_bsm1_fixed_kla,
)
from .classification import classify_candidate, persistent_threshold, wilson_interval
from .config import IDX, SCENARIOS
from .configuration import load_surrogate_params
from .fixed_aeration import commanded_aeration_index
from .surrogate import InfluentPoint, ReducedASM1Surrogate
from .timebase import SimulationWindow


BSM1_REFERENCE_COD_PROXY = 69.5 + 202.32
SURROGATE_REFERENCE_COD = 260.0
BSM1_REFERENCE_SNH = 31.56
SURROGATE_REFERENCE_NH4 = 30.0
BSM1_REFERENCE_FLOW = 18446.0


def map_bsm1_influent_to_surrogate(data_in: np.ndarray) -> pd.DataFrame:
    """Map the documented BSM1 influent states to the three surrogate inputs.

    The mapping is fixed before validation: SS+XS supplies the COD proxy, SNH
    supplies ammonium, and Q supplies the hydraulic multiplier. Reference-value
    ratios align the constant BSM1 row with the frozen surrogate baseline input.
    """
    data = np.asarray(data_in, dtype=float)
    if data.ndim != 2 or data.shape[1] < 22:
        raise ValueError("BSM1 influent mapping requires time plus 21 state columns.")
    cod = (data[:, 1 + SS] + data[:, 1 + XS]) * SURROGATE_REFERENCE_COD / BSM1_REFERENCE_COD_PROXY
    nh4 = data[:, 1 + SNH] * SURROGATE_REFERENCE_NH4 / BSM1_REFERENCE_SNH
    q_scale = data[:, 1 + Q] / BSM1_REFERENCE_FLOW
    frame = pd.DataFrame(
        {
            "time_d": data[:, 0],
            "surrogate_influent_cod_mgCOD_L": cod,
            "surrogate_influent_nh4_mgN_L": nh4,
            "surrogate_flow_scale": q_scale,
            "mapping_cod_formula": "(BSM1_SS + BSM1_XS) * 260 / 271.82",
            "mapping_nh4_formula": "BSM1_SNH * 30 / 31.56",
            "mapping_flow_formula": "BSM1_Q / 18446",
        }
    )
    if not np.all(np.isfinite(frame[["surrogate_influent_cod_mgCOD_L", "surrogate_influent_nh4_mgN_L", "surrogate_flow_scale"]])):
        raise FloatingPointError("Mapped dynamic influent contains NaN or Inf.")
    return frame


def run_dynamic_surrogate(
    *,
    profile: str,
    commanded_kla_d_1: float,
    warmup_days: float,
    evaluation_days: float,
    timestep_min: float,
    nh4_threshold_mgN_L: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    window = SimulationWindow(warmup_days, evaluation_days, timestep_min)
    data = build_bsm1_data(
        "baseline",
        total_days=warmup_days + evaluation_days,
        timestep_days=window.dt_days,
        inflow_mode=profile,
        tempmodel=False,
    )
    mapped = map_bsm1_influent_to_surrogate(data)
    if len(mapped) < window.total_steps:
        raise RuntimeError("Mapped dynamic influent is shorter than the requested simulation window.")
    model = ReducedASM1Surrogate(load_surrogate_params())
    scenario = SCENARIOS["baseline"]
    x = model.initial_state()
    records = []
    tic = time.perf_counter()
    for step in range(window.total_steps):
        input_row = mapped.iloc[step]
        influent = InfluentPoint(
            cod=float(input_row["surrogate_influent_cod_mgCOD_L"]),
            nh4=float(input_row["surrogate_influent_nh4_mgN_L"]),
            q_scale=float(input_row["surrogate_flow_scale"]),
        )
        if step >= window.warmup_steps:
            records.append(
                {
                    "time_d": float(step * window.dt_days),
                    "profile": profile,
                    "commanded_kla_d-1": float(commanded_kla_d_1),
                    "surrogate_SNH_mgN_L": float(x[IDX["SNH"]]),
                    "surrogate_SO_mgO2_L": float(x[IDX["SO"]]),
                    "surrogate_SNO_proxy_mgN_L": float(x[IDX["SNO"]]),
                    "surrogate_TIN_proxy_mgN_L": float(x[IDX["SNH"]] + x[IDX["SNO"]]),
                    "commanded_aeration_index": commanded_aeration_index(commanded_kla_d_1),
                }
            )
        x = model.step(x, float(commanded_kla_d_1), influent, scenario, window.dt_hours)
    runtime = time.perf_counter() - tic
    timeseries = pd.DataFrame(records)
    snh = timeseries["surrogate_SNH_mgN_L"].to_numpy(dtype=float)
    exceed = snh > float(nh4_threshold_mgN_L)
    summary = {
        "profile": profile,
        "commanded_kla_d-1": float(commanded_kla_d_1),
        "surrogate_mean_snh_mgN_L": float(np.mean(snh)),
        "surrogate_p95_snh_mgN_L": float(np.quantile(snh, 0.95)),
        "surrogate_max_snh_mgN_L": float(np.max(snh)),
        "surrogate_violation_h": float(np.sum(exceed) * window.dt_hours),
        "surrogate_violation_fraction": float(np.mean(exceed)),
        "surrogate_zero_violation": bool(not np.any(exceed)),
        "surrogate_mean_do_mgO2_L": float(timeseries["surrogate_SO_mgO2_L"].mean()),
        "surrogate_mean_oxidized_n_proxy_mgN_L": float(timeseries["surrogate_SNO_proxy_mgN_L"].mean()),
        "surrogate_mean_tin_proxy_mgN_L": float(timeseries["surrogate_TIN_proxy_mgN_L"].mean()),
        "surrogate_runtime_seconds": float(runtime),
        "commanded_aeration_index": commanded_aeration_index(commanded_kla_d_1),
    }
    return summary, timeseries, mapped


def run_dynamic_pairwise_grid(
    *,
    profiles: Sequence[str],
    kla_values: Sequence[float],
    warmup_days: float,
    evaluation_days: float,
    surrogate_timestep_min: float,
    bsm1_timestep_min: float,
    nh4_threshold_mgN_L: float,
    persistent_points: int,
    timeseries_kla: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    saved_timeseries = []
    mapped_inputs = []
    mapped_profiles: set[str] = set()
    for profile in profiles:
        for kla in kla_values:
            save_representative = (
                timeseries_kla is not None
                and abs(float(kla) - float(timeseries_kla)) <= 1e-9
            )
            surrogate, surrogate_ts, mapped = run_dynamic_surrogate(
                profile=profile,
                commanded_kla_d_1=float(kla),
                warmup_days=warmup_days,
                evaluation_days=evaluation_days,
                timestep_min=surrogate_timestep_min,
                nh4_threshold_mgN_L=nh4_threshold_mgN_L,
            )
            bsm1_output = run_bsm1_fixed_kla(
                scenario="baseline",
                kla=float(kla),
                warmup_days=warmup_days,
                eval_days=evaluation_days,
                nh4_limit=nh4_threshold_mgN_L,
                inflow_mode=profile,
                timestep_min=bsm1_timestep_min,
                oxygen_transfer_factor=0.70,
                tempmodel_policy="off",
                return_biomass_timeseries=save_representative,
            )
            if save_representative:
                bsm1, bsm1_ts = bsm1_output
            else:
                bsm1 = bsm1_output
                bsm1_ts = None
            row = dict(surrogate)
            row.update(
                {
                    "bsm1_mean_snh_mgN_L": bsm1["SNH_mean_eval"],
                    "bsm1_p95_snh_mgN_L": bsm1["SNH_p95_eval"],
                    "bsm1_max_snh_mgN_L": bsm1["SNH_max_eval"],
                    "bsm1_violation_h": bsm1["SNH_violation_hours_gt_limit"],
                    "bsm1_violation_fraction": bsm1["SNH_violation_ratio_gt_limit"],
                    "bsm1_zero_violation": bsm1["SNH_zero_violation"],
                    "bsm1_mean_tn_mgN_L": bsm1.get("TN_mean_eval", np.nan),
                    "bsm1_mean_do_mgO2_L": bsm1.get("DO_tank5_mean_eval", np.nan),
                    "bsm1_standard_aeration_energy_kWh_per_d": bsm1.get(
                        "bsm1_standard_aeration_energy_kWh_per_d", np.nan
                    ),
                    "bsm1_runtime_seconds": bsm1["runtime_seconds"],
                }
            )
            row["classification"] = classify_candidate(
                bool(row["surrogate_zero_violation"]), bool(row["bsm1_zero_violation"])
            )
            rows.append(row)
            if save_representative:
                representative = surrogate_ts.copy()
                bsm1_eval = bsm1_ts[bsm1_ts["phase"].eq("main")].copy()
                if bsm1_eval.empty:
                    raise RuntimeError("The representative dynamic BSM1 time series has no evaluation samples.")
                representative["bsm1_eff_NH4_mgN_L"] = np.interp(
                    representative["time_d"], bsm1_eval["time_d"], bsm1_eval["eff_NH4"]
                )
                representative["bsm1_eff_TN_mgN_L"] = np.interp(
                    representative["time_d"], bsm1_eval["time_d"], bsm1_eval["eff_TN"]
                )
                representative["bsm1_original_timestep_min"] = float(bsm1_timestep_min)
                representative["surrogate_original_timestep_min"] = float(surrogate_timestep_min)
                representative["pairing_method"] = "bsm1_linearly_interpolated_to_surrogate_reporting_times_for_plot_only"
                saved_timeseries.append(representative)
            if profile not in mapped_profiles:
                mapped_copy = mapped.copy()
                mapped_copy.insert(1, "profile", profile)
                mapped_inputs.append(mapped_copy)
                mapped_profiles.add(profile)
    pairwise = pd.DataFrame(rows)
    threshold_rows = []
    classification_rows = []
    for profile, group in pairwise.groupby("profile", sort=False):
        surrogate_threshold = persistent_threshold(
            group,
            kla_col="commanded_kla_d-1",
            compliant_col="surrogate_zero_violation",
            persistent_points=persistent_points,
        )
        bsm1_threshold = persistent_threshold(
            group,
            kla_col="commanded_kla_d-1",
            compliant_col="bsm1_zero_violation",
            persistent_points=persistent_points,
        )
        threshold_rows.append(
            {
                "profile": profile,
                "surrogate_threshold_status": surrogate_threshold.status,
                "surrogate_first_compliant_grid_point_d-1": surrogate_threshold.first_compliant_grid_point,
                "bsm1_threshold_status": bsm1_threshold.status,
                "bsm1_first_compliant_grid_point_d-1": bsm1_threshold.first_compliant_grid_point,
                "absolute_threshold_error_d-1": (
                    abs(float(surrogate_threshold.first_compliant_grid_point) - float(bsm1_threshold.first_compliant_grid_point))
                    if surrogate_threshold.first_compliant_grid_point is not None and bsm1_threshold.first_compliant_grid_point is not None
                    else np.nan
                ),
                "persistent_compliance_points": persistent_points,
            }
        )
        counts = group["classification"].value_counts()
        false_safe = int(counts.get("false_safe", 0))
        actual_reject = int(counts.get("true_reject", 0)) + false_safe
        fs_low, fs_high = wilson_interval(false_safe, actual_reject)
        classification_rows.append(
            {
                "profile": profile,
                "n_candidates": int(len(group)),
                "true_safe": int(counts.get("true_safe", 0)),
                "true_reject": int(counts.get("true_reject", 0)),
                "false_safe": false_safe,
                "conservative_reject": int(counts.get("conservative_reject", 0)),
                "false_safe_rate": false_safe / actual_reject if actual_reject else np.nan,
                "false_safe_wilson_low": fs_low,
                "false_safe_wilson_high": fs_high,
            }
        )
    timeseries = pd.concat(saved_timeseries, ignore_index=True) if saved_timeseries else pd.DataFrame()
    mapped_frame = pd.concat(mapped_inputs, ignore_index=True) if mapped_inputs else pd.DataFrame()
    return pairwise, pd.DataFrame(threshold_rows), pd.DataFrame(classification_rows), timeseries, mapped_frame
