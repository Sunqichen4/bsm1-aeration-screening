from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .classification import persistent_threshold
from .config import IDX, STATE_NAMES, normalize_scenario_name
from .configuration import load_scenario_config, load_surrogate_params
from .fixed_aeration import constant_influent_for_scenario
from .surrogate import ReducedASM1Surrogate, SurrogateParams
from .timebase import SimulationWindow


@dataclass
class DiagnosticRun:
    summary: dict
    state_timeseries: pd.DataFrame
    bounds: pd.DataFrame
    warmup_convergence: pd.DataFrame


def run_surrogate_diagnostic(
    *,
    scenario_name: str,
    kla_d_1: float,
    warmup_days: float,
    evaluation_days: float,
    timestep_min: float,
    nh4_threshold_mgN_L: float,
    params: SurrogateParams | None = None,
) -> DiagnosticRun:
    scenario_name = normalize_scenario_name(scenario_name)
    scenario = load_scenario_config(scenario_name)
    model = ReducedASM1Surrogate(params or load_surrogate_params())
    influent = constant_influent_for_scenario(scenario)
    window = SimulationWindow(warmup_days, evaluation_days, timestep_min)
    lower, upper = model.state_bounds()
    x = model.initial_state()

    all_states = np.empty((window.total_steps, len(STATE_NAMES)), dtype=float)
    times_d = window.time_days(interval_end=False)
    lower_counts = np.zeros(len(STATE_NAMES), dtype=int)
    upper_counts = np.zeros(len(STATE_NAMES), dtype=int)
    at_lower_counts = np.zeros(len(STATE_NAMES), dtype=int)
    at_upper_counts = np.zeros(len(STATE_NAMES), dtype=int)
    max_lower_violation = np.zeros(len(STATE_NAMES), dtype=float)
    max_upper_violation = np.zeros(len(STATE_NAMES), dtype=float)
    first_clip_step = np.full(len(STATE_NAMES), -1, dtype=int)

    for step in range(window.total_steps):
        all_states[step] = x
        bounded, info = model.step_with_bound_info(
            x,
            float(kla_d_1),
            influent,
            scenario,
            window.dt_hours,
        )
        candidate = np.asarray(info["candidate"], dtype=float)
        lower_clipped = np.asarray(info["lower_clipped"], dtype=bool)
        upper_clipped = np.asarray(info["upper_clipped"], dtype=bool)
        lower_counts += lower_clipped.astype(int)
        upper_counts += upper_clipped.astype(int)
        at_lower_counts += np.asarray(info["at_lower"], dtype=bool).astype(int)
        at_upper_counts += np.asarray(info["at_upper"], dtype=bool).astype(int)
        max_lower_violation = np.maximum(
            max_lower_violation,
            np.where(lower_clipped, lower - candidate, 0.0),
        )
        max_upper_violation = np.maximum(
            max_upper_violation,
            np.where(upper_clipped, candidate - upper, 0.0),
        )
        newly_clipped = (lower_clipped | upper_clipped) & (first_clip_step < 0)
        first_clip_step[newly_clipped] = step
        x = bounded

    if not np.all(np.isfinite(all_states)):
        raise FloatingPointError("Diagnostic surrogate states contain NaN or Inf.")
    if np.any(all_states < -1e-10):
        raise FloatingPointError("Bounded diagnostic surrogate states contain negative values.")

    evaluation = all_states[window.evaluation_slice]
    evaluation_time = times_d[window.evaluation_slice]
    snh = evaluation[:, IDX["SNH"]]
    exceed = snh > float(nh4_threshold_mgN_L)
    state_frame = pd.DataFrame(evaluation, columns=STATE_NAMES)
    state_frame.insert(0, "time_d", evaluation_time)
    state_frame.insert(1, "scenario", scenario_name)
    state_frame.insert(2, "commanded_kla_d-1", float(kla_d_1))
    state_frame.insert(3, "timestep_min", float(timestep_min))

    bounds_rows = []
    for index, state_name in enumerate(STATE_NAMES):
        total_clips = int(lower_counts[index] + upper_counts[index])
        bounds_rows.append(
            {
                "scenario": scenario_name,
                "commanded_kla_d-1": float(kla_d_1),
                "timestep_min": float(timestep_min),
                "state": state_name,
                "lower_bound": float(lower[index]),
                "upper_bound": float(upper[index]),
                "lower_clip_count": int(lower_counts[index]),
                "upper_clip_count": int(upper_counts[index]),
                "total_clip_count": total_clips,
                "fraction_of_steps_clipped": total_clips / window.total_steps,
                "clip_total_duration_h": total_clips * window.dt_hours,
                "fraction_of_steps_at_lower": at_lower_counts[index] / window.total_steps,
                "fraction_of_steps_at_upper": at_upper_counts[index] / window.total_steps,
                "max_preclip_lower_violation": float(max_lower_violation[index]),
                "max_preclip_upper_violation": float(max_upper_violation[index]),
                "max_preclip_violation": float(max(max_lower_violation[index], max_upper_violation[index])),
                "first_clip_time_d": (
                    float(first_clip_step[index] * window.dt_days) if first_clip_step[index] >= 0 else np.nan
                ),
            }
        )

    warmup_rows = []
    seven_day_steps = int(round(7.0 / window.dt_days))
    if window.warmup_steps >= 2 * seven_day_steps:
        prior = all_states[window.warmup_steps - 2 * seven_day_steps : window.warmup_steps - seven_day_steps]
        final = all_states[window.warmup_steps - seven_day_steps : window.warmup_steps]
        for index, state_name in enumerate(STATE_NAMES):
            prior_mean = float(np.mean(prior[:, index]))
            final_mean = float(np.mean(final[:, index]))
            warmup_rows.append(
                {
                    "scenario": scenario_name,
                    "commanded_kla_d-1": float(kla_d_1),
                    "timestep_min": float(timestep_min),
                    "state": state_name,
                    "previous_7d_mean": prior_mean,
                    "final_7d_mean": final_mean,
                    "absolute_change": final_mean - prior_mean,
                    "relative_change": (final_mean - prior_mean) / max(abs(prior_mean), 1e-12),
                }
            )

    summary = {
        "scenario": scenario_name,
        "commanded_kla_d-1": float(kla_d_1),
        "effective_kla_d-1": float(kla_d_1) * float(scenario.kla_scale),
        "timestep_min": float(timestep_min),
        "warmup_days": float(warmup_days),
        "evaluation_days": float(evaluation_days),
        "SNH_mean_eval": float(np.mean(snh)),
        "SNH_p95_eval": float(np.quantile(snh, 0.95)),
        "SNH_max_eval": float(np.max(snh)),
        "SNH_violation_hours_gt_limit": float(np.sum(exceed) * window.dt_hours),
        "SNH_violation_ratio_gt_limit": float(np.mean(exceed)),
        "SNH_zero_violation": bool(not np.any(exceed)),
        "total_clip_count": int(lower_counts.sum() + upper_counts.sum()),
        "any_clip": bool(np.any(lower_counts + upper_counts)),
    }
    for index, state_name in enumerate(STATE_NAMES):
        summary[f"{state_name}_mean_eval"] = float(np.mean(evaluation[:, index]))
        summary[f"{state_name}_p95_eval"] = float(np.quantile(evaluation[:, index], 0.95))
    return DiagnosticRun(
        summary=summary,
        state_timeseries=state_frame,
        bounds=pd.DataFrame(bounds_rows),
        warmup_convergence=pd.DataFrame(warmup_rows),
    )


def trajectory_errors(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference_time = reference["time_d"].to_numpy(dtype=float)
    candidate_time = candidate["time_d"].to_numpy(dtype=float)
    for state in STATE_NAMES:
        reference_values = np.interp(
            candidate_time,
            reference_time,
            reference[state].to_numpy(dtype=float),
        )
        candidate_values = candidate[state].to_numpy(dtype=float)
        residual = candidate_values - reference_values
        rows.append(
            {
                "state": state,
                "trajectory_rmse": float(np.sqrt(np.mean(residual * residual))),
                "trajectory_mae": float(np.mean(np.abs(residual))),
                "trajectory_max_abs_error": float(np.max(np.abs(residual))),
            }
        )
    return pd.DataFrame(rows)


def timestep_thresholds(case_metrics: pd.DataFrame, persistent_points: int) -> pd.DataFrame:
    rows = []
    for (scenario, timestep), group in case_metrics.groupby(["scenario", "timestep_min"], sort=False):
        result = persistent_threshold(
            group,
            kla_col="commanded_kla_d-1",
            compliant_col="SNH_zero_violation",
            persistent_points=persistent_points,
        )
        rows.append(
            {
                "scenario": scenario,
                "timestep_min": timestep,
                "threshold_status": result.status,
                "first_compliant_grid_point_d-1": result.first_compliant_grid_point,
                "threshold_interval_lower_d-1": result.interval_lower,
                "threshold_interval_upper_d-1": result.interval_upper,
                "grid_step_d-1": result.grid_step,
                "persistent_compliance_points": persistent_points,
            }
        )
    return pd.DataFrame(rows)


def run_timestep_and_bounds_grid(
    *,
    scenarios: Sequence[str],
    kla_values: Sequence[float],
    timesteps_min: Sequence[float],
    reference_timestep_min: float,
    warmup_days: float,
    evaluation_days: float,
    nh4_threshold_mgN_L: float,
    persistent_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = load_surrogate_params()
    case_rows: list[dict] = []
    error_rows: list[dict] = []
    bound_frames: list[pd.DataFrame] = []
    warmup_frames: list[pd.DataFrame] = []
    all_timesteps = [float(reference_timestep_min)] + [
        float(value) for value in timesteps_min if abs(float(value) - float(reference_timestep_min)) > 1e-12
    ]
    for scenario in scenarios:
        for kla in kla_values:
            runs: dict[float, DiagnosticRun] = {}
            for timestep in all_timesteps:
                run = run_surrogate_diagnostic(
                    scenario_name=scenario,
                    kla_d_1=float(kla),
                    warmup_days=warmup_days,
                    evaluation_days=evaluation_days,
                    timestep_min=timestep,
                    nh4_threshold_mgN_L=nh4_threshold_mgN_L,
                    params=params,
                )
                runs[timestep] = run
                case_rows.append(run.summary)
                bound_frames.append(run.bounds)
                if not run.warmup_convergence.empty:
                    warmup_frames.append(run.warmup_convergence)
            reference = runs[float(reference_timestep_min)]
            for timestep, run in runs.items():
                errors = trajectory_errors(reference.state_timeseries, run.state_timeseries)
                errors.insert(0, "scenario", normalize_scenario_name(scenario))
                errors.insert(1, "commanded_kla_d-1", float(kla))
                errors.insert(2, "timestep_min", float(timestep))
                errors.insert(3, "reference_timestep_min", float(reference_timestep_min))
                error_rows.extend(errors.to_dict(orient="records"))
    cases = pd.DataFrame(case_rows)
    thresholds = timestep_thresholds(cases, persistent_points=persistent_points)
    reference_threshold = thresholds[
        np.isclose(thresholds["timestep_min"], float(reference_timestep_min))
    ][["scenario", "first_compliant_grid_point_d-1"]].rename(
        columns={"first_compliant_grid_point_d-1": "reference_first_compliant_grid_point_d-1"}
    )
    thresholds = thresholds.merge(reference_threshold, on="scenario", how="left")
    thresholds["threshold_error_vs_reference_d-1"] = (
        thresholds["first_compliant_grid_point_d-1"]
        - thresholds["reference_first_compliant_grid_point_d-1"]
    )
    bounds = pd.concat(bound_frames, ignore_index=True) if bound_frames else pd.DataFrame()
    warmup = pd.concat(warmup_frames, ignore_index=True) if warmup_frames else pd.DataFrame()
    return cases, pd.DataFrame(error_rows), thresholds, bounds, warmup
