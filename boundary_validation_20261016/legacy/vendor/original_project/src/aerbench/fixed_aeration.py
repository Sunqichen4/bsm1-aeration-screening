from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from .config import IDX, SCENARIOS, ScenarioConfig, normalize_scenario_name
from .configuration import load_surrogate_params
from .surrogate import InfluentPoint, ReducedASM1Surrogate, SurrogateParams
from .timebase import SimulationWindow
from .steady_conditions import SteadyOperatingPoint


DEFAULT_AERATION_REFERENCE_KLA_D_1 = 240.0


def commanded_aeration_index(
    commanded_kla_d_1: float,
    reference_kla_d_1: float = DEFAULT_AERATION_REFERENCE_KLA_D_1,
) -> float:
    """Return a dimensionless commanded-aeration effort proxy.

    This index is not electrical energy.  It intentionally contains no blower,
    diffuser, reactor-volume, alpha-factor, or power-conversion assumptions.
    """
    reference = float(reference_kla_d_1)
    if reference <= 0.0:
        raise ValueError("reference_kla_d_1 must be positive.")
    commanded = float(commanded_kla_d_1)
    if commanded < 0.0:
        raise ValueError("commanded_kla_d_1 cannot be negative.")
    return commanded / reference


def validation_scenario(name: str, oxygen_transfer_factor: float) -> ScenarioConfig:
    """Return one manuscript scenario with the requested oxygen-transfer loss."""
    name = normalize_scenario_name(name)
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    scenario = SCENARIOS[name]
    if name == "low_oxygen_transfer":
        return replace(scenario, kla_scale=float(oxygen_transfer_factor))
    return scenario


def constant_influent_for_scenario(scenario: ScenarioConfig) -> InfluentPoint:
    """Return the steady influent used by the matched fixed-aeration checks."""
    return InfluentPoint(
        cod=float(260.0 * scenario.cod_scale),
        nh4=float(30.0 * scenario.nh4_scale),
        q_scale=float(scenario.flow_scale),
    )


def run_surrogate_fixed_aeration(
    *,
    scenario_name: str,
    kla: float,
    warmup_days: float,
    eval_days: float,
    dt_minutes: float,
    nh4_limit: float,
    oxygen_transfer_factor: float,
    params: SurrogateParams | None = None,
    operating_point: SteadyOperatingPoint | None = None,
) -> dict:
    """Run the reduced model under one constant nominal aeration command.

    This shared implementation is used both by the matched BSM1 validation and
    by the large-scale screening example.  Keeping one implementation prevents
    the candidate screen from silently using different equations or metrics.
    """
    if warmup_days < 0.0 or eval_days <= 0.0 or dt_minutes <= 0.0:
        raise ValueError("warmup_days must be non-negative; eval_days and dt_minutes must be positive.")

    scenario_name = normalize_scenario_name(scenario_name)
    if operating_point is not None and operating_point.name != scenario_name:
        raise ValueError("Operating-point name must match scenario_name.")
    scenario = (operating_point.surrogate_scenario() if operating_point is not None
                else validation_scenario(scenario_name, oxygen_transfer_factor))
    influent = constant_influent_for_scenario(scenario)
    model = ReducedASM1Surrogate(params or load_surrogate_params())
    window = SimulationWindow(
        warmup_days=float(warmup_days),
        evaluation_days=float(eval_days),
        timestep_min=float(dt_minutes),
    )
    dt_h = window.dt_hours

    x = model.initial_state()
    eval_snh: list[float] = []
    eval_so: list[float] = []
    eval_sno: list[float] = []
    eval_states: list[np.ndarray] = []

    tic = time.perf_counter()
    for k in range(window.total_steps):
        x_pre = x.copy()
        if k >= window.warmup_steps:
            eval_snh.append(float(x_pre[IDX["SNH"]]))
            eval_so.append(float(x_pre[IDX["SO"]]))
            eval_sno.append(float(x_pre[IDX["SNO"]]))
            eval_states.append(x_pre)
        x = model.step(x_pre, float(kla), influent, scenario, dt_h)
    runtime_seconds = time.perf_counter() - tic

    snh = np.asarray(eval_snh, dtype=float)
    so = np.asarray(eval_so, dtype=float)
    sno = np.asarray(eval_sno, dtype=float)
    if snh.size == 0:
        raise ValueError("The selected time settings produced no evaluation samples.")
    exceed = snh > float(nh4_limit)
    eval_duration_h = len(snh) * dt_h
    states = np.asarray(eval_states, dtype=float)
    if not np.all(np.isfinite(states)):
        raise FloatingPointError("The surrogate evaluation states contain NaN or Inf.")
    if np.any(states < -1e-10):
        raise FloatingPointError("The bounded surrogate evaluation states contain negative values.")

    return {
        "model": "Reduced surrogate",
        "experiment": "fixed_kla",
        "inflow_mode": "steady_const",
        "scenario": scenario_name,
        "kla": float(kla),
        "effective_kla": float(kla * scenario.kla_scale),
        "oxygen_transfer_factor": float(
            oxygen_transfer_factor if scenario_name == "low_oxygen_transfer" else 1.0
        ),
        "warmup_days": float(warmup_days),
        "eval_days": float(eval_days),
        "dt_minutes": float(dt_minutes),
        "nh4_limit": float(nh4_limit),
        "SNH_mean_eval": float(np.mean(snh)),
        "SNH_p95_eval": float(np.quantile(snh, 0.95)),
        "SNH_max_eval": float(np.max(snh)),
        "SNH_violation_hours_gt_limit": float(np.sum(exceed) * dt_h),
        "SNH_violation_ratio_gt_limit": float(np.mean(exceed)),
        "SNH_zero_violation": bool(not np.any(exceed)),
        "SNO_mean_eval": float(np.mean(sno)),
        "TIN_mean_eval": float(np.mean(snh + sno)),
        "DO_mean_eval": float(np.mean(so)),
        "DO_p05_eval": float(np.quantile(so, 0.05)),
        "commanded_aeration_index": commanded_aeration_index(float(kla)),
        "commanded_aeration_index_reference_kla_d-1": DEFAULT_AERATION_REFERENCE_KLA_D_1,
        "commanded_aeration_index_unit": "dimensionless",
        "cumulative_commanded_kla_d-1_times_d": float(kla * eval_days),
        "runtime_seconds": float(runtime_seconds),
        "n_eval_steps": int(len(snh)),
        "eval_duration_h": float(eval_duration_h),
        **{
            f"{state_name}_min_eval": float(np.min(states[:, index]))
            for index, state_name in enumerate(("XH", "SS", "XA", "SNO", "SNH", "SO"))
        },
        **{
            f"{state_name}_max_eval": float(np.max(states[:, index]))
            for index, state_name in enumerate(("XH", "SS", "XA", "SNO", "SNH", "SO"))
        },
        **(operating_point.metadata() if operating_point is not None else {}),
    }
