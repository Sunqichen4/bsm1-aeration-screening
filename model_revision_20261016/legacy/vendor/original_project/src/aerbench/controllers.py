from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .config import EnergyConfig, IDX, SafetyConfig
from .risk_cue import OnlineRiskCue, RiskCueConfig
from .surrogate import InfluentPoint, ReducedASM1Surrogate, ScenarioConfig


@dataclass
class ControllerContext:
    step: int
    dt_h: float
    meas: Dict[str, float]
    influent: InfluentPoint
    scenario: ScenarioConfig
    model: ReducedASM1Surrogate
    x_est: np.ndarray
    prev_u: float
    risk_cue: Optional[OnlineRiskCue] = None


class BaseController:
    name = "base"

    def reset(self) -> None:
        pass

    def compute_raw(self, ctx: ControllerContext) -> Tuple[float, Dict[str, float]]:
        raise NotImplementedError


@dataclass
class PIDConfig:
    kp: float = 55.0
    ki: float = 4.0
    kd: float = 0.0
    do_sp: float = 2.0
    u_min: float = 40.0
    u_max: float = 450.0
    rate_limit: float = 45.0
    integral_limit: float = 30.0


class DOPIDController(BaseController):
    name = "DO-PID"

    def __init__(self, cfg: PIDConfig | None = None):
        self.cfg = cfg or PIDConfig()
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_raw(self, ctx: ControllerContext) -> Tuple[float, Dict[str, float]]:
        err = self.cfg.do_sp - ctx.meas["meas_DO"]
        self.integral = float(np.clip(self.integral + err * ctx.dt_h, -self.cfg.integral_limit, self.cfg.integral_limit))
        deriv = (err - self.prev_error) / max(ctx.dt_h, 1e-12)
        self.prev_error = err
        raw = ctx.prev_u + self.cfg.kp * err + self.cfg.ki * self.integral + self.cfg.kd * deriv
        raw = float(np.clip(raw, ctx.prev_u - self.cfg.rate_limit, ctx.prev_u + self.cfg.rate_limit))
        raw = float(np.clip(raw, self.cfg.u_min, self.cfg.u_max))
        return raw, {"do_sp": self.cfg.do_sp, "controller_mode": 0.0}


@dataclass
class ABACConfig:
    nh4_ref: float = 8.0
    do_base: float = 1.8
    do_sp_min: float = 0.8
    do_sp_max: float = 3.2
    abac_gain: float = 0.18
    kp: float = 65.0
    ki: float = 5.0
    kd: float = 0.0
    u_min: float = 40.0
    u_max: float = 450.0
    rate_limit: float = 45.0
    integral_limit: float = 30.0


class ABACPIDController(BaseController):
    name = "ABAC-PID"

    def __init__(self, cfg: ABACConfig | None = None):
        self.cfg = cfg or ABACConfig()
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_do_sp(self, meas_nh4: float) -> float:
        do_sp = self.cfg.do_base + self.cfg.abac_gain * (meas_nh4 - self.cfg.nh4_ref)
        return float(np.clip(do_sp, self.cfg.do_sp_min, self.cfg.do_sp_max))

    def compute_raw(self, ctx: ControllerContext) -> Tuple[float, Dict[str, float]]:
        do_sp = self.compute_do_sp(ctx.meas["meas_NH4"])
        err = do_sp - ctx.meas["meas_DO"]
        self.integral = float(np.clip(self.integral + err * ctx.dt_h, -self.cfg.integral_limit, self.cfg.integral_limit))
        deriv = (err - self.prev_error) / max(ctx.dt_h, 1e-12)
        self.prev_error = err
        raw = ctx.prev_u + self.cfg.kp * err + self.cfg.ki * self.integral + self.cfg.kd * deriv
        raw = float(np.clip(raw, ctx.prev_u - self.cfg.rate_limit, ctx.prev_u + self.cfg.rate_limit))
        raw = float(np.clip(raw, self.cfg.u_min, self.cfg.u_max))
        return raw, {"do_sp": do_sp, "controller_mode": 1.0}


class HybridRiskPIDController(ABACPIDController):
    name = "Hybrid-Risk-PID"

    def compute_raw(self, ctx: ControllerContext) -> Tuple[float, Dict[str, float]]:
        raw, info = super().compute_raw(ctx)
        risk = np.nan
        if ctx.risk_cue is not None:
            risk = ctx.risk_cue.predict(ctx.step)
        if np.isfinite(risk):
            if risk > self.cfg.nh4_ref + 1.0:
                raw *= 1.15
                info["controller_mode"] = 2.0
            elif risk < self.cfg.nh4_ref - 1.5 and ctx.meas["meas_DO"] > info["do_sp"]:
                raw *= 0.92
                info["controller_mode"] = -1.0
        info["nh4_future"] = float(risk) if np.isfinite(risk) else np.nan
        return float(raw), info


@dataclass
class MPCConfig:
    horizon: int = 12
    move_blocks: int = 3
    u_min: float = 40.0
    u_max: float = 430.0
    rate_limit: float = 45.0
    nh4_ref: float = 8.0
    do_ref: float = 2.0
    w_nh4: float = 1600.0
    w_do: float = 100.0
    w_energy: float = 2.5
    w_smooth: float = 0.25
    maxiter: int = 40
    recompute_every: int = 2


class MPCController(BaseController):
    name = "MPC"

    def __init__(self, cfg: MPCConfig | None = None, energy_cfg: EnergyConfig | None = None):
        self.cfg = cfg or MPCConfig()
        self.energy_cfg = energy_cfg or EnergyConfig()
        self.last_solution: Optional[np.ndarray] = None
        self.hold_u: Optional[float] = None
        self.hold_until: int = -1

    def reset(self) -> None:
        self.last_solution = None
        self.hold_u = None
        self.hold_until = -1

    def _expand_moves(self, z: np.ndarray) -> np.ndarray:
        reps = int(np.ceil(self.cfg.horizon / len(z)))
        return np.repeat(z, reps)[: self.cfg.horizon]

    def _energy_power(self, u: float) -> float:
        e = self.energy_cfg
        return e.alpha * (u / e.u_ref) ** e.beta

    def _risk_schedule(self, ctx: ControllerContext, base_cfg: MPCConfig) -> MPCConfig:
        return base_cfg

    def compute_raw(self, ctx: ControllerContext) -> Tuple[float, Dict[str, float]]:
        if self.hold_u is not None and ctx.step < self.hold_until:
            return float(self.hold_u), {"do_sp": self.cfg.do_ref, "controller_mode": 10.0}

        cfg = self._risk_schedule(ctx, self.cfg)
        block_n = cfg.move_blocks
        if self.last_solution is None:
            z0 = np.full(block_n, ctx.prev_u, dtype=float)
        else:
            z0 = np.roll(self.last_solution, -1)
            z0[-1] = z0[-2] if len(z0) > 1 else ctx.prev_u
        z0 = np.clip(z0, cfg.u_min, cfg.u_max)
        bounds = [(cfg.u_min, cfg.u_max)] * block_n

        def objective(z: np.ndarray) -> float:
            u_seq = self._expand_moves(z)
            x = ctx.x_est.copy()
            prev = ctx.prev_u
            cost = 0.0
            for i, u in enumerate(u_seq):
                u_rate = float(np.clip(u, prev - cfg.rate_limit, prev + cfg.rate_limit))
                # Persistence forecast for influent within the short horizon.
                x = ctx.model.step(x, u_rate, ctx.influent, ctx.scenario, ctx.dt_h)
                nh4 = x[IDX["SNH"]]
                do = x[IDX["SO"]]
                cost += cfg.w_nh4 * max(0.0, nh4 - cfg.nh4_ref) ** 2
                cost += cfg.w_do * max(0.0, cfg.do_ref - do) ** 2
                cost += cfg.w_energy * self._energy_power(u_rate) * ctx.dt_h
                cost += cfg.w_smooth * (u_rate - prev) ** 2
                prev = u_rate
            return float(cost)

        res = minimize(objective, z0, method="SLSQP", bounds=bounds, options={"maxiter": cfg.maxiter, "ftol": 1e-4, "disp": False})
        z = np.array(res.x if res.success else z0, dtype=float)
        self.last_solution = z
        raw = float(np.clip(z[0], ctx.prev_u - cfg.rate_limit, ctx.prev_u + cfg.rate_limit))
        raw = float(np.clip(raw, cfg.u_min, cfg.u_max))
        self.hold_u = raw
        self.hold_until = ctx.step + max(1, cfg.recompute_every)
        return raw, {"do_sp": cfg.do_ref, "controller_mode": 11.0, "mpc_success": float(res.success)}


class RiskCueMPCController(MPCController):
    name = "Risk-cue-MPC"

    def _risk_schedule(self, ctx: ControllerContext, base_cfg: MPCConfig) -> MPCConfig:
        risk = np.nan
        if ctx.risk_cue is not None:
            risk = ctx.risk_cue.predict(ctx.step)
        cfg = base_cfg
        if np.isfinite(risk):
            if risk > base_cfg.nh4_ref + 1.0:
                cfg = replace(
                    base_cfg,
                    do_ref=min(3.1, base_cfg.do_ref + 0.5),
                    u_min=max(base_cfg.u_min, 120.0),
                    w_nh4=base_cfg.w_nh4 * 1.35,
                    w_energy=base_cfg.w_energy * 0.85,
                )
            elif risk < base_cfg.nh4_ref - 1.0:
                cfg = replace(
                    base_cfg,
                    do_ref=max(1.5, base_cfg.do_ref - 0.25),
                    w_energy=base_cfg.w_energy * 1.25,
                )
        return cfg

    def compute_raw(self, ctx: ControllerContext) -> Tuple[float, Dict[str, float]]:
        raw, info = super().compute_raw(ctx)
        info["nh4_future"] = ctx.risk_cue.last_prediction if ctx.risk_cue is not None else np.nan
        info["controller_mode"] = 12.0
        return raw, info


def apply_safety_and_actuator(
    u_raw: float,
    prev_u: float,
    meas: Dict[str, float],
    safety: SafetyConfig,
) -> Tuple[float, Dict[str, float]]:
    hard = False
    u_safe = u_raw
    if meas["meas_DO"] < safety.do_floor:
        u_safe = max(u_safe, safety.hard_safety_floor_u)
        hard = True
    if meas["meas_NH4"] > safety.nh4_limit * safety.nh4_safety_factor:
        u_safe = max(u_safe, prev_u + safety.nh4_boost)
        hard = True
    rate_limited = float(np.clip(u_safe, prev_u - safety.rate_limit, prev_u + safety.rate_limit))
    clipped = float(np.clip(rate_limited, safety.u_min, safety.u_max))
    actuator = (abs(clipped - u_safe) > 1e-9) or (abs(rate_limited - u_safe) > 1e-9)
    return clipped, {"hard_safety": float(hard), "actuator_intervention": float(actuator)}


def default_controllers(nh4_limit: float = 8.0) -> Dict[str, BaseController]:
    return {
        "DO-PID": DOPIDController(PIDConfig(do_sp=2.0)),
        "ABAC-PID": ABACPIDController(ABACConfig(nh4_ref=nh4_limit)),
        "Hybrid-Risk-PID": HybridRiskPIDController(ABACConfig(nh4_ref=nh4_limit)),
        "MPC": MPCController(MPCConfig(nh4_ref=nh4_limit)),
        "Risk-cue-MPC": RiskCueMPCController(MPCConfig(nh4_ref=nh4_limit)),
    }
