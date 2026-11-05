from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import IDX, ScenarioConfig


@dataclass
class SurrogateParams:
    """Reduced ASM/BSM1-inspired parameter set.

    The model is intentionally reduced. It is used for high-throughput, auditable
    controller comparison. Use bsm1_validation.py for the external BSM1
    model-structure check.
    """

    tau_h: float = 8.0
    mu_h: float = 0.75
    mu_a: float = 1.05
    k_s: float = 30.0
    k_nh: float = 1.0
    k_oh: float = 0.35
    k_oa: float = 3.00
    hill_n: float = 2.0
    so_sat: float = 8.0
    oxygen_gain: float = 0.55
    y_no: float = 3.0
    ss_to_nh: float = 0.030
    nitrification_rate_cap: float = 1.60
    nh4_nitrification_factor: float = 4.8
    # Biomass states are reduced-model proxies rather than concentrations that
    # are numerically identical to BSM1.  The following parameters make that
    # proxy mapping explicit and give the states a finite, auditable dynamic
    # equilibrium instead of relying on a persistent lower-bound intervention.
    biomass_tau_h: float = 120.0
    biomass_tau_a: float = 120.0
    xh_reference: float = 1200.0
    xa_reference: float = 534.0
    bsm1_xba_asymptote: float = 152.5
    bsm1_xba_logistic_slope: float = 0.080
    bsm1_xba_logistic_midpoint: float = 123.7
    xh_oxygen_response_weight: float = 0.018
    biomass_load_exponent: float = 0.50
    biomass_temperature_exponent: float = 1.0
    # Kept for backwards-compatible manifests and old parameter files.  The
    # biomass equations below now use calibrated target relaxation, so these
    # legacy first-order decay coefficients are no longer applied to XH/XA.
    decay_h: float = 0.015
    decay_a: float = 0.010
    min_vals: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    max_vals: Tuple[float, ...] = (5000.0, 500.0, 1200.0, 80.0, 80.0, 8.0)


@dataclass
class InfluentPoint:
    cod: float
    nh4: float
    q_scale: float


class InfluentGenerator:
    def __init__(self, scenario: ScenarioConfig, seed: int):
        self.scenario = scenario
        self.rng = np.random.default_rng(seed + 10_000)
        self.cod_noise = self.rng.normal(0.0, 1.0, 20000)
        self.nh4_noise = self.rng.normal(0.0, 1.0, 20000)

    def at(self, k: int, t_days: float) -> InfluentPoint:
        # Smooth daily and weekly components plus reproducible noise.
        daily = np.sin(2 * np.pi * t_days - 0.5)
        daily2 = np.sin(4 * np.pi * t_days + 0.8)
        weekly = np.sin(2 * np.pi * t_days / 7.0 + 0.2)
        cod = 260.0 + 45.0 * daily + 15.0 * daily2 + 20.0 * weekly
        nh4 = 30.0 + 5.0 * np.sin(2 * np.pi * t_days - 0.2) + 2.0 * weekly

        # A transient high-load pulse; present in every scenario but scaled later.
        if 4.2 <= t_days <= 4.8 or 7.1 <= t_days <= 7.4:
            cod += 45.0
            nh4 += 7.0

        cod += 8.0 * self.cod_noise[k % len(self.cod_noise)]
        nh4 += 1.0 * self.nh4_noise[k % len(self.nh4_noise)]
        cod = max(20.0, cod * self.scenario.cod_scale)
        nh4 = max(0.5, nh4 * self.scenario.nh4_scale)
        return InfluentPoint(cod=cod, nh4=nh4, q_scale=self.scenario.flow_scale)


class ReducedASM1Surrogate:
    def __init__(self, params: SurrogateParams | None = None):
        self.params = params or SurrogateParams()

    def initial_state(self) -> np.ndarray:
        return np.array([1200.0, 75.0, 110.0, 6.0, 8.0, 2.0], dtype=float)

    def measure(self, x_pre: np.ndarray, scenario: ScenarioConfig, rng: np.random.Generator) -> Dict[str, float]:
        do = x_pre[IDX["SO"]] + scenario.do_bias + rng.normal(0.0, scenario.do_noise)
        nh4 = x_pre[IDX["SNH"]] + scenario.nh4_bias + rng.normal(0.0, scenario.nh4_noise)
        return {
            "meas_DO": float(max(0.0, do)),
            "meas_NH4": float(max(0.0, nh4)),
            "meas_COD": float(max(0.0, x_pre[IDX["SS"]])),
        }

    def rates(self, x: np.ndarray, u: float, influent: InfluentPoint, scenario: ScenarioConfig) -> Dict[str, float]:
        p = self.params
        XH, SS, XA, SNO, SNH, SO = x
        mu_h = p.mu_h * scenario.mu_h_scale
        mu_a = p.mu_a * scenario.mu_a_scale
        oxygen_h = SO / (p.k_oh + SO + 1e-9)
        oxygen_a = (SO ** p.hill_n) / (p.k_oa ** p.hill_n + SO ** p.hill_n + 1e-9)
        r_h = mu_h * (SS / (p.k_s + SS + 1e-9)) * oxygen_h * (XH / 1000.0)
        r_a_unlimited = mu_a * (SNH / (p.k_nh + SNH + 1e-9)) * oxygen_a * (XA / 100.0)
        r_a = min(r_a_unlimited, p.nitrification_rate_cap * scenario.nitrification_cap_scale)
        kla_eff = scenario.kla_scale * p.oxygen_gain * (u / 240.0)
        r_o2 = kla_eff * (p.so_sat - SO)
        return {"r_h": float(r_h), "r_a": float(r_a), "r_o2": float(r_o2)}

    def biomass_equilibrium_targets(
        self,
        u: float,
        influent: InfluentPoint,
        scenario: ScenarioConfig,
    ) -> Dict[str, float]:
        """Return calibrated equilibrium targets for the two biomass proxies.

        The BSM1 biomass export showed that heterotrophic biomass varies only
        weakly over the tested KLa range, whereas autotrophic biomass follows a
        saturating KLa response.  A smooth logistic response is used here so
        the reduced model remains well behaved between and outside the BSM1
        calibration points.  Influent-load factors and the scenario activity
        scales provide the intended dynamic adaptation for stress cases.

        ``XH`` and ``XA`` are proxy states.  Their scales (1200 and 534) are
        deliberately explicit and should not be interpreted as direct BSM1
        concentrations; the BSM1 values enter through the normalized response
        curve and its reference point.
        """
        p = self.params
        effective_kla = max(0.0, float(u)) * max(float(scenario.kla_scale), 0.0)

        # Smooth approximation to the exported BSM1 XBA response at KLa
        # = 120--180: approximately 65, 94, 119, 136, 145, 149 and 151.
        logistic_arg = p.bsm1_xba_logistic_slope * (
            effective_kla - p.bsm1_xba_logistic_midpoint
        )
        logistic_arg = float(np.clip(logistic_arg, -60.0, 60.0))
        xba_activity = p.bsm1_xba_asymptote / (1.0 + np.exp(-logistic_arg))
        ref_arg = p.bsm1_xba_logistic_slope * (160.0 - p.bsm1_xba_logistic_midpoint)
        ref_arg = float(np.clip(ref_arg, -60.0, 60.0))
        xba_reference_activity = p.bsm1_xba_asymptote / (1.0 + np.exp(-ref_arg))
        oxygen_response = xba_activity / max(xba_reference_activity, 1e-12)

        cod_load = max(float(influent.cod) * max(float(influent.q_scale), 1e-6) / 260.0, 1e-6)
        nh4_load = max(float(influent.nh4) * max(float(influent.q_scale), 1e-6) / 30.0, 1e-6)
        load_h = cod_load ** p.biomass_load_exponent
        load_a = nh4_load ** p.biomass_load_exponent
        temperature_activity = max(float(scenario.mu_a_scale), 1e-6) ** p.biomass_temperature_exponent

        xh_eq = p.xh_reference * load_h * (
            1.0 + p.xh_oxygen_response_weight * (oxygen_response - 1.0)
        )
        xa_eq = p.xa_reference * oxygen_response * load_a * temperature_activity
        return {"XH_eq": float(max(xh_eq, 0.0)), "XA_eq": float(max(xa_eq, 0.0))}

    def state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return validated lower and upper state bounds.

        Infinite bounds are allowed so that a single state constraint can be
        disabled during a diagnostic experiment without changing the equations
        or the constraints on the other states.
        """
        lower = np.asarray(self.params.min_vals, dtype=float)
        upper = np.asarray(self.params.max_vals, dtype=float)
        expected_shape = self.initial_state().shape
        if lower.shape != expected_shape or upper.shape != expected_shape:
            raise ValueError(
                "Surrogate state bounds must contain one value for each state; "
                f"expected shape {expected_shape}, got {lower.shape} and {upper.shape}."
            )
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("Surrogate state bounds cannot contain NaN values.")
        if np.any(lower >= upper):
            raise ValueError("Every surrogate lower state bound must be smaller than its upper bound.")
        return lower, upper

    def unbounded_step(
        self,
        x: np.ndarray,
        u: float,
        influent: InfluentPoint,
        scenario: ScenarioConfig,
        dt_h: float,
    ) -> np.ndarray:
        """Return one explicit-Euler candidate before applying state bounds."""
        p = self.params
        XH, SS, XA, SNO, SNH, SO = x
        tau = p.tau_h / max(influent.q_scale, 1e-6)
        r = self.rates(x, u, influent, scenario)
        r_h, r_a, r_o2 = r["r_h"], r["r_a"], r["r_o2"]

        targets = self.biomass_equilibrium_targets(u, influent, scenario)
        dXH = (targets["XH_eq"] - XH) / max(p.biomass_tau_h, 1e-6)
        dXA = (targets["XA_eq"] - XA) / max(p.biomass_tau_a, 1e-6)
        dSS = (influent.cod - SS) / tau - 18.0 * r_h
        dSNH = (
            (influent.nh4 - SNH) / tau
            + p.ss_to_nh * max(SS, 0.0) / tau
            - p.nh4_nitrification_factor * r_a
        )
        dSNO = (2.0 - SNO) / tau + p.y_no * r_a - 0.25 * r_h
        dSO = (0.4 - SO) / tau + r_o2 - 0.40 * r_h - 2.50 * r_a

        dx = np.array([dXH, dSS, dXA, dSNO, dSNH, dSO], dtype=float)
        return x + dt_h * dx

    def step_with_bound_info(
        self,
        x: np.ndarray,
        u: float,
        influent: InfluentPoint,
        scenario: ScenarioConfig,
        dt_h: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Advance one step and report which constraints altered the candidate.

        ``lower_clipped`` and ``upper_clipped`` identify a genuine numerical
        intervention: the Euler candidate lies outside a configured bound.
        ``at_lower`` and ``at_upper`` identify states that finish the step on a
        finite bound, including exact equality. These two concepts are reported
        separately because a state can remain on a bound for many consecutive
        steps.
        """
        candidate = self.unbounded_step(x, u, influent, scenario, dt_h)
        lower, upper = self.state_bounds()
        bounded = np.clip(candidate, lower, upper)
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        lower_clipped = finite_lower & (candidate < lower)
        upper_clipped = finite_upper & (candidate > upper)
        at_lower = finite_lower & np.isclose(bounded, lower, rtol=0.0, atol=1e-10)
        at_upper = finite_upper & np.isclose(bounded, upper, rtol=0.0, atol=1e-10)
        return bounded, {
            "candidate": candidate,
            "lower_clipped": lower_clipped,
            "upper_clipped": upper_clipped,
            "at_lower": at_lower,
            "at_upper": at_upper,
        }

    def step(self, x: np.ndarray, u: float, influent: InfluentPoint, scenario: ScenarioConfig, dt_h: float) -> np.ndarray:
        # Keep the normal simulation path lightweight. The extra masks are only
        # constructed when the dedicated audit calls step_with_bound_info().
        candidate = self.unbounded_step(x, u, influent, scenario, dt_h)
        return np.clip(candidate, np.asarray(self.params.min_vals), np.asarray(self.params.max_vals))

    def clone(self) -> "ReducedASM1Surrogate":
        return ReducedASM1Surrogate(self.params)
