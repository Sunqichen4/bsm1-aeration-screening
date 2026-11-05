from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


STATE_NAMES = ["XH", "SS", "XA", "SNO", "SNH", "SO"]
IDX = {name: i for i, name in enumerate(STATE_NAMES)}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    flow_scale: float = 1.0
    cod_scale: float = 1.0
    nh4_scale: float = 1.0
    mu_h_scale: float = 1.0
    mu_a_scale: float = 1.0
    kla_scale: float = 1.0
    do_noise: float = 0.02
    nh4_noise: float = 0.05
    do_bias: float = 0.0
    nh4_bias: float = 0.0
    nitrification_cap_scale: float = 1.0
    temperature_c: float = 20.0


SCENARIO_ALIASES = {
    # Legacy command-line aliases are accepted only at the input boundary.
    # Every result table and manifest must use the canonical ET&I names below.
    "low_temp": "low_temperature",
    "low_aeration_eff": "low_oxygen_transfer",
}


def normalize_scenario_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return SCENARIO_ALIASES.get(normalized, normalized)


SCENARIOS: Dict[str, ScenarioConfig] = {
    "baseline": ScenarioConfig("baseline"),
    "high_load": ScenarioConfig("high_load", cod_scale=1.20, nh4_scale=1.35, mu_a_scale=1.05),
    "low_temperature": ScenarioConfig(
        "low_temperature",
        mu_h_scale=0.75,
        mu_a_scale=0.60,
        do_noise=0.05,
        nitrification_cap_scale=0.50,
        temperature_c=10.0,
    ),
    "low_oxygen_transfer": ScenarioConfig("low_oxygen_transfer", kla_scale=0.70),
    "dilution_rain": ScenarioConfig("dilution_rain", flow_scale=1.35, cod_scale=0.65, nh4_scale=0.80),
    "sensor_fault": ScenarioConfig(
        "sensor_fault",
        do_noise=0.15,
        nh4_noise=0.30,
        do_bias=0.10,
        nh4_bias=0.20,
    ),
}


MANUSCRIPT_SCENARIOS = ("baseline", "high_load", "low_oxygen_transfer", "low_temperature")
COARSE_KLA_VALUES = (40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0, 360.0)
FINE_KLA_VALUES_BY_SCENARIO = {
    "baseline": (120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0),
    "high_load": (180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0),
    "low_oxygen_transfer": (200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0),
    "low_temperature": (360.0, 400.0, 440.0, 480.0, 520.0),
}


def _unique_sorted(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(sorted({float(v) for v in values}))


KLA_GRID_PRESETS = {
    "coarse": {"default": COARSE_KLA_VALUES},
    "fine": FINE_KLA_VALUES_BY_SCENARIO,
    "coarse_plus_fine": {
        scenario: _unique_sorted(COARSE_KLA_VALUES + FINE_KLA_VALUES_BY_SCENARIO.get(scenario, ()))
        for scenario in MANUSCRIPT_SCENARIOS
    },
}


def kla_values_for_scenario(
    scenario: str,
    grid_preset: str = "coarse",
    custom_values: Sequence[float] | None = None,
) -> list[float]:
    """Return the fixed-aeration KLa grid for a manuscript scenario.

    Use ``grid_preset='coarse'`` to reproduce the current manuscript grid and
    ``grid_preset='fine'`` to run only threshold-neighborhood refinement points.
    Passing ``custom_values`` applies the same user-specified grid to every
    scenario and is intended for quick PyCharm tests or reviewer-requested grids.
    """
    scenario = normalize_scenario_name(scenario)
    if custom_values is not None:
        return list(_unique_sorted(custom_values))
    preset = str(grid_preset).strip().lower()
    if preset not in KLA_GRID_PRESETS:
        raise ValueError(f"Unknown KLa grid preset: {grid_preset}")
    values = KLA_GRID_PRESETS[preset].get(scenario, KLA_GRID_PRESETS[preset].get("default"))
    if values is None:
        raise ValueError(f"No KLa grid configured for scenario={scenario!r}, preset={grid_preset!r}")
    return list(values)


@dataclass
class BenchmarkConfig:
    days: float = 10.0
    dt_minutes: float = 15.0
    warmup_fraction: float = 0.25
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    scenarios: List[str] = field(default_factory=lambda: list(SCENARIOS.keys()))
    output_dir: Path = Path("outputs")
    nh4_limit: float = 4.0
    do_low_limit: float = 0.8
    do_target: float = 2.0
    save_timeseries: bool = True
    # Set to False for a fast smoke test.
    use_risk_cue: bool = True

    @property
    def dt_hours(self) -> float:
        return self.dt_minutes / 60.0

    @property
    def dt_days(self) -> float:
        return self.dt_hours / 24.0

    @property
    def n_steps(self) -> int:
        return int(round(self.days * 24.0 / self.dt_hours))

    @property
    def warmup_steps(self) -> int:
        return int(round(self.n_steps * self.warmup_fraction))


@dataclass
class EnergyConfig:
    alpha: float = 60.0
    u_ref: float = 240.0
    beta: float = 1.35


@dataclass
class SafetyConfig:
    u_min: float = 40.0
    u_max: float = 450.0
    rate_limit: float = 45.0
    do_floor: float = 0.7
    nh4_limit: float = 4.0
    nh4_boost: float = 80.0
    # Safety intervention is deliberately separated from the evaluation limit.
    # This avoids forcing every controller to max aeration whenever NH4 is just
    # above the reporting threshold; interventions only occur for severe risk.
    nh4_safety_factor: float = 2.2
    hard_safety_floor_u: float = 160.0
