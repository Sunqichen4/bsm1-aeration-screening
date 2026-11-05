"""Explicit load/transfer combinations; no mutation of the legacy scenarios."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .config import ScenarioConfig, normalize_scenario_name


@dataclass(frozen=True)
class SteadyOperatingPoint:
    name: str
    load_coordinate: float
    oxygen_transfer_factor: float

    def __post_init__(self) -> None:
        if not self.name or normalize_scenario_name(self.name) != self.name:
            raise ValueError("Use a nonempty canonical operating-point name.")
        if not math.isfinite(self.load_coordinate) or self.load_coordinate < 0:
            raise ValueError("Load coordinate must be finite and nonnegative.")
        if not math.isfinite(self.oxygen_transfer_factor) or not 0 < self.oxygen_transfer_factor <= 1:
            raise ValueError("Oxygen-transfer factor must be in (0, 1].")

    def surrogate_scenario(self) -> ScenarioConfig:
        # Interpolate the OLD high-load assumption, including its activity
        # multiplier. This is not new calibration or a BSM1 kinetic change.
        return ScenarioConfig(
            self.name, cod_scale=1 + .20 * self.load_coordinate,
            nh4_scale=1 + .35 * self.load_coordinate,
            mu_a_scale=1 + .05 * self.load_coordinate,
            kla_scale=self.oxygen_transfer_factor,
        )

    def metadata(self) -> dict:
        op = self.surrogate_scenario()
        return {
            "load_coordinate": self.load_coordinate,
            "cod_scale": op.cod_scale, "nh4_scale": op.nh4_scale,
            "flow_scale": 1.0, "surrogate_mu_a_scale": op.mu_a_scale,
            "oxygen_transfer_factor": self.oxygen_transfer_factor,
            "operating_point_definition": "steady_load_transfer_v1",
        }
