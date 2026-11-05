from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MINUTES_PER_HOUR = 60.0
HOURS_PER_DAY = 24.0
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY


def minutes_to_hours(minutes: float) -> float:
    value = float(minutes)
    if value <= 0.0:
        raise ValueError("A time step must be positive.")
    return value / MINUTES_PER_HOUR


def minutes_to_days(minutes: float) -> float:
    return minutes_to_hours(minutes) / HOURS_PER_DAY


def days_to_steps(days: float, timestep_min: float) -> int:
    duration = float(days)
    if duration < 0.0:
        raise ValueError("Duration cannot be negative.")
    return int(round(duration / minutes_to_days(timestep_min)))


@dataclass(frozen=True)
class SimulationWindow:
    warmup_days: float
    evaluation_days: float
    timestep_min: float

    def __post_init__(self) -> None:
        if self.warmup_days < 0.0 or self.evaluation_days <= 0.0:
            raise ValueError("warmup_days must be non-negative and evaluation_days positive.")
        minutes_to_days(self.timestep_min)

    @property
    def warmup_steps(self) -> int:
        return days_to_steps(self.warmup_days, self.timestep_min)

    @property
    def evaluation_steps(self) -> int:
        return days_to_steps(self.evaluation_days, self.timestep_min)

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.evaluation_steps

    @property
    def dt_hours(self) -> float:
        return minutes_to_hours(self.timestep_min)

    @property
    def dt_days(self) -> float:
        return minutes_to_days(self.timestep_min)

    @property
    def evaluation_slice(self) -> slice:
        return slice(self.warmup_steps, self.total_steps)

    def time_days(self, *, interval_end: bool = False) -> np.ndarray:
        offset = 1 if interval_end else 0
        return (np.arange(self.total_steps, dtype=float) + offset) * self.dt_days
