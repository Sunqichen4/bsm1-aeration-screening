from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import re
import time
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .utils import write_json
from .config import COARSE_KLA_VALUES, kla_values_for_scenario, normalize_scenario_name
from .fixed_aeration import commanded_aeration_index
from .steady_conditions import SteadyOperatingPoint

# BSM1 state indices used by bsm2-python examples.
SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK, TSS, Q, TEMP, SD1, SD2, SD3, XD4, XD5 = range(21)

# ASM1/BSM1 nitrogen fractions used by bsm2-python's BSM1 initialisation.
# These values convert the COD-based biomass/product states to g(N) per
# g(COD) when constructing a model-consistent total nitrogen concentration.
BSM1_I_XB = 0.08
BSM1_I_XP = 0.06

# Constant influent used in the BSM2-Python BSM1 reference test.
# Columns: time + 21 ASM/BSM components
# [SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK, TSS, Q, TEMP, SD1, SD2, SD3, XD4, XD5]
BSM1_CONST_INFLUENT_ROW = np.array(
    [30, 69.5, 51.2, 202.32, 28.17, 0, 0, 0, 0, 31.56, 6.95, 10.59, 7, 211.2675, 18446, 15, 0, 0, 0, 0, 0],
    dtype=float,
)

BSM1_REACTOR_NUMBERS = (1, 2, 3, 4, 5)
BSM1_AERATED_REACTOR_NUMBERS = (3, 4, 5)

# The packaged BSM1 dry-weather influent is supplied at 15-min resolution.
# The plant can still be integrated at 1 min: the source series is resampled
# with zero-order hold, so no artificial concentration values are created
# between two influent observations.  Do not confuse this file with the
# bsm2-python default ``dyninfluent_bsm2.csv`` input.
BSM1_DYNAMIC_SOURCE_TIMESTEP_MIN = 15.0
BSM1_DYNAMIC_WEATHER_PROFILES = {
    "dynamic_dry": {
        "label": "dry",
        "provenance": "packaged_bsm1_dry_reference",
        "source_file": "dryinfluent.csv",
        "source_kind": "direct_packaged_file",
        "source_timestep_note": "Package timestamps represent the nominal 15-min BSM1 source resolution.",
        "description": "Packaged BSM1 dry-weather influent repeated on its native 14-day cycle.",
        "base_q_scale": 1.00,
        "base_cod_scale": 1.00,
        "base_n_scale": 1.00,
        "base_alkalinity_scale": 1.00,
        "base_tss_scale": 1.00,
        "event_windows_days_mod_7": [],
    },
    "dynamic_rain": {
        "label": "rain",
        "provenance": "packaged_bsm1_rain_reference",
        "source_file": "raininfluent.csv",
        "source_kind": "direct_packaged_file",
        "source_timestep_note": "Nominal 15-min BSM1 source; package timestamps are stored at 0.01-d precision (approximately 14.4 min).",
        "description": "Packaged BSM1 rain-weather influent repeated on its native 14-day cycle.",
        "base_q_scale": 1.00,
        "base_cod_scale": 1.00,
        "base_n_scale": 1.00,
        "base_alkalinity_scale": 1.00,
        "base_tss_scale": 1.00,
        "event_windows_days_mod_7": [],
        "event_q_scale": 1.00,
        "event_cod_scale": 1.00,
        "event_n_scale": 1.00,
        "event_alkalinity_scale": 1.00,
        "event_tss_scale": 1.00,
    },
    "dynamic_storm": {
        "label": "storm",
        "provenance": "deterministic_storm_transformation_of_packaged_bsm1_dry_reference",
        "source_file": "dryinfluent.csv",
        "source_kind": "synthetic_transformation",
        "source_timestep_note": "Uses the nominal 15-min dry BSM1 source resolution before the synthetic transformation.",
        "description": "Packaged BSM1 dry-weather influent with a documented short, severe synthetic storm pulse.",
        "base_q_scale": 1.00,
        "base_cod_scale": 1.00,
        "base_n_scale": 1.00,
        "base_alkalinity_scale": 1.00,
        "base_tss_scale": 1.00,
        "event_windows_days_mod_7": [(3.0, 3.25)],
        "event_q_scale": 2.50,
        "event_cod_scale": 0.35,
        "event_n_scale": 0.50,
        "event_alkalinity_scale": 0.55,
        "event_tss_scale": 0.40,
    },
}


@dataclass
class BSM1ABACConfig:
    """Transparent ABAC-PID controller for BSM1 model-structure checks.

    The controller is intentionally simple and auditable. It should not be
    presented as a new BSM1 controller; it is only used to check whether a
    conventional ammonium-based aeration loop behaves plausibly in BSM1.
    """

    nh4_ref: float = 4.0
    do_base: float = 1.8
    do_sp_min: float = 0.8
    do_sp_max: float = 4.0
    abac_gain: float = 0.35
    kp: float = 110.0
    ki: float = 14.0
    kd: float = 0.0
    kla_min: float = 0.0
    kla_max: float = 300.0
    rate_limit: float = 45.0
    integral_limit: float = 12.0


class BSM1ABACPID:
    def __init__(self, cfg: BSM1ABACConfig, dt_d: float):
        self.cfg = cfg
        self.dt_d = dt_d
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_kla = 120.0

    def compute(self, meas_do: float, meas_nh4: float) -> tuple[float, float]:
        # NH4 raises the DO target, making the ABAC loop risk-aware while still
        # relying on a conventional DO-PID inner loop.
        do_sp = self.cfg.do_base + self.cfg.abac_gain * (meas_nh4 - self.cfg.nh4_ref)
        do_sp = float(np.clip(do_sp, self.cfg.do_sp_min, self.cfg.do_sp_max))

        err = do_sp - meas_do
        self.integral = float(np.clip(self.integral + err * self.dt_d, -self.cfg.integral_limit, self.cfg.integral_limit))
        deriv = (err - self.prev_error) / max(self.dt_d, 1e-12)
        self.prev_error = err

        raw = self.last_kla + self.cfg.kp * err + self.cfg.ki * self.integral + self.cfg.kd * deriv
        raw = float(np.clip(raw, self.last_kla - self.cfg.rate_limit, self.last_kla + self.cfg.rate_limit))
        kla = float(np.clip(raw, self.cfg.kla_min, self.cfg.kla_max))
        self.last_kla = kla
        return kla, do_sp


def require_bsm1ol():
    try:
        from bsm2_python.bsm1_ol import BSM1OL
    except Exception as exc:
        raise ImportError(
            "Could not import bsm2_python.bsm1_ol.BSM1OL. Install bsm2-python and its dependencies first, for example:\n"
            "  python -m pip install bsm2-python\n"
            "If you installed from a downloaded GitHub folder, run this inside that folder instead:\n"
            "  python -m pip install -e .\n"
            f"Original import error: {exc}\n"
            "The surrogate benchmark can run without this optional package."
        ) from exc
    return BSM1OL


def installed_bsm2_python_version() -> str:
    """Return the installed distribution version without importing the model."""
    try:
        return package_version("bsm2-python")
    except PackageNotFoundError:
        return "unknown"


def parse_float_list(values: str | Sequence[float]) -> list[float]:
    if isinstance(values, str):
        return [float(x.strip()) for x in values.split(",") if x.strip()]
    return [float(x) for x in values]


def make_constant_influent(total_days: float) -> np.ndarray:
    """Return constant BSM1 influent over the requested simulation horizon."""
    end_marker = max(float(total_days) + 1.0, 200.1)
    return np.vstack(
        [
            np.concatenate([[0.0], BSM1_CONST_INFLUENT_ROW.copy()]),
            np.concatenate([[end_marker], BSM1_CONST_INFLUENT_ROW.copy()]),
        ]
    )


def make_operating_point_data(total_days: float, point: SteadyOperatingPoint) -> np.ndarray:
    """Apply load scaling once to the unmodified BSM1 reference influent.

    As in the old high_load scenario, COD scaling applies to SS and XS only;
    NH4 scaling applies to SNH. Flow, other fractions and kinetics stay fixed.
    """
    data = make_constant_influent(total_days)
    scenario = point.surrogate_scenario()
    data[:, 1 + SS] *= scenario.cod_scale
    data[:, 1 + XS] *= scenario.cod_scale
    data[:, 1 + SNH] *= scenario.nh4_scale
    return data


def make_scenario_data(base_data_in: np.ndarray, scenario: str) -> np.ndarray:
    """Apply simple scenario perturbations to BSM1 influent data.

    For low_oxygen_transfer the influent is unchanged; the scenario is implemented
    by lowering effective KLa when the model is stepped.
    """
    scenario = normalize_scenario_name(scenario)
    data = base_data_in.copy()
    if scenario == "baseline":
        return data
    if scenario == "high_load":
        data[:, 1 + SS] *= 1.20
        data[:, 1 + XS] *= 1.20
        data[:, 1 + SNH] *= 1.35
        return data
    if scenario == "dilution_rain":
        data[:, 1 + Q] *= 1.35
        data[:, 1 + SS] *= 0.65
        data[:, 1 + XS] *= 0.65
        data[:, 1 + SNH] *= 0.80
        return data
    if scenario == "low_temperature":
        data[:, 1 + TEMP] = 10.0
        return data
    if scenario == "low_oxygen_transfer":
        return data
    raise ValueError(f"Unknown BSM1 scenario: {scenario}")


def resample_influent_zero_order_hold(
    data_in: np.ndarray,
    target_timestep_days: float,
    total_days: float,
) -> np.ndarray:
    """Resample a time-stamped influent series using a zero-order hold.

    BSM1 dynamic observations are treated as piecewise constant between two
    source observations.  This makes the relationship between a 15-min source
    series and a 1-min integration step explicit and reproducible.
    """
    data = np.asarray(data_in, dtype=float)
    if data.ndim != 2 or data.shape[1] < 22:
        raise ValueError(f"BSM1 influent must have shape (n, >=22); got {data.shape}.")
    step = float(target_timestep_days)
    total = float(total_days)
    if step <= 0.0 or total < 0.0:
        raise ValueError("target_timestep_days must be positive and total_days cannot be negative.")
    source_times = np.asarray(data[:, 0], dtype=float)
    if np.any(~np.isfinite(source_times)) or np.any(np.diff(source_times) < 0.0):
        raise ValueError("Influent source times must be finite and non-decreasing.")
    target_times = np.arange(0.0, total + 0.5 * step, step, dtype=float)
    if target_times.size == 0 or target_times[-1] < total - 1e-12:
        target_times = np.append(target_times, total)
    target_times = np.minimum(target_times, total)
    source_indices = np.searchsorted(source_times, target_times, side="right") - 1
    source_indices = np.clip(source_indices, 0, len(source_times) - 1)
    result = data[source_indices].copy()
    result[:, 0] = target_times
    return result


_LOCALE_FLOW_TOKEN = re.compile(
    r"^(?P<sign>[+-]?)(?P<thousands>\d{1,3})\.(?P<hundreds>\d{3})(?:\.(?P<decimal>\d+))?$"
)


def parse_packaged_bsm1_flow_token(token: str) -> float:
    """Parse a BSM1 influent flow token, including locale-formatted values.

    The bsm2-python 0.0.16 ``raininfluent.csv`` file contains a few flow values
    written with a thousands separator and decimal point in the same token,
    e.g. ``30.044.50`` for 30044.50 m3/d.  Treating this as an ordinary float
    would fail and silently dropping the suffix would lose the decimal part.
    This parser is restricted to the flow column and leaves all other columns
    under the normal floating-point parser.
    """
    value = str(token).strip()
    match = _LOCALE_FLOW_TOKEN.fullmatch(value)
    if match:
        normalized = (
            f"{match.group('sign')}{match.group('thousands')}{match.group('hundreds')}"
        )
        if match.group("decimal") is not None:
            normalized += f".{match.group('decimal')}"
        return float(normalized)
    return float(value)


def _load_packaged_bsm1_csv(data_path: Path) -> np.ndarray:
    """Load a package influent CSV and normalize the flow column only."""
    rows: list[list[float]] = []
    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 22:
                raise ValueError(
                    f"Unexpected column count in {data_path.name} line {line_number}: "
                    f"expected 22, got {len(row)}."
                )
            parsed: list[float] = []
            for column, token in enumerate(row):
                try:
                    parsed.append(
                        parse_packaged_bsm1_flow_token(token)
                        if column == 1 + Q
                        else float(token)
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value in {data_path.name} line {line_number}, "
                        f"column {column + 1}: {token!r}."
                    ) from exc
            rows.append(parsed)
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != 22 or data.shape[0] < 2:
        raise ValueError(f"Unexpected packaged BSM1 influent shape: {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Packaged BSM1 influent contains non-finite values: {data_path}.")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"Packaged BSM1 influent times must be strictly increasing: {data_path}.")
    if np.any(data[:, Q] <= 0.0):
        raise ValueError(f"Packaged BSM1 influent flow must be positive: {data_path}.")
    return data


def load_packaged_bsm1_weather_influent(BSM1OL, inflow_mode: str) -> np.ndarray:
    """Load the package-supplied dry or rain BSM1 influent at nominal 15 min."""
    mode = normalize_inflow_mode(inflow_mode)
    profile = BSM1_DYNAMIC_WEATHER_PROFILES.get(mode)
    if profile is None:
        raise ValueError(f"No packaged BSM1 weather profile is defined for {inflow_mode!r}.")
    module = import_module(BSM1OL.__module__)
    data_path = Path(module.__file__).resolve().parent / "data" / profile["source_file"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"The installed bsm2-python package does not contain {data_path}. "
            f"The {profile['label']} weather validation requires that source file."
        )
    return _load_packaged_bsm1_csv(data_path)


def load_packaged_bsm1_dry_influent(BSM1OL) -> np.ndarray:
    """Backward-compatible wrapper for the packaged BSM1 dry-weather input."""
    return load_packaged_bsm1_weather_influent(BSM1OL, "dynamic_dry")


def repeat_periodic_influent(
    source_data: np.ndarray,
    total_days: float,
    period_days: float = 14.0,
) -> np.ndarray:
    """Repeat one time-stamped influent cycle until it covers the horizon."""
    source = np.asarray(source_data, dtype=float)
    total = float(total_days)
    period = float(period_days)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] < 22:
        raise ValueError(f"Periodic BSM1 source must have shape (n>=2, >=22); got {source.shape}.")
    if total < 0.0 or period <= 0.0:
        raise ValueError("total_days cannot be negative and period_days must be positive.")
    base = source.copy()
    base[:, 0] -= float(base[0, 0])
    if np.any(np.diff(base[:, 0]) <= 0.0):
        raise ValueError("Periodic BSM1 source times must be strictly increasing.")
    n_cycles = max(1, int(np.ceil((total + period) / period)))
    blocks = []
    for cycle in range(n_cycles):
        block = base.copy()
        block[:, 0] += cycle * period
        blocks.append(block)
    repeated = np.vstack(blocks)
    # Retain one source observation beyond the requested end when possible;
    # it provides a safe final hold value for downstream resampling.
    beyond = np.flatnonzero(repeated[:, 0] > total)
    stop = int(beyond[0] + 1) if beyond.size else len(repeated)
    return repeated[:stop]


def _scale_influent_columns(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    q_scale: float,
    cod_scale: float,
    n_scale: float,
    alkalinity_scale: float,
    tss_scale: float,
) -> None:
    """Apply a weather multiplier to the concentration/flow columns in place."""
    if not np.any(mask):
        return
    data[mask, 1 + Q] *= float(q_scale)
    for state in (SI, SS, XI, XS):
        data[mask, 1 + state] *= float(cod_scale)
    for state in (SNH, SND, XND):
        data[mask, 1 + state] *= float(n_scale)
    data[mask, 1 + SALK] *= float(alkalinity_scale)
    data[mask, 1 + TSS] *= float(tss_scale)


def apply_dynamic_weather_profile(data_in: np.ndarray, inflow_mode: str) -> np.ndarray:
    """Apply a deterministic dry/rain/storm profile to dynamic BSM1 data.

    The underlying time-varying BSM1 influent is retained.  Profiles only add
    transparent flow/concentration multipliers; event windows repeat weekly
    so that long warm-up periods remain reproducible.
    """
    mode = normalize_inflow_mode(inflow_mode)
    data = np.asarray(data_in, dtype=float).copy()
    profile = BSM1_DYNAMIC_WEATHER_PROFILES.get(mode)
    if profile is None:
        if mode == "dynamic_default":
            return data
        raise ValueError(
            "Unknown dynamic weather mode. Use dynamic_default, dynamic_dry, "
            "dynamic_rain or dynamic_storm."
        )
    all_rows = np.ones(len(data), dtype=bool)
    _scale_influent_columns(
        data,
        all_rows,
        q_scale=profile["base_q_scale"],
        cod_scale=profile["base_cod_scale"],
        n_scale=profile["base_n_scale"],
        alkalinity_scale=profile["base_alkalinity_scale"],
        tss_scale=profile["base_tss_scale"],
    )
    if profile["event_windows_days_mod_7"]:
        phase = np.mod(data[:, 0], 7.0)
        event_mask = np.zeros(len(data), dtype=bool)
        for start, end in profile["event_windows_days_mod_7"]:
            event_mask |= (phase >= float(start)) & (phase < float(end))
        # Remove the base scaling before applying event scaling so the event
        # values are relative to the unmodified standard dynamic influent.
        _scale_influent_columns(
            data,
            event_mask,
            q_scale=1.0 / max(float(profile["base_q_scale"]), 1e-12),
            cod_scale=1.0 / max(float(profile["base_cod_scale"]), 1e-12),
            n_scale=1.0 / max(float(profile["base_n_scale"]), 1e-12),
            alkalinity_scale=1.0 / max(float(profile["base_alkalinity_scale"]), 1e-12),
            tss_scale=1.0 / max(float(profile["base_tss_scale"]), 1e-12),
        )
        _scale_influent_columns(
            data,
            event_mask,
            q_scale=profile["event_q_scale"],
            cod_scale=profile["event_cod_scale"],
            n_scale=profile["event_n_scale"],
            alkalinity_scale=profile["event_alkalinity_scale"],
            tss_scale=profile["event_tss_scale"],
        )
    return data


def normalize_inflow_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    aliases = {
        "steady": "steady_const",
        "constant": "steady_const",
        "const": "steady_const",
        "dynamic": "dynamic_default",
        "default": "dynamic_default",
        "dry": "dynamic_dry",
        "rain": "dynamic_rain",
        "storm": "dynamic_storm",
    }
    return aliases.get(mode, mode)


def inflow_resolution_metadata(inflow_mode: str, timestep_min: float) -> dict:
    """Return auditable source-vs-integration time-resolution metadata."""
    mode = normalize_inflow_mode(inflow_mode)
    is_dynamic = mode == "dynamic_default" or mode in BSM1_DYNAMIC_WEATHER_PROFILES
    is_weather = mode in BSM1_DYNAMIC_WEATHER_PROFILES
    profile = BSM1_DYNAMIC_WEATHER_PROFILES.get(mode)
    if profile is not None:
        source_description = (
            f"packaged BSM1 {profile['label']}-weather influent ({profile['source_file']})"
            if profile["source_kind"] == "direct_packaged_file"
            else f"packaged BSM1 dryinfluent.csv with documented {profile['label']} transformation"
        )
        source_file = str(profile["source_file"])
        provenance = str(profile["provenance"])
    else:
        source_description = (
            "bsm2-python default BSM2 dynamic influent" if is_dynamic else "constant BSM1 reference row"
        )
        source_file = "dyninfluent_bsm2.csv" if is_dynamic else "constant_reference_row"
        provenance = "none"
    return {
        "influent_source": source_description,
        "influent_source_file": source_file,
        "influent_source_period_days": 14.0 if is_weather else None,
        "influent_source_timestep_min": (
            float(BSM1_DYNAMIC_SOURCE_TIMESTEP_MIN) if is_dynamic else None
        ),
        "influent_model_timestep_min": float(timestep_min),
        "influent_resampling_method": (
            "zero_order_hold" if is_dynamic else "constant_piecewise_hold"
        ),
        "weather_profile": (
            BSM1_DYNAMIC_WEATHER_PROFILES[mode]["label"]
            if mode in BSM1_DYNAMIC_WEATHER_PROFILES
            else "none"
        ),
        "weather_profile_provenance": provenance,
        "weather_profile_source_kind": profile["source_kind"] if profile is not None else "none",
        "influent_source_timestep_note": (
            profile["source_timestep_note"] if profile is not None else "none"
        ),
    }


def tempmodel_for_scenario(scenario: str, policy: str) -> bool:
    scenario = normalize_scenario_name(scenario)
    policy = str(policy).strip().lower()
    if policy in {"on", "true", "yes", "all"}:
        return True
    if policy in {"off", "false", "no", "none"}:
        return False
    if policy in {"scenario", "per_scenario", "low_temp_only"}:
        return scenario == "low_temperature"
    raise ValueError("tempmodel_policy must be one of: scenario, all, off")


def build_bsm1_data(
    scenario: str,
    total_days: float,
    timestep_days: float,
    inflow_mode: str = "steady_const",
    tempmodel: bool = False,
) -> np.ndarray:
    """Create the BSM1 influent matrix for the selected validation mode."""
    scenario = normalize_scenario_name(scenario)
    mode = normalize_inflow_mode(inflow_mode)
    if mode == "steady_const":
        base_data = make_constant_influent(total_days)
    elif mode == "dynamic_default" or mode in BSM1_DYNAMIC_WEATHER_PROFILES:
        BSM1OL = require_bsm1ol()
        if mode in BSM1_DYNAMIC_WEATHER_PROFILES:
            source_data = repeat_periodic_influent(
                load_packaged_bsm1_weather_influent(BSM1OL, mode),
                total_days=total_days,
                period_days=14.0,
            )
        else:
            # Backward-compatible mode: bsm2-python 0.0.16 documents that this
            # default is the BSM2 dynamic influent, not a BSM1 weather file.
            source_timestep_days = BSM1_DYNAMIC_SOURCE_TIMESTEP_MIN / (60.0 * 24.0)
            base = BSM1OL(
                timestep=source_timestep_days,
                endtime=total_days,
                tempmodel=tempmodel,
                activate=False,
            )
            source_data = np.asarray(base.data_in, dtype=float)
        base_data = resample_influent_zero_order_hold(
            source_data,
            target_timestep_days=timestep_days,
            total_days=total_days,
        )
        if mode in BSM1_DYNAMIC_WEATHER_PROFILES and BSM1_DYNAMIC_WEATHER_PROFILES[mode]["source_kind"] == "synthetic_transformation":
            base_data = apply_dynamic_weather_profile(base_data, mode)
    else:
        raise ValueError(
            "inflow_mode must be 'steady_const', 'dynamic_default', "
            "'dynamic_dry', 'dynamic_rain' or 'dynamic_storm'"
        )
    return make_scenario_data(base_data, scenario)


def evaluation_mask_from_time(time_d: np.ndarray, eval_start_d: float, eval_end_d: float) -> np.ndarray:
    return (np.asarray(time_d, dtype=float) >= eval_start_d) & (np.asarray(time_d, dtype=float) < eval_end_d)


def bsm1_nitrogen_components(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return model-consistent BSM1 nitrogen components from ASM1 states.

    ``SNH``, ``SNO``, ``SND`` and ``XND`` are already expressed as nitrogen
    concentrations. ``XBH`` and ``XBA`` are COD-based biomass states and ``XP``
    is a COD-based particulate product state; the BSM1 fractions ``I_XB`` and
    ``I_XP`` convert those states to nitrogen. The resulting ``TN`` is the
    total nitrogen represented by the ASM1 state vector, not an independent
    laboratory measurement with additional unmodelled fractions.
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape[-1] <= XND:
        raise ValueError(
            f"BSM1 state array must have at least {XND + 1} columns; got shape {arr.shape}."
        )
    tin = arr[..., SNH] + arr[..., SNO]
    organic_n = (
        arr[..., SND]
        + arr[..., XND]
        + BSM1_I_XB * (arr[..., XBH] + arr[..., XBA])
        + BSM1_I_XP * arr[..., XP]
    )
    return {
        "TIN": tin,
        "organic_N": organic_n,
        "TN": tin + organic_n,
    }


def total_nitrogen_from_bsm1_states(values: np.ndarray) -> np.ndarray:
    """Return the model-consistent total nitrogen represented by BSM1 states."""
    return bsm1_nitrogen_components(values)["TN"]


def _plant_history(
    plant,
    attribute: str,
    expected_rows: int,
    minimum_columns: int = SNH + 1,
) -> np.ndarray:
    """Return one validated BSM1 history array with a useful error message."""
    if not hasattr(plant, attribute):
        raise AttributeError(
            f"The installed bsm2-python BSM1OL object does not provide {attribute!r}. "
            "The biomass export requires y_out1_all through y_out5_all and ys_eff_all."
        )
    values = np.asarray(getattr(plant, attribute), dtype=float)
    if values.ndim != 2 or values.shape[0] < expected_rows or values.shape[1] < minimum_columns:
        raise ValueError(
            f"Unexpected shape for BSM1 history {attribute!r}: {values.shape}; "
            f"expected at least ({expected_rows}, {minimum_columns})."
        )
    return values[:expected_rows]


def extract_bsm1_biomass_timeseries(
    *,
    plant,
    time_d: np.ndarray,
    eval_start: float,
    eval_end: float,
) -> pd.DataFrame:
    """Extract reactor biomass and calibration covariates from a completed run.

    ``XBH`` is the BSM1 heterotrophic biomass state and is the closest reference
    for the reduced model's ``XH`` proxy. The quantities should not be assumed
    to share an identical absolute scale; the exported trajectories support a
    later, explicit mapping or normalized calibration.
    """
    time_d = np.asarray(time_d, dtype=float)
    n_rows = len(time_d)
    histories = {
        tank: _plant_history(plant, f"y_out{tank}_all", n_rows)
        for tank in BSM1_REACTOR_NUMBERS
    }
    effluent = _plant_history(plant, "ys_eff_all", n_rows)
    phase = np.full(n_rows, "post_eval", dtype=object)
    phase[time_d < eval_start] = "warmup"
    phase[evaluation_mask_from_time(time_d, eval_start, eval_end)] = "main"
    frame = pd.DataFrame(
        {
            "time_d": time_d,
            "phase": phase,
            "tank5_SS": histories[5][:, SS],
            "tank5_DO": histories[5][:, SO],
            "tank5_SNO": histories[5][:, SNO],
            "tank5_SNH": histories[5][:, SNH],
            "eff_NH4": effluent[:, SNH],
            "eff_SNO": effluent[:, SNO],
        }
    )
    tank5_n = bsm1_nitrogen_components(histories[5])
    effluent_n = bsm1_nitrogen_components(effluent)
    frame["tank5_TIN"] = tank5_n["TIN"]
    frame["tank5_organic_N"] = tank5_n["organic_N"]
    frame["tank5_TN"] = tank5_n["TN"]
    frame["eff_TIN"] = effluent_n["TIN"]
    frame["eff_organic_N"] = effluent_n["organic_N"]
    frame["eff_TN"] = effluent_n["TN"]
    for tank in BSM1_REACTOR_NUMBERS:
        xbh = histories[tank][:, XBH]
        xba = histories[tank][:, XBA]
        frame[f"tank{tank}_XBH"] = xbh
        frame[f"tank{tank}_XBA"] = xba
        frame[f"tank{tank}_active_biomass"] = xbh + xba

    for state in ("XBH", "XBA", "active_biomass"):
        columns = [f"tank{tank}_{state}" for tank in BSM1_AERATED_REACTOR_NUMBERS]
        frame[f"aerated_reactors_{state}_mean"] = frame[columns].mean(axis=1)
    return frame


def _safe_relative_change(start: float, end: float) -> float:
    if not np.isfinite(start) or not np.isfinite(end) or abs(start) < 1e-12:
        return np.nan
    return float((end - start) / start)


def summarise_bsm1_biomass(biomass_ts: pd.DataFrame) -> dict:
    """Summarise BSM1 biomass during the explicitly marked evaluation period."""
    eval_df = biomass_ts[biomass_ts["phase"].eq("main")]
    if eval_df.empty:
        raise ValueError("Cannot summarise BSM1 biomass because the evaluation period is empty.")

    out: dict[str, float | int] = {"biomass_n_eval_steps": int(len(eval_df))}
    for tank in BSM1_REACTOR_NUMBERS:
        for state in ("XBH", "XBA", "active_biomass"):
            column = f"tank{tank}_{state}"
            out[f"{column}_mean_eval"] = float(eval_df[column].mean())

    detailed_scopes = {
        "tank5": "tank5",
        "aerated_reactors": "aerated_reactors",
    }
    for output_scope, column_scope in detailed_scopes.items():
        for state in ("XBH", "XBA", "active_biomass"):
            column = (
                f"{column_scope}_{state}"
                if column_scope == "tank5"
                else f"{column_scope}_{state}_mean"
            )
            values = pd.to_numeric(eval_df[column], errors="coerce").to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            start = float(values[0])
            end = float(values[-1])
            prefix = f"{output_scope}_{state}"
            out[f"{prefix}_mean_eval"] = mean
            out[f"{prefix}_std_eval"] = std
            out[f"{prefix}_cv_eval"] = float(std / abs(mean)) if abs(mean) > 1e-12 else np.nan
            out[f"{prefix}_p05_eval"] = float(np.quantile(values, 0.05))
            out[f"{prefix}_p95_eval"] = float(np.quantile(values, 0.95))
            out[f"{prefix}_start_eval"] = start
            out[f"{prefix}_end_eval"] = end
            out[f"{prefix}_relative_change_eval"] = _safe_relative_change(start, end)
    return out


def kla_filename_token(kla: float) -> str:
    """Return a filesystem-safe, stable token such as 160 or 160p5."""
    return f"{float(kla):g}".replace("-", "m").replace(".", "p")


def violation_duration_from_timeseries(
    time_d: np.ndarray,
    values: np.ndarray,
    threshold: float,
    eval_start_d: float,
    eval_end_d: float,
) -> tuple[float, float, float]:
    """Calculate violation duration directly from the saved time series.

    This avoids the time-unit ambiguity that can occur when using helper
    functions with internal timestep assumptions.
    """
    time_d = np.asarray(time_d, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = evaluation_mask_from_time(time_d, eval_start_d, eval_end_d)
    if np.sum(mask) == 0:
        return 0.0, 0.0, np.nan
    dt_days = float(np.median(np.diff(time_d))) if len(time_d) > 1 else 0.0
    exceed = values[mask] > threshold
    days = float(np.sum(exceed) * dt_days)
    hours = float(days * 24.0)
    ratio = float(np.mean(exceed))
    return days, hours, ratio


def bsm1_time_settings(warmup_days: float, eval_days: float, endtime: float | None = None) -> tuple[float, float, float]:
    if endtime is not None:
        total = float(endtime)
        eval_start = total * 0.5
        eval_end = total
    else:
        eval_start = float(warmup_days)
        eval_end = float(warmup_days + eval_days)
        total = eval_end
    return total, eval_start, eval_end


def bsm1_internal_evaltime(
    data_in: np.ndarray,
    timestep_days: float,
    endtime_days: float,
    eval_start_days: float,
) -> np.ndarray:
    """Return an evaltime accepted by bsm2-python's strict time check.

    ``bsm2-python`` constructs ``simtime`` with ``np.arange``.  When the last
    influent timestamp equals ``endtime``, the final simulated point is one
    integration step earlier.  The package therefore rejects ``evaltime`` if
    its end is set directly to ``endtime``.  This helper mirrors the package's
    time-grid construction and selects its actual final point.  Our own summary
    mask still uses the requested evaluation window.
    """
    data = np.asarray(data_in, dtype=float)
    step = float(timestep_days)
    total = float(endtime_days)
    start = float(eval_start_days)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("data_in must contain at least one time-stamped influent row.")
    if step <= 0.0:
        raise ValueError("timestep_days must be positive.")
    simtime = np.arange(0.0, float(data[-1, 0]), step, dtype=float)
    simtime = simtime[simtime <= total]
    if simtime.size == 0:
        raise ValueError("The selected BSM1 time settings produce no simulation point.")
    internal_end = float(simtime[-1])
    if start > internal_end:
        raise ValueError(
            "The evaluation period is shorter than one BSM1 integration step; "
            "increase eval_days or reduce timestep_min."
        )
    return np.array([start, internal_end], dtype=float)


def bsm1_executable_step_count(plant: object) -> int:
    """Return the number of step indices accepted by a BSM1 plant.

    Depending on the influent construction and floating-point endpoint, either
    ``simtime`` or ``timesteps`` can contain one extra entry.  ``plant.step(i)``
    reads both arrays, so only indices present in both are executable.
    """
    n_steps = min(len(plant.simtime), len(plant.timesteps))
    if n_steps < 1:
        raise ValueError("BSM1 produced no executable simulation step.")
    return int(n_steps)


def effective_kla_for_scenario(kla: float, scenario: str, oxygen_transfer_factor: float = 0.70) -> float:
    """Return effective KLa.

    A moderate default factor of 0.70 is used for low_oxygen_transfer. The earlier
    value 0.25 was intentionally severe but caused all controllers and all KLa
    settings to fail, which is not useful for a manuscript validation figure.
    """
    scenario = normalize_scenario_name(scenario)
    return float(kla * oxygen_transfer_factor if scenario == "low_oxygen_transfer" else kla)


def applied_oxygen_transfer_factor(scenario: str, oxygen_transfer_factor: float = 0.70) -> float:
    """Return the scenario-specific oxygen-transfer factor actually applied."""
    scenario = normalize_scenario_name(scenario)
    return float(oxygen_transfer_factor if scenario == "low_oxygen_transfer" else 1.0)




def aeration_energy_scaling(commanded_kla: float, effective_kla: float) -> float:
    """Return multiplier converting BSM1 internal effective-KLa energy to commanded-KLa energy.

    BSM2-Python's internal aeration energy is calculated from the KLa values
    passed to the plant. In the low_oxygen_transfer scenario we intentionally pass
    a reduced effective KLa to the plant to mimic poorer oxygen transfer. For
    engineering interpretation, however, blower effort should be reported on the
    commanded/nominal KLa basis: if the oxygen-transfer factor is 0.70, the same
    effective KLa requires about 1/0.70 times more commanded blower effort.
    """
    commanded = float(commanded_kla)
    effective = float(effective_kla)
    if abs(effective) < 1e-12:
        return 1.0 if abs(commanded) < 1e-12 else np.nan
    return commanded / effective


def add_commanded_energy_columns(summary: dict, commanded_kla: float, effective_kla: float) -> dict:
    """Separate the dimensionless command index from BSM1's internal energy metric.

    The BSM2-Python performance-factor output is retained as a model-specific
    kWh/d metric based on the effective KLa passed to BSM1.  It is not rescaled
    and relabelled as measured blower electricity for the commanded KLa.
    """
    row = dict(summary)
    internal_effective = float(row.get("aeration_kWh_per_d", np.nan))
    scale = float(aeration_energy_scaling(commanded_kla, effective_kla))
    row["effective_aeration_kWh_per_d"] = internal_effective
    row["bsm1_standard_aeration_energy_kWh_per_d"] = internal_effective
    row["commanded_aeration_index"] = commanded_aeration_index(commanded_kla)
    row["commanded_aeration_index_unit"] = "dimensionless"
    row["commanded_aeration_index_reference_kla_d-1"] = 240.0
    # Preserve the package-native name without altering its numerical basis.
    row["aeration_kWh_per_d"] = internal_effective
    row["bsm1_internal_aeration_kWh_per_d"] = internal_effective
    row["aeration_energy_correction_factor"] = scale
    row["aeration_energy_basis"] = "bsm2_python_internal_effective_kla_metric"
    return row

def summarise_eval(
    *,
    time_d: np.ndarray,
    snh: np.ndarray,
    tn: np.ndarray | None = None,
    tin: np.ndarray | None = None,
    organic_n: np.ndarray | None = None,
    do_tank5: np.ndarray | None,
    perf_factors_all: np.ndarray | None,
    eval_start: float,
    eval_end: float,
    nh4_limit: float,
) -> dict:
    eval_mask = evaluation_mask_from_time(time_d, eval_start, eval_end)
    viol_days, viol_hours, viol_ratio = violation_duration_from_timeseries(time_d, snh, nh4_limit, eval_start, eval_end)
    out = {
        "SNH_mean_eval": float(np.mean(snh[eval_mask])),
        "SNH_p95_eval": float(np.quantile(snh[eval_mask], 0.95)),
        "SNH_max_eval": float(np.max(snh[eval_mask])),
        "SNH_violation_days_gt_limit": viol_days,
        "SNH_violation_hours_gt_limit": viol_hours,
        "SNH_violation_ratio_gt_limit": viol_ratio,
        "SNH_zero_violation": bool(viol_hours <= 1e-12),
    }
    if tn is not None:
        tn_eval = np.asarray(tn, dtype=float)[eval_mask]
        out["TN_mean_eval"] = float(np.mean(tn_eval))
        out["TN_p95_eval"] = float(np.quantile(tn_eval, 0.95))
    if tin is not None:
        tin_eval = np.asarray(tin, dtype=float)[eval_mask]
        out["TIN_mean_eval"] = float(np.mean(tin_eval))
        out["TIN_p95_eval"] = float(np.quantile(tin_eval, 0.95))
    if organic_n is not None:
        organic_eval = np.asarray(organic_n, dtype=float)[eval_mask]
        out["organic_N_mean_eval"] = float(np.mean(organic_eval))
        out["organic_N_p95_eval"] = float(np.quantile(organic_eval, 0.95))
    if do_tank5 is not None:
        out["DO_tank5_mean_eval"] = float(np.mean(do_tank5[eval_mask]))
        out["DO_tank5_p05_eval"] = float(np.quantile(do_tank5[eval_mask], 0.05))
    if perf_factors_all is not None:
        perf_eval = perf_factors_all[eval_mask, :]
        out["pumping_kWh_per_d"] = float(np.mean(perf_eval[:, 0]))
        out["aeration_kWh_per_d"] = float(np.mean(perf_eval[:, 1]))
        out["mixing_kWh_per_d"] = float(np.mean(perf_eval[:, 2]))
    return out


def run_bsm1_fixed_kla(
    scenario: str,
    kla: float,
    warmup_days: float = 60.0,
    eval_days: float = 7.0,
    endtime: float | None = None,
    nh4_limit: float = 4.0,
    inflow_mode: str = "steady_const",
    timestep_min: float = 1.0,
    oxygen_transfer_factor: float = 0.70,
    tempmodel_policy: str = "scenario",
    return_biomass_timeseries: bool = False,
    *,
    operating_point: SteadyOperatingPoint | None = None,
) -> dict | tuple[dict, pd.DataFrame]:
    scenario = normalize_scenario_name(scenario)
    BSM1OL = require_bsm1ol()
    timestep_days = float(timestep_min) / (60.0 * 24.0)
    total, eval_start, eval_end = bsm1_time_settings(warmup_days, eval_days, endtime=endtime)
    if operating_point is not None:
        if operating_point.name != scenario:
            raise ValueError("Operating-point name must match scenario.")
        if normalize_inflow_mode(inflow_mode) != "steady_const" or tempmodel_policy != "off":
            raise ValueError("Custom load/transfer points require steady_const and tempmodel_policy='off'.")
        tempmodel = False
        data = make_operating_point_data(total, operating_point)
    else:
        tempmodel = tempmodel_for_scenario(scenario, tempmodel_policy)
        data = build_bsm1_data(scenario, total, timestep_days, inflow_mode=inflow_mode, tempmodel=tempmodel)
    plant_evaltime = bsm1_internal_evaltime(data, timestep_days, total, eval_start)
    plant = BSM1OL(
        data_in=data,
        timestep=timestep_days,
        endtime=total,
        evaltime=plant_evaltime,
        tempmodel=tempmodel,
        activate=False,
    )

    effective_kla = (float(kla * operating_point.oxygen_transfer_factor) if operating_point is not None
                     else effective_kla_for_scenario(kla, scenario, oxygen_transfer_factor=oxygen_transfer_factor))
    klas = np.array([0.0, 0.0, effective_kla, effective_kla, effective_kla], dtype=float)
    n_steps = bsm1_executable_step_count(plant)
    tic = time.perf_counter()
    for idx in range(n_steps):
        plant.step(idx, klas=klas)
    runtime_seconds = time.perf_counter() - tic

    # BSM2-Python endpoint construction may leave either time array with one
    # extra entry. State histories at index i are the response after interval
    # i, so label executable entries with the interval end.
    time_d = (
        np.asarray(plant.simtime[:n_steps], dtype=float)
        + np.asarray(plant.timesteps[:n_steps], dtype=float)
    )
    effluent = np.asarray(plant.ys_eff_all[:n_steps], dtype=float)
    snh = np.asarray(effluent[:, SNH], dtype=float)
    nitrogen = bsm1_nitrogen_components(effluent)
    do_tank5 = np.asarray(plant.y_out5_all[:n_steps, SO], dtype=float)
    biomass_ts = extract_bsm1_biomass_timeseries(
        plant=plant,
        time_d=time_d,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    summary = summarise_eval(
        time_d=time_d,
        snh=snh,
        tn=nitrogen["TN"],
        tin=nitrogen["TIN"],
        organic_n=nitrogen["organic_N"],
        do_tank5=do_tank5,
        perf_factors_all=np.asarray(plant.perf_factors_all[:n_steps], dtype=float),
        eval_start=eval_start,
        eval_end=eval_end,
        nh4_limit=nh4_limit,
    )
    summary = add_commanded_energy_columns(summary, commanded_kla=kla, effective_kla=effective_kla)
    summary.update(summarise_bsm1_biomass(biomass_ts))
    inflow_meta = inflow_resolution_metadata(inflow_mode, timestep_min)
    row = {
        "model": "BSM2-Python/BSM1",
        "bsm2_python_version": installed_bsm2_python_version(),
        "experiment": "fixed_kla",
        "inflow_mode": normalize_inflow_mode(inflow_mode),
        "scenario": scenario,
        "tempmodel": bool(tempmodel),
        "timestep_min": float(timestep_min),
        "kla": float(kla),
        "effective_kla": float(effective_kla),
        "oxygen_transfer_factor": applied_oxygen_transfer_factor(scenario, oxygen_transfer_factor),
        "configured_low_oxygen_transfer_factor": float(oxygen_transfer_factor),
        "warmup_days": float(eval_start),
        "eval_days": float(eval_end - eval_start),
        "requested_eval_end_d": float(eval_end),
        "plant_internal_eval_end_d": float(plant_evaltime[1]),
        "nh4_limit": float(nh4_limit),
        "runtime_seconds": float(runtime_seconds),
        **inflow_meta,
        **summary,
        **(operating_point.metadata() if operating_point is not None else {}),
    }
    if return_biomass_timeseries:
        biomass_ts.insert(2, "inflow_mode", normalize_inflow_mode(inflow_mode))
        biomass_ts.insert(3, "scenario", scenario)
        biomass_ts.insert(4, "tempmodel", bool(tempmodel))
        biomass_ts.insert(5, "timestep_min", float(timestep_min))
        biomass_ts.insert(6, "kla", float(kla))
        biomass_ts.insert(7, "effective_kla", float(effective_kla))
        biomass_ts.insert(
            8,
            "oxygen_transfer_factor",
            row["oxygen_transfer_factor"],
        )
        insert_at = 9
        for key, value in inflow_meta.items():
            biomass_ts.insert(insert_at, key, value)
            insert_at += 1
        return row, biomass_ts
    return row


def run_bsm1_abac(
    scenario: str,
    warmup_days: float = 60.0,
    eval_days: float = 7.0,
    endtime: float | None = None,
    seed: int = 42,
    cfg: BSM1ABACConfig | None = None,
    nh4_limit: float = 4.0,
    inflow_mode: str = "steady_const",
    timestep_min: float = 1.0,
    oxygen_transfer_factor: float = 0.70,
    tempmodel_policy: str = "scenario",
) -> tuple[pd.DataFrame, dict]:
    scenario = normalize_scenario_name(scenario)
    BSM1OL = require_bsm1ol()
    cfg = cfg or BSM1ABACConfig(nh4_ref=nh4_limit)
    rng = np.random.default_rng(seed)
    timestep_days = float(timestep_min) / (60.0 * 24.0)
    total, eval_start, eval_end = bsm1_time_settings(warmup_days, eval_days, endtime=endtime)
    tempmodel = tempmodel_for_scenario(scenario, tempmodel_policy)
    data = build_bsm1_data(scenario, total, timestep_days, inflow_mode=inflow_mode, tempmodel=tempmodel)
    plant_evaltime = bsm1_internal_evaltime(data, timestep_days, total, eval_start)
    plant = BSM1OL(
        data_in=data,
        timestep=timestep_days,
        endtime=total,
        evaltime=plant_evaltime,
        tempmodel=tempmodel,
        activate=False,
    )
    ctrl = BSM1ABACPID(cfg, dt_d=timestep_days)
    inflow_meta = inflow_resolution_metadata(inflow_mode, timestep_min)
    records = []

    n_steps = bsm1_executable_step_count(plant)
    tic = time.perf_counter()
    for idx in range(n_steps):
        t_start = float(plant.simtime[idx])
        t_end = t_start + float(plant.timesteps[idx])
        phase = "warmup" if t_end < eval_start else "main"
        # Use previous-step plant outputs as sensor measurements. Values after
        # plant.step are saved as the realized process response at this time.
        true_do_prev = float(plant.y_out5[SO])
        true_nh4_prev = float(plant.ys_eff[SNH])
        meas_do = float(max(0.0, true_do_prev + rng.normal(0.0, 0.02)))
        meas_nh4 = float(max(0.0, true_nh4_prev + rng.normal(0.0, 0.05)))

        kla, do_sp = ctrl.compute(meas_do=meas_do, meas_nh4=meas_nh4)
        effective_kla = effective_kla_for_scenario(kla, scenario, oxygen_transfer_factor=oxygen_transfer_factor)
        plant.step(idx, klas=np.array([0.0, 0.0, effective_kla, effective_kla, effective_kla], dtype=float))
        energy_scale = aeration_energy_scaling(kla, effective_kla)
        internal_effective_aeration = float(plant.perf_factors_all[idx, 1])
        command_index = commanded_aeration_index(kla)
        nitrogen = bsm1_nitrogen_components(plant.ys_eff)

        records.append(
            {
                "time_d": t_end,
                "phase": phase,
                "inflow_mode": normalize_inflow_mode(inflow_mode),
                "scenario": scenario,
                "controller": "ABAC-PID",
                "tempmodel": bool(tempmodel),
                "timestep_min": float(timestep_min),
                **inflow_meta,
                "meas_DO": meas_do,
                "meas_NH4": meas_nh4,
                "true_tank5_DO": float(plant.y_out5[SO]),
                "true_eff_NH4": float(plant.ys_eff[SNH]),
                "true_eff_TIN": float(nitrogen["TIN"]),
                "true_eff_organic_N": float(nitrogen["organic_N"]),
                "true_eff_TN": float(nitrogen["TN"]),
                "do_sp": float(do_sp),
                "kla": float(kla),
                "effective_kla": float(effective_kla),
                "oxygen_transfer_factor": applied_oxygen_transfer_factor(scenario, oxygen_transfer_factor),
                "eff_SNO": float(plant.ys_eff[SNO]),
                "pumping_kWh_per_d_step": float(plant.perf_factors_all[idx, 0]),
                "effective_aeration_kWh_per_d_step": internal_effective_aeration,
                "bsm1_standard_aeration_energy_kWh_per_d_step": internal_effective_aeration,
                "commanded_aeration_index_step": command_index,
                "aeration_kWh_per_d_step": internal_effective_aeration,
                "bsm1_internal_aeration_kWh_per_d_step": internal_effective_aeration,
                "aeration_energy_correction_factor_step": float(energy_scale),
                "mixing_kWh_per_d_step": float(plant.perf_factors_all[idx, 2]),
            }
        )

    df = pd.DataFrame(records)
    runtime_seconds = time.perf_counter() - tic
    eval_mask = df["phase"].eq("main")
    viol_days, viol_hours, viol_ratio = violation_duration_from_timeseries(
        df["time_d"].to_numpy(), df["true_eff_NH4"].to_numpy(), nh4_limit, eval_start, eval_end
    )
    summary = {
        "model": "BSM2-Python/BSM1",
        "bsm2_python_version": installed_bsm2_python_version(),
        "experiment": "ABAC-PID",
        "inflow_mode": normalize_inflow_mode(inflow_mode),
        "scenario": scenario,
        "controller": "ABAC-PID",
        "tempmodel": bool(tempmodel),
        "timestep_min": float(timestep_min),
        "warmup_days": float(eval_start),
        "eval_days": float(eval_end - eval_start),
        "requested_eval_end_d": float(eval_end),
        "plant_internal_eval_end_d": float(plant_evaltime[1]),
        "nh4_limit": float(nh4_limit),
        "runtime_seconds": float(runtime_seconds),
        **inflow_meta,
        "SNH_mean_eval": float(df.loc[eval_mask, "true_eff_NH4"].mean()),
        "SNH_p95_eval": float(df.loc[eval_mask, "true_eff_NH4"].quantile(0.95)),
        "SNH_max_eval": float(df.loc[eval_mask, "true_eff_NH4"].max()),
        "SNH_violation_days_gt_limit": viol_days,
        "SNH_violation_hours_gt_limit": viol_hours,
        "SNH_violation_ratio_gt_limit": viol_ratio,
        "SNH_zero_violation": bool(viol_hours <= 1e-12),
        "TN_mean_eval": float(df.loc[eval_mask, "true_eff_TN"].mean()),
        "TN_p95_eval": float(df.loc[eval_mask, "true_eff_TN"].quantile(0.95)),
        "TIN_mean_eval": float(df.loc[eval_mask, "true_eff_TIN"].mean()),
        "TIN_p95_eval": float(df.loc[eval_mask, "true_eff_TIN"].quantile(0.95)),
        "organic_N_mean_eval": float(df.loc[eval_mask, "true_eff_organic_N"].mean()),
        "organic_N_p95_eval": float(df.loc[eval_mask, "true_eff_organic_N"].quantile(0.95)),
        "DO_tank5_mean_eval": float(df.loc[eval_mask, "true_tank5_DO"].mean()),
        "DO_tank5_p05_eval": float(df.loc[eval_mask, "true_tank5_DO"].quantile(0.05)),
        "mean_kla_eval": float(df.loc[eval_mask, "kla"].mean()),
        "mean_effective_kla_eval": float(df.loc[eval_mask, "effective_kla"].mean()),
        "oxygen_transfer_factor": applied_oxygen_transfer_factor(scenario, oxygen_transfer_factor),
        "configured_low_oxygen_transfer_factor": float(oxygen_transfer_factor),
        "pumping_kWh_per_d": float(df.loc[eval_mask, "pumping_kWh_per_d_step"].mean()),
        "effective_aeration_kWh_per_d": float(df.loc[eval_mask, "effective_aeration_kWh_per_d_step"].mean()),
        "bsm1_standard_aeration_energy_kWh_per_d": float(df.loc[eval_mask, "bsm1_standard_aeration_energy_kWh_per_d_step"].mean()),
        "commanded_aeration_index": float(df.loc[eval_mask, "commanded_aeration_index_step"].mean()),
        "commanded_aeration_index_unit": "dimensionless",
        "commanded_aeration_index_reference_kla_d-1": 240.0,
        "aeration_kWh_per_d": float(df.loc[eval_mask, "aeration_kWh_per_d_step"].mean()),
        "bsm1_internal_aeration_kWh_per_d": float(df.loc[eval_mask, "bsm1_internal_aeration_kWh_per_d_step"].mean()),
        "aeration_energy_correction_factor": float(df.loc[eval_mask, "aeration_energy_correction_factor_step"].mean()),
        "aeration_energy_basis": "bsm2_python_internal_effective_kla_metric",
        "mixing_kWh_per_d": float(df.loc[eval_mask, "mixing_kWh_per_d_step"].mean()),
        "abac_config": asdict(cfg),
    }
    return df, summary


def run_bsm1_validation(
    output_dir: Path = Path("outputs"),
    scenarios: Iterable[str] = ("baseline", "low_temperature", "low_oxygen_transfer"),
    warmup_days: float = 60.0,
    eval_days: float = 7.0,
    endtime: float | None = None,
    nh4_limit: float = 4.0,
    inflow_mode: str = "steady_const",
    timestep_min: float = 1.0,
    oxygen_transfer_factor: float = 0.70,
    tempmodel_policy: str = "scenario",
    kla_values: Sequence[float] | None = None,
    grid_preset: str = "coarse",
    seed: int = 42,
    run_fixed_kla: bool = True,
    run_abac: bool = True,
    save_fixed_kla_biomass_timeseries: bool = False,
) -> pd.DataFrame:
    """Run selected BSM1 model-structure checks and save auditable CSV outputs.

    Recommended manuscript workflow:
    1. Use inflow_mode='steady_const' for the main BSM1 validation figure.
    2. Use timestep_min=1 because the BSM2-Python documentation warns that
       sensor-sensitive simulations should not use time steps larger than 1 min.
    3. Use tempmodel_policy='scenario' so low_temperature activates BSM1 temperature
       dependencies while baseline remains comparable to the standard test.
    4. Use a moderate oxygen_transfer_factor such as 0.70 or 0.50. A factor of
       0.25 is too severe and usually makes every KLa/controller fail.
    """
    output_dir = Path(output_dir)
    out_dir = output_dir / "bsm1_validation"
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    mode = normalize_inflow_mode(inflow_mode)
    scenario_list = [normalize_scenario_name(value) for value in scenarios]
    shared_kla_values = parse_float_list(kla_values) if kla_values is not None else None
    rows: List[dict] = []

    for scenario in scenario_list:
        scenario_kla_values = kla_values_for_scenario(
            scenario,
            grid_preset=grid_preset,
            custom_values=shared_kla_values,
        )
        if run_fixed_kla:
            for kla in scenario_kla_values:
                print(f"BSM1 fixed KLa mode={mode} scenario={scenario} kla={kla}")
                fixed_result = run_bsm1_fixed_kla(
                    scenario,
                    kla,
                    warmup_days=warmup_days,
                    eval_days=eval_days,
                    endtime=endtime,
                    nh4_limit=nh4_limit,
                    inflow_mode=mode,
                    timestep_min=timestep_min,
                    oxygen_transfer_factor=oxygen_transfer_factor,
                    tempmodel_policy=tempmodel_policy,
                    return_biomass_timeseries=save_fixed_kla_biomass_timeseries,
                )
                if save_fixed_kla_biomass_timeseries:
                    fixed_row, biomass_ts = fixed_result
                    biomass_path = out_dir / (
                        f"bsm1_fixed_kla_biomass_timeseries_{mode}_{scenario}_"
                        f"kla{kla_filename_token(kla)}.csv"
                    )
                    biomass_ts.to_csv(biomass_path, index=False)
                else:
                    fixed_row = fixed_result
                rows.append(fixed_row)

        if run_abac:
            print(f"BSM1 ABAC mode={mode} scenario={scenario}")
            ts, summary = run_bsm1_abac(
                scenario,
                warmup_days=warmup_days,
                eval_days=eval_days,
                endtime=endtime,
                seed=seed,
                nh4_limit=nh4_limit,
                inflow_mode=mode,
                timestep_min=timestep_min,
                oxygen_transfer_factor=oxygen_transfer_factor,
                tempmodel_policy=tempmodel_policy,
            )
            ts.to_csv(out_dir / f"bsm1_abac_timeseries_{mode}_{scenario}.csv", index=False)
            # Backward-compatible filename for users who already expect this name.
            ts.to_csv(out_dir / f"bsm1_abac_timeseries_{scenario}.csv", index=False)
            rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bsm1_validation_summary.csv", index=False)
    df.to_csv(table_dir / "bsm1_validation_summary.csv", index=False)

    biomass_metric_columns = [
        column
        for column in df.columns
        if "XBH" in column or "XBA" in column or "active_biomass" in column or column == "biomass_n_eval_steps"
    ]
    biomass_id_columns = [
        column
        for column in [
            "model",
            "experiment",
            "inflow_mode",
            "scenario",
            "tempmodel",
            "timestep_min",
            "bsm2_python_version",
            "kla",
            "effective_kla",
            "warmup_days",
            "eval_days",
            "weather_profile",
            "influent_source_timestep_min",
            "influent_model_timestep_min",
            "influent_resampling_method",
        ]
        if column in df.columns
    ]
    fixed_mask = df["experiment"].eq("fixed_kla") if "experiment" in df.columns else np.zeros(len(df), dtype=bool)
    biomass_summary = df.loc[fixed_mask, biomass_id_columns + biomass_metric_columns].copy()
    biomass_summary.to_csv(out_dir / "bsm1_biomass_summary.csv", index=False)
    biomass_summary.to_csv(table_dir / "bsm1_biomass_summary.csv", index=False)

    write_json(
        manifest_dir / "bsm1_validation_manifest.json",
        {
            "scenarios": scenario_list,
            "warmup_days": warmup_days,
            "eval_days": eval_days,
            "endtime_compatibility_mode": endtime,
            "nh4_limit": nh4_limit,
            "inflow_mode": mode,
            "inflow_resolution": inflow_resolution_metadata(mode, timestep_min),
            "dynamic_source_timestep_min": BSM1_DYNAMIC_SOURCE_TIMESTEP_MIN,
            "dynamic_weather_profiles": BSM1_DYNAMIC_WEATHER_PROFILES,
            "timestep_min": timestep_min,
            "oxygen_transfer_factor": oxygen_transfer_factor,
            "tempmodel_policy": tempmodel_policy,
            "grid_preset": grid_preset,
            "custom_kla_values": shared_kla_values,
            "default_coarse_kla_values": list(COARSE_KLA_VALUES),
            "kla_values_by_scenario": {
                str(s): kla_values_for_scenario(str(s), grid_preset=grid_preset, custom_values=shared_kla_values)
                for s in scenario_list
            },
            "seed": seed,
            "run_fixed_kla": run_fixed_kla,
            "run_abac": run_abac,
            "save_fixed_kla_biomass_timeseries": save_fixed_kla_biomass_timeseries,
            "bsm2_python_version": installed_bsm2_python_version(),
            "biomass_reference": {
                "BSM1_XBH_role": "heterotrophic biomass reference for the reduced-model XH proxy",
                "BSM1_XBA_role": "autotrophic biomass reference for the reduced-model XA proxy",
                "mapping_caution": (
                    "Reduced proxy states and BSM1 component concentrations are not assumed to share an identical absolute scale. "
                    "Use the exported trajectories to fit an explicit scale mapping or normalized dynamics."
                ),
                "aerated_reactor_aggregate": "arithmetic mean of equal-layout reactors 3, 4, and 5",
            },
            "nitrogen_definition": {
                "formula": "TN = SNH + SNO + SND + XND + I_XB*(XBH + XBA) + I_XP*XP",
                "TIN_formula": "TIN = SNH + SNO",
                "organic_N_formula": "SND + XND + I_XB*(XBH + XBA) + I_XP*XP",
                "I_XB": BSM1_I_XB,
                "I_XP": BSM1_I_XP,
                "units": "mg N/L",
                "scope": "BSM1 model-consistent nitrogen represented by the ASM1 state vector; not an independent laboratory assay.",
            },
            "note": (
                "SNH violation duration is computed directly from saved time series using the actual time step. "
                "The dry and rain checks load the package-supplied BSM1 dryinfluent.csv and raininfluent.csv files, respectively, "
                "at their nominal 15-min source resolution (the rain timestamps are stored at 0.01-d precision), repeat the "
                "native 14-day cycle when needed, and resample to the "
                "integration step by zero-order hold. The package rain file contains three locale-formatted flow tokens; these "
                "are normalized by the documented parser without changing any other state column. The storm check is explicitly "
                "a deterministic transformation of the packaged dry reference because no storminfluent.csv file is supplied by "
                "the installed bsm2-python distribution; it must be reported as a synthetic storm stress test, not an official "
                "BSM1 storm-weather file. "
                "The backward-compatible dynamic_default mode instead uses bsm2-python's dyninfluent_bsm2.csv and must not be "
                "reported as a standard BSM1 weather result. "
                "BSM1 is used as an external model-structure check, not as a full replacement for the surrogate benchmark. "
                "TN is calculated from the explicitly represented ASM1 nitrogen states using the recorded BSM1 I_XB and I_XP fractions. "
                "Commanded aeration is reported as a dimensionless index referenced to KLa=240 d^-1. "
                "The kWh/d aeration field is the BSM2-Python internal performance metric based on the effective KLa "
                "passed to BSM1; it is not relabelled as measured plant electricity or rescaled commanded energy."
            ),
        },
    )
    return df
