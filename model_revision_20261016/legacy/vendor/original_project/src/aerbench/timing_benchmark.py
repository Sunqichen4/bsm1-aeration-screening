from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ctypes
import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .bsm1_validation import run_bsm1_fixed_kla
from .config import normalize_scenario_name
from .fixed_aeration import run_surrogate_fixed_aeration
from .provenance import stable_record_hash
from .utils import write_json


DEFAULT_REPRESENTATIVE_KLA = {
    "baseline": 160.0,
    "high_load": 190.0,
    "low_oxygen_transfer": 220.0,
    "low_temperature": 440.0,
}

TIMING_SCOPE_COLUMNS = {
    "simulation_kernel": "simulation_kernel_seconds",
    "end_to_end": "end_to_end_seconds",
}


@dataclass(frozen=True)
class TimingModelSpec:
    label: str
    family: str
    timestep_min: float
    solver: str


def _number_token(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g").replace(".", "p")


def build_model_specs(
    surrogate_dt_minutes: Sequence[float],
    bsm1_timestep_min: float,
    include_bsm1: bool = True,
) -> list[TimingModelSpec]:
    dts = []
    for value in surrogate_dt_minutes:
        dt = float(value)
        if dt <= 0.0:
            raise ValueError("All surrogate time steps must be positive.")
        if not any(abs(dt - existing) <= 1e-12 for existing in dts):
            dts.append(dt)
    if not dts:
        raise ValueError("At least one surrogate time step is required.")
    if float(bsm1_timestep_min) <= 0.0:
        raise ValueError("The BSM1 time step must be positive.")

    specs = [
        TimingModelSpec(
            label=f"surrogate_{_number_token(dt)}min",
            family="reduced_surrogate",
            timestep_min=dt,
            solver="explicit Euler followed by component-wise numpy.clip bounds",
        )
        for dt in dts
    ]
    if include_bsm1:
        specs.append(
            TimingModelSpec(
                label=f"bsm1_{_number_token(float(bsm1_timestep_min))}min",
                family="BSM2-Python/BSM1",
                timestep_min=float(bsm1_timestep_min),
                solver=(
                    "bsm2-python BSM1OL; reactor and settler ODEs use "
                    "scipy.integrate.odeint (LSODA) with package-default solver settings"
                ),
            )
        )
    return specs


def parse_kla_mapping(text: str | None) -> dict[str, float]:
    if text is None or not str(text).strip():
        return dict(DEFAULT_REPRESENTATIVE_KLA)
    mapping: dict[str, float] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("KLa mapping entries must use scenario=value syntax.")
        scenario, value = item.split("=", 1)
        scenario = normalize_scenario_name(scenario.strip())
        kla = float(value.strip())
        if not scenario or kla < 0.0:
            raise ValueError("KLa mapping requires a scenario name and a non-negative value.")
        mapping[scenario] = kla
    if not mapping:
        raise ValueError("The KLa mapping cannot be empty.")
    return mapping


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _cpu_model() -> str:
    candidates = [platform.processor(), os.environ.get("PROCESSOR_IDENTIFIER", "")]
    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                candidates.insert(0, str(winreg.QueryValueEx(key, "ProcessorNameString")[0]))
        except (OSError, ImportError):
            pass
    return next((value.strip() for value in candidates if str(value).strip()), platform.machine())


def _total_memory_gb() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().total / 1024**3)
    except (ImportError, AttributeError):
        pass
    if os.name == "nt":
        try:
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.ullTotalPhys / 1024**3)
        except (AttributeError, OSError):
            return None
    return None


def _physical_core_count() -> int | None:
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        return int(count) if count is not None else None
    except (ImportError, AttributeError):
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _operating_system_metadata() -> dict[str, str]:
    """Return a readable Windows name while preserving the raw kernel report."""
    raw_release = platform.release()
    raw_version = platform.version()
    raw_platform = platform.platform()
    display_release = raw_release
    display_platform = raw_platform
    if os.name == "nt" and str(raw_release) == "10":
        match = re.search(r"10\.0\.(\d+)", str(raw_version))
        if match and int(match.group(1)) >= 22000:
            display_release = "11"
            display_platform = str(raw_platform).replace("Windows-10", "Windows-11", 1)
    return {
        "system": platform.system(),
        "release": display_release,
        "kernel_release": raw_release,
        "version": raw_version,
        "platform": display_platform,
        "raw_platform": raw_platform,
    }


def collect_environment_metadata(project_root: Path | None = None) -> dict:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    clock = time.get_clock_info("perf_counter")
    source_paths = [
        root / "src" / "aerbench" / "surrogate.py",
        root / "src" / "aerbench" / "fixed_aeration.py",
        root / "src" / "aerbench" / "bsm1_validation.py",
        root / "src" / "aerbench" / "timing_benchmark.py",
        root / "scripts" / "run_standardized_timing.py",
    ]
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cpu_model": _cpu_model(),
            "physical_cpu_cores": _physical_core_count(),
            "logical_cpu_cores": os.cpu_count(),
            "total_memory_gb": _total_memory_gb(),
            "machine_architecture": platform.machine(),
        },
        "operating_system": _operating_system_metadata(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "software_versions": {
            "bsm2-python": _package_version("bsm2-python"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "scikit-learn": _package_version("scikit-learn"),
            "numba": _package_version("numba"),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "timer": {
            "name": "time.perf_counter",
            "implementation": clock.implementation,
            "resolution_seconds": clock.resolution,
            "monotonic": clock.monotonic,
        },
        "source_sha256": {str(path.relative_to(root)): _file_sha256(path) for path in source_paths},
    }


def descriptive_timing_stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Timing statistics require at least one finite value.")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "cv_percent": float(100.0 * std / mean) if mean > 0.0 else np.nan,
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _result_is_consistent(group: pd.DataFrame, tolerance: float = 1e-10) -> bool:
    for column in ["SNH_mean_eval", "SNH_violation_hours_gt_limit"]:
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size > 1 and float(np.max(values) - np.min(values)) > tolerance:
            return False
    return True


def _model_summary_rows(measured: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    frames = [(scenario, group) for scenario, group in measured.groupby("scenario", sort=False)]
    frames.append(("ALL", measured))
    for scenario, frame in frames:
        for model_label, group in frame.groupby("model_label", sort=False):
            if scenario == "ALL":
                result_consistent = all(
                    _result_is_consistent(scenario_group)
                    for _, scenario_group in group.groupby("scenario", sort=False)
                )
            else:
                result_consistent = _result_is_consistent(group)
            for scope, column in TIMING_SCOPE_COLUMNS.items():
                stats = descriptive_timing_stats(pd.to_numeric(group[column], errors="coerce"))
                rows.append(
                    {
                        "scenario": scenario,
                        "model_label": model_label,
                        "model_family": group["model_family"].iloc[0],
                        "timestep_min": float(group["timestep_min"].iloc[0]),
                        "timing_scope": scope,
                        "n_repeats": stats["n"],
                        "mean_seconds": stats["mean"],
                        "std_seconds": stats["std"],
                        "cv_percent": stats["cv_percent"],
                        "median_seconds": stats["median"],
                        "p25_seconds": stats["p25"],
                        "p75_seconds": stats["p75"],
                        "iqr_seconds": stats["iqr"],
                        "min_seconds": stats["minimum"],
                        "max_seconds": stats["maximum"],
                        "p05_seconds": stats["p05"],
                        "p95_seconds": stats["p95"],
                        "result_consistent_across_repeats": result_consistent,
                    }
                )
    return rows


def _comparison_definitions(specs: Sequence[TimingModelSpec]) -> list[tuple[str, str, str, str]]:
    surrogates = [spec for spec in specs if spec.family == "reduced_surrogate"]
    bsm1 = next((spec for spec in specs if spec.family == "BSM2-Python/BSM1"), None)
    comparisons: list[tuple[str, str, str, str]] = []
    if bsm1 is not None and surrogates:
        default = surrogates[0]
        comparisons.append(
            (
                bsm1.label,
                default.label,
                "operational_default",
                "Combined effect of model equations, solver/implementation, and configured time step.",
            )
        )
        matched = min(surrogates, key=lambda spec: abs(spec.timestep_min - bsm1.timestep_min))
        if matched.label != default.label:
            comparisons.append(
                (
                    bsm1.label,
                    matched.label,
                    "matched_timestep",
                    "Models compared at the closest available time step; implementation and solver still differ.",
                )
            )
    if len(surrogates) >= 2:
        fine = min(surrogates, key=lambda spec: spec.timestep_min)
        coarse = max(surrogates, key=lambda spec: spec.timestep_min)
        if fine.label != coarse.label:
            comparisons.append(
                (
                    fine.label,
                    coarse.label,
                    "surrogate_timestep_effect",
                    "Same reduced-model equations and implementation; isolates the effect of time-step coarsening.",
                )
            )
    return comparisons


def _speedup_rows(measured: pd.DataFrame, specs: Sequence[TimingModelSpec]) -> list[dict]:
    rows: list[dict] = []
    scenarios = list(measured["scenario"].drop_duplicates()) + ["ALL"]
    spec_map = {spec.label: spec for spec in specs}
    for reference_label, candidate_label, comparison_type, interpretation in _comparison_definitions(specs):
        for scenario in scenarios:
            frame = measured if scenario == "ALL" else measured[measured["scenario"].eq(scenario)]
            reference = frame[frame["model_label"].eq(reference_label)]
            candidate = frame[frame["model_label"].eq(candidate_label)]
            keys = ["scenario", "repeat_index"] if scenario == "ALL" else ["repeat_index"]
            for scope, column in TIMING_SCOPE_COLUMNS.items():
                paired = pd.merge(
                    reference[keys + [column]],
                    candidate[keys + [column]],
                    on=keys,
                    suffixes=("_reference", "_candidate"),
                    how="inner",
                )
                if paired.empty:
                    continue
                ref_values = pd.to_numeric(paired[f"{column}_reference"], errors="coerce").to_numpy(dtype=float)
                cand_values = pd.to_numeric(paired[f"{column}_candidate"], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(ref_values) & np.isfinite(cand_values) & (cand_values > 0.0)
                ref_values = ref_values[valid]
                cand_values = cand_values[valid]
                if ref_values.size == 0:
                    continue
                paired_stats = descriptive_timing_stats(ref_values / cand_values)
                rows.append(
                    {
                        "scenario": scenario,
                        "comparison_type": comparison_type,
                        "timing_scope": scope,
                        "reference_model": reference_label,
                        "candidate_model": candidate_label,
                        "reference_timestep_min": spec_map[reference_label].timestep_min,
                        "candidate_timestep_min": spec_map[candidate_label].timestep_min,
                        "n_paired_repeats": paired_stats["n"],
                        "reference_mean_seconds": float(np.mean(ref_values)),
                        "candidate_mean_seconds": float(np.mean(cand_values)),
                        "speedup_ratio_of_means": float(np.mean(ref_values) / np.mean(cand_values)),
                        "paired_speedup_mean": paired_stats["mean"],
                        "paired_speedup_std": paired_stats["std"],
                        "paired_speedup_cv_percent": paired_stats["cv_percent"],
                        "paired_speedup_median": paired_stats["median"],
                        "paired_speedup_p25": paired_stats["p25"],
                        "paired_speedup_p75": paired_stats["p75"],
                        "paired_speedup_iqr": paired_stats["iqr"],
                        "paired_speedup_p05": paired_stats["p05"],
                        "paired_speedup_p95": paired_stats["p95"],
                        "paired_speedup_min": paired_stats["minimum"],
                        "paired_speedup_max": paired_stats["maximum"],
                        "interpretation": interpretation,
                    }
                )
    return rows


def build_timing_summaries(
    raw_runs: pd.DataFrame,
    specs: Sequence[TimingModelSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    measured = raw_runs[raw_runs["measurement_phase"].eq("measured")].copy()
    if measured.empty:
        raise ValueError("No measured timing runs were provided.")
    model_summary = pd.DataFrame(_model_summary_rows(measured))
    speedup_summary = pd.DataFrame(_speedup_rows(measured, specs))
    return model_summary, speedup_summary


def _execute_one(
    *,
    spec: TimingModelSpec,
    scenario: str,
    kla: float,
    warmup_days: float,
    eval_days: float,
    nh4_limit: float,
    oxygen_transfer_factor: float,
    phase: str,
    run_index: int,
    order_position: int,
    surrogate_runner: Callable[..., dict],
    bsm1_runner: Callable[..., dict],
) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    tic = time.perf_counter()
    if spec.family == "reduced_surrogate":
        result = surrogate_runner(
            scenario_name=scenario,
            kla=kla,
            warmup_days=warmup_days,
            eval_days=eval_days,
            dt_minutes=spec.timestep_min,
            nh4_limit=nh4_limit,
            oxygen_transfer_factor=oxygen_transfer_factor,
        )
    else:
        result = bsm1_runner(
            scenario=scenario,
            kla=kla,
            warmup_days=warmup_days,
            eval_days=eval_days,
            nh4_limit=nh4_limit,
            inflow_mode="steady_const",
            timestep_min=spec.timestep_min,
            oxygen_transfer_factor=oxygen_transfer_factor,
            tempmodel_policy="scenario",
            return_biomass_timeseries=False,
        )
    end_to_end_seconds = float(time.perf_counter() - tic)
    kernel_seconds = float(result["runtime_seconds"])
    if kernel_seconds <= 0.0 or end_to_end_seconds <= 0.0:
        raise ValueError("Every timed run must report a positive duration.")
    n_steps = int(round((float(warmup_days) + float(eval_days)) * 24.0 * 60.0 / spec.timestep_min))
    record = {
        "started_utc": started_utc,
        "timestamp": started_utc,
        "measurement_phase": phase,
        "repeat_index": int(run_index),
        "order_position": int(order_position),
        "run_order": int(order_position),
        "scenario": scenario,
        "kla": float(kla),
        "model_label": spec.label,
        "model_family": spec.family,
        "solver": spec.solver,
        "timestep_min": float(spec.timestep_min),
        "n_configured_steps": n_steps,
        "warmup_days": float(warmup_days),
        "eval_days": float(eval_days),
        "simulation_kernel_seconds": kernel_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "setup_postprocess_seconds": float(max(0.0, end_to_end_seconds - kernel_seconds)),
        "SNH_mean_eval": float(result["SNH_mean_eval"]),
        "SNH_violation_hours_gt_limit": float(result["SNH_violation_hours_gt_limit"]),
    }
    record["result_hash"] = stable_record_hash(
        record,
        excluded=[
            "started_utc",
            "timestamp",
            "simulation_kernel_seconds",
            "end_to_end_seconds",
            "setup_postprocess_seconds",
            "order_position",
            "run_order",
        ],
    )
    return record


def _format_value(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def write_timing_report(
    *,
    path: Path,
    settings: Mapping[str, object],
    environment: Mapping[str, object],
    speedup_summary: pd.DataFrame,
) -> None:
    hardware = environment["hardware"]
    software = environment["software_versions"]
    lines = [
        "# Standardized timing benchmark",
        "",
        "## Protocol",
        "",
        f"- Recorded repeats per model/scenario: {settings['repeats']}",
        f"- Warm-up runs excluded from timing summaries per model/scenario: {settings['warmup_runs']}",
        f"- Physical simulation window: {settings['warmup_days']} d warm-up + {settings['eval_days']} d evaluation",
        "- Model order is randomized between repeats with the recorded fixed seed to reduce order bias.",
        "- Timer: Python time.perf_counter (monotonic high-resolution wall clock).",
        "- simulation_kernel includes only each model's simulation loop.",
        "- end_to_end includes input/setup, model construction, simulation, and in-memory metric post-processing.",
        "- Module import, warm-up runs, CSV/JSON/Markdown writing, and this report generation are excluded.",
        "",
        "## Hardware and software",
        "",
        f"- CPU: {hardware.get('cpu_model')}",
        f"- Cores: {hardware.get('physical_cpu_cores')} physical, {hardware.get('logical_cpu_cores')} logical",
        f"- RAM: {_format_value(hardware.get('total_memory_gb'), 3)} GB",
        f"- OS: {environment['operating_system'].get('platform')} "
        f"(kernel release {environment['operating_system'].get('kernel_release')})",
        f"- Python: {environment['python'].get('version')} ({environment['python'].get('implementation')})",
        f"- bsm2-python: {software.get('bsm2-python')}",
        f"- NumPy/Pandas/SciPy: {software.get('numpy')} / {software.get('pandas')} / {software.get('scipy')}",
        "",
        "## Solver and implementation",
        "",
        "- Reduced surrogate: six-state explicit Euler update followed by component-wise clipping.",
        "- BSM1: BSM2-Python BSM1OL; reactor and settler ODEs are integrated with scipy.integrate.odeint (LSODA) using package defaults.",
        "",
        "## Speedup results",
        "",
        "| Scenario | Comparison | Scope | Reference mean (s) | Candidate mean (s) | Ratio of means | Paired median | Paired IQR | Paired P5–P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in speedup_summary.to_dict(orient="records"):
        lines.append(
            "| {scenario} | {comparison_type} | {timing_scope} | {reference_mean} | "
            "{candidate_mean} | {speedup} | {speedup_median} | {speedup_iqr} | {speedup_range} |".format(
                scenario=row["scenario"],
                comparison_type=row["comparison_type"],
                timing_scope=row["timing_scope"],
                reference_mean=_format_value(row["reference_mean_seconds"]),
                candidate_mean=_format_value(row["candidate_mean_seconds"]),
                speedup=_format_value(row["speedup_ratio_of_means"]),
                speedup_median=_format_value(row["paired_speedup_median"]),
                speedup_iqr=_format_value(row["paired_speedup_iqr"]),
                speedup_range=(
                    f"{_format_value(row['paired_speedup_p05'])}–{_format_value(row['paired_speedup_p95'])}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limit",
            "",
            "The operational speedup combines model structure, numerical solver, Python/package implementation, and time-step choice. "
            "It must not be attributed solely to the reduction in state dimension. The matched-timestep and surrogate-timestep comparisons "
            "are reported separately to expose these contributions. Exact values are specific to this hardware and software environment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_standardized_timing(
    *,
    output_dir: Path,
    scenarios: Sequence[str],
    kla_by_scenario: Mapping[str, float],
    repeats: int = 30,
    warmup_runs: int = 1,
    warmup_days: float = 60.0,
    eval_days: float = 7.0,
    surrogate_dt_minutes: Sequence[float] = (15.0, 1.0),
    bsm1_timestep_min: float = 1.0,
    nh4_limit: float = 4.0,
    oxygen_transfer_factor: float = 0.70,
    include_bsm1: bool = True,
    random_seed: int = 20260731,
    surrogate_runner: Callable[..., dict] = run_surrogate_fixed_aeration,
    bsm1_runner: Callable[..., dict] = run_bsm1_fixed_kla,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if int(repeats) < 2:
        raise ValueError("Use at least two measured repeats so timing variation can be reported.")
    if int(warmup_runs) < 0:
        raise ValueError("warmup_runs cannot be negative.")
    if float(warmup_days) < 0.0 or float(eval_days) <= 0.0:
        raise ValueError("warmup_days must be non-negative and eval_days must be positive.")
    scenario_list = [normalize_scenario_name(value) for value in scenarios]
    if not scenario_list:
        raise ValueError("At least one scenario is required.")
    missing = [scenario for scenario in scenario_list if scenario not in kla_by_scenario]
    if missing:
        raise ValueError(f"Missing representative KLa values for: {', '.join(missing)}")

    specs = build_model_specs(surrogate_dt_minutes, bsm1_timestep_min, include_bsm1=include_bsm1)
    records: list[dict] = []
    rng = np.random.default_rng(int(random_seed))
    for scenario in scenario_list:
        kla = float(kla_by_scenario[scenario])
        for warmup_index in range(1, int(warmup_runs) + 1):
            offset = (warmup_index - 1) % len(specs)
            ordered = specs[offset:] + specs[:offset]
            for order_position, spec in enumerate(ordered, start=1):
                print(f"Warm-up scenario={scenario} model={spec.label} run={warmup_index}/{warmup_runs}")
                records.append(
                    _execute_one(
                        spec=spec,
                        scenario=scenario,
                        kla=kla,
                        warmup_days=warmup_days,
                        eval_days=eval_days,
                        nh4_limit=nh4_limit,
                        oxygen_transfer_factor=oxygen_transfer_factor,
                        phase="warmup",
                        run_index=warmup_index,
                        order_position=order_position,
                        surrogate_runner=surrogate_runner,
                        bsm1_runner=bsm1_runner,
                    )
                )
        for repeat_index in range(1, int(repeats) + 1):
            ordered = [specs[index] for index in rng.permutation(len(specs))]
            for order_position, spec in enumerate(ordered, start=1):
                print(f"Timing scenario={scenario} model={spec.label} repeat={repeat_index}/{repeats}")
                records.append(
                    _execute_one(
                        spec=spec,
                        scenario=scenario,
                        kla=kla,
                        warmup_days=warmup_days,
                        eval_days=eval_days,
                        nh4_limit=nh4_limit,
                        oxygen_transfer_factor=oxygen_transfer_factor,
                        phase="measured",
                        run_index=repeat_index,
                        order_position=order_position,
                        surrogate_runner=surrogate_runner,
                        bsm1_runner=bsm1_runner,
                    )
                )

    raw = pd.DataFrame(records)
    model_summary, speedup_summary = build_timing_summaries(raw, specs)
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(table_dir / "standardized_timing_raw.csv", index=False)
    model_summary.to_csv(table_dir / "standardized_timing_summary.csv", index=False)
    speedup_summary.to_csv(table_dir / "standardized_speedup_summary.csv", index=False)

    settings = {
        "scenarios": scenario_list,
        "representative_kla_by_scenario": {scenario: float(kla_by_scenario[scenario]) for scenario in scenario_list},
        "representative_kla_basis": (
            "Rounded threshold-neighborhood points from the matched fine-grid BSM1 validation; "
            "these points are used only to standardize the timing comparison."
        ),
        "repeats": int(repeats),
        "warmup_runs": int(warmup_runs),
        "warmup_days": float(warmup_days),
        "eval_days": float(eval_days),
        "surrogate_dt_minutes": [float(value) for value in surrogate_dt_minutes],
        "bsm1_timestep_min": float(bsm1_timestep_min),
        "nh4_limit": float(nh4_limit),
        "oxygen_transfer_factor": float(oxygen_transfer_factor),
        "random_seed": int(random_seed),
        "inflow_mode": "steady_const",
        "models": [asdict(spec) for spec in specs],
    }
    environment = collect_environment_metadata()
    write_json(
        manifest_dir / "standardized_timing_manifest.json",
        {
            "settings": settings,
            "environment": environment,
            "timing_scope": {
                "simulation_kernel": (
                    "Includes only the simulation loop reported by each model runner; setup and final metric aggregation are excluded."
                ),
                "end_to_end": (
                    "Includes in-memory input/setup, model construction, simulation, and metric aggregation. "
                    "Imports, warm-up runs, and result-file writing are excluded."
                ),
            },
            "ordering_control": "Model order is randomized for every measured repeat using the recorded fixed seed.",
            "outlier_policy": "All finite valid runs are retained; median, IQR, mean, SD, P5 and P95 are reported.",
            "interpretation_limit": (
                "Speedup depends on model structure, time step, solver, code/package implementation, hardware, and system load; "
                "it is not attributable solely to state-dimension reduction."
            ),
            "command": " ".join(sys.argv),
        },
    )
    write_timing_report(
        path=output_dir / "standardized_timing_report.md",
        settings=settings,
        environment=environment,
        speedup_summary=speedup_summary,
    )
    return raw, model_summary, speedup_summary
