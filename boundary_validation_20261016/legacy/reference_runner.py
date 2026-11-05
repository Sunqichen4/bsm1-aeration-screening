"""Fresh constant-condition BSM1 references, with the frozen 20261013 definition.

This module never reads old reference labels.  It reuses the original influent,
time-grid and nitrogen-accounting helpers, and retains native initial states.
The reporting mask is on output endpoints, [60, 67), exactly as run_revision.py.
The 74-d challenge runner and its explicit temperature initialization are not
used.  Parallel audit elapsed times are not serial performance measurements.

Public APIs: source_hashes(), run_reference_case(), run_reference_audit().
The ``check`` and ``self-test`` commands do not instantiate BSM1 or run ODEs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
from typing import Any, Mapping
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent

SI, SS, XI, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK, TSS, Q, TEMP, SD1, SD2, SD3, XD4, XD5 = range(21)
STATE_COLUMNS = ["SI", "SS", "XI", "XS", "XBH", "XBA", "XP", "SO", "SNO", "SNH", "SND", "XND", "SALK", "TSS", "Q", "TEMP", "SD1", "SD2", "SD3", "XD4", "XD5"]
REFERENCE_DEFINITION = "constant_native_initial_states_endpoint_output_mask_60_le_t_lt_67_v1"
EXPECTED_UNITS = {"time": "d", "kla": "d^-1", "SNH_SNO_SND_XND_TN": "g N/m3 = mg N/L", "SO": "g O2/m3 = mg O2/L", "SS_XBH_XBA_COD": "g COD/m3 = mg COD/L", "flow": "m3/d", "temperature": "degC"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def array_sha(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(json.dumps(list(a.shape)).encode("ascii") + a.tobytes()).hexdigest()


def json_save(path: str | Path, value: Any, *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def base_project(protocol=None):
    return Path(__file__).resolve().parents[1] / "vendor/original_project"



def _float(case: Mapping[str, Any], key: str) -> float:
    value = float(case[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite case field {key}")
    return value


def normalize_case(case: Mapping[str, Any], dt_min: float = 1.0, nh4_limit: float = 4.0) -> dict:
    identity, scenario = str(case["candidate_id"]), str(case["scenario"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity) or identity in {".", ".."}:
        raise ValueError("candidate_id must be a safe, unique file stem")
    if not scenario or scenario != scenario.strip().lower():
        raise ValueError("scenario must be a nonempty canonical lowercase name")
    load, alpha, kla = (_float(case, k) for k in ("load_coordinate", "oxygen_transfer_factor", "kla"))
    dt, limit = float(dt_min), float(nh4_limit)
    if load < 0 or not 0 < alpha <= 1 or kla < 0:
        raise ValueError("Invalid load/alpha/KLa")
    if not math.isfinite(dt) or not 0 < dt <= 1 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("dt_min must be in (0,1] and NH4 limit positive")
    if any(not math.isclose(days * 1440. / dt, round(days * 1440. / dt), abs_tol=1e-7, rel_tol=0.) for days in (60., 7., 67.)):
        raise ValueError("Timestep must divide the frozen 60+7-day boundaries")
    effective = kla * alpha
    if "effective_kla" in case and not math.isclose(float(case["effective_kla"]), effective, rel_tol=0., abs_tol=1e-10):
        raise ValueError("effective_kla must equal alpha * commanded KLa once")
    for key, expected in (("cod_scale", 1 + .2 * load), ("nh4_scale", 1 + .35 * load), ("flow_scale", 1.), ("surrogate_mu_a_scale", 1 + .05 * load)):
        if key in case and not math.isclose(float(case[key]), expected, rel_tol=0., abs_tol=1e-10):
            raise ValueError(f"Case metadata disagrees with original definition: {key}")
    for key, expected in (("warmup_days", 60.), ("eval_days", 7.), ("temperature_C", 15.), ("nh4_limit", limit)):
        if key in case and not math.isclose(float(case[key]), expected, rel_tol=0., abs_tol=1e-10):
            raise ValueError(f"This runner requires {key}={expected}")
    if "tempmodel" in case and str(case["tempmodel"]).lower() not in {"false", "0", "0.0"}:
        raise ValueError("This runner retains tempmodel=False")
    normalized = {"candidate_id": identity, "scenario": scenario, "load_coordinate": load,
            "oxygen_transfer_factor": alpha, "kla": kla, "effective_kla": effective,
            "kla_hex": kla.hex(), "load_hex": load.hex(), "alpha_hex": alpha.hex(),
            "cod_scale": 1 + .2 * load, "nh4_scale": 1 + .35 * load,
            "flow_scale": 1., "surrogate_mu_a_scale": 1 + .05 * load,
            "operating_point_definition": "steady_load_transfer_v1",
            "warmup_days": 60., "eval_days": 7., "total_days": 67., "nh4_limit": limit,
            "timestep_min": dt, "tempmodel": False, "temperature_C": 15.,
            "reference_definition": REFERENCE_DEFINITION,
            "within_marginal_ranges": bool(0 <= load <= 1 and .7 <= alpha <= 1),
            "within_joint_condition_hull": bool(0 <= load <= 1 and .7 + .3 * load - 1e-12 <= alpha <= 1),
            "within_model_feature_hull": bool(0 <= load <= 1 and 28 + 12 * load - 1e-12 <= effective <= 520)}
    for key in ("grid_kind", "selection_reason"):
        if key in case:
            normalized[key] = str(case[key])
    return normalized


def summarize_evaluation(eff: np.ndarray, tank5: np.ndarray, performance: np.ndarray, *, dt_min: float, nh4_limit: float) -> dict:
    """Pure aggregation shared with checkpoint verification; no model execution."""
    eff, tank5, performance = (np.asarray(a, dtype=float) for a in (eff, tank5, performance))
    if eff.ndim != 2 or tank5.shape != eff.shape or eff.shape[1] != 21 or performance.shape != (len(eff), 3) or len(eff) < 1:
        raise ValueError("Evaluation histories have unexpected shape")
    if not all(np.isfinite(a).all() for a in (eff, tank5, performance)):
        raise FloatingPointError("Nonfinite reference evaluation states/performance")
    snh, violations = eff[:, SNH], eff[:, SNH] > nh4_limit
    cod = eff[:, :7].sum(axis=1)
    tn = eff[:, SNH] + eff[:, SNO] + eff[:, SND] + eff[:, XND] + .08 * (eff[:, XBH] + eff[:, XBA]) + .06 * eff[:, XP]
    result = {"SNH_mean_eval": float(snh.mean()), "SNH_max_eval": float(snh.max()),
              "SNH_min_eval": float(snh.min()), "SNH_p95_eval": float(np.quantile(snh, .95)),
              "SNH_zero_violation": bool(not violations.any()), "SNH_violation_ratio_gt_limit": float(violations.mean()),
              "SNH_violation_ratio": float(violations.mean()), "SNH_violation_hours_gt_limit": float(violations.sum() * dt_min / 60.),
              "SNH_violation_days_gt_limit": float(violations.sum() * dt_min / 1440.),
              "COD_mean_eval": float(cod.mean()), "COD_max_eval": float(cod.max()),
              "TN_mean_eval": float(tn.mean()), "TN_max_eval": float(tn.max()),
              "DO_mean_eval": float(tank5[:, SO].mean()), "DO_min_eval": float(tank5[:, SO].min()),
              "DO_tank5_mean_eval": float(tank5[:, SO].mean()), "DO_tank5_p05_eval": float(np.quantile(tank5[:, SO], .05)),
              "effluent_Q_mean_eval": float(eff[:, Q].mean()), "evaluation_samples": len(eff),
              "minimum_effluent_component": float(eff.min()),
              "minimum_evaluation_concentration": float(min(eff[:, :14].min(), tank5[:, :14].min())),
              "negative_evaluation_concentration_count": int(np.count_nonzero(eff[:, :14] < 0) + np.count_nonzero(tank5[:, :14] < 0)),
              "pumping_kWh_per_d": float(performance[:, 0].mean()), "aeration_kWh_per_d": float(performance[:, 1].mean()),
              "mixing_kWh_per_d": float(performance[:, 2].mean()), "total_model_energy_kWh_per_d": float(performance.sum(axis=1).mean())}
    for prefix, history in (("eff", eff), ("tank5", tank5)):
        for state, index in (("SO", SO), ("SNO", SNO), ("SS", SS), ("SNH", SNH), ("XBH", XBH), ("XBA", XBA), ("SND", SND), ("XND", XND)):
            values = history[:, index]
            result.update({f"{prefix}_{state}_{stat}_eval": float(function(values)) for stat, function in (("mean", np.mean), ("min", np.min), ("max", np.max))})
    return result


def _native_dependencies(base: Path):
    source = str(base / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from aerbench.steady_conditions import SteadyOperatingPoint
    from aerbench.bsm1_validation import require_bsm1ol, make_operating_point_data, bsm1_internal_evaltime, bsm1_executable_step_count
    return SteadyOperatingPoint, require_bsm1ol, make_operating_point_data, bsm1_internal_evaltime, bsm1_executable_step_count


def run_reference_case(case: Mapping[str, Any], dt_min: float = 1.0, output_dir: str | Path | None = None,
                       freeze_sha256: str | None = None, nh4_limit: float = 4.0, base: str | Path | None = None) -> dict:
    """Integrate a new native instance and optionally save immutable raw output.

    Existing successes, failures, raw files or locks are never overwritten.
    Each case receives fresh native initial states; no state or label cache is
    accepted.  Retrying a failed or interrupted attempt needs a new directory.
    """
    entered, started = time.perf_counter(), now()
    spec = normalize_case(case, dt_min, nh4_limit)
    case_digest = object_sha(spec)
    project = Path(base).resolve() if base is not None else base_project()
    folder = Path(output_dir).resolve() if output_dir is not None else None
    identity, lock, trajectory_folder = spec["candidate_id"], None, None
    if folder is not None:
        if not freeze_sha256:
            raise ValueError("Writing references requires the current freeze SHA256")
        folder.mkdir(parents=True, exist_ok=True)
        trajectory_folder = folder.parent / "reference_trajectories"
        trajectory_folder.mkdir(parents=True, exist_ok=True)
        outputs = [folder / f"{identity}{suffix}" for suffix in (".json", "_failure.json", ".lock")]
        outputs += [trajectory_folder / f"{identity}{suffix}" for suffix in ("_evaluation.npz", "_evaluation.npz.tmp")]
        if any(p.exists() for p in outputs):
            raise FileExistsError(f"Reference attempt already exists for {identity}; successful checkpoints are validated by run_reference_audit")
        lock = folder / (identity + ".lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_utc": started, "case_sha256": case_digest, "freeze_sha256": freeze_sha256}, stream)
    try:
        SteadyOperatingPoint, require_bsm1ol, make_data, internal_evaltime, executable_count = _native_dependencies(project)
        point = SteadyOperatingPoint(spec["scenario"], spec["load_coordinate"], spec["oxygen_transfer_factor"])
        data = make_data(67., point)
        step = spec["timestep_min"] / 1440.
        plant = require_bsm1ol()(data_in=data, timestep=step, endtime=67.,
                                evaltime=internal_evaltime(data, step, 67., 60.), tempmodel=False, activate=False)
        n = executable_count(plant)
        klas = np.array([0., 0., spec["effective_kla"], spec["effective_kla"], spec["effective_kla"]])
        kernel_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(n):
                plant.step(i, klas=klas)
        kernel = time.perf_counter() - kernel_start
        interval_starts = np.asarray(plant.simtime[:n])
        times = interval_starts + np.asarray(plant.timesteps[:n])
        mask = (times >= 60.) & (times < 67.)
        eff = np.asarray(plant.ys_eff_all[:n])[mask].copy()
        tank5 = np.asarray(plant.y_out5_all[:n])[mask].copy()
        performance = np.asarray(plant.perf_factors_all[:n])[mask].copy()
        expected = int(round(7. * 1440. / spec["timestep_min"]))
        if len(eff) != expected or not np.allclose(np.diff(times[mask]), step, atol=1e-10, rtol=0.):
            raise ValueError("Native reporting window does not match the frozen regular 7-day endpoint mask")
        metrics = summarize_evaluation(eff, tank5, performance, dt_min=spec["timestep_min"], nh4_limit=spec["nh4_limit"])
        metrics.update(spec)
        metrics.update({"case_sha256": case_digest, "freeze_sha256": freeze_sha256,
                        "reference_runner_source_sha256": sha(__file__), "fresh_reference": True,
                        "reference_result_origin": "new_native_bsm1_integration_no_result_cache",
                        "computed_at_utc": now(), "reference_started_utc": started, "pid": os.getpid(),
                        "input_sha256": array_sha(data), "input_shape": list(data.shape),
                        "native_initialization": "original_native_defaults_no_explicit_temperature_or_state_changes",
                        "tempmodel_semantics": "False_uses_15C_influent_for_native_kinetics_and_oxygen_transfer",
                        "load_application": "only_influent_SS_XS_times_1_plus_0.2L_and_SNH_times_1_plus_0.35L;_native_kinetics_and_flow_unchanged",
                        "alpha_application": "effective_KLa=alpha_times_commanded_KLa_once_in_reactors_3_4_5",
                        "simulation_steps": n, "evaluation_alignment": "output_endpoint_60_le_t_lt_67_matching_frozen_run_revision",
                        "first_eval_interval_start_day": float(interval_starts[mask][0]), "first_eval_output_end_day": float(times[mask][0]),
                        "last_eval_output_end_day": float(times[mask][-1]),
                        "kernel_seconds": float(kernel), "elapsed_seconds": float(time.perf_counter() - entered),
                        "energy_basis": "native_effective_kla_model_metric_not_measured_electricity",
                        "commanded_aeration_index": spec["kla"] / 240., "units": EXPECTED_UNITS,
                        "solver_warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                        "solver_warning_count": len(caught), "native_internal_clipping": "native_intermediate_clipping_is_unchanged_and_not_instrumented",
                        "timing_scope": "audit_cost;_parallel_elapsed_seconds_are_not_serial_performance",
                        "complete": True})
        writing = time.perf_counter()
        if folder is not None:
            raw = trajectory_folder / (identity + "_evaluation.npz")
            temporary = raw.with_name(raw.name + ".tmp")
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, interval_start_day=interval_starts[mask], time_day=times[mask],
                                    effluent_states=eff, tank5_states=tank5, performance=performance,
                                    input_data=data, state_columns=np.asarray(STATE_COLUMNS),
                                    nh4_exceeds_limit=eff[:, SNH] > spec["nh4_limit"])
            temporary.replace(raw)
            metrics["raw_file"] = str(raw)
            metrics["raw_file_sha256"] = sha(raw)
        metrics["output_write_seconds"] = float(time.perf_counter() - writing) if folder is not None else 0.
        metrics["total_with_output_seconds"] = float(time.perf_counter() - entered)
        if folder is not None:
            json_save(folder / (identity + ".json"), metrics)
        return metrics
    except BaseException as exc:
        if folder is not None:
            json_save(folder / (identity + "_failure.json"), {"candidate_id": identity, "case": spec,
                      "case_sha256": case_digest, "freeze_sha256": freeze_sha256, "pid": os.getpid(),
                      "reference_started_utc": started, "failed_at_utc": now(), "fresh_reference_attempted": True,
                      "elapsed_seconds": float(time.perf_counter() - entered), "complete": False,
                      "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc(),
                      "reference_runner_source_sha256": sha(__file__)})
        raise
    finally:
        if lock is not None and lock.exists():
            lock.unlink()

