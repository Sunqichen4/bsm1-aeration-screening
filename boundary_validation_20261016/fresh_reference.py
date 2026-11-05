"""Run and independently audit new BSM1 labels after model/design freezing.

The original project and its reference runner are read-only.  This wrapper only
reads candidate rows and frozen artifacts in the selected study directory; it
never imports historical reference labels.  Elapsed parallel work is an audit
cost, not a serial performance benchmark.

Required freeze schema::

    {"files": [{"path": "new_candidates.csv", "sha256": "..."}, ...],
     "source_hashes": {
       "design": {"path": "design.json", "sha256": "..."},
       "protocol": {"path": "protocol.json", "sha256": "..."}}}

Design/protocol entries may name JSON artifacts or their generating source.
All paths in ``files`` and ``source_hashes`` are study-relative.  Both sources
must also appear in ``files``.  The freeze should include models, predictions,
routes, and this wrapper, in addition to the candidate/design/protocol files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd


PACKAGE_PATH = Path(__file__).resolve().parent
RUNNER_PATH = PACKAGE_PATH / "legacy" / "reference_runner.py"
VENDOR_PATH = PACKAGE_PATH / "legacy" / "vendor" / "original_project"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")
ORIGIN = "new_native_bsm1_integration_no_result_cache"
INITIAL_ISOLATION = "owned_ndarray_copies_of_native_reactor_settler_performance_defaults_values_unchanged_v1"
DT_MIN = 1.0
NH4_LIMIT = 4.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _load_runner():
    """Import the historical runner under an isolated module identity."""
    name = "_model_revision_20261016_native_reference_runner"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != RUNNER_PATH.resolve():
            raise ValueError("Cached reference runner came from a different package; restart Python")
        return existing
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load BSM1 runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _relative_file(study: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Freeze paths must be nonempty study-relative paths")
    path = (study / relative).resolve()
    if not path.is_relative_to(study) or not path.is_file():
        raise ValueError(f"Frozen file missing or outside study: {relative}")
    return path


def _source_entry(freeze: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    # The explicit aliases permit small differences in the parent workflow's
    # manifest, while retaining a verifiable path and digest for both sources.
    sources = freeze.get("source_hashes", {})
    entry = sources.get(role) if isinstance(sources, dict) else None
    if entry is None:
        entry = freeze.get(f"{role}_source_sha256", freeze.get(f"{role}_source"))
    if isinstance(entry, str):
        path = freeze.get(f"{role}_source_path", freeze.get(f"{role}_path"))
        entry = {"path": path, "sha256": entry}
    if not isinstance(entry, dict) or not {"path", "sha256"} <= entry.keys():
        raise ValueError(f"freeze.json must bind the {role} source path and sha256")
    return entry


def verify_freeze(study_dir: Path, *, candidate_filename: str = "new_candidates.csv",
                  dt_min: float = DT_MIN) -> tuple[dict, str, list[dict]]:
    """Verify all immutable inputs before reading a checkpoint or running ODEs."""
    study = Path(study_dir).resolve()
    freeze_path = study / "freeze.json"
    freeze = _read_json(freeze_path)
    files = freeze.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("freeze.json files must be a nonempty list")
    digests = {}
    for entry in files:
        if not isinstance(entry, dict) or not {"path", "sha256"} <= entry.keys():
            raise ValueError("Every frozen file needs path and sha256")
        path = _relative_file(study, entry["path"])
        if path in digests:
            raise ValueError(f"Duplicate frozen path: {entry['path']}")
        if _sha(path) != entry["sha256"]:
            raise ValueError(f"Frozen file changed: {entry['path']}")
        digests[path] = entry["sha256"]
    for role in ("design", "protocol"):
        entry = _source_entry(freeze, role)
        path = _relative_file(study, entry["path"])
        if digests.get(path) != entry["sha256"] or _sha(path) != entry["sha256"]:
            raise ValueError(f"Frozen {role} source changed or is absent from files")
    if candidate_filename not in {"new_candidates.csv", "half_step_candidates.csv"} or dt_min not in {1., .5}:
        raise ValueError("Only frozen main/half-step candidate files and 1/0.5-min grids are supported")
    candidate_path = study / candidate_filename
    if candidate_path not in digests:
        raise ValueError(f"{candidate_filename} must be included in freeze.json files")
    candidates = pd.read_csv(candidate_path, float_precision="round_trip", dtype={"candidate_id": str, "scenario": str})
    if candidates.empty or not {"candidate_id", "scenario", "kla"} <= set(candidates.columns):
        raise ValueError("Candidates need nonempty candidate_id, scenario, kla rows")
    forbidden = {"SNH_max_eval", "SNH_zero_violation", "case_sha256",
                 "raw_file", "raw_file_sha256", "reference_result_origin"}
    if forbidden.intersection(candidates.columns):
        raise ValueError(f"{candidate_filename} contains reference labels/provenance")
    if candidates["candidate_id"].duplicated().any() or candidates[["candidate_id", "scenario", "kla"]].isna().any().any():
        raise ValueError("Candidates have duplicate IDs or missing required fields")
    runner = _load_runner()
    cases = []
    for row in candidates.to_dict(orient="records"):
        if "load_coordinate" not in row:
            row["load_coordinate"] = row.get("L")
        if "oxygen_transfer_factor" not in row:
            row["oxygen_transfer_factor"] = row.get("alpha")
        if "L" in row and not math.isclose(float(row["L"]), float(row["load_coordinate"]), abs_tol=1e-12):
            raise ValueError("Candidate L and load_coordinate disagree")
        if "alpha" in row and not math.isclose(float(row["alpha"]), float(row["oxygen_transfer_factor"]), abs_tol=1e-12):
            raise ValueError("Candidate alpha and oxygen_transfer_factor disagree")
        cases.append(runner.normalize_case(row, dt_min, NH4_LIMIT))
    return freeze, _sha(freeze_path), cases


def reference_source_hashes() -> dict[str, str]:
    """Sources actually used; these are recorded before producing new labels."""
    sources = [Path(__file__).resolve(), RUNNER_PATH]
    sources.extend(sorted((VENDOR_PATH / "src" / "aerbench").rglob("*.py")))
    sources.extend(path for path in sorted((VENDOR_PATH / "configs").rglob("*")) if path.is_file())
    sources.extend(path for path in sorted((PACKAGE_PATH / "legacy" / "third_party").glob("*")) if path.is_file())
    sources.append(PACKAGE_PATH / "legacy" / "THIRD_PARTY_NOTICES.md")
    # Stable relative labels let a copied package/results directory be audited
    # in a different location with equivalent installed native sources.
    hashes = {"package/" + path.relative_to(PACKAGE_PATH).as_posix(): _sha(path) for path in sources}
    # Bind the installed native implementation too, not only its version string.
    native = importlib.util.find_spec("bsm2_python")
    if native is not None and native.submodule_search_locations:
        for location in native.submodule_search_locations:
            native_root = Path(location)
            for path in sorted(native_root.rglob("*.py")):
                hashes["dependency/bsm2_python/" + path.relative_to(native_root).as_posix()] = _sha(path)
    return hashes


def _provenance(study: Path, freeze_sha: str, *, create: bool, dt_min: float = DT_MIN) -> dict:
    path = study / "reference_provenance.json"
    sources = reference_source_hashes()
    if path.exists():
        provenance = _read_json(path)
        if provenance.get("freeze_sha256") != freeze_sha:
            raise ValueError("Reference provenance belongs to a different freeze")
        if provenance.get("source_hashes") != sources:
            raise ValueError("Reference implementation changed since references began")
        if provenance.get("dt_min") != dt_min or provenance.get("native_initial_state_isolation") != INITIAL_ISOLATION:
            raise ValueError("Reference integration grid or native initialization policy changed")
        return provenance
    if not create:
        raise ValueError("Missing pre-execution reference_provenance.json")
    for folder in (study / "reference", study / "reference_trajectories"):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Existing reference output has no pre-execution provenance")
    try:
        version = importlib.metadata.version("bsm2-python")
    except importlib.metadata.PackageNotFoundError:
        version = "not_installed"
    provenance = {"freeze_sha256": freeze_sha, "created_at_utc": _now(),
                  "source_hashes": sources, "bsm2_python_version": version,
                  "reference_definition": _load_runner().REFERENCE_DEFINITION,
                  "dt_min": dt_min, "nh4_limit": NH4_LIMIT,
                  "reference_origin": ORIGIN, "old_label_imports": False,
                  "native_initial_state_isolation": INITIAL_ISOLATION}
    _write_json(path, provenance)
    return provenance


def _expected_input(case: Mapping[str, Any]) -> np.ndarray:
    """Rebuild input via separately imported vendor helpers; never run BSM1."""
    name = "_revision_reference_vendor_aerbench"
    if name not in sys.modules:
        package_dir = VENDOR_PATH / "src" / "aerbench"
        spec = importlib.util.spec_from_file_location(name, package_dir / "__init__.py",
                                                    submodule_search_locations=[str(package_dir)])
        if spec is None or spec.loader is None:
            raise ImportError("Cannot import original vendor package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    validation = importlib.import_module(name + ".bsm1_validation")
    conditions = importlib.import_module(name + ".steady_conditions")
    point = conditions.SteadyOperatingPoint(case["scenario"], case["load_coordinate"], case["oxygen_transfer_factor"])
    return validation.make_operating_point_data(67., point)


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, (bool, np.bool_)):
        equal = isinstance(actual, (bool, np.bool_)) and bool(actual) == bool(expected)
    elif isinstance(expected, (float, int, np.number)):
        try:
            equal = math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10)
        except (TypeError, ValueError):
            equal = False
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(f"Reference mismatch for {label}: {actual!r} != {expected!r}")


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict:
    manifest = {}
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        if not np.isfinite(array).all():
            raise ValueError(f"Native default array has nonfinite entries: {name}")
        manifest[name] = {"shape": list(array.shape), "dtype": array.dtype.str,
                          "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return manifest


def _native_default_array_manifest() -> dict:
    """Read pristine native defaults under isolated names, without a plant/ODE.

    The installed 0.0.16 initialization modules contain NumPy constant arrays.
    Reloading those source files under private names avoids trusting globals
    that a previous native plant in this Python process might have aliased.
    """
    cache_name = "_native_default_manifest_cache"
    if cache_name in globals():
        return globals()[cache_name]
    native_base = importlib.import_module("bsm2_python.bsm1_base")
    templates = {}
    for role in ("asm1init", "settler1dinit", "pp_init"):
        source = Path(getattr(native_base, role).__file__)
        name = "_revision_pristine_defaults_" + role
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot read pristine native defaults: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        templates[role] = module
    asm, settler, performance = (templates[role] for role in ("asm1init", "settler1dinit", "pp_init"))
    arrays = {}
    for i in range(1, 6):
        arrays[f"reactor{i}.y0"] = getattr(asm, f"YINIT{i}")
        arrays[f"reactor{i}.asm1par"] = getattr(asm, f"PAR{i}")
    arrays.update({"settler.dim": settler.DIM, "settler.layer": settler.LAYER,
                   "settler.ys0": settler.settlerinit, "settler.sedpar": settler.SETTLERPAR,
                   "settler.asm1par": asm.PAR1, "performance.pp_par": performance.PP_PAR})
    manifest = _array_manifest(arrays)
    globals()[cache_name] = manifest
    return manifest


def _copy_native_initial_arrays(plant: Any) -> dict:
    """Detach all mutable native component arrays without changing any value."""
    arrays, owners = {}, {}
    for name in [f"reactor{i}" for i in range(1, 6)] + ["settler", "performance"]:
        component = getattr(plant, name)
        for attribute, value in vars(component).items():
            if isinstance(value, np.ndarray):
                label = name + "." + attribute
                arrays[label] = value
                owners[label] = (component, attribute)
    before = _array_manifest(arrays)
    if before != _native_default_array_manifest():
        raise ValueError("Native initial arrays differ from pristine installed source defaults")
    detached = {}
    for name, value in arrays.items():
        component, attribute = owners[name]
        owned = value.copy(order="K")
        if np.shares_memory(value, owned) or not np.array_equal(value, owned):
            raise ValueError(f"Could not isolate native initial array: {name}")
        setattr(component, attribute, owned)
        detached[name] = owned
    after = _array_manifest(detached)
    if before != after:
        raise ValueError("Native initial values changed during ownership isolation")
    return {"array_count": len(before), "before": before, "after": after,
            "values_preserved": True, "shares_memory_after_copy": False}


def _owned_native_factory(native_constructor: Any, audit_records: list):
    def construct(*args, **kwargs):
        plant = native_constructor(*args, **kwargs)
        audit_records.append(_copy_native_initial_arrays(plant))
        return plant
    return construct


def _verify_success(study: Path, case: dict, freeze_sha: str, provenance: dict) -> dict:
    identity = case["candidate_id"]
    record = _read_json(study / "reference" / f"{identity}.json")
    runner = _load_runner()
    dt_min = float(case["timestep_min"])
    if dt_min != provenance["dt_min"]:
        raise ValueError(f"{identity}: candidate/reference provenance timestep mismatch")
    for key, value in case.items():
        _assert_equal(record.get(key), value, f"{identity}.{key}")
    for key, value in {"case_sha256": runner.object_sha(case), "freeze_sha256": freeze_sha,
                       "complete": True, "fresh_reference": True,
                       "reference_result_origin": ORIGIN,
                       "reference_runner_source_sha256": _sha(RUNNER_PATH),
                       "native_initial_state_isolation": INITIAL_ISOLATION,
                       "native_initialization": "original_native_defaults_no_explicit_temperature_or_state_changes",
                       "evaluation_alignment": "output_endpoint_60_le_t_lt_67_matching_frozen_run_revision"}.items():
        _assert_equal(record.get(key), value, f"{identity}.{key}")
    expected_defaults = _native_default_array_manifest()
    isolation = {"array_count": len(expected_defaults), "before": expected_defaults,
                 "after": expected_defaults, "values_preserved": True, "shares_memory_after_copy": False}
    _assert_equal(record.get("native_initial_arrays"), isolation, f"{identity}.native_initial_arrays")
    if datetime.fromisoformat(record["reference_started_utc"]) < datetime.fromisoformat(provenance["created_at_utc"]):
        raise ValueError(f"{identity}: reference predates this frozen execution")
    raw = study / "reference_trajectories" / f"{identity}_evaluation.npz"
    recorded_raw = Path(record.get("raw_file", ""))
    if recorded_raw.resolve() != raw.resolve() and not (
            recorded_raw.name == raw.name and recorded_raw.parent.name == "reference_trajectories"):
        raise ValueError(f"{identity}: raw path does not identify this execution's candidate trajectory")
    if not raw.is_file() or _sha(raw) != record.get("raw_file_sha256"):
        raise ValueError(f"{identity}: raw trajectory missing or hash mismatch")
    with np.load(raw, allow_pickle=False) as archive:
        required = {"interval_start_day", "time_day", "effluent_states", "tank5_states",
                    "performance", "input_data", "state_columns", "nh4_exceeds_limit"}
        if not required <= set(archive.files):
            raise ValueError(f"{identity}: incomplete raw trajectory")
        ends = archive["time_day"]
        starts = archive["interval_start_day"]
        eff, tank5, performance = (archive[key] for key in ("effluent_states", "tank5_states", "performance"))
        expected_n = int(7 * 1440 / dt_min)
        expected_ends = 60. + np.arange(expected_n) * dt_min / 1440.
        if ends.shape != (expected_n,) or starts.shape != ends.shape:
            raise ValueError(f"{identity}: incorrect evaluation sample count")
        if not (np.isfinite(ends).all() and np.isfinite(starts).all() and
                ((ends >= 60.) & (ends < 67.)).all() and
                np.allclose(ends, expected_ends, atol=2e-10, rtol=0.) and
                np.allclose(ends - starts, dt_min / 1440., atol=2e-10, rtol=0.)):
            raise ValueError(f"{identity}: incorrect [60,67) endpoint reporting window")
        if archive["state_columns"].tolist() != runner.STATE_COLUMNS:
            raise ValueError(f"{identity}: state column order changed")
        input_data = archive["input_data"]
        expected_input = _expected_input(case)
        if input_data.shape != expected_input.shape or not np.array_equal(input_data, expected_input):
            raise ValueError(f"{identity}: influent differs from original operating point definition")
        _assert_equal(record.get("input_shape"), list(input_data.shape), f"{identity}.input_shape")
        _assert_equal(record.get("input_sha256"), runner.array_sha(input_data), f"{identity}.input_sha256")
        recomputed = runner.summarize_evaluation(eff, tank5, performance, dt_min=dt_min, nh4_limit=NH4_LIMIT)
        # SNH is additionally recalculated directly, independently of the
        # historical runner's aggregation helper.
        snh = eff[:, runner.SNH]
        exceeds = snh > NH4_LIMIT
        if archive["nh4_exceeds_limit"].dtype.kind != "b" or not np.array_equal(archive["nh4_exceeds_limit"], exceeds):
            raise ValueError(f"{identity}: raw exceedance labels disagree with SNH")
        direct = {"SNH_max_eval": float(np.max(snh)), "SNH_mean_eval": float(np.mean(snh)),
                  "SNH_zero_violation": bool(not np.any(exceeds)), "evaluation_samples": expected_n,
                  "first_eval_interval_start_day": float(starts[0]),
                  "first_eval_output_end_day": float(ends[0]), "last_eval_output_end_day": float(ends[-1])}
        recomputed.update(direct)
        for key, value in recomputed.items():
            _assert_equal(record.get(key), value, f"{identity}.{key}")
    for key in ("kernel_seconds", "elapsed_seconds", "output_write_seconds", "total_with_output_seconds"):
        if not math.isfinite(float(record.get(key, float("nan")))) or float(record[key]) < 0:
            raise ValueError(f"{identity}: invalid timing {key}")
    return record


def _verify_failure(study: Path, case: dict, freeze_sha: str) -> dict:
    record = _read_json(study / "reference" / f"{case['candidate_id']}_failure.json")
    runner = _load_runner()
    for key, value in {"candidate_id": case["candidate_id"], "case_sha256": runner.object_sha(case),
                       "freeze_sha256": freeze_sha, "complete": False,
                       "reference_runner_source_sha256": _sha(RUNNER_PATH)}.items():
        _assert_equal(record.get(key), value, key)
    return record


def _check_inventory(study: Path, cases: list[dict]) -> None:
    ids = {row["candidate_id"] for row in cases}
    allowed = {f"{identity}{suffix}" for identity in ids for suffix in (".json", "_failure.json")}
    folder = study / "reference"
    if folder.exists():
        unexpected = {path.name for path in folder.iterdir()} - allowed
        if unexpected:
            raise ValueError(f"Unexpected/incomplete reference artifacts: {sorted(unexpected)}")
        for identity in ids:
            if (folder / f"{identity}.json").exists() and (folder / f"{identity}_failure.json").exists():
                raise ValueError(f"Both success and failure records exist for {identity}")
    raw_folder = study / "reference_trajectories"
    if raw_folder.exists():
        allowed_raw = {f"{identity}_evaluation.npz" for identity in ids}
        unexpected = {path.name for path in raw_folder.iterdir()} - allowed_raw
        if unexpected:
            raise ValueError(f"Unexpected/incomplete raw artifacts: {sorted(unexpected)}")
        for identity in ids:
            if (raw_folder / f"{identity}_evaluation.npz").exists() and not (folder / f"{identity}.json").exists():
                raise ValueError(f"Orphan raw trajectory for {identity}; preserve it for investigation")


def _worker_init() -> None:
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    try:
        from threadpoolctl import threadpool_limits
        # Keep the limiter alive for the worker lifetime (NumPy imports can
        # happen before a ProcessPool initializer on Windows).
        globals()["_thread_limiter"] = threadpool_limits(limits=1)
    except ImportError:
        pass


def _reference_worker(case: dict, output_dir: str, freeze_sha: str) -> dict:
    _worker_init()
    runner = _load_runner()
    native_dependencies = runner._native_dependencies
    dependencies = native_dependencies(VENDOR_PATH)
    expected_package = (VENDOR_PATH / "src" / "aerbench").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "aerbench" or name.startswith("aerbench."):
            source = getattr(module, "__file__", None)
            if source is not None and not Path(source).resolve().is_relative_to(expected_package):
                raise ValueError(f"Native reference helper imported from a different project: {name}={source}")
    initialization_records = []
    owned_constructor = _owned_native_factory(dependencies[1](), initialization_records)

    def owned_dependencies(base):
        if Path(base).resolve() != VENDOR_PATH.resolve():
            raise ValueError("Native construction attempted to use a different vendor project")
        return (dependencies[0], lambda: owned_constructor, *dependencies[2:])

    original_save = runner.json_save

    def save_with_initialization(path, value, *, replace=False):
        if value.get("complete") is True:
            if len(initialization_records) != 1:
                raise ValueError("Each reference must construct exactly one isolated native instance")
            value["native_initial_state_isolation"] = INITIAL_ISOLATION
            value["native_initial_arrays"] = initialization_records[0]
        elif initialization_records:
            value["native_initial_state_isolation"] = INITIAL_ISOLATION
            value["native_initial_arrays"] = initialization_records[0]
        return original_save(path, value, replace=replace)

    # The copied runner remains byte-identical to the historical source.  Its
    # construction dependency is adapted only for ndarray ownership, and its
    # immutable JSON record receives the checked initial-array provenance.
    runner._native_dependencies = owned_dependencies
    runner.json_save = save_with_initialization
    try:
        return runner.run_reference_case(case, dt_min=float(case.get("timestep_min", DT_MIN)), output_dir=output_dir,
                                         freeze_sha256=freeze_sha, nh4_limit=NH4_LIMIT,
                                         base=VENDOR_PATH)
    finally:
        runner._native_dependencies = native_dependencies
        runner.json_save = original_save


def _summary_row(case: dict, record: dict | None, status: str) -> dict:
    row = {"candidate_id": case["candidate_id"], "scenario": case["scenario"],
           "L": case["load_coordinate"], "alpha": case["oxygen_transfer_factor"],
           "kla": case["kla"], "effective_kla": case["effective_kla"], "status": status}
    if record is not None:
        keys = ("SNH_max_eval", "SNH_mean_eval", "SNH_zero_violation", "SNH_violation_ratio_gt_limit",
                "kernel_seconds", "elapsed_seconds", "total_with_output_seconds", "evaluation_samples",
                "case_sha256", "freeze_sha256", "input_sha256", "raw_file_sha256", "error_type", "error_message")
        row.update({key: record[key] for key in keys if key in record})
    return row


def run_frozen_references(study_dir: Path, workers: int = 4) -> pd.DataFrame:
    """Execute new references, resuming only independently verified successes.

    Failed/interrupted attempts are preserved, not silently retried.  An audit
    must pass before any result is used as complete scientific validation.
    """
    return _run_references(study_dir, workers)


def _run_references(study_dir: Path, workers: int, *, candidate_filename: str = "new_candidates.csv",
                    dt_min: float = DT_MIN, output_subdir: str = "",
                    summary_filename: str = "reference_summary.csv") -> pd.DataFrame:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer from 1 to 4")
    freeze_study = Path(study_dir).resolve()
    study = freeze_study / output_subdir
    _, freeze_sha, cases = verify_freeze(freeze_study, candidate_filename=candidate_filename, dt_min=dt_min)
    provenance = _provenance(study, freeze_sha, create=True, dt_min=dt_min)
    _check_inventory(study, cases)
    started, clock_start = _now(), time.perf_counter()
    records, statuses, pending = {}, {}, []
    for case in cases:
        identity = case["candidate_id"]
        if (study / "reference" / f"{identity}.json").exists():
            records[identity] = _verify_success(study, case, freeze_sha, provenance)
            statuses[identity] = "success"
        elif (study / "reference" / f"{identity}_failure.json").exists():
            records[identity] = _verify_failure(study, case, freeze_sha)
            statuses[identity] = "failed"
        else:
            pending.append(case)
    resumed = len(cases) - len(pending)
    execution_path = study / "reference_execution.json"
    previous_sessions = []
    if execution_path.exists():
        previous = _read_json(execution_path)
        if previous.get("freeze_sha256") != freeze_sha:
            raise ValueError("Reference execution history belongs to a different freeze")
        previous_sessions = previous.get("sessions", [])
    exceptions = []
    # Environment is set before spawning, so child imports inherit one thread.
    # Each native BSM1 instance retains its own full integration histories.
    previous_env = {key: os.environ.get(key) for key in THREAD_KEYS}
    try:
        for key in THREAD_KEYS:
            os.environ[key] = "1"
        if workers == 1:
            for case in pending:
                try:
                    _reference_worker(case, str(study / "reference"), freeze_sha)
                except Exception as exc:
                    exceptions.append({"candidate_id": case["candidate_id"], "type": type(exc).__name__, "message": str(exc)})
        elif pending:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
                futures = {pool.submit(_reference_worker, case, str(study / "reference"), freeze_sha): case for case in pending}
                for future in as_completed(futures):
                    case = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        exceptions.append({"candidate_id": case["candidate_id"], "type": type(exc).__name__, "message": str(exc)})
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    # Detect any model/design changes made while the expensive integrations ran.
    _, final_freeze_sha, _ = verify_freeze(freeze_study, candidate_filename=candidate_filename, dt_min=dt_min)
    if final_freeze_sha != freeze_sha:
        raise ValueError("freeze.json changed while reference integrations were running")
    _provenance(study, freeze_sha, create=False, dt_min=dt_min)
    for case in pending:
        identity = case["candidate_id"]
        if (study / "reference" / f"{identity}.json").exists():
            records[identity] = _verify_success(study, case, freeze_sha, provenance)
            statuses[identity] = "success"
        elif (study / "reference" / f"{identity}_failure.json").exists():
            records[identity] = _verify_failure(study, case, freeze_sha)
            statuses[identity] = "failed"
        else:
            statuses[identity] = "missing"
    frame = pd.DataFrame([_summary_row(case, records.get(case["candidate_id"]), statuses[case["candidate_id"]]) for case in cases])
    frame.to_csv(study / summary_filename, index=False, encoding="utf-8-sig")
    session = {"started_at_utc": started, "finished_at_utc": _now(), "workers": workers,
               "wall_seconds": time.perf_counter() - clock_start, "new_attempts": len(pending),
               "resumed_records": resumed, "exceptions": exceptions}
    success = [record for identity, record in records.items() if statuses[identity] == "success"]
    execution = {"freeze_sha256": freeze_sha, "sessions": previous_sessions + [session],
                 "expected_count": len(cases), "successful_count": len(success),
                 "failed_count": int((frame.status == "failed").sum()),
                 "missing_count": int((frame.status == "missing").sum()),
                 "successful_kernel_work_seconds": sum(record["kernel_seconds"] for record in success),
                 "successful_total_work_seconds": sum(record["total_with_output_seconds"] for record in success),
                 "reported_failure_work_seconds": sum(record.get("elapsed_seconds", 0.) for identity, record in records.items() if statuses[identity] == "failed"),
                 "timing_scope": "reference_generation_cost_parallel_wall_and_summed_work_no_serial_speedup_claim",
                 "worker_thread_limit": 1, "old_label_imports": False, "timestep_min": dt_min,
                 "candidate_filename": candidate_filename}
    _write_json(execution_path, execution, replace=True)
    return frame


def audit_reference_outputs(study_dir: Path) -> dict:
    """Recompute all raw metrics, labels, windows, and input hashes independently."""
    return _audit_references(study_dir)


def _audit_references(study_dir: Path, *, candidate_filename: str = "new_candidates.csv",
                       dt_min: float = DT_MIN, output_subdir: str = "",
                       summary_filename: str = "reference_summary.csv",
                       audit_filename: str = "reference_audit.json") -> dict:
    freeze_study = Path(study_dir).resolve()
    study = freeze_study / output_subdir
    _, freeze_sha, cases = verify_freeze(freeze_study, candidate_filename=candidate_filename, dt_min=dt_min)
    provenance = _provenance(study, freeze_sha, create=False, dt_min=dt_min)
    errors, successes, failures = [], {}, []
    try:
        _check_inventory(study, cases)
    except (ValueError, FileNotFoundError) as exc:
        errors.append(str(exc))
    for case in cases:
        identity = case["candidate_id"]
        try:
            if (study / "reference" / f"{identity}.json").exists():
                successes[identity] = _verify_success(study, case, freeze_sha, provenance)
            elif (study / "reference" / f"{identity}_failure.json").exists():
                _verify_failure(study, case, freeze_sha)
                failures.append(identity)
                errors.append(f"{identity}: reference execution failed")
            else:
                errors.append(f"{identity}: missing reference result")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append(f"{identity}: {exc}")
    summary_path = study / summary_filename
    if not summary_path.exists():
        errors.append(f"Missing {summary_filename}")
    else:
        try:
            summary = pd.read_csv(summary_path, float_precision="round_trip", dtype={"candidate_id": str})
            if "candidate_id" not in summary or summary.candidate_id.duplicated().any() or set(summary.candidate_id) != {case["candidate_id"] for case in cases}:
                raise ValueError(f"{summary_filename} coverage/IDs disagree with frozen candidates")
            summary = summary.set_index("candidate_id")
            for case in cases:
                identity = case["candidate_id"]
                if identity not in successes:
                    continue
                expected = _summary_row(case, successes[identity], "success")
                for key, value in expected.items():
                    if key != "candidate_id":
                        _assert_equal(summary.at[identity, key], value, f"summary.{identity}.{key}")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append(str(exc))
    result = {"passed": not errors and len(successes) == len(cases), "freeze_sha256": freeze_sha,
              "audited_at_utc": _now(), "expected_count": len(cases), "successful_count": len(successes),
              "failed_count": len(failures), "errors": errors, "old_label_imports": False,
              "timestep_min": dt_min, "expected_evaluation_samples_per_case": int(7 * 1440 / dt_min),
              "checks": ["frozen_model_prediction_route_design_protocol_hashes", "candidate_and_freeze_identity",
                         "native_source_hashes", "raw_trajectory_hashes", "full_candidate_coverage",
                         "raw_SNH_peak_mean_and_zero_violation", "strict_endpoint_reporting_60_le_t_lt_67",
                         "native_influent_reconstruction_and_hash", "all_saved_evaluation_metrics", "summary_consistency"]}
    _write_json(study / audit_filename, result, replace=True)
    return result


def _half_step_inputs(study_dir: Path) -> tuple[str, list[dict], dict[str, dict]]:
    study = Path(study_dir).resolve()
    _, freeze_sha, main_cases = verify_freeze(study)
    _, half_freeze_sha, half_cases = verify_freeze(study, candidate_filename="half_step_candidates.csv", dt_min=.5)
    if freeze_sha != half_freeze_sha:
        raise ValueError("Main and half-step files must belong to the same unchanged freeze")
    main_by_id = {case["candidate_id"]: case for case in main_cases}
    selected = {}
    main_provenance = _provenance(study, freeze_sha, create=False)
    for half in half_cases:
        identity = half["candidate_id"]
        if identity not in main_by_id:
            raise ValueError(f"Half-step candidate is absent from the main frozen design: {identity}")
        main = main_by_id[identity]
        for key in ("scenario", "load_coordinate", "oxygen_transfer_factor", "kla", "effective_kla"):
            _assert_equal(half[key], main[key], f"half_step.{identity}.{key}")
        selected[identity] = _verify_success(study, main, freeze_sha, main_provenance)
    return freeze_sha, half_cases, selected


def _half_step_comparisons(frame: pd.DataFrame, main_records: dict[str, dict], freeze_sha: str) -> tuple[pd.DataFrame, dict]:
    comparisons = []
    for row in frame.to_dict(orient="records"):
        main = main_records[row["candidate_id"]]
        row["main_timestep_min"] = 1.
        row["half_timestep_min"] = .5
        row["main_SNH_max_eval"] = main["SNH_max_eval"]
        row["main_SNH_zero_violation"] = main["SNH_zero_violation"]
        row["absolute_peak_difference"] = abs(float(row["SNH_max_eval"]) - main["SNH_max_eval"]) if row["status"] == "success" else None
        row["label_changed"] = bool(row["SNH_zero_violation"]) != main["SNH_zero_violation"] if row["status"] == "success" else None
        comparisons.append(row)
    result = pd.DataFrame(comparisons)
    valid = result[result.status == "success"]
    diagnostics = {"freeze_sha256": freeze_sha, "main_timestep_min": 1., "half_timestep_min": .5,
                   "expected_count": len(result), "compared_count": len(valid),
                   "maximum_absolute_peak_difference": float(valid.absolute_peak_difference.max()) if len(valid) else None,
                   "label_change_count": int(valid.label_changed.sum()) if len(valid) else 0,
                   "complete": bool(len(valid) == len(result)),
                   "result_role": "independent_half_step_numerical_check_main_labels_not_replaced",
                   "expected_evaluation_samples_per_case": 20160}
    return result, diagnostics


def run_half_step_references(study_dir: Path, workers: int = 2) -> pd.DataFrame:
    """Run new 0.5-min references for a pre-frozen subset of main candidate IDs.

    Existing main references provide only the numerical comparison.  Every
    half-step trajectory/label is integrated independently, with the same native
    default values and owned initial arrays, in ``numerical/``.
    """
    study = Path(study_dir).resolve()
    freeze_sha, _, main_records = _half_step_inputs(study)
    frame = _run_references(study, workers, candidate_filename="half_step_candidates.csv", dt_min=.5,
                            output_subdir="numerical", summary_filename="half_step_summary.csv")
    frame, diagnostics = _half_step_comparisons(frame, main_records, freeze_sha)
    frame.to_csv(study / "numerical" / "half_step_summary.csv", index=False, encoding="utf-8-sig")
    _write_json(study / "numerical" / "half_step_diagnostics.json", diagnostics, replace=True)
    return frame


def audit_half_step_outputs(study_dir: Path) -> dict:
    """Independently audit raw 20160-sample half-step outputs and comparisons."""
    study = Path(study_dir).resolve()
    result = _audit_references(study, candidate_filename="half_step_candidates.csv", dt_min=.5,
                               output_subdir="numerical", summary_filename="half_step_summary.csv",
                               audit_filename="half_step_audit.json")
    if result["passed"]:
        try:
            freeze_sha, cases, main_records = _half_step_inputs(study)
            provenance = _provenance(study / "numerical", freeze_sha, create=False, dt_min=.5)
            frame = pd.DataFrame([_summary_row(case, _verify_success(study / "numerical", case, freeze_sha, provenance), "success") for case in cases])
            expected_frame, expected_diagnostics = _half_step_comparisons(frame, main_records, freeze_sha)
            actual_frame = pd.read_csv(study / "numerical" / "half_step_summary.csv", float_precision="round_trip", dtype={"candidate_id": str}).set_index("candidate_id")
            comparison_keys = ("main_timestep_min", "half_timestep_min", "main_SNH_max_eval",
                               "main_SNH_zero_violation", "absolute_peak_difference", "label_changed")
            for row in expected_frame.to_dict(orient="records"):
                for key in comparison_keys:
                    _assert_equal(actual_frame.at[row["candidate_id"], key], row[key], f"half_comparison.{row['candidate_id']}.{key}")
            diagnostics = _read_json(study / "numerical" / "half_step_diagnostics.json")
            _assert_equal(diagnostics, expected_diagnostics, "half_step_diagnostics")
            result.update({key: expected_diagnostics[key] for key in ("maximum_absolute_peak_difference", "label_change_count")})
            result["checks"].extend(["frozen_half_ids_match_main_inputs", "fresh_half_step_raw_labels",
                                      "main_half_peak_and_label_differences", "half_step_diagnostic_summary"])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            result["errors"].append(str(exc))
            result["passed"] = False
    _write_json(study / "numerical" / "half_step_audit.json", result, replace=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--audit", action="store_true")
    actions.add_argument("--run-half-step", action="store_true")
    actions.add_argument("--audit-half-step", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.run or args.run_half_step:
        runner = run_half_step_references if args.run_half_step else run_frozen_references
        frame = runner(args.study_dir, args.workers)
        print(frame.to_string(index=False))
        return 0 if (frame.status == "success").all() else 1
    auditor = audit_half_step_outputs if args.audit_half_step else audit_reference_outputs
    result = auditor(args.study_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
