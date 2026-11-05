"""Resumable exhaustive audit, with predictions frozen before test simulation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from .bsm1_validation import run_bsm1_fixed_kla
from .fixed_aeration import run_surrogate_fixed_aeration
from .heldout_validation import (
    METHODS, calibrate_buffers, candidate_grid, evaluate, fit_logistic,
    logistic_predictions, parse_bool, screening_decisions,
)
from .provenance import environment_snapshot, sha256_file, stable_record_hash
from .steady_conditions import SteadyOperatingPoint


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records(frame: pd.DataFrame) -> list[dict]:
    # Preserve nulls in optional metadata; never serialize non-standard NaN.
    return json.loads(frame.to_json(orient="records", double_precision=15))


@contextmanager
def exclusive_run_lock(output: Path):
    """OS releases the lock on crash; the harmless lock file may remain."""
    handle = (output / ".run.lock").open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("This output directory is already in use by another run.") from None
    try:
        yield
    finally:
        handle.close()


def verify_hashes(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen source or output changed: {name}. Do not mix runs.")


def save_checked(path: Path, payload: dict, context: str) -> None:
    atomic_json(path, {"context_sha256": context, "payload_sha256": stable_record_hash(payload), "payload": payload})


def load_checked(path: Path, context: str) -> dict:
    data = read_json(path)
    if data["context_sha256"] != context or stable_record_hash(data["payload"]) != data["payload_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {path}.")
    return data["payload"]


def check_simulation(row: dict, case: dict, p: dict, kind: str) -> None:
    for name in ("SNH_mean_eval", "SNH_max_eval", "SNH_violation_hours_gt_limit", "elapsed_seconds"):
        if not np.isfinite(row[name]) or row[name] < 0:
            raise ValueError(f"Invalid {kind} result: {name}.")
    if row["elapsed_seconds"] <= 0:
        raise ValueError("Measured elapsed time must be positive.")
    if row["scenario"] != case["scenario"] or row["candidate_id"] != case["candidate_id"]:
        raise ValueError("Cached simulation has the wrong candidate identity.")
    for name, expected in (("kla", case["kla"]), ("effective_kla", case["effective_kla"]),
                           ("warmup_days", p["warmup_days"]), ("eval_days", p["eval_days"]),
                           ("nh4_limit", p["nh4_limit"]),
                           ("dt_minutes" if kind == "surrogate" else "timestep_min",
                            p["surrogate_timestep_min" if kind == "surrogate" else "bsm1_timestep_min"])):
        if not np.isclose(row[name], expected, rtol=0, atol=1e-9):
            raise ValueError(f"Cached simulation metadata mismatch: {name}.")
    flag = bool(parse_bool([row["SNH_zero_violation"]])[0])
    if flag != (row["SNH_max_eval"] <= p["nh4_limit"]) or flag != (row["SNH_violation_hours_gt_limit"] == 0):
        raise ValueError("Simulation concentrations, duration and compliance flag disagree.")


def cached_simulation(case: dict, p: dict, kind: str, folder: Path, context: str, runner) -> dict:
    path = folder / f"{case['candidate_id']}.json"
    if path.exists():
        row = load_checked(path, context)
    else:
        point = SteadyOperatingPoint(case["scenario"], case["load_coordinate"], case["oxygen_transfer_factor"])
        common = {"kla": case["kla"], "warmup_days": p["warmup_days"], "eval_days": p["eval_days"],
                  "nh4_limit": p["nh4_limit"], "oxygen_transfer_factor": point.oxygen_transfer_factor,
                  "operating_point": point}
        started = time.perf_counter()
        if kind == "surrogate":
            row = runner(scenario_name=point.name, dt_minutes=p["surrogate_timestep_min"], **common)
        else:
            row = runner(scenario=point.name, timestep_min=p["bsm1_timestep_min"],
                         tempmodel_policy="off", inflow_mode="steady_const", **common)
        elapsed = time.perf_counter() - started
        row = dict(row, candidate_id=case["candidate_id"], elapsed_seconds=elapsed)
        check_simulation(row, case, p, kind)
        # Only optional, non-primary metadata may contain missing values.
        row = records(pd.DataFrame([row]))[0]
        save_checked(path, row, context)
    check_simulation(row, case, p, kind)
    return row


def environment_key(root: Path) -> dict:
    env = environment_snapshot(root)
    env["packages"]["numba"] = version("numba")
    return {k: env[k] for k in ("python_version", "python_executable", "packages", "thread_settings", "operating_system")}


def prepare_predictions(root: Path, output: Path, p: dict, dev: pd.DataFrame, grid: pd.DataFrame,
                        context: str, source_hashes: dict, surrogate_runner) -> dict:
    # No call or access to the NEW reference runner occurs in this function.
    all_surrogate = []
    for phase, frame in (("development", dev), ("new_predictions", grid)):
        rows = []
        for index, case in enumerate(records(frame), 1):
            if index == 1 or index % 25 == 0:
                verify_hashes(source_hashes)
                print(f"{phase}: surrogate {index}/{len(frame)}", flush=True)
            rows.append(cached_simulation(case, p, "surrogate", output / "cases" / phase,
                                          context, surrogate_runner))
        all_surrogate.append(pd.DataFrame(rows))
    development_surrogate, new_surrogate = all_surrogate
    fit_started = time.perf_counter()
    model = fit_logistic(dev, p["logistic_l2_penalty"])
    logistic_fit_seconds = time.perf_counter() - fit_started
    dev_predictions = {
        "frozen_surrogate": parse_bool(development_surrogate.SNH_zero_violation),
        "condition_logistic": logistic_predictions(dev, model) >= .5,
    }
    fit_started = time.perf_counter()
    buffers = {m: calibrate_buffers(dev, dev_predictions[m], p) for m in METHODS}
    buffer_fit_seconds = time.perf_counter() - fit_started
    started = time.perf_counter()
    probabilities = logistic_predictions(grid, model)
    logistic_prediction_seconds = time.perf_counter() - started
    predictions = {"frozen_surrogate": parse_bool(new_surrogate.SNH_zero_violation),
                   "condition_logistic": probabilities >= .5}
    decisions, costs = [], {}
    for method in METHODS:
        started = time.perf_counter()
        decision = screening_decisions(grid, predictions[method], buffers[method], p)
        policy_seconds = time.perf_counter() - started
        decision["method"] = method
        if method == "condition_logistic":
            decision["model_probability"] = probabilities
        else:
            decision["surrogate_max_snh"] = new_surrogate.SNH_max_eval.to_numpy(float)
        costs[method] = {"prediction_seconds": float(new_surrogate.elapsed_seconds.sum()) if method == "frozen_surrogate" else logistic_prediction_seconds,
                         "decision_seconds": policy_seconds}
        decisions.extend(records(decision))
    verify_hashes(source_hashes)
    return {
        "frozen_at_utc": now(), "test_bsm1_calls_at_freeze": 0,
        "protocol": p, "grid": records(grid), "logistic_model": model, "buffers": buffers,
        "decisions": decisions, "operational_component_seconds": costs,
        "development_costs": {
            "old_bsm1_labels_used": len(dev),
            "old_bsm1_loop_seconds_not_current_wall_time": float(dev.runtime_seconds.sum()),
            "development_surrogate_seconds": float(development_surrogate.elapsed_seconds.sum()),
            "logistic_fit_seconds": logistic_fit_seconds, "buffer_calibration_seconds": buffer_fit_seconds,
            "surrogate_original_training": "25 baseline points; other previously examined results are development knowledge",
            "information_comparison": "same old 750 BSM1 cases available for new policy development; model training histories differ",
        },
        "development_predictions": records(dev[["candidate_id", "scenario", "kla", "effective_kla", "target"]].assign(
            surrogate_compliant=dev_predictions["frozen_surrogate"], logistic_compliant=dev_predictions["condition_logistic"])),
    }


INTERPRETATION = """# New-condition validation: interpretation

This is a follow-up BSM1 simulation audit with new steady load/transfer
combinations. It is not independent plant validation, formal preregistration,
or proof of dynamic-weather safety. New combinations are within the marginal
parameter ranges but need not lie inside the convex hull of development points.

The six-state parameters remain frozen. Both NEW decision policies use old
development data for error buffers; the logistic model fits all 750 old labels.
The original surrogate was optimized against 25 baseline points and has other
development history. These are common-development-access comparisons, not
identical-training-history comparisons. Neither new policy is the old paper's
post-audit gate. Fixed 20 effective-KLa guard is a design choice, not a bound
guaranteeing unseen-condition safety. A nonmonotone or unresolved threshold
forces reference evaluation of the whole corresponding scenario.

Load coordinate L means BSM1 SS and XS x (1 + 0.20L), SNH x (1 + 0.35L).
Other input fractions and flow stay fixed. The surrogate retains its original
260 COD/30 NH4 mapping and linearly extends the old high-load activity multiplier
mu_a x (1 + 0.05L); this multiplier does not change BSM1 kinetics. Effective
KLa = commanded KLa x transfer factor in both simulators. Temperature model is
off, matching the old non-low-temperature runs. Each candidate starts afresh.
Simulation windows remain 60-day warm-up and 7-day evaluation, surrogate 15 min
and BSM1 1 min. Compliance means no NH4 sample exceeds 4 mg N/L over evaluation.
This does not establish numerical convergence or total-nitrogen compliance.

All new predictions and BSM1-recheck selections were saved in a hashed frozen
bundle before the first new BSM1 simulation. Only selected reference calls can
override a policy decision. Direct-release errors and lost feasible candidates
remain visible. A bracketed threshold requires a noncompliant point followed
by at least three compliant points and no subsequent noncompliant point.
Threshold intervals and errors use the stated finite grid, not a continuous
threshold. Counts on deterministic grids are descriptive; no independent-trial
confidence interval or acceptance probability is inferred.

All 150 BSM1 cases are actually computed for exhaustive scientific auditing.
'Calls avoided' refers to hypothetical deployment of the frozen screening
workflow. Reconstructed workflow seconds = measured prescreen/policy seconds
+ summed measured whole-function BSM1 times for selected cases. This is a
case-time reconstruction, not a separately timed end-to-end workflow and not
time saved by running this validation. File I/O, provenance, environment imports,
the exhaustive audit and offline development costs are not operational costs.
Per-case times may include initialization/JIT costs; single-pass measurements
do not justify precision or timing significance claims. Original fitting costs
are not fully known. Additional development costs are reported separately.

Negative results are valid results: do not revise gates using these test labels
and continue calling the same cases held-out tests. Dynamic weather and low
temperature require BSM1; this experiment does not validate direct surrogate
screening there. Do not modify source/configuration/results during a run.
"""


def execute_study(*, root: Path, output: Path, protocol: dict, dev: pd.DataFrame,
                  source_files: list[Path], resume: bool = False,
                  surrogate_runner=run_surrogate_fixed_aeration,
                  bsm1_runner=run_bsm1_fixed_kla) -> None:
    root, output = root.resolve(), output.resolve()
    grid = candidate_grid(protocol)
    if set(grid.scenario) & set(dev.scenario):
        raise ValueError("Development and new scenario identifiers overlap.")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output is nonempty: {output}. Use --resume for the same run.")
    if resume and not (output / "run_plan.json").is_file():
        raise FileNotFoundError("No run_plan.json to resume; use a new output folder.")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(output):
        source_hashes = {str(f.resolve()): sha256_file(f) for f in source_files}
        signature = {"protocol": protocol, "sources": source_hashes, "environment": environment_key(root)}
        context = stable_record_hash(signature)
        if resume:
            plan = read_json(output / "run_plan.json")
            if plan["context_sha256"] != context:
                raise RuntimeError("Source, protocol, input data or environment changed since this run. Resume refused.")
            verify_hashes(plan["snapshot_sha256"])
        else:
            snapshots = {}
            for source in source_files:
                relative = source.resolve().relative_to(root)
                destination = output / "source_snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                snapshots[str(destination)] = sha256_file(destination)
            plan = {"created_at_utc": now(), "context_sha256": context, **signature,
                    "snapshot_sha256": snapshots, "expected_cases": len(grid), "grid": records(grid),
                    "environment_details": environment_snapshot(root)}
            atomic_json(output / "run_plan.json", plan)
        status_path = output / "run_status.json"
        if resume and status_path.exists():
            previous = read_json(status_path)
            if previous.get("status") == "complete":
                verify_hashes(previous["output_sha256"])
                print(f"Already complete; output hashes verified: {output}", flush=True)
                return
        status = {"status": "preparing", "completed_bsm1_cases": 0, "expected_bsm1_cases": len(grid),
                  "context_sha256": context, "updated_at_utc": now()}
        atomic_json(status_path, status)
        try:
            frozen_path = output / "frozen_predictions.json"
            if frozen_path.exists():
                frozen = load_checked(frozen_path, context)
            else:
                if (output / "cases" / "bsm1").exists():
                    raise RuntimeError("Test-reference checkpoints exist without frozen predictions; refusing refit.")
                frozen = prepare_predictions(root, output, protocol, dev, grid, context, source_hashes, surrogate_runner)
                save_checked(frozen_path, frozen, context)
            frozen_digest = sha256_file(frozen_path)
            print("Predictions and recheck decisions FROZEN. Starting exhaustive new BSM1 audit.", flush=True)
            status.update(status="running_bsm1", frozen_predictions_sha256=frozen_digest)
            atomic_json(status_path, status)
            references = []
            ordered = grid.iloc[np.random.default_rng(protocol["bsm1_order_seed"]).permutation(len(grid))]
            for index, case in enumerate(records(ordered), 1):
                verify_hashes({**source_hashes, str(frozen_path): frozen_digest})
                row = cached_simulation(case, protocol, "bsm1", output / "cases" / "bsm1", context, bsm1_runner)
                references.append(row)
                status.update(completed_bsm1_cases=index, updated_at_utc=now())
                atomic_json(status_path, status)
                hours = np.mean([r["elapsed_seconds"] for r in references]) * (len(grid) - index) / 3600
                print(f"BSM1 {index}/{len(grid)} | {case['candidate_id']} KLa={case['kla']:g} | "
                      f"NH4max={row['SNH_max_eval']:.4f} | remaining estimate {hours:.2f} h", flush=True)
            verify_hashes({**source_hashes, str(frozen_path): frozen_digest})
            ref = pd.DataFrame(references).sort_values("candidate_id")
            decision = pd.DataFrame(frozen["decisions"])
            final, summary, thresholds = evaluate(decision, ref, frozen["operational_component_seconds"],
                                                  protocol["minimum_compliant_tail_points"])
            tables = {"new_candidate_grid.csv": grid, "bsm1_new_conditions.csv": ref,
                      "frozen_screening_decisions.csv": decision, "development_predictions.csv": pd.DataFrame(frozen["development_predictions"]),
                      "candidate_audit.csv": final, "method_summary.csv": summary, "threshold_comparison.csv": thresholds}
            for name, table in tables.items():
                table.to_csv(output / name, index=False, encoding="utf-8-sig")
            atomic_json(output / "development_costs.json", frozen["development_costs"])
            (output / "INTERPRETATION.md").write_text(INTERPRETATION, encoding="utf-8")
            outputs = [output / name for name in tables] + [frozen_path, output / "development_costs.json", output / "INTERPRETATION.md"]
            outputs += sorted((output / "cases").rglob("*.json"))
            status.update(status="complete", updated_at_utc=now(),
                          output_sha256={str(f): sha256_file(f) for f in outputs})
            atomic_json(status_path, status)
            print(summary.loc[summary.scenario.eq("ALL"), ["method", "raw_false_safe", "final_false_safe",
                  "lost_feasible_candidates", "bsm1_rechecks", "reconstructed_reduction_percent"]].to_string(index=False), flush=True)
            print(f"COMPLETE: {output}", flush=True)
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=f"{type(error).__name__}: {error}", updated_at_utc=now())
            atomic_json(status_path, status)
            raise
