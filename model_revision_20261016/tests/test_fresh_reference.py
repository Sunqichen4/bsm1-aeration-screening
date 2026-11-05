"""Provenance/aggregation tests: no native BSM1 integration is executed."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fresh_reference as fr


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def study(tmp_path, monkeypatch):
    pd.DataFrame([{"candidate_id": "unseen_001", "scenario": "new_condition",
                   "L": 0.31, "alpha": 0.91, "kla": 140.}]).to_csv(tmp_path / "new_candidates.csv", index=False)
    (tmp_path / "design.json").write_text('{"seed": 17}', encoding="utf-8")
    (tmp_path / "protocol.json").write_text('{"frozen": true}', encoding="utf-8")
    entries = [{"path": name, "sha256": sha(tmp_path / name)}
               for name in ("new_candidates.csv", "design.json", "protocol.json")]
    freeze = {"files": entries, "source_hashes": {role: next(entry for entry in entries if entry["path"] == role + ".json")
                                                   for role in ("design", "protocol")}}
    write_json(tmp_path / "freeze.json", freeze)
    monkeypatch.setattr(fr, "reference_source_hashes", lambda: {"fixture_native_source": "unchanged"})
    data = np.zeros((2, 22))
    data[1, 0] = 200.1
    monkeypatch.setattr(fr, "_expected_input", lambda case: data.copy())
    return tmp_path


def fake_success(case, output_dir, freeze_sha):
    """Tiny compressed raw fixture with the real fixed 10080-point window."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir.parent / "reference_trajectories"
    raw_dir.mkdir(exist_ok=True)
    runner = fr._load_runner()
    dt_min = float(case["timestep_min"])
    n = int(7 * 1440 / dt_min)
    ends = 60. + np.arange(n) * dt_min / 1440.
    starts = ends - dt_min / 1440.
    eff = np.ones((n, 21))
    eff[:, runner.SNH] = np.linspace(2., 3.5, n)
    tank5, performance = np.ones_like(eff), np.ones((n, 3))
    input_data = fr._expected_input(case)
    raw = raw_dir / (case["candidate_id"] + "_evaluation.npz")
    np.savez_compressed(raw, interval_start_day=starts, time_day=ends,
                        effluent_states=eff, tank5_states=tank5, performance=performance,
                        input_data=input_data, state_columns=np.asarray(runner.STATE_COLUMNS),
                        nh4_exceeds_limit=eff[:, runner.SNH] > 4.)
    result = runner.summarize_evaluation(eff, tank5, performance, dt_min=dt_min, nh4_limit=4.)
    result.update(case)
    defaults = fr._native_default_array_manifest()
    result.update({"case_sha256": runner.object_sha(case), "freeze_sha256": freeze_sha,
                   "complete": True, "fresh_reference": True,
                   "reference_result_origin": fr.ORIGIN,
                   "native_initial_state_isolation": fr.INITIAL_ISOLATION,
                   "native_initial_arrays": {"array_count": len(defaults), "before": defaults,
                                             "after": defaults, "values_preserved": True,
                                             "shares_memory_after_copy": False},
                   "reference_runner_source_sha256": sha(fr.RUNNER_PATH),
                   "reference_started_utc": datetime.now(timezone.utc).isoformat(),
                   "native_initialization": "original_native_defaults_no_explicit_temperature_or_state_changes",
                   "evaluation_alignment": "output_endpoint_60_le_t_lt_67_matching_frozen_run_revision",
                   "raw_file": str(raw.resolve()), "raw_file_sha256": sha(raw),
                   "input_shape": list(input_data.shape), "input_sha256": runner.array_sha(input_data),
                   "first_eval_interval_start_day": float(starts[0]),
                   "first_eval_output_end_day": float(ends[0]), "last_eval_output_end_day": float(ends[-1]),
                   "kernel_seconds": 0.01, "elapsed_seconds": 0.02,
                   "output_write_seconds": 0.001, "total_with_output_seconds": 0.021})
    write_json(output_dir / (case["candidate_id"] + ".json"), result)
    return result


@pytest.fixture
def completed(study, monkeypatch):
    monkeypatch.setattr(fr, "_reference_worker", fake_success)
    frame = fr.run_frozen_references(study, workers=1)
    assert frame.status.tolist() == ["success"]
    return study


def record_path(study):
    return study / "reference" / "unseen_001.json"


def rewrite_raw(study, updater):
    record = json.loads(record_path(study).read_text(encoding="utf-8"))
    raw = Path(record["raw_file"])
    with np.load(raw, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    updater(arrays)
    np.savez_compressed(raw, **arrays)
    record["raw_file_sha256"] = sha(raw)
    write_json(record_path(study), record)


def test_frozen_candidate_tampering_rejected_before_simulation(study, monkeypatch):
    (study / "new_candidates.csv").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(fr, "_reference_worker", lambda *args: pytest.fail("Must not simulate tampered inputs"))
    with pytest.raises(ValueError, match="Frozen file changed"):
        fr.run_frozen_references(study, 1)
    assert not (study / "reference_provenance.json").exists()


def test_separate_protocol_source_digest_is_checked(study):
    freeze_path = study / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["source_hashes"]["protocol"]["sha256"] = "0" * 64
    write_json(freeze_path, freeze)
    with pytest.raises(ValueError, match="Frozen file changed|protocol source"):
        fr.verify_freeze(study)


def test_freeze_changed_during_integration_is_rejected(study, monkeypatch):
    def mutate_manifest(case, output_dir, freeze_sha):
        result = fake_success(case, output_dir, freeze_sha)
        path = study / "freeze.json"
        freeze = json.loads(path.read_text(encoding="utf-8"))
        freeze["post_start_change"] = True
        write_json(path, freeze)
        return result
    monkeypatch.setattr(fr, "_reference_worker", mutate_manifest)
    with pytest.raises(ValueError, match="freeze.json changed while"):
        fr.run_frozen_references(study, 1)
    assert not (study / "reference_summary.csv").exists()


def test_success_is_audited_and_resume_never_reruns(completed, monkeypatch):
    audit = fr.audit_reference_outputs(completed)
    assert audit["passed"], audit
    before = sha(record_path(completed))
    monkeypatch.setattr(fr, "_reference_worker", lambda *args: pytest.fail("Verified checkpoint must be reused"))
    resumed = fr.run_frozen_references(completed, 1)
    assert resumed.status.tolist() == ["success"]
    assert sha(record_path(completed)) == before
    execution = json.loads((completed / "reference_execution.json").read_text(encoding="utf-8"))
    assert execution["sessions"][-1]["new_attempts"] == 0
    assert execution["sessions"][-1]["resumed_records"] == 1
    assert execution["successful_kernel_work_seconds"] == pytest.approx(0.01)


def test_checkpoint_from_different_freeze_is_rejected(completed, monkeypatch):
    path = record_path(completed)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["freeze_sha256"] = "a" * 64
    write_json(path, record)
    monkeypatch.setattr(fr, "_reference_worker", lambda *args: pytest.fail("Foreign checkpoint must not be overwritten"))
    with pytest.raises(ValueError, match="freeze_sha256"):
        fr.run_frozen_references(completed, 1)


def test_saved_peak_error_detected_from_raw(completed):
    path = record_path(completed)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["SNH_max_eval"] = 2.9
    write_json(path, record)
    audit = fr.audit_reference_outputs(completed)
    assert not audit["passed"]
    assert any("SNH_max_eval" in error for error in audit["errors"])


def test_raw_peak_and_label_change_detected_even_if_raw_hash_updated(completed):
    def change(arrays):
        arrays["effluent_states"][3, 9] = 5.
        arrays["nh4_exceeds_limit"][3] = True
    rewrite_raw(completed, change)
    audit = fr.audit_reference_outputs(completed)
    assert not audit["passed"]
    assert any("SNH_" in error for error in audit["errors"])


def test_reporting_window_error_detected(completed):
    rewrite_raw(completed, lambda arrays: arrays["time_day"].__setitem__(0, 59.99))
    audit = fr.audit_reference_outputs(completed)
    assert not audit["passed"]
    assert any("reporting window" in error for error in audit["errors"])


def test_input_reconstruction_catches_forged_input_hash(completed):
    rewrite_raw(completed, lambda arrays: arrays["input_data"].__setitem__((0, 10), 123.))
    path = record_path(completed)
    record = json.loads(path.read_text(encoding="utf-8"))
    with np.load(record["raw_file"], allow_pickle=False) as archive:
        record["input_sha256"] = fr._load_runner().array_sha(archive["input_data"])
    write_json(path, record)
    audit = fr.audit_reference_outputs(completed)
    assert not audit["passed"]
    assert any("influent differs" in error for error in audit["errors"])


def test_summary_label_tampering_is_detected(completed):
    path = completed / "reference_summary.csv"
    frame = pd.read_csv(path)
    frame["SNH_zero_violation"] = False
    frame.to_csv(path, index=False)
    audit = fr.audit_reference_outputs(completed)
    assert not audit["passed"]
    assert any("summary.unseen_001.SNH_zero_violation" in error for error in audit["errors"])


def test_changed_native_sources_reject_checkpoint_reuse(completed, monkeypatch):
    monkeypatch.setattr(fr, "reference_source_hashes", lambda: {"fixture_native_source": "changed"})
    with pytest.raises(ValueError, match="implementation changed"):
        fr.run_frozen_references(completed, 1)


def test_historical_reference_labels_forbidden_in_candidate_file(study):
    path = study / "new_candidates.csv"
    frame = pd.read_csv(path)
    frame["SNH_max_eval"] = 3.
    frame.to_csv(path, index=False)
    freeze_path = study / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    next(entry for entry in freeze["files"] if entry["path"] == path.name)["sha256"] = sha(path)
    write_json(freeze_path, freeze)
    with pytest.raises(ValueError, match="reference labels"):
        fr.verify_freeze(study)


def test_failed_run_is_preserved_and_excluded_from_complete_validation(study, monkeypatch):
    calls = []
    def fail(case, output_dir, freeze_sha):
        calls.append(case["candidate_id"])
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        record = {"candidate_id": case["candidate_id"], "case_sha256": fr._load_runner().object_sha(case),
                  "freeze_sha256": freeze_sha, "complete": False,
                  "reference_runner_source_sha256": sha(fr.RUNNER_PATH), "elapsed_seconds": 0.03,
                  "error_type": "RuntimeError", "error_message": "fixture solver failure"}
        write_json(output_dir / (case["candidate_id"] + "_failure.json"), record)
        raise RuntimeError("fixture solver failure")
    monkeypatch.setattr(fr, "_reference_worker", fail)
    assert fr.run_frozen_references(study, 1).status.tolist() == ["failed"]
    assert fr.run_frozen_references(study, 1).status.tolist() == ["failed"]
    assert calls == ["unseen_001"]
    assert not fr.audit_reference_outputs(study)["passed"]


def test_packaged_reference_sources_are_local_and_portable():
    assert fr.RUNNER_PATH == Path(fr.__file__).resolve().parent / "legacy" / "reference_runner.py"
    assert fr.VENDOR_PATH.is_relative_to(Path(fr.__file__).resolve().parent / "legacy")
    hashes = fr.reference_source_hashes()
    assert "package/legacy/reference_runner.py" in hashes
    assert "package/legacy/vendor/original_project/src/aerbench/bsm1_validation.py" in hashes
    assert "package/legacy/third_party/LICENSE" in hashes
    assert all(key.startswith(("package/", "dependency/")) for key in hashes)


def test_native_defaults_are_independent_between_two_real_constructions_without_ode():
    """Construct only; no step(), output(), or ODE function is called."""
    native = pytest.importorskip("bsm2_python.bsm1_ol")
    input_data = np.vstack([np.r_[0., np.ones(21)], np.r_[200.1, np.ones(21)]])
    input_data[:, 1 + 15] = 15.
    input_data[:, 1 + 14] = 18446.
    audits = []
    factory = fr._owned_native_factory(native.BSM1OL, audits)
    kwargs = {"data_in": input_data, "timestep": 1 / 1440., "endtime": 0.01,
              "evaltime": np.array([0., 0.005]), "tempmodel": False, "activate": False}
    first = factory(**kwargs)
    expected = fr._native_default_array_manifest()
    # Deliberate mutations stand in for any implementation that updates its
    # initial arrays in-place during integration.
    for i in range(1, 6):
        getattr(first, f"reactor{i}").y0[:] = -123.
        getattr(first, f"reactor{i}").asm1par[:] = -456.
    first.settler.ys0[:] = -789.
    first.settler.asm1par[:] = -432.
    first.settler.dim[:] = -1.
    first.performance.pp_par[:] = -999.
    second = factory(**kwargs)
    assert len(audits) == 2
    assert audits[0]["before"] == audits[1]["before"] == expected
    assert np.isfinite(second.reactor1.y0).all()
    assert not np.shares_memory(first.reactor1.y0, second.reactor1.y0)
    assert not np.shares_memory(first.settler.ys0, second.settler.ys0)
    assert not np.shares_memory(first.performance.pp_par, second.performance.pp_par)


def test_native_initialization_provenance_tampering_detected(completed):
    path = record_path(completed)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["native_initial_arrays"]["shares_memory_after_copy"] = True
    write_json(path, record)
    audit = fr.audit_reference_outputs(completed)
    assert not audit["passed"]
    assert any("native_initial_arrays" in error for error in audit["errors"])


def test_completed_study_remains_auditable_after_relocation(completed):
    relocated = completed.parent / (completed.name + "_relocated")
    shutil.copytree(completed, relocated)
    audit = fr.audit_reference_outputs(relocated)
    assert audit["passed"], audit


def test_worker_injects_ownership_and_records_it_without_running_ode(tmp_path, monkeypatch):
    native = pytest.importorskip("bsm2_python.bsm1_ol")
    runner, saves = SimpleNamespace(), []
    def dependencies(base):
        return (object, lambda: native.BSM1OL, object, object, object)
    def save(path, record, *, replace=False):
        saves.append(dict(record))
    def no_ode_reference(case, **kwargs):
        constructor = runner._native_dependencies(kwargs["base"])[1]()
        data = np.vstack([np.r_[0., np.ones(21)], np.r_[200.1, np.ones(21)]])
        constructor(data_in=data, timestep=1 / 1440., endtime=0.01,
                    evaltime=np.array([0., 0.005]), tempmodel=False, activate=False)
        result = {"complete": True}
        runner.json_save(tmp_path / "mock.json", result)
        return result
    runner._native_dependencies = dependencies
    runner.json_save = save
    runner.run_reference_case = no_ode_reference
    monkeypatch.setattr(fr, "_load_runner", lambda: runner)
    monkeypatch.setattr(fr, "_worker_init", lambda: None)
    result = fr._reference_worker({"candidate_id": "mock"}, str(tmp_path), "f" * 64)
    assert result["native_initial_state_isolation"] == fr.INITIAL_ISOLATION
    assert result["native_initial_arrays"]["array_count"] == 16
    assert saves[0] == result
    assert runner._native_dependencies is dependencies
    assert runner.json_save is save


@pytest.fixture
def completed_with_half_selection(study, monkeypatch):
    name = "half_step_candidates.csv"
    shutil.copy2(study / "new_candidates.csv", study / name)
    freeze_path = study / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["files"].append({"path": name, "sha256": sha(study / name)})
    write_json(freeze_path, freeze)
    monkeypatch.setattr(fr, "_reference_worker", fake_success)
    fr.run_frozen_references(study, 1)
    return study


def test_half_step_uses_independent_raw_20160_sample_window_and_same_freeze(completed_with_half_selection, monkeypatch):
    study = completed_with_half_selection
    main_hash = sha(record_path(study))
    frame = fr.run_half_step_references(study, 1)
    assert frame.evaluation_samples.tolist() == [20160]
    assert frame.absolute_peak_difference.tolist() == [0.]
    assert frame.label_changed.tolist() == [False]
    assert frame.freeze_sha256.tolist() == [sha(study / "freeze.json")]
    with np.load(study / "numerical" / "reference_trajectories" / "unseen_001_evaluation.npz") as archive:
        assert archive["time_day"].shape == (20160,)
        assert np.diff(archive["time_day"]).mean() == pytest.approx(.5 / 1440.)
    assert sha(record_path(study)) == main_hash
    audit = fr.audit_half_step_outputs(study)
    assert audit["passed"], audit
    assert audit["expected_evaluation_samples_per_case"] == 20160
    assert fr.audit_reference_outputs(study)["passed"]
    monkeypatch.setattr(fr, "_reference_worker", lambda *args: pytest.fail("Half-step successful checkpoint must resume"))
    resumed = fr.run_half_step_references(study, 1)
    assert resumed.status.tolist() == ["success"]


def test_half_step_candidates_must_be_frozen(completed, monkeypatch):
    shutil.copy2(completed / "new_candidates.csv", completed / "half_step_candidates.csv")
    monkeypatch.setattr(fr, "_reference_worker", lambda *args: pytest.fail("Unfrozen half candidates cannot run"))
    with pytest.raises(ValueError, match="half_step_candidates.csv must be included"):
        fr.run_half_step_references(completed, 1)


def test_half_step_comparison_tampering_is_detected(completed_with_half_selection):
    study = completed_with_half_selection
    fr.run_half_step_references(study, 1)
    path = study / "numerical" / "half_step_summary.csv"
    frame = pd.read_csv(path)
    frame["absolute_peak_difference"] = 99.
    frame.to_csv(path, index=False)
    audit = fr.audit_half_step_outputs(study)
    assert not audit["passed"]
    assert any("absolute_peak_difference" in error for error in audit["errors"])


def test_half_step_main_raw_cannot_be_reused_as_half_output(completed_with_half_selection):
    study = completed_with_half_selection
    fr.run_half_step_references(study, 1)
    raw = study / "numerical" / "reference_trajectories" / "unseen_001_evaluation.npz"
    shutil.copy2(study / "reference_trajectories" / raw.name, raw)
    path = study / "numerical" / "reference" / "unseen_001.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["raw_file_sha256"] = sha(raw)
    write_json(path, record)
    audit = fr.audit_half_step_outputs(study)
    assert not audit["passed"]
    assert any("evaluation sample count" in error for error in audit["errors"])
