"""Freeze integrity and prediction-only candidate selection; no ODE execution."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REAL_ROOT))
import run_study as study


SCIENTIFIC_FILES = (
    "process_model.py", "statistical_models.py", "run_study.py", "design.json",
    "protocol.json", "data/development_750.csv", "legacy/controlled_models.py",
    "legacy/M2_full_parameters.json",
)
NEW_MODEL_NAMES = (*study.pm.MODELS, "poly_ridge", "rbf_ridge")
ALL_MODEL_NAMES = (*NEW_MODEL_NAMES, "legacy_M2_binary")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def isolated_study(tmp_path, monkeypatch):
    """Minimal source-bound study with deterministic input-only predictions."""
    for name in SCIENTIFIC_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable source fixture", encoding="utf-8")
    protocol = {
        "new_conditions": [{"load": .22, "q": .4}, {"load": .22, "q": .85},
                           {"load": .48, "q": .4}, {"load": .48, "q": .85},
                           {"load": .78, "q": .4}, {"load": .78, "q": .85}],
        "fixed_coarse_kla": [40, 100, 160, 220, 300, 400, 520],
    }
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "design.json", {"selection": "predictions only"})
    monkeypatch.setattr(study, "ROOT", tmp_path)
    out = tmp_path / "outputs" / "development"
    models = {name: {"kind": name, "threshold": 102. + 12. * i}
              for i, name in enumerate(NEW_MODEL_NAMES)}
    # A very different historical boundary must not affect candidate selection.
    models["legacy_M2_binary"] = {"kind": "legacy_M2_binary", "threshold": 202.}
    write_json(out / "models.json", models)
    write_json(out / "gates.json", {name: {"absolute_peak_buffer_mgN_L": .5}
                                     for name in ALL_MODEL_NAMES})
    write_json(out / "development_checks.json", {"pass": True})
    write_json(out / "run_started.json", {"scientific_source_hashes": study.scientific_sources()})
    for name in ("completed.json", "identifiability.json"):
        write_json(out / name, {"fixture": True})
    for name in ("development_summary.csv", "rho_profile.csv", "development_predictions.csv"):
        (out / name).write_text("fixture\n", encoding="utf-8")
    # Deliberately present labels that selection is forbidden from reading.
    write_json(tmp_path / "reference" / "poison.json", {"SNH_max_eval": -1e99})
    (tmp_path / "old_test_244.csv").write_text("SNH_max_eval\n-1e99\n", encoding="utf-8")
    calls = []

    def peak(frame, model):
        allowed = {"candidate_id", "scenario", "load_coordinate", "oxygen_transfer_factor", "kla",
                   "effective_kla", "grid_kind", "selection_reason"}
        assert set(frame.columns) <= allowed, "A reference/target column entered the freeze predictor"
        calls.append((model["kind"], len(frame)))
        return np.maximum(0., 4. + (model["threshold"] - frame.kla.to_numpy(float)) / 2.)

    def finite(frame, model):
        return frame[["candidate_id", "scenario"]].copy().assign(
            model=model["kind"], predicted_peak=peak(frame, model), critical_NH4_DO_clips=0,
            lower_clips_total=0, upper_clips_total=0)

    def legacy(frame, model, finite=False):
        if finite:
            return frame[["candidate_id", "scenario"]].copy().assign(
                model=model["kind"], predicted_peak=peak(frame, model),
                lower_clips_total=0, upper_clips_total=0)
        return peak(frame, model)

    monkeypatch.setattr(study.pm, "predict_steady", peak)
    monkeypatch.setattr(study.pm, "predict_finite", finite)
    monkeypatch.setattr(study, "predict_baseline", peak)
    monkeypatch.setattr(study, "legacy_predict", legacy)
    return {"root": tmp_path, "protocol": protocol, "models": models, "calls": calls}


@pytest.mark.parametrize("changed_file", SCIENTIFIC_FILES)
def test_source_change_rejected_before_prediction_or_freeze(isolated_study, changed_file):
    root = isolated_study["root"]
    path = root / changed_file
    # Whitespace preserves valid JSON, so refusal must come from the digest.
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(RuntimeError, match="fit-start snapshot"):
        study.verify_development_sources()
    with pytest.raises(RuntimeError, match="fit-start snapshot"):
        study.freeze()
    assert isolated_study["calls"] == []
    assert not (root / "freeze.json").exists()
    assert not (root / "new_candidates.csv").exists()


def test_freeze_selection_is_complete_and_uses_five_new_predictions_only(isolated_study, monkeypatch):
    root = isolated_study["root"]
    original_read = study.read
    allowed_reads = {root / "protocol.json", root / "outputs/development/run_started.json",
                     root / "outputs/development/development_checks.json",
                     root / "outputs/development/models.json", root / "outputs/development/gates.json"}

    def guarded_read(path):
        assert Path(path) in allowed_reads, "Freeze attempted to read historical/new evaluation results"
        return original_read(path)

    monkeypatch.setattr(study, "read", guarded_read)
    study.verify_development_sources()
    study.freeze()
    candidates = pd.read_csv(root / "new_candidates.csv")
    frozen = pd.read_csv(root / "frozen_predictions.csv")
    selection = pd.read_csv(root / "boundary_selection.csv")
    manifest = json.loads((root / "freeze.json").read_text(encoding="utf-8"))

    # Ten neighboring points minus the coarse point 100 give nine extras;
    # the prespecified evenly spaced thinning retains eight of them.
    coarse = set(isolated_study["protocol"]["fixed_coarse_kla"])
    extras = sorted({threshold + offset for threshold in (102., 114., 126., 138., 150.)
                     for offset in (-2., 2.)} - coarse)
    extras = {extras[i] for i in np.rint(np.linspace(0, len(extras)-1, 8)).astype(int)}
    expected_kla = coarse | extras
    assert len(candidates) == 6 * 15
    assert candidates.candidate_id.is_unique
    assert set(selection.model) == set(NEW_MODEL_NAMES)
    assert len(selection) == 6 * 5
    assert 200. not in set(candidates.kla) and 204. not in set(candidates.kla)
    for _, group in candidates.groupby("scenario"):
        assert set(group.kla) == expected_kla
        assert set(group.loc[group.grid_kind.eq("coarse"), "kla"]) == coarse
        assert group[["kla", "load_coordinate", "oxygen_transfer_factor"]].duplicated().sum() == 0
        np.testing.assert_allclose(group.effective_kla, group.kla * group.oxygen_transfer_factor)
    assert len(frozen) == len(candidates) * len(ALL_MODEL_NAMES)
    assert set(frozen.model) == set(ALL_MODEL_NAMES)
    assert not frozen.duplicated(["candidate_id", "model"]).any()
    # Each finite predictor omits operating columns; the merge must restore
    # exactly one unambiguous copy for process, statistical and legacy rows.
    for name in ("kla", "load_coordinate", "oxygen_transfer_factor", "effective_kla"):
        assert name in frozen and not frozen[name].isna().any()
        assert name + "_x" not in frozen and name + "_y" not in frozen
    assert frozen.in_joint_domain.all() and frozen.in_feature_domain.all()
    assert ((frozen.direct_release.astype(int) + frozen.direct_reject.astype(int)
             + frozen.recheck.astype(int)) == 1).all()
    assert manifest["new_reference_labels_read"] is False
    assert manifest["n_candidates"] == len(candidates)
    for entry in manifest["files"]:
        assert study.sha(root / entry["path"]) == entry["sha256"]


def test_half_step_ids_frozen_from_full_model_distance_then_kla(isolated_study):
    root = isolated_study["root"]
    study.freeze()
    frozen = pd.read_csv(root / "frozen_predictions.csv")
    fine = pd.read_csv(root / "half_step_candidates.csv")
    assert len(fine) == 6 and fine.scenario.is_unique
    expected = []
    for _, group in frozen.loc[frozen.model.eq("full_calibrated")].groupby("scenario"):
        selected = group.assign(distance=np.abs(group.predicted_peak-4.)).sort_values(["distance", "kla"]).iloc[0]
        expected.append(selected.candidate_id)
        assert selected.kla == 100.  # Exact equal distance at 100 and 104.
    assert list(fine.candidate_id) == expected


def test_existing_prospective_freeze_cannot_be_overwritten(isolated_study):
    root = isolated_study["root"]
    study.freeze()
    before = (root / "freeze.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        study.freeze()
    assert (root / "freeze.json").read_bytes() == before
