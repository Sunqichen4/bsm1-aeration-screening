"""Independent M3B design checks; synthetic predictions only, no native ODE."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import boundary_study as bs
import fresh_reference as fr


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_legacy(monkeypatch, *, root=170., constant=None):
    """The selector can use old-model steady predictions, never new labels."""
    calls = []
    def legacy(frame, model, finite):
        assert model == {"fixture": "already_frozen_legacy"}
        assert finite is False
        assert not any("SNH_max" in c or "reference" in c for c in frame.columns)
        calls.append(frame.copy())
        z = frame.effective_kla.to_numpy()
        return np.full(len(frame), constant) if constant is not None else 4. + (root-z)/10.
    monkeypatch.setattr(bs.prediction_library, "legacy_predict", legacy)
    return {"legacy_M2_binary": {"fixture": "already_frozen_legacy"}}, calls


def test_fixed_grid_has_108_model_independent_primary_cases(monkeypatch):
    models, calls = _mock_legacy(monkeypatch)
    candidates, selection = bs.make_candidates(models)
    fixed = candidates[candidates.grid_kind.eq("coarse")]
    assert bs.FIXED_Z == [112.] + list(np.arange(136., 197., 4.)) + [220.]
    assert len(fixed) == 108
    assert len(candidates) == 120
    assert len(selection) == 6
    for _, frame in fixed.groupby("scenario"):
        assert sorted(frame.effective_kla.tolist()) == bs.FIXED_Z
    assert calls


def test_conditions_and_every_command_are_within_frozen_domain(monkeypatch):
    models, _ = _mock_legacy(monkeypatch)
    candidates, _ = bs.make_candidates(models)
    assert candidates.scenario.nunique() == 6
    assert sorted(candidates.load_coordinate.unique()) == bs.LOADS
    for i, (load, q) in enumerate(zip(bs.LOADS, bs.QS), 1):
        frame = candidates[candidates.scenario.str.startswith(f"m3b_n{i:02d}_")]
        alpha_min = .7 + .3*load
        alpha = alpha_min + q*(1-alpha_min)
        assert np.all(frame.oxygen_transfer_factor.to_numpy() == alpha)
        assert alpha_min <= alpha <= 1.
    assert candidates.kla.between(40., 520.).all()
    np.testing.assert_allclose(candidates.kla*candidates.oxygen_transfer_factor,
                               candidates.effective_kla, rtol=1e-15, atol=0.)
    assert not candidates.duplicated(["load_coordinate", "oxygen_transfer_factor", "kla"]).any()
    assert candidates.candidate_id.is_unique


def test_enrichment_and_half_identities_follow_only_frozen_prediction(monkeypatch):
    models, _ = _mock_legacy(monkeypatch, root=170.)
    candidates, selection = bs.make_candidates(models)
    assert np.allclose(selection.root_z, 170., atol=1e-9)
    half = candidates[candidates.half_step_selected]
    assert len(half) == 12
    assert set(half.candidate_id) <= set(candidates.candidate_id)
    for _, frame in half.groupby("scenario"):
        np.testing.assert_allclose(sorted(frame.effective_kla), [169., 171.], atol=1e-9)
    # Moving only the frozen prediction changes enrichment, never the fixed grid.
    models, _ = _mock_legacy(monkeypatch, root=180.)
    alternate, _ = bs.make_candidates(models)
    cols = ["scenario", "load_coordinate", "oxygen_transfer_factor", "kla", "effective_kla"]
    pd.testing.assert_frame_equal(candidates.loc[candidates.grid_kind.eq("coarse"), cols].reset_index(drop=True),
                                  alternate.loc[alternate.grid_kind.eq("coarse"), cols].reset_index(drop=True))
    assert set(alternate.loc[alternate.half_step_selected, "effective_kla"]) == {179., 181.}


@pytest.mark.parametrize("root,expected_new", [(169., 114), (112., 114), (220., 114)])
def test_coincident_or_clipped_enrichment_is_one_physical_reference(monkeypatch, root, expected_new):
    models, _ = _mock_legacy(monkeypatch, root=root)
    candidates, _ = bs.make_candidates(models)
    assert len(candidates) == expected_new
    assert candidates.grid_kind.eq("coarse").sum() == 108
    assert not candidates.duplicated(["load_coordinate", "effective_kla"]).any()
    assert candidates.half_step_selected.sum() == 12
    assert candidates.loc[candidates.half_step_selected, "effective_kla"].between(112., 220.).all()
    assert candidates.selection_reason.str.contains("also_legacy_enrichment").sum() == 6


@pytest.mark.parametrize("constant,expected_z", [(5., 112.), (3., 112.)])
def test_no_crossing_skips_enrichment_and_half_fallback_uses_predictions(monkeypatch, constant, expected_z):
    models, _ = _mock_legacy(monkeypatch, constant=constant)
    candidates, selection = bs.make_candidates(models)
    assert len(candidates) == 108
    assert selection.root_z.isna().all()
    assert selection.selection_status.eq("no_endpoint_sign_change_skip_enrichment").all()
    # Only one predicted side exists; equal-distance fallback ties use lower z.
    assert candidates.half_step_selected.sum() == 6
    assert candidates.loc[candidates.half_step_selected, "effective_kla"].eq(expected_z).all()


def test_threshold_equality_and_strict_gate_boundaries():
    peaks = [3., 3.-1e-9, 5., 5.+1e-9, 4., 4.+1e-9]
    frame = pd.DataFrame(dict(model=["m"]*6, predicted_peak=peaks,
                              load_coordinate=[.5]*6, oxygen_transfer_factor=[.95]*6,
                              effective_kla=[170.]*6))
    out = bs.apply_gates(frame, {"m": {"absolute_peak_buffer_mgN_L": 1.}})
    assert out.raw_compliant.tolist() == [True, True, False, False, True, False]
    assert out.direct_release.tolist() == [False, True, False, False, False, False]
    assert out.direct_reject.tolist() == [False, False, False, True, False, False]
    assert out.recheck.tolist() == [True, False, True, False, True, True]


@pytest.mark.parametrize("load,alpha,z", [(-.01,.95,170.), (1.01,.95,170.),
                                         (.5,.84,170.), (.5,1.01,170.),
                                         (.5,.95,33.9), (.5,.95,520.1)])
def test_out_of_domain_forces_reference_for_apparent_safe_prediction(load, alpha, z):
    out = bs.apply_gates(pd.DataFrame([dict(model="m", predicted_peak=0.,
        load_coordinate=load, oxygen_transfer_factor=alpha, effective_kla=z)]),
        {"m": {"absolute_peak_buffer_mgN_L": .1}})
    assert bool(out.recheck.iloc[0])
    assert not bool(out.direct_release.iloc[0])
    assert not bool(out.direct_reject.iloc[0])


ORIGIN_FILES = ["outputs/development/models.json", "outputs/development/gates.json",
                "process_model.py", "statistical_models.py", "run_study.py",
                "legacy/controlled_models.py", "legacy/M2_full_parameters.json"]


@pytest.fixture
def origin_copy(tmp_path):
    root = tmp_path/"study"
    for name in ORIGIN_FILES + ["source_M3_manifest.json", "source_M3_freeze.json"]:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bs.ROOT/name, target)
    return root


def test_actual_origin_verifies():
    assert bs.verify_model_origin()["passed"]


@pytest.mark.parametrize("name", ORIGIN_FILES + ["source_M3_freeze.json"])
def test_model_gate_or_prediction_source_mutation_is_rejected(origin_copy, name):
    path = origin_copy/name
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError):
        bs.verify_model_origin(origin_copy)


def test_coordinated_model_and_manifest_mutation_cannot_rebind_old_origin(origin_copy):
    path = origin_copy/"outputs/development/models.json"
    path.write_bytes(path.read_bytes()+b" ")
    manifest_path = origin_copy/"source_M3_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for entry in manifest["files"]:
        if entry["path"] == "outputs/development/models.json":
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        bs.verify_model_origin(origin_copy)


def test_freeze_refuses_existing_freeze_or_any_native_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "ROOT", tmp_path)
    (tmp_path/"freeze.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        bs.freeze()
    (tmp_path/"freeze.json").unlink()
    (tmp_path/"reference").mkdir()
    with pytest.raises(ValueError, match="References already exist"):
        bs.freeze()


def test_protocol_explicitly_preserves_conditional_scope_and_termination(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "ROOT", tmp_path)
    bs.initialize_protocol()
    protocol = json.loads((tmp_path/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["fixed_effective_kla"] == bs.FIXED_Z
    assert len(protocol["conditions"]) == 6
    assert "M3 90" in protocol["old_evidence_used"]
    assert "Do not add cases" in protocol["stopping"]
    assert "independent identically distributed" in " ".join(protocol["inference_limits"])
    assert "keep 1-min main labels" in protocol["half_role"]
