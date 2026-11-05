"""Check aggregation and fail-closed analysis without running any ODEs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_results as ar


@pytest.fixture
def frames():
    candidates = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "scenario": ["one"] * 4,
        "kla": [100.0, 150.0, 200.0, 250.0],
        "grid_kind": ["coarse", "prediction_enriched", "coarse", "prediction_enriched"],
    })
    references = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "SNH_max_eval": [8.0, 6.0, 4.0, 2.0],
        "SNH_zero_violation": ["False", "False", "True", "True"],
        "status": ["success"] * 4,
    })
    predictions = pd.concat([
        candidates.assign(model=model, predicted_peak=[2.0, 3.0, 5.0, 6.0],
                          raw_compliant=["True", "True", "False", "False"],
                          recheck=["True", "False", "True", "False"])
        for model in ar.MODEL_NAMES
    ], ignore_index=True)
    return predictions, references, candidates


def test_errors_gate_corrections_and_strata(frames):
    joined = ar.join_evaluation(*frames)
    metrics = ar.aggregate_metrics(joined)
    assert len(metrics) == 5 * 3
    row = metrics.loc[(metrics.model == "full_calibrated") & (metrics.stratum == "all")].iloc[0]
    assert row.rmse == pytest.approx(np.sqrt((36 + 9 + 1 + 16) / 4))
    assert row.mae == 3.5
    assert row.raw_false_releases == row.raw_false_rejections == 2
    assert row.gate_selected == row.gate_corrected_errors == row.gate_missed_errors == 2
    assert row.final_false_releases == row.final_false_rejections == 1
    coarse = metrics.loc[(metrics.model == "full_calibrated") & (metrics.stratum == "coarse")].iloc[0]
    enriched = metrics.loc[(metrics.model == "full_calibrated") & (metrics.stratum == "prediction_enriched")].iloc[0]
    assert coarse.final_errors == 0
    assert enriched.final_errors == 2
    assert enriched.gate_selected == 0
    assert set(ar.aggregate_metrics(joined, by_condition=True).scenario) == {"one"}


def test_boundary_sign_and_missing_compliance(frames):
    joined = ar.join_evaluation(*frames)
    boundaries = ar.boundary_table(joined)
    row = boundaries.loc[(boundaries.model == "full_calibrated") & (boundaries.stratum == "all")].iloc[0]
    assert row.reference_first_compliant_kla == 200
    assert row.raw_first_compliant_kla == 100
    assert row.final_first_compliant_kla == 150
    assert row.raw_minus_reference_kla == -100
    assert row.final_minus_reference_kla == -50
    predictions, references, candidates = frames
    predictions["predicted_peak"] = 9.0
    predictions["raw_compliant"] = "False"
    predictions["recheck"] = "False"
    boundaries = ar.boundary_table(ar.join_evaluation(predictions, references, candidates))
    assert not boundaries.raw_has_compliant_candidate.any()
    assert boundaries.raw_minus_reference_kla.isna().all()


@pytest.mark.parametrize("bad", ["yes", "nan", "", None, 2, 0.5, np.nan])
def test_invalid_boolean_is_not_truthy(bad):
    with pytest.raises(ValueError, match="invalid boolean"):
        ar.strict_bool_series(pd.Series(["False", bad]), "flag")


def test_false_boolean_string_is_false():
    assert ar.strict_bool_series(pd.Series(["False", "false", "0", False, "True", "1"]), "flag").tolist() == [False, False, False, False, True, True]


def test_dynamic_historical_model_gets_own_metrics(frames):
    predictions, references, candidates = frames
    historical = predictions.loc[predictions.model.eq("full_calibrated")].assign(model="legacy_M2_binary")
    predictions = pd.concat([predictions, historical], ignore_index=True)
    joined = ar.join_evaluation(predictions, references, candidates)
    metrics = ar.aggregate_metrics(joined)
    assert set(metrics.model) == set(ar.MODEL_NAMES) | {"legacy_M2_binary"}
    assert len(metrics) == 6 * 3


@pytest.mark.parametrize("defect", ["missing_reference", "extra_reference", "duplicate_reference", "missing_prediction", "duplicate_prediction", "model_name", "scenario", "grid_kind", "reference_failure", "raw_label", "reference_label", "nonfinite"])
def test_invalid_coverage_status_and_labels_fail(frames, defect):
    predictions, references, candidates = (frame.copy() for frame in frames)
    if defect == "missing_reference": references = references.iloc[:-1]
    elif defect == "extra_reference": references = pd.concat([references, references.iloc[[0]].assign(candidate_id="extra")])
    elif defect == "duplicate_reference": references = pd.concat([references, references.iloc[[0]]])
    elif defect == "missing_prediction": predictions = predictions.iloc[1:]
    elif defect == "duplicate_prediction": predictions = pd.concat([predictions, predictions.iloc[[0]]])
    elif defect == "model_name": predictions.loc[0, "model"] = "unexpected"
    elif defect == "scenario": predictions.loc[0, "scenario"] = "wrong"
    elif defect == "grid_kind": predictions.loc[0, "grid_kind"] = "prediction_enriched"
    elif defect == "reference_failure": references.loc[0, "status"] = "failed"
    elif defect == "raw_label": predictions.loc[0, "raw_compliant"] = "False"
    elif defect == "reference_label": references.loc[0, "SNH_zero_violation"] = "True"
    elif defect == "nonfinite": predictions.loc[0, "predicted_peak"] = np.inf
    with pytest.raises(ValueError):
        ar.join_evaluation(predictions, references, candidates, expected_models=ar.MODEL_NAMES)


def test_analysis_requires_passing_audit_before_reading_results(tmp_path, monkeypatch):
    events = []
    def fake_verify(path):
        events.append("verify")
        return {"files": [{"path": "frozen_predictions.csv"}]}, "abc", [{"candidate_id": "a"}]
    def fake_audit(path):
        events.append("audit")
        return {"passed": False, "errors": ["raw data mismatch"]}
    monkeypatch.setattr(ar.fr, "verify_freeze", fake_verify)
    monkeypatch.setattr(ar.fr, "audit_reference_outputs", fake_audit)
    with pytest.raises(ValueError, match="audit failed"):
        ar.analyze(tmp_path)
    assert events == ["verify", "audit"]
    assert not (tmp_path / "analysis_summary.json").exists()


def test_analyze_saves_checked_outputs_without_fitting(frames, tmp_path, monkeypatch):
    predictions, references, candidates = frames
    candidates.to_csv(tmp_path / "new_candidates.csv", index=False)
    predictions.to_csv(tmp_path / "frozen_predictions.csv", index=False)
    references.to_csv(tmp_path / "reference_summary.csv", index=False)
    cases = [{"candidate_id": identity} for identity in candidates.candidate_id]
    monkeypatch.setattr(ar.fr, "verify_freeze", lambda _: ({"files": [{"path": "frozen_predictions.csv"}]}, "abc", cases))
    monkeypatch.setattr(ar.fr, "audit_reference_outputs", lambda _: {"passed": True, "freeze_sha256": "abc", "expected_count": 4, "successful_count": 4, "failed_count": 0})
    summary = ar.analyze(tmp_path)
    assert summary["candidate_count"] == 4
    assert summary["post_test_model_or_gate_adaptation"] is False
    assert summary["old_244_96_test_data_access"] is False
    for filename in summary["output_files"]:
        assert (tmp_path / filename).is_file()
    restored = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    assert restored["source_sha256"]["reference_summary.csv"] == hashlib.sha256((tmp_path / "reference_summary.csv").read_bytes()).hexdigest()
