"""Hand-calculated edge cases for post-freeze descriptive boundary metrics."""
import numpy as np
import pandas as pd
import pytest

import analyze_results as standard
import boundary_analysis as boundary


def test_coverage_counts_candidates_once_and_keeps_nonmonotone_changes():
    candidates = pd.DataFrame({"candidate_id": ["p1", "p2", "p3", "p4"],
                               "scenario": ["condition"] * 4, "kla": [100., 104., 108., 112.],
                               "effective_kla": [100., 104., 108., 112.], "grid_kind": ["coarse"] * 4})
    refs = pd.DataFrame({"candidate_id": candidates.candidate_id, "SNH_max_eval": [5., 3., 5., 4.],
                        "SNH_zero_violation": [False, True, False, True], "status": ["success"] * 4})
    truth = boundary.candidate_truth(candidates, refs)
    coverage, changes = boundary.coverage_tables(truth)
    coarse = coverage.loc[coverage.stratum.eq("coarse")].iloc[0]
    assert coarse.n_candidates == 4
    assert coarse.compliant_count == coarse.noncompliant_count == 2
    assert coarse.has_both_sides
    assert coarse.nearest_compliant_distance_mgN_L == 0.
    assert coarse.nearest_noncompliant_distance_mgN_L == 1.
    assert coarse.within_0p25_count == 1
    assert coarse.adjacent_label_change_count == 3
    assert coarse.adjacent_strict_sign_change_count == 2
    assert coarse.compliant_to_noncompliant_count == 1
    assert coarse.noncompliant_to_compliant_count == 2
    assert len(changes.loc[changes.stratum.eq("coarse")]) == 3
    missing = coverage.loc[coverage.stratum.eq("prediction_enriched")].iloc[0]
    assert missing.n_candidates == 0
    assert not missing.has_both_sides
    assert pd.isna(missing.nearest_compliant_distance_mgN_L)


def example_joined():
    candidates = pd.DataFrame({"candidate_id": ["p1", "p2", "p3"],
                               "scenario": ["condition"] * 3, "kla": [100., 104., 108.],
                               "effective_kla": [100., 104., 108.], "grid_kind": ["coarse"] * 3})
    refs = pd.DataFrame({"candidate_id": candidates.candidate_id, "SNH_max_eval": [3.75, 4., 4.5],
                        "SNH_zero_violation": [True, True, False], "status": ["success"] * 3})
    predictions = []
    for name in boundary.MODELS:
        values = [4.1, 4.1, 3.9] if name == "full_fixed_ratio" else [3.95, 4.2, 4.7]
        for i, candidate in enumerate(candidates.to_dict(orient="records")):
            predictions.append(dict(candidate, model=name, predicted_peak=values[i],
                                    raw_compliant=values[i] <= 4., recheck=[True, False, True][i]))
    return standard.join_evaluation(pd.DataFrame(predictions), refs, candidates, expected_models=boundary.MODELS)


def test_near_windows_include_equality_and_preserve_gate_errors():
    near = boundary.near_threshold_metrics(example_joined())
    row = near.loc[near.stratum.eq("coarse") & near.window_mgN_L.eq(.25) & near.model.eq("full_fixed_ratio")].iloc[0]
    assert row.n_candidates == 2
    assert row.reference_compliant_count == 2
    assert row.raw_false_releases == 0
    assert row.raw_false_rejections == 2
    assert row.gate_selected == 1
    assert row.gate_corrected_errors == row.gate_missed_errors == 1
    assert row.final_false_rejections == 1
    wider = near.loc[near.stratum.eq("coarse") & near.window_mgN_L.eq(.5) & near.model.eq("full_fixed_ratio")].iloc[0]
    assert wider.n_candidates == 3
    assert wider.raw_false_releases == 1
    assert wider.gate_corrected_errors == 2
    empty = near.loc[near.stratum.eq("prediction_enriched")]
    assert empty.n_candidates.eq(0).all()
    assert empty.rmse.isna().all()


def test_matched_pair_is_paired_on_identity_not_row_order():
    joined = example_joined().sample(frac=1., random_state=12)
    pair = boundary.matched_pair_metrics(joined)
    row = pair.loc[pair.stratum.eq("coarse")].iloc[0]
    expected_full_rmse = np.sqrt((.35 ** 2 + .1 ** 2 + .6 ** 2) / 3.)
    assert row.full_fixed_ratio_rmse == pytest.approx(expected_full_rmse)
    assert row.no_feedback_rmse == pytest.approx(.2)
    assert row.difference_rmse == pytest.approx(expected_full_rmse - .2)
    assert row.full_lower_absolute_error_count == 1
    assert row.no_feedback_lower_absolute_error_count == 2
    assert row.n_candidates == 3
    missing = joined.loc[~(joined.model.eq("no_feedback") & joined.candidate_id.eq("p2"))]
    with pytest.raises(ValueError, match="identical candidates"):
        boundary.matched_pair_metrics(missing)


def test_false_reference_label_cannot_enter_boundary_coverage():
    candidates = pd.DataFrame([dict(candidate_id="p", scenario="condition", kla=100., effective_kla=100., grid_kind="coarse")])
    refs = pd.DataFrame([dict(candidate_id="p", SNH_max_eval=5., SNH_zero_violation=True, status="success")])
    with pytest.raises(ValueError, match="peak and compliance disagree"):
        boundary.candidate_truth(candidates, refs)
