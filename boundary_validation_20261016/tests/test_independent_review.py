"""Small adversarial checks for the post-freeze independent artifact auditor."""
import numpy as np
import pandas as pd
import pytest

from verification import independent_review as ir


def test_independent_classification_accounting_preserves_missed_error():
    frame = pd.DataFrame(dict(predicted_peak=[3.,5.,6.,2.,5.], SNH_max_eval=[5.,5.,2.,2.,3.],
        truth=[False,False,True,True,True], raw=[True,False,False,True,False],
        check=[True,False,True,False,False], final=[False,False,True,True,False]))
    row = ir.summary_row(frame)
    assert row["rmse"] == pytest.approx(np.sqrt(24./5.))
    assert row["mae"] == pytest.approx(8./5.)
    assert row["mean_error"] == pytest.approx(4./5.)
    assert row["raw_false_releases"] == 1
    assert row["raw_false_rejections"] == 2
    assert row["gate_selected"] == 2
    assert row["gate_corrected_false_releases"] == 1
    assert row["gate_corrected_false_rejections"] == 1
    assert row["gate_missed_false_rejections"] == 1
    assert row["final_false_releases"] == 0
    assert row["final_false_rejections"] == 1


def test_coverage_is_per_candidate_with_closed_near_threshold_windows():
    candidates = pd.DataFrame(dict(candidate_id=["a","b","c"], scenario=["one"]*3,
        effective_kla=[140.,144.,148.], grid_kind=["coarse","coarse","prediction_enriched"]))
    ref = pd.DataFrame(dict(candidate_id=["a","b","c"], SNH_max_eval=[3.75,4.25,5.]))
    result = ir.coverage_table(candidates,ref).set_index("stratum")
    assert bool(result.loc["coarse","has_both_sides"])
    assert result.loc["coarse","within_0p25_count"] == 2
    assert result.loc["coarse","within_0p25_compliant_count"] == 1
    assert result.loc["coarse","within_0p25_noncompliant_count"] == 1
    assert result.loc["all","within_1p0_count"] == 3
    assert result.loc["coarse","compliant_to_noncompliant_count"] == 1
    assert result.loc["coarse","noncompliant_to_compliant_count"] == 0
    assert pd.isna(result.loc["prediction_enriched","nearest_compliant_distance_mgN_L"])
    assert not bool(result.loc["prediction_enriched","has_both_sides"])


@pytest.mark.parametrize("wrong", [True, "True", "nonsense", np.nan])
def test_recorded_metric_drift_or_truthy_text_is_rejected(wrong):
    expected = pd.DataFrame(dict(key=["a"], has_both_sides=[False], rmse=[2.]))
    actual = expected.copy()
    actual["has_both_sides"] = [wrong]
    with pytest.raises(ValueError):
        ir.compare_table(expected, actual, ["key"], "synthetic")


def test_missing_side_must_remain_null_not_zero():
    expected = pd.DataFrame(dict(key=["a","b"], distance=[None,.5]))
    actual = expected.fillna(0.)
    with pytest.raises(ValueError,match="missing-value mismatch"):
        ir.compare_table(expected, actual, ["key"], "synthetic")


def test_mismatched_row_id_cannot_be_hidden_by_equal_metrics():
    expected = pd.DataFrame(dict(key=["a"], rmse=[2.]))
    actual = pd.DataFrame(dict(key=["b"], rmse=[2.]))
    with pytest.raises(ValueError,match="row identities differ"):
        ir.compare_table(expected, actual, ["key"], "synthetic")
