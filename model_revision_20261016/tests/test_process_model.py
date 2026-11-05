"""Independent mathematical and numerical checks for the M3 process model.

Only synthetic operating inputs are used. The preserved M2 implementation
provides an independently versioned oracle at the canonical fixed ratio.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import process_model as m

_spec = importlib.util.spec_from_file_location("m3_legacy_oracle", ROOT / "legacy" / "controlled_models.py")
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
_spec.loader.exec_module(legacy)


def model(kind="full_fixed_ratio", values=None):
    p = [8., 1.05 * m.CANONICAL_KNA, 3., .55, .03, m.FIXED_RATIO]
    return {"kind": kind, "values": p if values is None else list(values),
            "effective_dimension": 6 if kind == "full_calibrated" else 5}


def legacy_parameters(p):
    old = legacy.default_parameters()
    old.update(tau_h=float(p[0]), mu_a=float(p[1] / m.CANONICAL_KNA),
               k_oa=float(p[2]), oxygen_gain=float(p[3]), ss_to_nh=float(p[4]),
               nitrification_rate_cap=m.REMOVAL_CAP / m.CANONICAL_KNA,
               nh4_nitrification_factor=m.CANONICAL_KNA)
    return legacy.parameter_array(old)


@pytest.fixture
def cases():
    return pd.DataFrame({
        "candidate_id": ["synthetic_low", "synthetic_mid", "synthetic_high", "synthetic_load"],
        "scenario": ["arbitrary"] * 4,
        "load_coordinate": [0., .4, 1., 1.],
        "oxygen_transfer_factor": [.7, .88, 1., .7],
        "kla": [40., 165., 520., 100.],
    })


@pytest.mark.parametrize("kind,old_kind", [("full_fixed_ratio", "full"), ("no_feedback", "no_oxygen_demand_feedback")])
@pytest.mark.parametrize("values", [None, [2.5, 20., .4, 1.2, .09, m.FIXED_RATIO]])
def test_canonical_steady_states_equal_preserved_implementation(cases, kind, old_kind, values):
    current = model(kind, values)
    p = m.parameters(current)
    old = legacy_parameters(p)
    code = int(kind == "no_feedback")
    old_code = int(old_kind == "no_oxygen_demand_feedback")
    for row in m.inputs(cases):
        expected = legacy._steady_values(old, *row, old_code)
        np.testing.assert_allclose(m.steady_one(p, row, code), expected, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(m.predict_steady(cases, current),
                               legacy.steady_predictions(cases, old, old_kind), rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("kind", ["full_fixed_ratio", "no_feedback"])
@pytest.mark.parametrize("state", [[1200., 75., 110., 6., 8., 2.], [1600., 180., 1100., 15., 40., 7.]])
def test_canonical_rhs_equal_preserved_implementation_including_active_cap(kind, state):
    current = model(kind, [5., 24., 2., .7, .05, m.FIXED_RATIO])
    p = m.parameters(current)
    old = legacy_parameters(p)
    case = np.array([.7, .85, 200.])
    closure = m.closure(*case)
    expected = legacy._rhs(np.asarray(state), old, *closure, case[1], case[2], int(kind == "no_feedback"))
    actual = m.rhs(np.asarray(state), p, *closure, case[1], case[2], int(kind == "no_feedback"))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("kind", m.MODELS)
def test_steady_solution_satisfies_all_six_equations(cases, kind):
    current = model(kind)
    if kind == "full_calibrated":
        current["values"][5] = 1.4
    p = m.parameters(current)
    code = int(kind == "no_feedback")
    for row in m.inputs(cases):
        x = m.steady_one(p, row, code)
        residual = m.rhs(x, p, *m.closure(*row), row[1], row[2], code)
        np.testing.assert_allclose(residual, np.zeros(6), rtol=0., atol=2e-10)


@pytest.mark.parametrize("kind", m.MODELS)
def test_predictions_ignore_target_and_noninput_columns(cases, kind):
    current = model(kind)
    bare = cases[["load_coordinate", "oxygen_transfer_factor", "kla"]]
    poison = cases.assign(SNH_max_eval=np.nan, target="do not read", DO_tank5_mean_eval=np.inf,
                          SNH_mean_eval=-1e200, scenario="irrelevant name")
    np.testing.assert_array_equal(m.predict_steady(bare, current), m.predict_steady(poison, current))
    a = m.predict_finite(bare.iloc[:1], current, warmup_days=0., eval_days=2./1440.)
    b = m.predict_finite(poison.iloc[:1], current, warmup_days=0., eval_days=2./1440.)
    columns = ["predicted_peak", "predicted_mean", "predicted_do", "predicted_sno_canonical", "raw_compliant"]
    pd.testing.assert_frame_equal(a[columns], b[columns])


def test_reported_peak_is_window_maximum_not_mean(cases):
    current = model()
    case = cases.iloc[[1]].copy()
    p = m.parameters(current)
    row = m.inputs(case)[0]
    lo, hi, avg, *_ = m.simulate(p, row, 0, 1., 0., .1)
    prediction = m.predict_finite(case, current, warmup_days=0., eval_days=.1).iloc[0]
    assert prediction.predicted_peak == hi[4]
    assert prediction.predicted_mean == avg[4]
    assert prediction.raw_compliant == (hi[4] <= 4.)
    assert hi[4] >= avg[4] >= lo[4]
    assert hi[4] - avg[4] > 1e-3  # A nonsteady case distinguishes peak from mean.


@pytest.mark.parametrize("kind,old_kind", [("full_fixed_ratio", "full"), ("no_feedback", "no_oxygen_demand_feedback")])
def test_single_case_67_day_rk4_matches_preserved_model_and_half_step(kind, old_kind):
    frame = pd.DataFrame({"load_coordinate": [.4], "oxygen_transfer_factor": [.88], "kla": [165.]})
    current = model(kind)
    minute = m.predict_finite(frame, current).iloc[0]
    half = m.predict_finite(frame, current, dt_minutes=.5).iloc[0]
    expected = legacy.predict_cases(frame, legacy_parameters(m.parameters(current)), old_kind).iloc[0]
    assert minute.predicted_peak == pytest.approx(expected.SNH_max_eval, abs=2e-9)
    assert minute.predicted_mean == pytest.approx(expected.SNH_mean_eval, abs=2e-9)
    assert minute.predicted_do == pytest.approx(expected.DO_mean_eval, abs=2e-9)
    assert minute.predicted_sno_canonical == pytest.approx(expected.SNO_mean_eval, abs=2e-9)
    assert minute.predicted_peak == pytest.approx(half.predicted_peak, abs=1e-6)
    assert minute.predicted_peak == pytest.approx(m.predict_steady(frame, current)[0], abs=1e-3)
    assert minute.raw_compliant == half.raw_compliant
    assert minute.critical_NH4_DO_clips == 0


@pytest.mark.parametrize("values", [[np.nan, 5., 3., .5, .03, m.FIXED_RATIO],
                                     [0., 5., 3., .5, .03, m.FIXED_RATIO],
                                     [8., 26., 3., .5, .03, m.FIXED_RATIO],
                                     [8., 5., 3., .5, -.01, m.FIXED_RATIO],
                                     [8., 5., 3., .5, .03]])
def test_invalid_parameters_rejected(cases, values):
    with pytest.raises(ValueError):
        m.predict_steady(cases, model(values=values))


def test_invalid_kind_and_changed_fixed_ratio_rejected(cases):
    with pytest.raises(ValueError):
        m.predict_steady(cases, model("unrecognized"))
    for kind in ("full_fixed_ratio", "no_feedback"):
        values = model(kind)["values"]
        values[5] = 1.
        with pytest.raises(ValueError):
            m.predict_steady(cases, model(kind, values))


@pytest.mark.parametrize("column,bad", [("load_coordinate", -1.), ("oxygen_transfer_factor", 0.),
                                       ("oxygen_transfer_factor", 1.01), ("kla", -1.), ("kla", np.inf)])
def test_invalid_operating_inputs_rejected(cases, column, bad):
    broken = cases.copy()
    broken.loc[0, column] = bad
    with pytest.raises(ValueError):
        m.predict_steady(broken, model())


@pytest.mark.parametrize("windows", [dict(dt_minutes=0.), dict(dt_minutes=np.inf),
                                     dict(warmup_days=-1.), dict(eval_days=0.),
                                     dict(eval_days=1.5/1440.), dict(eval_days=1e-12)])
def test_invalid_or_zero_sample_integration_windows_rejected(cases, windows):
    with pytest.raises(ValueError):
        m.predict_finite(cases.iloc[:1], model(), **windows)


@pytest.mark.parametrize("kind,dimension", [("full_fixed_ratio", 5), ("no_feedback", 5), ("full_calibrated", 6)])
def test_fit_uses_prespecified_effective_dimensions(cases, tmp_path, kind, dimension):
    # Tiny synthetic development table checks the optimizer interface only.
    training = cases.copy()
    training["SNH_max_eval"] = m.predict_steady(training, model(kind))
    training["target"] = "unusable binary labels"
    fitted = m.fit_model(training, kind, tmp_path / kind, n_starts=1, max_nfev=2)
    assert fitted["effective_dimension"] == dimension
    assert fitted["n_training_cases"] == len(training)
    assert fitted["fit_names"] == list(m.NAMES[:dimension])
    assert len(fitted["runs"][0]["normalized_solution"]) == dimension
    assert len(m.parameters(fitted)) == 6
    if dimension == 5:
        assert fitted["values"][5] == m.FIXED_RATIO


def test_valid_reoptimized_profile_is_five_dimensional_and_serializable(cases, tmp_path):
    training = cases.assign(SNH_max_eval=10.)
    fitted = m.fit_model(training, "full_calibrated", tmp_path / "profile", n_starts=1,
                         max_nfev=1, fixed_rho=1.4)
    assert fitted["kind"] == "full_calibrated"
    assert fitted["effective_dimension"] == 5
    assert fitted["values"][5] == 1.4
    assert len(m.parameters(fitted)) == 6


def test_existing_fit_cannot_be_silently_overwritten(cases, tmp_path):
    training = cases.assign(SNH_max_eval=10.)
    destination = tmp_path / "preserved"
    destination.mkdir()
    artifact = destination / "model.json"
    artifact.write_text('{"preserved": true}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        m.fit_model(training, "full_calibrated", destination, n_starts=1, max_nfev=1)
    assert artifact.read_text(encoding="utf-8") == '{"preserved": true}'


@pytest.mark.parametrize("kind", ["full_fixed_ratio", "no_feedback"])
def test_fixed_pair_cannot_fit_an_inconsistent_ratio(cases, tmp_path, kind):
    training = cases.assign(SNH_max_eval=10.)
    with pytest.raises(ValueError):
        m.fit_model(training, kind, tmp_path / kind, n_starts=1, max_nfev=1, fixed_rho=1.)


@pytest.mark.parametrize("rho", [np.nan, np.inf, -.1, 3.1])
def test_profile_rejects_invalid_fixed_ratio(cases, tmp_path, rho):
    training = cases.assign(SNH_max_eval=10.)
    with pytest.raises(ValueError):
        m.fit_model(training, "full_calibrated", tmp_path / "invalid_profile", n_starts=1,
                    max_nfev=1, fixed_rho=rho)
