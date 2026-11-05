"""Protocol tests: leakage controls, serialization, and input validation."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from statistical_models import fit_baselines, predict_baseline


@pytest.fixture
def development():
    parts = []
    for scenario, load, alpha in (("base", 0.0, 1.0), ("load", 1.0, 1.0), ("transfer", 0.0, 0.7)):
        kla = np.linspace(40.0, 520.0, 8)
        z = kla * alpha
        parts.append(pd.DataFrame({
            "scenario": scenario,
            "load_coordinate": load,
            "oxygen_transfer_factor": alpha,
            "kla": kla,
            "SNH_max_eval": 2 + 8 * load + 32 * np.exp(-z / 130),
            "target": np.nan,  # Binary labels are deliberately unusable.
            "DO_tank5_mean_eval": np.inf,
        }))
    return pd.concat(parts, ignore_index=True)


def test_prediction_uses_only_allowed_inputs_and_survives_json(development, tmp_path):
    models = fit_baselines(development, tmp_path)
    inputs = development[["load_coordinate", "oxygen_transfer_factor", "kla"]].copy()
    poisoned = development.copy()
    poisoned["SNH_max_eval"] = -1e99
    poisoned["scenario"] = "unseen name that must not affect predictions"
    poisoned["target"] = "leak"
    poisoned["DO_tank5_mean_eval"] = "leak"
    for name, model in models.items():
        expected = predict_baseline(inputs, model)
        np.testing.assert_array_equal(predict_baseline(poisoned, model), expected)
        restored = json.loads(json.dumps(model, allow_nan=False))
        np.testing.assert_array_equal(predict_baseline(inputs, restored), expected)
        assert model["nonnegative_prediction_clip"] is True
        assert np.all(expected >= 0)
        assert "continuous" in model["supervision"]
    disk_models = json.loads((tmp_path / "statistical_models.json").read_text(encoding="utf-8"))
    for name in models:
        np.testing.assert_array_equal(predict_baseline(inputs, disk_models[name]), predict_baseline(inputs, models[name]))


def test_training_ignores_all_unapproved_columns(development, tmp_path):
    models = fit_baselines(development, tmp_path / "first")
    altered = development.copy()
    altered["target"] = np.arange(len(altered)) ** 3
    altered["DO_tank5_mean_eval"] = np.arange(len(altered)) * -999
    altered["SNH_mean_eval"] = altered["SNH_max_eval"] * 1e20
    other = fit_baselines(altered, tmp_path / "second")
    for name in models:
        assert models[name]["hyperparameters"] == other[name]["hyperparameters"]
        np.testing.assert_array_equal(predict_baseline(development, models[name]), predict_baseline(development, other[name]))


def test_nonnegative_clip_is_applied_to_negative_raw_predictions(development, tmp_path):
    models = fit_baselines(development, tmp_path)
    for model in models.values():
        model["weights"] = [0.0] * len(model["weights"])
        model["target_mean"] = -2.0
        np.testing.assert_array_equal(predict_baseline(development, model), np.zeros(len(development)))


def test_cv_scaling_and_target_center_are_fit_on_training_rows_only(development, tmp_path):
    fit_baselines(development, tmp_path)
    folds = pd.read_csv(tmp_path / "statistical_cv_folds.csv")
    assert len(folds) == (12 + 9) * 3
    for record in folds.to_dict("records"):
        train = development[development.scenario != record["held_out_scenario"]]
        x = np.column_stack((train.load_coordinate, train.oxygen_transfer_factor * train.kla))
        expected_scale = x.std(axis=0)
        expected_scale[expected_scale == 0] = 1
        np.testing.assert_allclose(json.loads(record["training_input_mean"]), x.mean(axis=0))
        np.testing.assert_allclose(json.loads(record["training_input_scale"]), expected_scale)
        assert record["training_target_mean"] == pytest.approx(train.SNH_max_eval.mean())
        if record["held_out_scenario"] == "load":
            assert json.loads(record["training_input_mean"])[0] == 0
            assert json.loads(record["training_constant_columns"]) == [0]
    selections = json.loads((tmp_path / "statistical_model_selection.json").read_text(encoding="utf-8"))
    for family in ("poly_ridge", "rbf_kernel_ridge"):
        candidates = [candidate for candidate in selections if candidate["model_name"] == family]
        selected = [candidate for candidate in candidates if candidate["selected"]]
        assert len(selected) == 1
        assert selected[0]["mean_scenario_mse_mgN_per_L_squared"] == min(candidate["mean_scenario_mse_mgN_per_L_squared"] for candidate in candidates)
        for candidate in candidates:
            expected = np.mean(list(candidate["scenario_mses"].values()))
            assert candidate["mean_scenario_mse_mgN_per_L_squared"] == pytest.approx(expected)


@pytest.mark.parametrize("column,value", [("load_coordinate", np.nan), ("kla", np.inf), ("kla", -1), ("oxygen_transfer_factor", 0), ("SNH_max_eval", -1), ("SNH_max_eval", np.nan)])
def test_rejects_invalid_training_values(development, tmp_path, column, value):
    development.loc[0, column] = value
    with pytest.raises(ValueError):
        fit_baselines(development, tmp_path)
    assert not (tmp_path / "statistical_models.json").exists()


def test_missing_inputs_scenarios_and_invalid_model(development, tmp_path):
    with pytest.raises(ValueError, match="Missing input"):
        fit_baselines(development.drop(columns="kla"), tmp_path)
    with pytest.raises(ValueError, match="three"):
        fit_baselines(development[development.scenario != "load"], tmp_path)
    with pytest.raises(ValueError, match="scenario"):
        fit_baselines(development.drop(columns="scenario"), tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        predict_baseline(development, {"schema_version": 999})
