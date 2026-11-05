"""New runtime input validator/router. No predictions, labels or BSM1 execution."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping
import numpy as np

VERSION = "runtime-routing-reconstructed-20261104-v1"
COLUMNS = ["time", "SI", "SS", "XI", "XS", "XBH", "XBA", "XP", "SO", "SNO", "SNH", "SND", "XND", "SALK", "TSS", "Q", "TEMP", "SD1", "SD2", "SD3", "XD4", "XD5"]
UNITS = ["d", "g COD/m3", "g COD/m3", "g COD/m3", "g COD/m3", "g COD/m3", "g COD/m3", "g COD/m3", "g O2/m3", "g N/m3", "g N/m3", "g N/m3", "g N/m3", "mol HCO3/m3", "g SS/m3", "m3/d", "degC", "native_dummy", "native_dummy", "native_dummy", "native_dummy", "native_dummy"]
BASELINE = np.array([30., 69.5, 51.2, 202.32, 28.17, 0., 0., 0., 0., 31.56, 6.95, 10.59, 7., 211.2675, 18446., 15., 0., 0., 0., 0., 0.])
ATOL = 1e-9

def _number(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))

def check_input(data, metadata: Mapping) -> dict:
    """Inspect all rows/22 columns before routing; never read reference labels.

    `in_scope` permits the *next* frozen model/Gate stage. It is not a direct
    acceptance or compliance decision. `fallback_required` is not execution.
    """
    result = {"implementation": VERSION, "status": "invalid_input", "reasons": [],
              "reference_executed": False, "final_compliant": None,
              "required_action": "stop_and_correct_input"}
    invalid = result["reasons"]
    if not isinstance(metadata, Mapping):
        invalid.append("metadata_must_be_an_object")
        return result
    if metadata.get("columns") != COLUMNS:
        invalid.append("column_order_must_match_exact_22_column_schema")
    if metadata.get("units") != UNITS:
        invalid.append("units_must_match_explicit_native_schema")
    if type(metadata.get("tempmodel")) is not bool:
        invalid.append("tempmodel_must_be_an_explicit_boolean")
    for name in ("load_coordinate", "oxygen_transfer_factor", "commanded_kla_d_1"):
        if not _number(metadata.get(name)):
            invalid.append("finite_numeric_metadata_required:" + name)
    if invalid:
        return result
    load = float(metadata["load_coordinate"])
    alpha = float(metadata["oxygen_transfer_factor"])
    kla = float(metadata["commanded_kla_d_1"])
    if load < 0 or not 0 < alpha <= 1 or kla < 0:
        invalid.append("invalid_operating_metadata:require_L_ge_0_0_lt_alpha_le_1_KLa_ge_0")
    try:
        if isinstance(data, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for row in data if isinstance(row, (list, tuple)) for value in row):
            invalid.append("array_must_not_contain_boolean_values")
            return result
        original = np.asarray(data)
        if original.dtype.kind not in "iuf":
            invalid.append("array_must_contain_numeric_values_not_booleans_or_strings")
            return result
        array = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        invalid.append("array_must_be_rectangular_numeric")
        return result
    if array.ndim != 2 or array.shape[1] != 22 or array.shape[0] < 2:
        invalid.append("array_must_have_at_least_two_rows_and_exactly_22_columns")
        return result
    if not np.isfinite(array).all():
        invalid.append("array_contains_nonfinite_values")
        return result
    if array[0, 0] < 0 or not np.all(np.diff(array[:, 0]) > 0):
        invalid.append("time_must_be_nonnegative_and_strictly_increasing")
    concentration_columns = [i for i in range(1, 22) if i not in (15, 16)]
    if (array[:, concentration_columns] < 0).any():
        invalid.append("negative_concentration_or_dummy_state")
    if (array[:, 15] <= 0).any():
        invalid.append("flow_must_be_positive")
    if (array[:, 16] <= -273.15).any():
        invalid.append("temperature_at_or_below_absolute_zero")
    if invalid:
        return result
    reasons = []
    changing = [COLUMNS[i] for i in range(1, 22) if not np.allclose(array[:, i], array[0, i], rtol=0, atol=ATOL)]
    if changing:
        reasons.append("time_varying_inputs:" + ",".join(changing))
    if not np.allclose(array[:, 16], 15., rtol=0, atol=ATOL):
        reasons.append("temperature_outside_fixed_15_degC_scope")
    if metadata["tempmodel"]:
        reasons.append("temperature_state_balance_enabled")
    expected = BASELINE.copy()
    expected[1] *= 1 + .2 * load
    expected[3] *= 1 + .2 * load
    expected[9] *= 1 + .35 * load
    # Compare every state column, including flow, temperature and placeholders.
    # A valid mismatch is unsupported and must not be silently remapped.
    mismatch = [COLUMNS[i + 1] for i in range(21) if not np.allclose(array[:, i + 1], expected[i], rtol=0, atol=ATOL)]
    if mismatch:
        reasons.append("outside_declared_constant_load_family:" + ",".join(mismatch))
    joint = 0 <= load <= 1 and .7 + .3 * load - 1e-12 <= alpha <= 1
    feature = 0 <= load <= 1 and 28 + 12 * load - 1e-12 <= alpha * kla <= 520 + 1e-12
    if not joint:
        reasons.append("outside_joint_condition_hull")
    if not feature:
        reasons.append("outside_model_feature_hull")
    result.update({"status": "fallback_required" if reasons else "in_scope",
                   "reasons": reasons,
                   "required_action": "run_fresh_successful_native_BSM1_before_final_decision" if reasons else "apply_frozen_model_and_verification_rule",
                   "rows_checked": len(array), "columns_checked": 22,
                   "effective_kla_d_1": alpha * kla,
                   "in_joint_condition_hull": joint, "in_model_feature_hull": feature,
                   "numerical_comparison_atol": ATOL})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="JSON object with data and metadata")
    parser.add_argument("--output", type=Path, help="New report file; refuses overwrite")
    args = parser.parse_args()
    try:
        payload = json.loads(args.case.read_text(encoding="utf-8"))
        report = check_input(payload["data"], payload["metadata"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Could not read case: " + str(error), file=sys.stderr)
        return 2
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 2 if report["status"] == "invalid_input" else 0

if __name__ == "__main__":
    raise SystemExit(main())
