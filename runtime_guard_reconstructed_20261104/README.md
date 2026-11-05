# Runtime input checks and routing

Version: `runtime-routing-reconstructed-20261104-v1`, written 2026-11-04.

This new implementation turns the declared input checks in manuscript Section 2.4 and Supplementary Section S5 into a runnable routing interface. It is separate from the unavailable original dynamic-input adapter. It does not replace frozen publication results or claim to reproduce the historical 96-case challenge.

## Run

Use Python 3.11 and NumPy from the parent release's pinned environment. From this directory:

```powershell
python -B input_guard.py examples/constant_in_scope.json
python -B input_guard.py examples/cold_dynamic_fallback.json
python -B input_guard.py examples/invalid_units.json
python -B -m unittest -v test_input_guard
```

The invalid example exits with code 2. Valid cases exit 0 regardless of routing; a zero exit code is not a compliance decision. `--output new_report.json` saves a report and refuses to overwrite an existing file.

## Meaning of the three statuses

| Status | Required next action |
| --- | --- |
| `invalid_input` | Stop and correct the array or metadata. No model should run. |
| `in_scope` | Apply the frozen predictor and its verification rule. This is not acceptance. |
| `fallback_required` | Obtain a fresh successful native BSM1 result before a final decision. |

Every report keeps `reference_executed=false` and `final_compliant=null`. **This module checks and routes inputs; it does not execute BSM1.** The existing M1/M2 `reference-smoke` command executes a fresh constant-condition reference. Its retained `legacy/reference_runner.py` only supports the declared constant 15 °C conditions and cannot be used as a cold/dynamic runner. A validated cold/dynamic integration adapter remains needed to automate those fallbacks. Historical summaries retain their identity as historical results.

## Checks and scope

The JSON declares the exact native order of time plus 21 BSM1 state columns, one unit token per column, explicit Boolean `tempmodel`, load L, oxygen-transfer factor alpha and commanded KLa. All rows and all state columns are inspected. The checks enforce finite numeric values, increasing nonnegative time, nonnegative concentrations/placeholders, positive flow, physically possible temperature and operating metadata.

Units are compared with the declared schema; physical units cannot be inferred from numbers. The five unused dummy states carry the schema token `native_dummy` and remain fixed at their native zero values in the supported load family. The baseline state row and load mapping come directly from the supplied `legacy/vendor/original_project/src/aerbench/bsm1_validation.py`: SS and XS scale by `1 + 0.2L`, SNH by `1 + 0.35L`, all other states stay fixed.

The supported scope is constant 15 °C input with `tempmodel=false`, the declared load family, `0 <= L <= 1`, `0.7 + 0.3L <= alpha <= 1`, and `28 + 12L <= alpha*KLa <= 520` d^-1. Enabling the temperature-state balance, changing any state/flow/temperature over time, using another valid composition, or leaving either hull requests fallback. Array comparisons use absolute tolerance 1e-9; hull edges use 1e-12. These are explicit implementation tolerances in this new version, not newly estimated scientific bounds.

The interface reads no reference labels and calculates no surrogate prediction. Tests cover every state column, low temperature and dynamics, both hulls, label independence, NaN/Inf, invalid flow/concentrations, column count/order, declared units and nonmonotonic time. Finite test coverage is software verification rather than a universal reliability guarantee.
