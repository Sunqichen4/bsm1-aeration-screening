# M1/M2 reproduction package — reconstructed-20261103-v1

This is a new local reconstruction of an executable interface. It is **not** the deleted `program_completion_20261015/portable_repro` release. The retained `controlled_models.py`, original research interfaces and evidence are copied byte for byte, with origins and SHA256 values in `copied_source_manifest.json`. The wrapper `reproduce.py` was written on 2026-11-03. Original manuscript data and frozen parameters remain authoritative.

## Environment and commands

Use Python 3.11 and the versions in `requirements-lock.txt`. This package was executed with Python 3.11.15, NumPy 2.4.3, pandas 2.3.3, SciPy 1.16.3, Numba 0.65.1 and bsm2-python 0.0.16 on Windows. An existing matching environment can be used directly. To create a separate environment, run these commands from this package directory:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe reproduce.py verify --output runs/integrity
.venv\Scripts\python.exe reproduce.py reproduce --output runs/tables
.venv\Scripts\python.exe reproduce.py predict --version M1 --all-cases --output runs/M1_predictions
.venv\Scripts\python.exe reproduce.py predict --version M2 --model full --all-cases --output runs/M2_full_predictions
.venv\Scripts\python.exe reproduce.py predict --version M2 --model no_oxygen_demand_feedback --all-cases --output runs/M2_ablated_predictions
.venv\Scripts\python.exe reproduce.py gate-audit --output runs/Gate_audit
.venv\Scripts\python.exe reproduce.py fit-m2 --output runs/M2_fit
.venv\Scripts\python.exe reproduce.py fit-m1 --output runs/M1_reconstructed_fit
.venv\Scripts\python.exe reproduce.py fit-logistic --output runs/Logistic_reconstructed_fit
.venv\Scripts\python.exe reproduce.py reference-smoke --output runs/constant_reference
.venv\Scripts\python.exe -m unittest discover -s tests
```

Every output directory must be new or empty. An existing result is never overwritten. All input paths are relative to this package; the original absolute locations preserved inside historical JSON/CSV metadata are provenance records and are not executable path dependencies. Copy the complete package to any other directory and use the same commands.

## What each command establishes

| Command | Scope |
| --- | --- |
| `verify` | Verifies copied scientific sources and frozen input bytes; records actual runtime versions. |
| `predict` | Runs all 132 M1 or 244 M2 process candidates with the retained RK4 kernel, 60 d warm-up and 7 d evaluation at 1 min. Compares numerical outputs and classifications against frozen evidence. M1 also checks fixed-weight Logistic predictions. |
| `reproduce` | Recomputes M2 screening/threshold-reuse counts and M1/Gate counts from frozen evidence. Historical measured runtime and reference labels are reused evidence, not newly measured. |
| `gate-audit` | Recomputes 228 M1 process predictions, both Logistic decision sets and hull/monotonicity/buffer routing on the 132-point Block A and 96-point prospective Gate sets. Reference answers are reused only when the reconstructed Gate requests them. Individual and shared buffers are checked separately. This command does not claim a raw-trajectory audit. |
| `fit-m2` | Runs the retained original 3-start controlled M2 fitting implementation, including all 750 finite-time checks for each process model. Compares the new fits with frozen parameters and development classifications. |
| `fit-m1` | Runs a **new explicitly reconstructed initialization protocol**: historical 25-case parameter initialization, plus two normalized Gaussian perturbations of SD 0.1, seed 20261013. The original perturbation-generator source was not retained. The seven identifiable steady parameters are fitted; the unidentifiable relaxation time retains its historical value. This is not an exact replay of the historical M1 optimizer run. |
| `fit-logistic` | Fits the documented M1 cross-entropy/L2 and M2 Brier objectives on the 750 development cases. M2 starts are reconstructed (zero plus two uniform [-1, 1] coefficient vectors, seed 20261014). Frozen weights are never starts. Numerical differences from original fits are reported instead of hidden. |
| `reference-smoke` | Performs one fresh native BSM1 constant-condition integration, saves its new raw trajectory and compares its metrics with the matching frozen M2 case. It does not rerun every historical BSM1 experiment. |

`verification/` contains actual runs of this new implementation. `verification_summary.json` records the completed checks and numerical differences. Existing `inputs/portable_verification.json`, `inputs/clean_environment_verification.json`, `inputs/final_relocation_verification.json` and `inputs/program_completion.json` remain historical records from the deleted package. Their successful checks are not re-attributed to this reconstruction.

## Results actually verified on 2026-11-03

- Copied-source/input integrity: all 81 files passed.
- Frozen-parameter RK4 replay: M1 132 cases, maximum reported concentration difference 1.42 × 10⁻¹⁴; M2 full and ablated 244 cases each, zero numerical differences; no classification changes.
- Primary M2 error counts reproduced: full 0/0, ablated 10/1, basic Logistic 0/1, shared-closure Logistic 6/0 (false acceptances/false rejections).
- Gate replay: all raw decisions, reference routes and final decisions matched on the 132-case and 96-case sets. Each model requested 88 and 80 references, respectively. Shared/individual buffers on the 132-case set also matched.
- Original M2 fitting kernel: both fitted parameter vectors exactly matched frozen values; all 750 finite-time checks per model passed.
- Reconstructed M1 fit: all 750 development classifications matched the frozen model; maximum steady ammonia difference was 0.0001239314 mg N/L. This fit does not replace the published parameter file.
- Eleven reconstructed Logistic fits: all checked frozen classifications matched; the largest probability difference was 9.0192 × 10⁻⁶.
- One new native BSM1 reference smoke run: ammonia peak and mean exactly matched the corresponding frozen case.
- Seven tests passed, including rejection of an M1 ablation request before output creation and detection of corrupted Logistic fixtures in integrity and prediction checks. M1 prediction passes only when both process and Logistic comparisons pass. No frozen inputs, old outputs or manuscript values were overwritten.

## Scientific and implementation boundaries

No new candidates, labels, parameter bounds, model selection rules or published results are introduced by prediction/table/Gate replay. Fitting outputs are written separately and do not replace frozen publication parameters. Coefficients 18, 0.25, 0.4 and 2.5 retain their disclosed fixed empirical status. This package does not supply a newly discovered physical derivation or independent coefficient estimate.

The complete old command set is not restored: the deleted `guard-smoke` implementation, original dynamic-input guard adapter, original M1 perturbation generator, original Logistic initialization code and coefficient-sensitivity executable are unavailable. The archived sensitivity results and protocols remain in `inputs/`. Dynamic challenge, half-flow, historical timing and all historical raw trajectories are not fully re-integrated by this wrapper. M3 and M3B have their own complete packages in the parent delivery directory.

Third-party notices and preserved bsm2-python license/authorship files are under `legacy/`. No public repository, DOI or author-code license is created by this local reconstruction.
