# Aeration screening research code and data

Research materials accompanying **Decision-oriented aeration screening in activated sludge systems: comparing mechanism-inspired and statistical models with targeted BSM1 verification**.

This release candidate provides 106 supplementary CSV/JSON files, a versioned reconstructed M1/M2 executable core, and the complete retained M3/M3B scientific packages. See [RUN_STATUS.md](RUN_STATUS.md) for the exact distinction between executable workflows, checks completed in this release, and historical results.

This directory is prepared locally for publication. A public repository URL and an archived release DOI have not yet been assigned. Author manuscript drafts and reviewer correspondence are not included in this software/data repository.

## Contents

| Path | Purpose |
| --- | --- |
| `supplement_data/` | All 106 supplementary evidence files; historical records retain their identity |
| `ATTACHMENT_INDEX.csv` | Relative file paths, manuscript locations, sizes and SHA-256 hashes for the 106 evidence files |
| `m1_m2_reconstructed_20261103/` | Reconstructed M1/M2 interface, frozen inputs, retained numerical kernels, tests and executed check records |
| `model_revision_20261016/` | M3 source, frozen models, saved native BSM1 trajectories, analyses and tests |
| `boundary_validation_20261016/` | M3B source, frozen boundary candidates, saved native BSM1 trajectories, analyses and tests |
| `coefficient_evidence/` | Provenance audit of the empirical settings 18, 0.25, 0.4 and 2.5 |
| `verification/` | Read-only M3/M3B replay entry point and prior delivery check records |
| `runtime_guard_reconstructed_20261104/` | New 22-column validator and router, three examples and ten tests; unsupported inputs are routed without executing BSM1 |
| `run_checks.py` | Runs release checks and writes logs to a new output directory |
| `release_validation.json` | Results of eleven successful check groups in a newly installed isolated Python 3.11.15 environment on 2026-11-04 |
| `RELEASE_MANIFEST.json` | Hashes of this publication package; distinct from historical scientific freeze files |
| `RELEASE_CONTENTS.csv` | Directory inventory |
| `CITATION.cff` | Citation metadata drawn from the current author manuscript |
| `LICENSE_STATUS.md` | Author-code license status and locations of third-party notices |

## Quick start

Use a Python **3.11** interpreter and the pinned dependencies. On 2026-11-04, this relocated package passed eleven check groups on Windows x86-64 with a newly installed isolated Python 3.11.15 environment. Lock-file installation and `pip check` succeeded. The executed checks include process/Gate replay, M3/M3B saved-result audits, seventeen software tests and one fresh native constant-condition example. See `release_validation.json` and `verification/release_20261104/`. Full optimizer and historical native study batches were not repeated in this release check.

From this repository root in PowerShell:

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r m1_m2_reconstructed_20261103\requirements-lock.txt
.venv\Scripts\python.exe verify_release.py
.venv\Scripts\python.exe m1_m2_reconstructed_20261103\reproduce.py verify --output runs\M1_M2_integrity
.venv\Scripts\python.exe m1_m2_reconstructed_20261103\reproduce.py reproduce --output runs\M1_M2_tables
.venv\Scripts\python.exe -X utf8 -B verification\verify_m3_m3b.py --output runs\M3_M3B_replay
```

The combined command `python -B run_checks.py --output runs/check_01` runs integrity checks, table reproduction, software tests and M3/M3B replay. Add `--extended` for M1/M2 process predictions, M1 Gate replay and one fresh constant-condition native BSM1 example. It uses the Python interpreter that launches it; run it with the configured Python 3.11 environment. Every combined-run output directory must be new.

`verify_release.py` requires only the Python standard library and checks the supplied-file hashes. It does not run scientific models. Every scientific output path above must be new or empty; the M3/M3B replay requires a new path. Keep new outputs in `runs/` so frozen scientific evidence is preserved.

The M1/M2 package README documents prediction, Gate replay, refitting, a fresh one-case native BSM1 reference demonstration, and tests. The M3 and M3B READMEs document their original full scientific workflows. A full rerun of long native BSM1 integrations should use a separate version directory; the quick-start replay checks saved trajectories and frozen results and performs no scientific refit or new native integration.

## Reproduction scope

M1/M2 is explicitly versioned `reconstructed-20261103-v1`. It recovers an executable core from retained sources and inputs; it does not claim to recover the deleted historical portable release. The dynamic-input guard adapter, original M1 perturbation generator, original M2 Logistic starts and coefficient-sensitivity executor were not recovered. Their surviving protocols/results remain historical evidence. The newly reconstructed M1/Logistic initialization protocols and their numerical differences are disclosed in that package.

The separate `runtime_guard_reconstructed_20261104/` is a new input-checking and routing implementation. It checks all 22 input columns and the declared units, validity, load family, temperature and domains. It reports `invalid_input`, `in_scope`, or `fallback_required`, with no final compliance label. It does not execute a cold/dynamic BSM1 fallback. See its README and tested examples.

M3/M3B retain code for full study execution as well as saved primary and half-step raw trajectories. The read-only verification entry point audits those trajectories, replays 1,260 predictions and Gate fields, and independently reaggregates the metrics. This check is different from refitting all models and repeating all native long integrations.

The empirical constants 18, 0.25, 0.4 and 2.5 are traceable to historical code. An independent derivation or original estimation record was not located. The coefficient audit does not assign them a new physical basis. Conclusions remain specific to the tested empirical configuration and reported sensitivity analyses.

## Historical paths and file integrity

Executable entry points locate inputs relative to their package. Some immutable historical JSON, CSV, source snapshots and logs contain original local paths. These strings are provenance records, not a request to create the old drive layout. Scientific snapshots and internal freeze/source manifests were preserved byte for byte. The two M3/M3B README command examples were made portable for this publication package; those documentation-only changes are listed in `RELEASE_NOTES.md`.

Git line-ending conversion is disabled for tracked files because changing line endings would invalidate byte-level source/evidence hashes. Keep `.gitattributes` in the repository. Use `verify_release.py` after download or clone. New run outputs and caches are not part of the supplied-file manifest.

## Citation and permissions

Use the associated manuscript as the scientific citation; `CITATION.cff` records its current title and author names. Add the real repository/release URL only after publication. No DOI is claimed here. Author-code licensing remains to be selected by the authors; see `LICENSE_STATUS.md`. Third-party license/authorship files under each package's `legacy/` directory are retained and continue to apply.
