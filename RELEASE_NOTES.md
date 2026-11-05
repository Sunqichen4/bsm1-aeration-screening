# Release preparation notes

This 2026-11-04 local staging directory was copied from the 2026-11-03 delivery. It is not an additional scientific experiment or a new model version.

## Changes for publication

- Added a repository-level English README, Chinese usage guide, executable-scope table, citation metadata, license-status notice, byte-preserving Git attributes and cache/run exclusions.
- Retained all 106 supplementary CSV/JSON evidence files unchanged. Author manuscript drafts and reviewer correspondence remain local and are not included in this software/data repository.
- Retained all six scientific/evidence directories, including historical logs, failed checks, internal freeze files, raw trajectories, third-party notices and license records. Runtime caches and temporary Word lock files are excluded.
- Changed environment and location examples in `model_revision_20261016/README.md` and `boundary_validation_20261016/README.md` to generic Python and relative workspace commands. This is documentation only; scientific source, inputs, frozen outputs and internal scientific manifests were not edited.
- Excluded local editorial change ledgers, delivery-copy audit files, old top-level delivery instructions and the old delivery manifest. A new `RELEASE_MANIFEST.json` records publication-package hashes.
- Added a separately versioned 22-column runtime input validator/router with three examples and ten tests. It identifies unsupported inputs but does not execute cold/dynamic BSM1 or change publication results.
- Added `run_checks.py` for a combined check with a new output directory; `--extended` additionally executes frozen-parameter surrogate predictions, Gate replay and one fresh constant-condition native reference example.

## Identity and scope

M1/M2 retains version `reconstructed-20261103-v1`. M3 and M3B retain their existing scientific identities. No measured results, frozen model parameters, candidate sets, Gate rules or reference labels were changed. No new DOI, public repository URL or author-code license is asserted.

`tools/build_release_manifest.py` regenerates the publication manifest and directory inventory after an intentional packaging change. It does not modify internal scientific freeze or source manifests. New runs belong in ignored `runs/` and should not be added to the supplied-file manifest.
