# Third-party notices and publication status

## bsm2-python

Runtime dependency: bsm2-python 0.0.16. Installed distribution metadata identifies BSD-3-Clause and project source https://github.com/FAU-EVT/bsm2-python and documentation https://bsm2-python.readthedocs.io/ . The actual installed distribution LICENSE and AUTHORS.md are preserved in `third_party/`. The package implementation is obtained through normal dependency installation, rather than silently copied without attribution.

Native dry/rain influent files are accessed from that installed distribution at runtime. Their inclusion in the installed package establishes provenance, not independent confirmation that the manuscript authors own all rights in the underlying benchmark data. Before a public dataset release, authors should confirm the benchmark data attribution and any additional source conditions. This portable package does not redistribute separate copies of the weather CSVs.

## Other dependencies

Pinned runtime dependencies and transitive package versions are listed in `requirements-lock.txt`. They are installed from their distributors; each retains its own license. No umbrella license in this directory overrides their terms. Source packages/distribution metadata should accompany any binary redistribution.

## Author research code and derived evidence

`core/`, the copied `vendor/original_project/src/aerbench/` research interfaces, custom CLI, and derived/frozen evidence are supplied locally for the author's review and reproduction. A research-software license for this work has not yet been selected by the author. **AUTHOR_INPUT_NEEDED:** choose the research-code license, confirm all coauthors/owners authorize public release, confirm benchmark input/data reuse terms, and decide whether/how to release complete raw trajectories. No public upload has been performed by this packaging task.

Original source and copied-source SHA256 pairs are recorded in `source_manifest.json`; path adaptations retain the scientific equations and numeric workflow. This is a core reproduction release candidate, not a claim of a fully archived, publicly licensed research repository.
