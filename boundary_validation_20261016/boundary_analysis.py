"""Post-freeze boundary diagnostics for M3B; no fitting or candidate selection.

All statements describe the observed frozen candidate set.  Adjacency is defined
within each separately reported stratum and never assumes a monotone response.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".runtime_cache" / "production"))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
import analyze_results as standard

LIMIT = 4.0
WINDOWS = (0.25, 0.5, 1.0)
STRATA = ("coarse", "prediction_enriched", "all")
MODELS = ("full_calibrated", "full_fixed_ratio", "no_feedback", "poly_ridge",
          "rbf_kernel_ridge", "legacy_M2_binary")
BOOL_COLUMNS = ("raw_compliant", "recheck", "reference_compliant", "final_compliant",
                "raw_false_release", "raw_false_rejection", "gate_corrected_error",
                "gate_missed_error", "final_false_release", "final_false_rejection")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratify(frame: pd.DataFrame, stratum: str) -> pd.DataFrame:
    if stratum not in STRATA:
        raise ValueError("Unknown candidate stratum")
    return frame if stratum == "all" else frame.loc[frame.grid_kind.eq(stratum)]


def candidate_truth(candidates: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Build one row per candidate; no model-dependent duplicate weighting."""
    required = {"candidate_id", "scenario", "kla", "effective_kla", "grid_kind"}
    if not required <= set(candidates):
        raise ValueError("Missing frozen candidate coordinates")
    if candidates.candidate_id.duplicated().any() or references.candidate_id.duplicated().any():
        raise ValueError("Candidate/reference identities must be unique")
    if set(candidates.candidate_id) != set(references.candidate_id):
        raise ValueError("Reference and candidate coverage differ")
    if not references.status.eq("success").all():
        raise ValueError("Only complete successful references can be analyzed")
    values = references[["candidate_id", "SNH_max_eval", "SNH_zero_violation"]].copy()
    values["reference_compliant"] = standard.strict_bool_series(values.pop("SNH_zero_violation"), "reference compliance")
    if not np.isfinite(values.SNH_max_eval.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite reference peak")
    if not np.array_equal(values.reference_compliant.to_numpy(), values.SNH_max_eval.to_numpy() <= LIMIT):
        raise ValueError("Reference peak and compliance disagree")
    result = candidates.merge(values, on="candidate_id", validate="one_to_one")
    if result.duplicated(["scenario", "effective_kla"]).any():
        raise ValueError("Repeated effective-KLa coordinates within a condition")
    result["reference_margin"] = result.SNH_max_eval - LIMIT
    return result.sort_values(["scenario", "effective_kla", "candidate_id"]).reset_index(drop=True)


def coverage_tables(truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, changes = [], []
    for stratum in STRATA:
        subset = stratify(truth, stratum)
        for scenario in sorted(truth.scenario.unique()):
            group = subset.loc[subset.scenario.eq(scenario)].sort_values(["effective_kla", "candidate_id"])
            safe = group.loc[group.reference_compliant]
            unsafe = group.loc[~group.reference_compliant]
            row = dict(stratum=stratum, scenario=scenario, n_candidates=len(group),
                       compliant_count=len(safe), noncompliant_count=len(unsafe),
                       has_both_sides=bool(len(safe) and len(unsafe)))
            for side, part in (("compliant", safe), ("noncompliant", unsafe)):
                nearest = part.iloc[np.argmin(np.abs(part.reference_margin.to_numpy()))] if len(part) else None
                row.update({f"nearest_{side}_distance_mgN_L": abs(float(nearest.reference_margin)) if nearest is not None else None,
                            f"nearest_{side}_candidate_id": str(nearest.candidate_id) if nearest is not None else None,
                            f"nearest_{side}_peak_mgN_L": float(nearest.SNH_max_eval) if nearest is not None else None,
                            f"nearest_{side}_effective_kla": float(nearest.effective_kla) if nearest is not None else None})
            for window in WINDOWS:
                suffix = str(window).replace(".", "p")
                near = group.loc[group.reference_margin.abs() <= window]
                row[f"within_{suffix}_count"] = len(near)
                row[f"within_{suffix}_compliant_count"] = int(near.reference_compliant.sum())
                row[f"within_{suffix}_noncompliant_count"] = int((~near.reference_compliant).sum())
            local_changes = []
            records = group.to_dict(orient="records")
            for left, right in zip(records, records[1:]):
                if left["reference_compliant"] == right["reference_compliant"]:
                    continue
                change = dict(stratum=stratum, scenario=scenario,
                              left_candidate_id=left["candidate_id"], right_candidate_id=right["candidate_id"],
                              left_effective_kla=float(left["effective_kla"]), right_effective_kla=float(right["effective_kla"]),
                              left_kla=float(left["kla"]), right_kla=float(right["kla"]),
                              left_peak_mgN_L=float(left["SNH_max_eval"]), right_peak_mgN_L=float(right["SNH_max_eval"]),
                              left_compliant=bool(left["reference_compliant"]), right_compliant=bool(right["reference_compliant"]),
                              effective_kla_gap=float(right["effective_kla"] - left["effective_kla"]),
                              direction="compliant_to_noncompliant" if left["reference_compliant"] else "noncompliant_to_compliant",
                              strict_sign_change=bool(left["reference_margin"] * right["reference_margin"] < 0.))
                local_changes.append(change)
            changes.extend(local_changes)
            row["adjacent_label_change_count"] = len(local_changes)
            row["adjacent_strict_sign_change_count"] = sum(x["strict_sign_change"] for x in local_changes)
            row["noncompliant_to_compliant_count"] = sum(x["direction"] == "noncompliant_to_compliant" for x in local_changes)
            row["compliant_to_noncompliant_count"] = sum(x["direction"] == "compliant_to_noncompliant" for x in local_changes)
            row["exact_threshold_count"] = int(group.reference_margin.eq(0.).sum())
            row["smallest_observed_label_change_z_gap"] = min((x["effective_kla_gap"] for x in local_changes), default=None)
            rows.append(row)
    change_columns = ("stratum", "scenario", "left_candidate_id", "right_candidate_id",
                      "left_effective_kla", "right_effective_kla", "left_kla", "right_kla",
                      "left_peak_mgN_L", "right_peak_mgN_L", "left_compliant", "right_compliant",
                      "effective_kla_gap", "direction", "strict_sign_change")
    return pd.DataFrame(rows), pd.DataFrame(changes, columns=change_columns)


def metrics(group: pd.DataFrame) -> dict:
    n = len(group)
    if n:
        error = group.predicted_peak.to_numpy() - group.SNH_max_eval.to_numpy()
        return dict(n_candidates=n, reference_compliant_count=int(group.reference_compliant.sum()),
                    reference_noncompliant_count=int((~group.reference_compliant).sum()),
                    rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                    mean_error=float(np.mean(error)), raw_false_releases=int(group.raw_false_release.sum()),
                    raw_false_rejections=int(group.raw_false_rejection.sum()), gate_selected=int(group.recheck.sum()),
                    gate_selected_fraction=float(group.recheck.mean()),
                    gate_corrected_errors=int(group.gate_corrected_error.sum()),
                    gate_missed_errors=int(group.gate_missed_error.sum()),
                    final_false_releases=int(group.final_false_release.sum()),
                    final_false_rejections=int(group.final_false_rejection.sum()))
    return dict(n_candidates=0, reference_compliant_count=0, reference_noncompliant_count=0,
                rmse=None, mae=None, mean_error=None, raw_false_releases=0, raw_false_rejections=0,
                gate_selected=0, gate_selected_fraction=None, gate_corrected_errors=0,
                gate_missed_errors=0, final_false_releases=0, final_false_rejections=0)


def near_threshold_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum in STRATA:
        subset = stratify(joined, stratum)
        for window in WINDOWS:
            near = subset.loc[(subset.SNH_max_eval - LIMIT).abs() <= window]
            for name in MODELS:
                rows.append(dict(stratum=stratum, window_mgN_L=window, model=name,
                                 **metrics(near.loc[near.model.eq(name)])))
    return pd.DataFrame(rows)


def matched_pair_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    """Descriptive paired five-parameter comparison, full minus no-feedback."""
    rows = []
    for stratum in STRATA:
        subset = stratify(joined, stratum)
        for scenario in sorted(joined.scenario.unique()):
            group = subset.loc[subset.scenario.eq(scenario)]
            full = group.loc[group.model.eq("full_fixed_ratio")].sort_values("candidate_id")
            none = group.loc[group.model.eq("no_feedback")].sort_values("candidate_id")
            if full.candidate_id.tolist() != none.candidate_id.tolist():
                raise ValueError("Matched pair must cover identical candidates")
            if not np.array_equal(full.SNH_max_eval.to_numpy(), none.SNH_max_eval.to_numpy()):
                raise ValueError("Matched pair reference values disagree")
            a, b = metrics(full), metrics(none)
            row = dict(stratum=stratum, scenario=scenario, n_candidates=len(full),
                       difference_definition="full_fixed_ratio_minus_no_feedback")
            for key in a:
                if key == "n_candidates":
                    continue
                row[f"full_fixed_ratio_{key}"] = a[key]
                row[f"no_feedback_{key}"] = b[key]
                row[f"difference_{key}"] = a[key] - b[key] if a[key] is not None and b[key] is not None else None
            absolute_full = np.abs(full.predicted_peak.to_numpy() - full.SNH_max_eval.to_numpy())
            absolute_none = np.abs(none.predicted_peak.to_numpy() - none.SNH_max_eval.to_numpy())
            row["full_lower_absolute_error_count"] = int((absolute_full < absolute_none).sum())
            row["no_feedback_lower_absolute_error_count"] = int((absolute_none < absolute_full).sum())
            row["equal_absolute_error_count"] = int((absolute_full == absolute_none).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_boundary(study_dir: Path = ROOT) -> dict:
    study = Path(study_dir).resolve()
    # The existing analyzer requires matching freeze, full coverage and a fresh
    # independent raw audit.  It never generates labels or adjusts a model.
    standard_summary = standard.analyze(study)
    sources = ("new_candidates.csv", "reference_summary.csv", "evaluation_joined.csv",
               "freeze.json", "outputs/development/models.json", "outputs/development/gates.json",
               "metrics_overall.csv")
    source_hashes = {name: sha(study / name) for name in sources}
    candidates, references, joined = (read_csv(study / name) for name in sources[:3])
    if set(joined.model) != set(MODELS):
        raise ValueError("Expected the six frozen M3 models")
    for column in BOOL_COLUMNS:
        joined[column] = standard.strict_bool_series(joined[column], column)
    truth = candidate_truth(candidates, references)
    coverage, changes = coverage_tables(truth)
    near = near_threshold_metrics(joined)
    matched = matched_pair_metrics(joined)
    if any(sha(study / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Source artifacts changed during boundary analysis")
    outputs = {"coverage_by_condition.csv": coverage, "observed_sign_changes.csv": changes,
               "near_threshold_metrics.csv": near, "matched_pair_by_condition.csv": matched,
               "boundary_candidate_truth.csv": truth}
    for filename, frame in outputs.items():
        frame.to_csv(study / filename, index=False, encoding="utf-8-sig")
    coarse = coverage.loc[coverage.stratum.eq("coarse")]
    summary = dict(
        version="M3B_descriptive_boundary_analysis_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
        analysis_source_sha256=sha(Path(__file__)), freeze_sha256=standard_summary["freeze_sha256"],
        source_sha256=source_hashes, output_sha256={name: sha(study / name) for name in outputs},
        n_candidates=len(truth), n_conditions=int(truth.scenario.nunique()),
        coarse_conditions_with_both_sides=int(coarse.has_both_sides.sum()),
        coarse_conditions_missing_a_side=coarse.loc[~coarse.has_both_sides, "scenario"].tolist(),
        windows_mgN_L=list(WINDOWS), compliance_definition="reference_peak <= 4 mg N/L",
        strict_sign_change_definition="Adjacent sampled reference margins have a strictly negative product; zero is not a strict sign.",
        adjacency_definition="Consecutive effective-KLa samples within each reported stratum, without monotonicity or interpolation.",
        nearest_distance_definition="Absolute observed peak-minus-4 distance; absent sides yield null, never zero.",
        matched_pair_definition="Fixed-ratio full and no-feedback both have five fitted parameters; differences are full minus no-feedback, descriptive only.",
        inference_limits=["Strata have different sampling rules and are reported separately.",
                          "Near-threshold subsets are defined by observed reference values; counts and errors are descriptive.",
                          "Coverage and sampled sign changes do not identify a continuous optimum or prove monotonicity.",
                          "No labels, parameters, Gates, conditions or candidates are changed by this analysis."])
    (study / "boundary_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze_boundary(args.study_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
