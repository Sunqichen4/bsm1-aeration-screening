"""Analyze already-frozen M3 predictions against independently audited labels.

No fitting, candidate selection, gate adjustment, or historical-test loading is
performed here.  Boundaries mean first compliant *sampled candidate*, without
assuming monotonicity or identifying a continuous optimum.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import fresh_reference as fr


MODEL_NAMES = ("full_calibrated", "full_fixed_ratio", "no_feedback", "poly_ridge", "rbf_kernel_ridge")
STRATA = ("all", "coarse", "prediction_enriched")
BOUNDARY_DEFINITION = "First compliant candidate in the sampled KLa set; differences are model minus reference, not a continuous optimum."


def strict_bool_series(values: pd.Series, label: str) -> pd.Series:
    """Parse actual booleans or true/false/1/0 tokens, never truthy strings."""
    result = []
    for index, value in values.items():
        if isinstance(value, (bool, np.bool_)):
            parsed = bool(value)
        elif isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            parsed = value.strip().lower() in {"true", "1"}
        elif isinstance(value, (int, np.integer)) and value in (0, 1):
            parsed = bool(value)
        else:
            raise ValueError(f"{label} has an invalid boolean at row {index}: {value!r}")
        result.append(parsed)
    return pd.Series(result, index=values.index, dtype=bool)


def _required(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"{label} must have unique DataFrame column names")
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} must not be empty")


def _identities(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        if frame[column].isna().any() or not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{label}.{column} needs nonempty string identities")


def _finite_numeric(values: pd.Series, label: str, *, nonnegative: bool = False) -> pd.Series:
    try:
        converted = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(converted).all() or (nonnegative and (converted < 0).any()):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return converted


def join_evaluation(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    expected_models: Sequence[str] | None = None,
    nh4_limit: float = fr.NH4_LIMIT,
) -> pd.DataFrame:
    """Pure checked aggregation input; each model must cover every candidate."""
    _required(candidates, {"candidate_id", "scenario", "kla", "grid_kind"}, "candidates")
    _required(predictions, {"model", "candidate_id", "scenario", "predicted_peak", "raw_compliant", "recheck", "grid_kind"}, "predictions")
    _required(references, {"candidate_id", "SNH_max_eval", "SNH_zero_violation", "status"}, "references")
    candidates, predictions, references = (frame.copy() for frame in (candidates, predictions, references))
    _identities(candidates, ("candidate_id", "scenario"), "candidates")
    _identities(predictions, ("candidate_id", "scenario", "model"), "predictions")
    _identities(references, ("candidate_id",), "references")
    if candidates.candidate_id.duplicated().any() or references.candidate_id.duplicated().any():
        raise ValueError("Candidate and reference IDs must be unique")
    if predictions.duplicated(["model", "candidate_id"]).any():
        raise ValueError("Predictions contain duplicate model/candidate IDs")
    expected_models = tuple(sorted(predictions.model.unique())) if expected_models is None else tuple(expected_models)
    models = set(expected_models)
    if not models or len(models) != len(expected_models) or set(predictions.model) != models:
        raise ValueError("Predicted model names must exactly cover the prescribed models")
    ids = set(candidates.candidate_id)
    if set(references.candidate_id) != ids:
        raise ValueError("Reference coverage must exactly match frozen candidates")
    for name, group in predictions.groupby("model", sort=False):
        if set(group.candidate_id) != ids:
            raise ValueError(f"Prediction coverage for {name} must exactly match frozen candidates")
    if not references.status.eq("success").all():
        raise ValueError("Every reference status must be success")
    if not candidates.grid_kind.isin(STRATA[1:]).all():
        raise ValueError("Candidate grid_kind must be coarse or prediction_enriched")

    candidates["kla"] = _finite_numeric(candidates.kla, "candidate KLa", nonnegative=True)
    if candidates.duplicated(["scenario", "kla"]).any():
        raise ValueError("Candidates contain duplicate scenario/KLa coordinates")
    predictions["predicted_peak"] = _finite_numeric(predictions.predicted_peak, "predicted_peak", nonnegative=True)
    references["SNH_max_eval"] = _finite_numeric(references.SNH_max_eval, "SNH_max_eval", nonnegative=True)
    predictions["raw_compliant"] = strict_bool_series(predictions.raw_compliant, "raw_compliant")
    predictions["recheck"] = strict_bool_series(predictions.recheck, "recheck")
    references["SNH_zero_violation"] = strict_bool_series(references.SNH_zero_violation, "SNH_zero_violation")
    if not np.array_equal(predictions.raw_compliant.to_numpy(), predictions.predicted_peak.to_numpy() <= nh4_limit):
        raise ValueError("Frozen raw_compliant disagrees with predicted_peak <= NH4 limit")
    if not np.array_equal(references.SNH_zero_violation.to_numpy(), references.SNH_max_eval.to_numpy() <= nh4_limit):
        raise ValueError("Reference compliance disagrees with SNH_max_eval <= NH4 limit")

    # Candidate coordinates remain authoritative; reject a mismatched scenario,
    # stratum, or KLa instead of silently accepting suffix columns after merge.
    canonical = candidates.set_index("candidate_id")
    for label, frame in (("predictions", predictions), ("references", references)):
        for column in ("scenario", "grid_kind", "kla"):
            if column not in frame:
                continue
            actual = frame[column].to_numpy()
            expected = frame.candidate_id.map(canonical[column]).to_numpy()
            if column == "kla":
                actual = _finite_numeric(frame[column], label + ".kla", nonnegative=True).to_numpy()
            if not np.array_equal(actual, expected):
                raise ValueError(f"{label}.{column} disagrees with frozen candidate coordinates")
            frame[column] = expected

    # Keep the full frozen prediction record for traceability, while joining
    # only authoritative reference fields and canonical candidate coordinates.
    extra_coordinates = [column for column in candidates if column not in predictions and column != "candidate_id"]
    joined = predictions.merge(candidates[["candidate_id"] + extra_coordinates], on="candidate_id", validate="many_to_one")
    joined = joined.merge(references[["candidate_id", "SNH_max_eval", "SNH_zero_violation", "status"]], on="candidate_id", validate="many_to_one")
    joined["reference_compliant"] = joined.SNH_zero_violation
    joined["final_compliant"] = np.where(joined.recheck, joined.reference_compliant, joined.raw_compliant).astype(bool)
    joined["peak_error"] = joined.predicted_peak - joined.SNH_max_eval
    joined["raw_false_release"] = joined.raw_compliant & ~joined.reference_compliant
    joined["raw_false_rejection"] = ~joined.raw_compliant & joined.reference_compliant
    joined["gate_corrected_error"] = joined.recheck & (joined.raw_compliant != joined.reference_compliant)
    joined["gate_missed_error"] = ~joined.recheck & (joined.raw_compliant != joined.reference_compliant)
    joined["final_false_release"] = joined.final_compliant & ~joined.reference_compliant
    joined["final_false_rejection"] = ~joined.final_compliant & joined.reference_compliant
    return joined.sort_values(["model", "scenario", "kla", "candidate_id"]).reset_index(drop=True)


def _strata(joined: pd.DataFrame):
    yield "all", joined
    for name in STRATA[1:]:
        subset = joined.loc[joined.grid_kind.eq(name)]
        if not subset.empty:
            yield name, subset


def aggregate_metrics(joined: pd.DataFrame, *, by_condition: bool = False) -> pd.DataFrame:
    """Calculate concentration and routing errors in each candidate stratum."""
    rows = []
    group_columns = ["model", "scenario"] if by_condition else ["model"]
    for stratum, subset in _strata(joined):
        for keys, group in subset.groupby(group_columns, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            raw_error = group.raw_false_release | group.raw_false_rejection
            row = {"stratum": stratum, **dict(zip(group_columns, keys))}
            row.update(
                n_candidates=int(len(group)),
                reference_compliant_count=int(group.reference_compliant.sum()),
                rmse=float(np.sqrt(np.mean(group.peak_error.to_numpy() ** 2))),
                mae=float(np.mean(np.abs(group.peak_error.to_numpy()))),
                mean_error=float(np.mean(group.peak_error.to_numpy())),
                raw_false_releases=int(group.raw_false_release.sum()),
                raw_false_rejections=int(group.raw_false_rejection.sum()),
                raw_errors=int(raw_error.sum()),
                gate_selected=int(group.recheck.sum()),
                gate_selected_fraction=float(group.recheck.mean()),
                gate_corrected_false_releases=int((group.recheck & group.raw_false_release).sum()),
                gate_corrected_false_rejections=int((group.recheck & group.raw_false_rejection).sum()),
                gate_corrected_errors=int(group.gate_corrected_error.sum()),
                gate_missed_false_releases=int((~group.recheck & group.raw_false_release).sum()),
                gate_missed_false_rejections=int((~group.recheck & group.raw_false_rejection).sum()),
                gate_missed_errors=int(group.gate_missed_error.sum()),
                gate_rechecks_of_correct_raw_predictions=int((group.recheck & ~raw_error).sum()),
                final_false_releases=int(group.final_false_release.sum()),
                final_false_rejections=int(group.final_false_rejection.sum()),
                final_errors=int((group.final_compliant != group.reference_compliant).sum()),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def boundary_table(joined: pd.DataFrame) -> pd.DataFrame:
    """First compliant sampled candidates; difference sign is model-reference."""
    rows = []
    for stratum, subset in _strata(joined):
        for (model, scenario), group in subset.groupby(["model", "scenario"], sort=True):
            row: dict[str, Any] = {"stratum": stratum, "model": model, "scenario": scenario, "n_candidates": int(len(group))}
            firsts = {}
            for kind, column in (("reference", "reference_compliant"), ("raw", "raw_compliant"), ("final", "final_compliant")):
                compliant = group.loc[group[column]].sort_values(["kla", "candidate_id"])
                first = compliant.iloc[0] if not compliant.empty else None
                firsts[kind] = float(first.kla) if first is not None else None
                row[f"{kind}_has_compliant_candidate"] = first is not None
                row[f"{kind}_first_compliant_kla"] = firsts[kind]
                row[f"{kind}_first_compliant_candidate_id"] = str(first.candidate_id) if first is not None else None
            for kind in ("raw", "final"):
                row[f"{kind}_minus_reference_kla"] = (
                    firsts[kind] - firsts["reference"]
                    if firsts[kind] is not None and firsts["reference"] is not None else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # Pandas handles numpy scalars and absent boundaries as JSON nulls.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def analyze(study_dir: Path) -> dict[str, Any]:
    """Audit fresh raw references, verify immutable predictions, and save tables."""
    study = Path(study_dir).resolve()
    freeze, freeze_sha, cases = fr.verify_freeze(study)
    frozen_paths = {entry["path"] for entry in freeze["files"]}
    if "frozen_predictions.csv" not in frozen_paths:
        raise ValueError("frozen_predictions.csv must be bound by freeze.json")
    audit = fr.audit_reference_outputs(study)
    if audit.get("passed") is not True:
        raise ValueError("Independent reference audit failed: " + json.dumps(audit.get("errors", []), ensure_ascii=False))
    if audit.get("freeze_sha256") != freeze_sha or audit.get("expected_count") != len(cases) or audit.get("successful_count") != len(cases) or audit.get("failed_count") != 0:
        raise ValueError("Reference audit identity/counts disagree with the current freeze")
    source_names = ("new_candidates.csv", "frozen_predictions.csv", "reference_summary.csv")
    source_hashes = {name: _sha(study / name) for name in source_names}
    candidates = pd.read_csv(study / "new_candidates.csv", float_precision="round_trip", dtype={"candidate_id": str, "scenario": str})
    predictions = pd.read_csv(study / "frozen_predictions.csv", float_precision="round_trip", dtype={"candidate_id": str, "scenario": str, "model": str, "raw_compliant": str, "recheck": str})
    references = pd.read_csv(study / "reference_summary.csv", float_precision="round_trip", dtype={"candidate_id": str, "SNH_zero_violation": str, "status": str})
    if set(candidates.candidate_id) != {case["candidate_id"] for case in cases}:
        raise ValueError("Frozen case identities disagree with candidate CSV")
    model_path = study / "outputs/development/models.json"
    if "outputs/development/models.json" in frozen_paths:
        saved_models = json.loads(model_path.read_text(encoding="utf-8-sig"))
        if not isinstance(saved_models, dict) or not saved_models:
            raise ValueError("Frozen models artifact must be a nonempty object")
        expected_models = sorted(saved_models)
        model_manifest_source = "frozen outputs/development/models.json keys"
    else:
        expected_models = sorted(predictions.model.unique())
        model_manifest_source = "hash-verified frozen_predictions.csv model names"
    joined = join_evaluation(predictions, references, candidates, expected_models=expected_models)
    overall = aggregate_metrics(joined)
    by_condition = aggregate_metrics(joined, by_condition=True)
    boundaries = boundary_table(joined)

    development = {}
    for name in ("development_summary.csv", "identifiability.json", "rho_profile.csv"):
        relative = "outputs/development/" + name
        path = study / relative
        if path.is_file():
            content = json.loads(path.read_text(encoding="utf-8-sig")) if path.suffix == ".json" else _json_records(pd.read_csv(path, float_precision="round_trip"))
            development[name] = {"path": relative, "sha256": _sha(path), "bound_by_freeze": relative in frozen_paths, "contents": content}
    # Detect changes between initial audit and writing derived analysis files.
    _, final_sha, _ = fr.verify_freeze(study)
    if final_sha != freeze_sha or any(_sha(study / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Frozen inputs or reference summary changed during analysis")
    summary = {
        "analysis_version": "M3B_audited_frozen_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha256": _sha(Path(__file__)),
        "freeze_sha256": freeze_sha,
        "source_sha256": source_hashes,
        "reference_audit": audit,
        "models": expected_models,
        "model_manifest_source": model_manifest_source,
        "model_comparison_roles": {
            name: (
                "historical M2 binary-supervision reference; different supervision; its steady crossing selects the separately reported enriched stratum"
                if name == "legacy_M2_binary" else "new continuous-target model"
            ) for name in expected_models
        },
        "candidate_count": int(len(candidates)),
        "condition_count": int(candidates.scenario.nunique()),
        "candidate_stratum_counts": {str(key): int(value) for key, value in candidates.grid_kind.value_counts().items()},
        "nh4_limit_mgN_L": fr.NH4_LIMIT,
        "peak_units": "mg N/L",
        "boundary_units": "KLa d^-1",
        "boundary_definition": BOUNDARY_DEFINITION,
        "gate_final_rule": "Use audited reference compliance for frozen recheck=True, otherwise retain frozen raw compliance.",
        "gate_cost_scope": "gate_selected counts simulated deployment rechecks; every candidate also has an independent scientific reference for evaluation.",
        "stratum_interpretation": "Fixed coarse and model-prediction-enriched strata are reported separately; the all stratum is their designed union, not a population sample.",
        "uncertainty_scope": "Descriptive results for the frozen conditions; no point-independent inferential or population guarantee.",
        "old_test_label_access_for_selection": False,
        "historical_input_only_duplicate_audit": True,
        "post_test_model_or_gate_adaptation": False,
        "metrics_overall": _json_records(overall),
        "development_context": development,
        "output_files": ["evaluation_joined.csv", "metrics_overall.csv", "metrics_by_condition.csv", "boundaries.csv", "analysis_summary.json"],
    }
    for name, frame in (("evaluation_joined.csv", joined), ("metrics_overall.csv", overall), ("metrics_by_condition.csv", by_condition), ("boundaries.csv", boundaries)):
        frame.to_csv(study / name, index=False, encoding="utf-8-sig")
    (study / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    summary = analyze(args.study_dir)
    print(json.dumps({key: summary[key] for key in ("freeze_sha256", "candidate_count", "condition_count", "models", "output_files")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
