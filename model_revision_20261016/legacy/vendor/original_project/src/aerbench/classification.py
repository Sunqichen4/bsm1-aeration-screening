from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import normalize_scenario_name
from .provenance import write_dataframe_with_hash, write_json


CLASSIFICATIONS = (
    "true_safe",
    "true_reject",
    "false_safe",
    "conservative_reject",
)


@dataclass(frozen=True)
class ThresholdResult:
    status: str
    first_compliant_grid_point: float | None
    interval_lower: float | None
    interval_upper: float | None
    grid_step: float | None
    persistent_points: int


def classify_candidate(surrogate_compliant: bool, bsm1_compliant: bool) -> str:
    if surrogate_compliant and bsm1_compliant:
        return "true_safe"
    if not surrogate_compliant and not bsm1_compliant:
        return "true_reject"
    if surrogate_compliant and not bsm1_compliant:
        return "false_safe"
    return "conservative_reject"


def wilson_interval(successes: int, trials: int, confidence_level: float = 0.95) -> tuple[float, float]:
    n = int(trials)
    x = int(successes)
    if n < 0 or x < 0 or x > n:
        raise ValueError("Wilson interval requires 0 <= successes <= trials.")
    if n == 0:
        return np.nan, np.nan
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one.")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    p = x / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    valid = normalized.isin(["true", "false", "1", "0"])
    if not bool(valid.all()):
        raise ValueError(f"Cannot parse Boolean values: {sorted(normalized[~valid].unique())}")
    return normalized.isin(["true", "1"])


def persistent_threshold(
    frame: pd.DataFrame,
    *,
    kla_col: str,
    compliant_col: str,
    persistent_points: int = 3,
) -> ThresholdResult:
    if persistent_points < 1:
        raise ValueError("persistent_points must be at least one.")
    ordered = frame.sort_values(kla_col).reset_index(drop=True)
    kla = pd.to_numeric(ordered[kla_col], errors="raise").to_numpy(dtype=float)
    compliant = _as_bool(ordered[compliant_col]).to_numpy(dtype=bool)
    if len(kla) == 0:
        return ThresholdResult("unresolved", None, None, None, None, persistent_points)
    if np.any(np.diff(kla) <= 0.0):
        raise ValueError("KLa grid must be strictly increasing within a scenario.")
    step = float(np.median(np.diff(kla))) if len(kla) > 1 else None
    for index in range(len(kla)):
        stop = index + persistent_points
        if stop <= len(kla) and bool(np.all(compliant[index:stop])):
            lower = float(kla[index - 1]) if index > 0 else float(kla[index])
            return ThresholdResult(
                "resolved",
                float(kla[index]),
                lower,
                float(kla[index]),
                step,
                persistent_points,
            )
    return ThresholdResult("unresolved", None, None, None, step, persistent_points)


def standardize_model_results(frame: pd.DataFrame, model_prefix: str, nh4_threshold: float) -> pd.DataFrame:
    aliases = {
        "candidate_id": "candidate_id",
        "scenario_candidate_index": "scenario_candidate_index",
        "scenario": "scenario",
        "kla": "commanded_kla_d-1",
        "commanded_kla_d-1": "commanded_kla_d-1",
        "effective_kla": "effective_kla_d-1",
        "effective_kla_d-1": "effective_kla_d-1",
        "SNH_mean_eval": f"{model_prefix}_mean_snh_mgN_L",
        "SNH_p95_eval": f"{model_prefix}_p95_snh_mgN_L",
        "SNH_max_eval": f"{model_prefix}_max_snh_mgN_L",
        "SNH_violation_hours_gt_limit": f"{model_prefix}_violation_h",
        "SNH_violation_ratio_gt_limit": f"{model_prefix}_violation_fraction",
        "runtime_seconds": f"{model_prefix}_runtime_seconds",
        "commanded_aeration_index": "commanded_aeration_index",
    }
    missing = [column for column in ["scenario", "SNH_mean_eval", "SNH_p95_eval", "SNH_violation_hours_gt_limit"] if column not in frame.columns]
    if "kla" not in frame.columns and "commanded_kla_d-1" not in frame.columns:
        missing.append("kla or commanded_kla_d-1")
    if missing:
        raise ValueError(f"Missing required {model_prefix} columns: {missing}")
    selected = [column for column in aliases if column in frame.columns]
    out = frame[selected].rename(columns={column: aliases[column] for column in selected}).copy()
    out["scenario"] = out["scenario"].map(normalize_scenario_name)
    if "effective_kla_d-1" not in out:
        out["effective_kla_d-1"] = out["commanded_kla_d-1"]
    violation = pd.to_numeric(out[f"{model_prefix}_violation_h"], errors="raise")
    out[f"{model_prefix}_zero_violation"] = violation <= 1e-12
    out[f"{model_prefix}_satisfies_study_benchmark"] = out[f"{model_prefix}_zero_violation"]
    out[f"{model_prefix}_nh4_benchmark_mgN_L"] = float(nh4_threshold)
    return out


def build_full_grid_classification(
    surrogate_results: pd.DataFrame,
    bsm1_results: pd.DataFrame,
    *,
    nh4_threshold: float = 4.0,
    persistent_points: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    surrogate = standardize_model_results(surrogate_results, "surrogate", nh4_threshold)
    bsm1 = standardize_model_results(bsm1_results, "bsm1", nh4_threshold)
    sort_key = ["scenario", "commanded_kla_d-1"]

    # Full-grid runs carry a stable candidate_id generated before either model
    # is executed. Match on that identifier rather than on an exact floating-
    # point KLa value: CSV round-tripping can change the last decimal place
    # (for example, ...38554 versus ...3856) even though both rows represent
    # the same planned candidate.
    match_by_candidate_id = "candidate_id" in surrogate.columns and "candidate_id" in bsm1.columns
    if match_by_candidate_id:
        for label, frame in (("surrogate", surrogate), ("BSM1", bsm1)):
            candidate_ids = frame["candidate_id"].astype(str).str.strip()
            if candidate_ids.eq("").any() or candidate_ids.str.lower().eq("nan").any():
                raise ValueError(f"{label} grid contains a missing candidate_id.")
            if candidate_ids.duplicated().any():
                duplicates = sorted(candidate_ids[candidate_ids.duplicated(keep=False)].unique())
                raise ValueError(f"{label} grid contains duplicate candidate_id values: {duplicates[:20]}")
            frame["candidate_id"] = candidate_ids
        merge_key = ["candidate_id"]
    else:
        merge_key = sort_key

    merged = pd.merge(
        surrogate,
        bsm1,
        on=merge_key,
        how="outer",
        validate="one_to_one",
        suffixes=("_surrogate", "_bsm1"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        unmatched = merged.loc[~merged["_merge"].eq("both"), merge_key + ["_merge"]]
        raise ValueError(f"Surrogate and BSM1 grids do not match:\n{unmatched.head(20).to_string(index=False)}")
    merged = merged.drop(columns="_merge")

    if match_by_candidate_id:
        surrogate_scenario = merged["scenario_surrogate"].map(normalize_scenario_name)
        bsm1_scenario = merged["scenario_bsm1"].map(normalize_scenario_name)
        if not surrogate_scenario.eq(bsm1_scenario).all():
            bad = merged.loc[
                ~surrogate_scenario.eq(bsm1_scenario),
                ["candidate_id", "scenario_surrogate", "scenario_bsm1"],
            ]
            raise ValueError(f"Candidate IDs map to different scenarios:\n{bad.head(20).to_string(index=False)}")
        merged["scenario"] = bsm1_scenario
        merged = merged.drop(columns=["scenario_surrogate", "scenario_bsm1"])

        surrogate_kla = pd.to_numeric(merged["commanded_kla_d-1_surrogate"], errors="raise")
        bsm1_kla = pd.to_numeric(merged["commanded_kla_d-1_bsm1"], errors="raise")
        kla_matches = np.isclose(surrogate_kla, bsm1_kla, rtol=0.0, atol=1e-9)
        if not bool(np.all(kla_matches)):
            bad = merged.loc[
                ~kla_matches,
                ["candidate_id", "commanded_kla_d-1_surrogate", "commanded_kla_d-1_bsm1"],
            ]
            raise ValueError(f"Candidate IDs map to materially different KLa values:\n{bad.head(20).to_string(index=False)}")
        merged["commanded_kla_d-1"] = surrogate_kla
        merged = merged.drop(columns=["commanded_kla_d-1_surrogate", "commanded_kla_d-1_bsm1"])

        if "scenario_candidate_index_surrogate" in merged.columns:
            surrogate_index = pd.to_numeric(merged["scenario_candidate_index_surrogate"], errors="raise")
            bsm1_index = pd.to_numeric(merged["scenario_candidate_index_bsm1"], errors="raise")
            if not surrogate_index.eq(bsm1_index).all():
                raise ValueError("Candidate IDs map to different scenario_candidate_index values.")
            merged["scenario_candidate_index"] = surrogate_index.astype(int)
            merged = merged.drop(
                columns=["scenario_candidate_index_surrogate", "scenario_candidate_index_bsm1"]
            )
    if "effective_kla_d-1_surrogate" in merged:
        merged["effective_kla_d-1"] = merged["effective_kla_d-1_bsm1"]
        merged = merged.drop(columns=["effective_kla_d-1_surrogate", "effective_kla_d-1_bsm1"])
    if "commanded_aeration_index_surrogate" in merged:
        left = pd.to_numeric(merged["commanded_aeration_index_surrogate"], errors="raise")
        right = pd.to_numeric(merged["commanded_aeration_index_bsm1"], errors="raise")
        if not np.allclose(left, right, rtol=0.0, atol=1e-12):
            raise ValueError("Surrogate and BSM1 commanded-aeration indices do not match.")
        merged["commanded_aeration_index"] = right
        merged = merged.drop(columns=["commanded_aeration_index_surrogate", "commanded_aeration_index_bsm1"])
    surrogate_ok = _as_bool(merged["surrogate_satisfies_study_benchmark"])
    bsm1_ok = _as_bool(merged["bsm1_satisfies_study_benchmark"])
    merged["classification"] = [
        classify_candidate(bool(left), bool(right)) for left, right in zip(surrogate_ok, bsm1_ok)
    ]

    threshold_rows: list[dict] = []
    threshold_map: dict[str, float | None] = {}
    for scenario, group in merged.groupby("scenario", sort=False):
        bsm_threshold = persistent_threshold(
            group,
            kla_col="commanded_kla_d-1",
            compliant_col="bsm1_satisfies_study_benchmark",
            persistent_points=persistent_points,
        )
        surrogate_threshold = persistent_threshold(
            group,
            kla_col="commanded_kla_d-1",
            compliant_col="surrogate_satisfies_study_benchmark",
            persistent_points=persistent_points,
        )
        threshold_map[scenario] = bsm_threshold.first_compliant_grid_point
        absolute_error = (
            abs(float(surrogate_threshold.first_compliant_grid_point) - float(bsm_threshold.first_compliant_grid_point))
            if surrogate_threshold.first_compliant_grid_point is not None and bsm_threshold.first_compliant_grid_point is not None
            else np.nan
        )
        relative_error = (
            absolute_error / float(bsm_threshold.first_compliant_grid_point)
            if np.isfinite(absolute_error) and float(bsm_threshold.first_compliant_grid_point) != 0.0
            else np.nan
        )
        threshold_rows.append(
            {
                "scenario": scenario,
                "bsm1_threshold_status": bsm_threshold.status,
                "bsm1_first_compliant_grid_point_d-1": bsm_threshold.first_compliant_grid_point,
                "bsm1_threshold_interval_lower_d-1": bsm_threshold.interval_lower,
                "bsm1_threshold_interval_upper_d-1": bsm_threshold.interval_upper,
                "surrogate_threshold_status": surrogate_threshold.status,
                "surrogate_first_compliant_grid_point_d-1": surrogate_threshold.first_compliant_grid_point,
                "surrogate_threshold_interval_lower_d-1": surrogate_threshold.interval_lower,
                "surrogate_threshold_interval_upper_d-1": surrogate_threshold.interval_upper,
                "absolute_threshold_error_d-1": absolute_error,
                "relative_threshold_error": relative_error,
                "grid_step_d-1": bsm_threshold.grid_step,
                "persistent_compliance_points": persistent_points,
                "applicability_boundary": bool(
                    surrogate_threshold.status == "unresolved" and bsm_threshold.status == "resolved"
                ),
            }
        )
    merged["distance_to_bsm1_threshold_kla_d-1"] = [
        float(kla) - float(threshold_map[scenario]) if threshold_map[scenario] is not None else np.nan
        for scenario, kla in zip(merged["scenario"], merged["commanded_kla_d-1"])
    ]
    return merged.sort_values(sort_key).reset_index(drop=True), pd.DataFrame(threshold_rows)


def confusion_summary(
    classification: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    rows: list[dict] = []
    groups = list(classification.groupby("scenario", sort=False)) + [("ALL", classification)]
    for scenario, group in groups:
        counts = group["classification"].value_counts()
        true_safe = int(counts.get("true_safe", 0))
        true_reject = int(counts.get("true_reject", 0))
        false_safe = int(counts.get("false_safe", 0))
        conservative = int(counts.get("conservative_reject", 0))
        n = int(len(group))
        actual_safe = true_safe + conservative
        actual_reject = true_reject + false_safe
        predicted_reject = true_reject + conservative
        false_safe_rate = false_safe / actual_reject if actual_reject else np.nan
        fs_low, fs_high = wilson_interval(false_safe, actual_reject, confidence_level)
        recall = true_safe / actual_safe if actual_safe else np.nan
        specificity = true_reject / actual_reject if actual_reject else np.nan
        rows.append(
            {
                "scenario": scenario,
                "n_candidates": n,
                "true_safe": true_safe,
                "true_reject": true_reject,
                "false_safe": false_safe,
                "conservative_reject": conservative,
                "false_safe_rate": false_safe_rate,
                "false_safe_wilson_low": fs_low,
                "false_safe_wilson_high": fs_high,
                "conservative_reject_rate": conservative / actual_safe if actual_safe else np.nan,
                "accuracy": (true_safe + true_reject) / n if n else np.nan,
                "sensitivity_recall_bsm1_compliant": recall,
                "specificity_bsm1_noncompliant": specificity,
                "negative_predictive_value": true_reject / predicted_reject if predicted_reject else np.nan,
                "balanced_accuracy": np.nanmean([recall, specificity]),
                "confidence_level": confidence_level,
            }
        )
    return pd.DataFrame(rows)


def distance_to_threshold_summary(
    classification: pd.DataFrame,
    bins: Sequence[float] = (-np.inf, -40.0, -20.0, 0.0, 20.0, 40.0, np.inf),
) -> pd.DataFrame:
    labels = ["<-40", "-40_to_-20", "-20_to_0", "0_to_20", "20_to_40", ">40"]
    if len(bins) != len(labels) + 1:
        raise ValueError("Distance bins must define six requested intervals.")
    frame = classification.copy()
    frame["distance_bin_kla_d-1"] = pd.cut(
        pd.to_numeric(frame["distance_to_bsm1_threshold_kla_d-1"], errors="coerce"),
        bins=list(bins),
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows: list[dict] = []
    for (scenario, distance_bin), group in frame.dropna(subset=["distance_bin_kla_d-1"]).groupby(
        ["scenario", "distance_bin_kla_d-1"], observed=True, sort=False
    ):
        counts = group["classification"].value_counts()
        row = {"scenario": scenario, "distance_bin_kla_d-1": str(distance_bin), "n_candidates": int(len(group))}
        row.update({label: int(counts.get(label, 0)) for label in CLASSIFICATIONS})
        rows.append(row)
    return pd.DataFrame(rows)


def apply_pre_registered_safety_margin(
    classification: pd.DataFrame,
    thresholds: pd.DataFrame,
    margins_d_1: Mapping[str, float],
    mandatory_recheck_scenarios: Iterable[str] = (),
) -> pd.DataFrame:
    frame = classification.copy()
    threshold_map = thresholds.set_index("scenario")["surrogate_first_compliant_grid_point_d-1"].to_dict()
    mandatory = {normalize_scenario_name(value) for value in mandatory_recheck_scenarios}
    corrected: list[bool] = []
    decisions: list[str] = []
    # Use column lookup because pandas sanitizes the hyphenated KLa name in itertuples().
    for _, row in frame.iterrows():
        scenario = normalize_scenario_name(row["scenario"])
        threshold = threshold_map.get(scenario)
        margin = float(margins_d_1.get(scenario, 0.0))
        if scenario in mandatory or threshold is None or not np.isfinite(float(threshold)):
            corrected.append(False)
            decisions.append("mandatory_bsm1_recheck")
            continue
        accept = bool(row["surrogate_satisfies_study_benchmark"]) and float(row["commanded_kla_d-1"]) >= float(threshold) + margin
        corrected.append(accept)
        decisions.append("direct_retain" if accept else "reject_or_recheck")
    frame["pre_registered_safety_margin_d-1"] = [
        float(margins_d_1.get(normalize_scenario_name(value), 0.0)) for value in frame["scenario"]
    ]
    frame["corrected_surrogate_satisfies_study_benchmark"] = corrected
    frame["corrected_screening_decision"] = decisions
    frame["corrected_classification"] = [
        classify_candidate(bool(left), bool(right))
        for left, right in zip(corrected, _as_bool(frame["bsm1_satisfies_study_benchmark"]))
    ]
    return frame


def write_full_grid_outputs(
    *,
    results_dir: Path,
    surrogate_results: pd.DataFrame,
    bsm1_results: pd.DataFrame,
    classification: pd.DataFrame,
    thresholds: pd.DataFrame,
    confidence_level: float,
    distance_bins: Sequence[float],
) -> dict[str, dict]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    confusion = confusion_summary(classification, confidence_level=confidence_level)
    distance = distance_to_threshold_summary(classification, bins=distance_bins)
    outputs = {}
    for name, frame in [
        ("full_grid_surrogate_results.csv", surrogate_results),
        ("full_grid_bsm1_results.csv", bsm1_results),
        ("full_grid_classification.csv", classification),
        ("confusion_matrix_by_scenario.csv", confusion),
        ("threshold_metrics_by_scenario.csv", thresholds),
        ("classification_by_threshold_distance.csv", distance),
    ]:
        outputs[name] = write_dataframe_with_hash(frame, results_dir / name)
    write_json(results_dir / "full_grid_output_hashes.json", outputs)
    return outputs
