from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import normalize_scenario_name


@dataclass(frozen=True)
class ScreeningPolicy:
    threshold_neighborhood_half_width_d_1: float = 20.0
    safety_margin_d_1: Mapping[str, float] = field(default_factory=dict)
    mandatory_bsm1_recheck_scenarios: tuple[str, ...] = ("low_temperature",)
    recheck_budget: int | None = 27
    budget_is_hard_cap: bool = False

    def __post_init__(self) -> None:
        if self.threshold_neighborhood_half_width_d_1 < 0.0:
            raise ValueError("threshold_neighborhood_half_width_d_1 cannot be negative.")
        if self.recheck_budget is not None and self.recheck_budget < 1:
            raise ValueError("recheck_budget must be positive or None.")


def deterministic_screening_decisions(
    candidates: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    policy: ScreeningPolicy,
) -> pd.DataFrame:
    required = {"scenario", "commanded_kla_d-1", "surrogate_satisfies_study_benchmark"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Missing screening candidate columns: {missing}")
    threshold_map = threshold_metrics.set_index("scenario")["surrogate_first_compliant_grid_point_d-1"].to_dict()
    mandatory = {normalize_scenario_name(value) for value in policy.mandatory_bsm1_recheck_scenarios}
    frame = candidates.copy()
    frame["scenario"] = frame["scenario"].map(normalize_scenario_name)
    decisions: list[str] = []
    reasons: list[str] = []
    priorities: list[float] = []
    for _, row in frame.iterrows():
        scenario = row["scenario"]
        kla = float(row["commanded_kla_d-1"])
        predicted = bool(row["surrogate_satisfies_study_benchmark"])
        threshold = threshold_map.get(scenario)
        margin = float(policy.safety_margin_d_1.get(scenario, 0.0))
        if scenario in mandatory:
            decisions.append("bsm1_recheck")
            reasons.append("mandatory_high_risk_or_applicability_boundary")
            priorities.append(-1e12 + kla)
            continue
        if threshold is None or not np.isfinite(float(threshold)):
            decisions.append("bsm1_recheck")
            reasons.append("unresolved_surrogate_transition")
            priorities.append(-1e11 + kla)
            continue
        distance = kla - float(threshold)
        if abs(distance) <= policy.threshold_neighborhood_half_width_d_1:
            decisions.append("bsm1_recheck")
            reasons.append("within_pre_registered_threshold_neighborhood")
            priorities.append(abs(distance))
        elif not predicted and distance < -policy.threshold_neighborhood_half_width_d_1:
            decisions.append("direct_reject")
            reasons.append("surrogate_noncompliant_below_threshold_neighborhood")
            priorities.append(1e12 + abs(distance))
        elif predicted and kla >= float(threshold) + margin + policy.threshold_neighborhood_half_width_d_1:
            decisions.append("direct_retain")
            reasons.append("surrogate_compliant_above_margin_and_neighborhood")
            priorities.append(1e12 + abs(distance))
        else:
            decisions.append("bsm1_recheck")
            reasons.append("safety_margin_ambiguity")
            priorities.append(abs(distance))
    frame["screening_decision"] = decisions
    frame["screening_reason"] = reasons
    frame["recheck_priority"] = priorities
    frame["selected_for_bsm1_recheck"] = frame["screening_decision"].eq("bsm1_recheck")

    required_rechecks = int(frame["selected_for_bsm1_recheck"].sum())
    budget = policy.recheck_budget
    frame["recheck_budget"] = budget
    frame["recheck_budget_exceeded"] = bool(budget is not None and required_rechecks > budget)
    if policy.budget_is_hard_cap and budget is not None and required_rechecks > budget:
        ordered = frame[frame["selected_for_bsm1_recheck"]].sort_values(
            ["recheck_priority", "scenario", "commanded_kla_d-1"]
        )
        kept = set(ordered.head(budget).index)
        dropped = frame["selected_for_bsm1_recheck"] & ~frame.index.isin(kept)
        frame.loc[dropped, "screening_decision"] = "unresolved_budget_exceeded"
        frame.loc[dropped, "screening_reason"] += ";not_safe_to_directly_accept_or_reject"
        frame.loc[dropped, "selected_for_bsm1_recheck"] = False
    return frame.sort_values(["scenario", "commanded_kla_d-1"]).reset_index(drop=True)


def workflow_efficiency_table(
    decisions: pd.DataFrame,
    *,
    surrogate_total_seconds: float | None = None,
    bsm1_runtime_seconds: Sequence[float] | None = None,
    bsm1_case_runtimes: pd.DataFrame | None = None,
    overhead_seconds: float | None = None,
) -> pd.DataFrame:
    working = decisions.copy()
    exact_case_runtime = False
    if bsm1_case_runtimes is not None:
        runtime_frame = bsm1_case_runtimes.copy()
        runtime_frame["scenario"] = runtime_frame["scenario"].map(normalize_scenario_name)
        runtime_frame["commanded_kla_d-1"] = pd.to_numeric(
            runtime_frame["commanded_kla_d-1"]
            if "commanded_kla_d-1" in runtime_frame
            else runtime_frame["kla"],
            errors="raise",
        )
        runtime_frame["observed_bsm1_runtime_seconds"] = pd.to_numeric(
            runtime_frame["runtime_seconds"], errors="raise"
        )
        if "candidate_id" in working.columns and "candidate_id" in runtime_frame.columns:
            # candidate_id is the stable key created before either model runs.
            # Do not require exact equality of CSV-round-tripped floating KLa
            # values merely to attach an observed runtime.
            for label, frame in (("screening decisions", working), ("BSM1 runtimes", runtime_frame)):
                candidate_ids = frame["candidate_id"].astype(str).str.strip()
                if candidate_ids.eq("").any() or candidate_ids.str.lower().eq("nan").any():
                    raise ValueError(f"{label} contain a missing candidate_id.")
                if candidate_ids.duplicated().any():
                    duplicates = sorted(candidate_ids[candidate_ids.duplicated(keep=False)].unique())
                    raise ValueError(f"{label} contain duplicate candidate_id values: {duplicates[:20]}")
                frame["candidate_id"] = candidate_ids

            audit = working[["candidate_id", "scenario", "commanded_kla_d-1"]].merge(
                runtime_frame[["candidate_id", "scenario", "commanded_kla_d-1"]],
                on="candidate_id",
                how="outer",
                validate="one_to_one",
                suffixes=("_decision", "_runtime"),
                indicator=True,
            )
            if not audit["_merge"].eq("both").all():
                unmatched = audit.loc[~audit["_merge"].eq("both"), ["candidate_id", "_merge"]]
                raise ValueError(
                    "Exact workflow timing candidate IDs do not match:\n"
                    f"{unmatched.head(20).to_string(index=False)}"
                )
            scenario_matches = audit["scenario_decision"].eq(audit["scenario_runtime"])
            decision_kla = pd.to_numeric(audit["commanded_kla_d-1_decision"], errors="raise")
            runtime_kla = pd.to_numeric(audit["commanded_kla_d-1_runtime"], errors="raise")
            kla_matches = np.isclose(decision_kla, runtime_kla, rtol=0.0, atol=1e-9)
            if not bool(scenario_matches.all()) or not bool(np.all(kla_matches)):
                raise ValueError("candidate_id maps to different scenario or KLa values in the runtime table.")

            working = working.merge(
                runtime_frame[["candidate_id", "observed_bsm1_runtime_seconds"]],
                on="candidate_id",
                how="left",
                validate="one_to_one",
            )
        else:
            # Legacy tables without candidate_id use a rounded numeric match
            # key, followed by one-to-one validation.
            working["_kla_runtime_match"] = pd.to_numeric(
                working["commanded_kla_d-1"], errors="raise"
            ).round(9)
            runtime_frame["_kla_runtime_match"] = runtime_frame["commanded_kla_d-1"].round(9)
            working = working.merge(
                runtime_frame[["scenario", "_kla_runtime_match", "observed_bsm1_runtime_seconds"]],
                on=["scenario", "_kla_runtime_match"],
                how="left",
                validate="one_to_one",
            ).drop(columns="_kla_runtime_match")
        if working["observed_bsm1_runtime_seconds"].isna().any():
            raise ValueError("Exact workflow timing requires a BSM1 runtime for every candidate.")
        exact_case_runtime = True
    groups = list(working.groupby("scenario", sort=False)) + [("ALL", working)]
    runtime_values = np.asarray(list(bsm1_runtime_seconds or []), dtype=float)
    mean_bsm1 = float(np.mean(runtime_values[np.isfinite(runtime_values)])) if np.any(np.isfinite(runtime_values)) else np.nan
    rows: list[dict] = []
    total_all = max(int(len(decisions)), 1)
    for scenario, group in groups:
        n = int(len(group))
        rechecks = int(group["selected_for_bsm1_recheck"].sum())
        surrogate_share = (
            float(surrogate_total_seconds) * n / total_all if surrogate_total_seconds is not None else np.nan
        )
        overhead_share = float(overhead_seconds) * n / total_all if overhead_seconds is not None else np.nan
        if exact_case_runtime:
            exhaustive = float(group["observed_bsm1_runtime_seconds"].sum())
            recheck_seconds = float(
                group.loc[group["selected_for_bsm1_recheck"], "observed_bsm1_runtime_seconds"].sum()
            )
        else:
            exhaustive = mean_bsm1 * n if np.isfinite(mean_bsm1) else np.nan
            recheck_seconds = mean_bsm1 * rechecks if np.isfinite(mean_bsm1) else np.nan
        two_stage = (
            surrogate_share + recheck_seconds + overhead_share
            if np.isfinite(recheck_seconds) and np.isfinite(surrogate_share) and np.isfinite(overhead_share)
            else np.nan
        )
        rows.append(
            {
                "scenario": scenario,
                "candidate_cases": n,
                "bsm1_rechecks": rechecks,
                "bsm1_calls_avoided": n - rechecks,
                "bsm1_call_reduction_fraction": (n - rechecks) / n if n else np.nan,
                "surrogate_prescreen_seconds": surrogate_share,
                "bsm1_recheck_seconds": recheck_seconds,
                "classification_overhead_seconds": overhead_share,
                "exhaustive_bsm1_seconds": exhaustive,
                "two_stage_workflow_seconds": two_stage,
                "workflow_speedup": exhaustive / two_stage if np.isfinite(two_stage) and two_stage > 0.0 else np.nan,
                "end_to_end_time_reduction_fraction": (exhaustive - two_stage) / exhaustive if np.isfinite(exhaustive) and exhaustive > 0.0 and np.isfinite(two_stage) else np.nan,
                "surrogate_overhead_fraction": surrogate_share / two_stage if np.isfinite(two_stage) and two_stage > 0.0 else np.nan,
                "runtime_status": (
                    "observed_full_grid_case_runtimes"
                    if exact_case_runtime and np.isfinite(two_stage)
                    else "estimated_from_mean_bsm1_runtime"
                    if np.isfinite(two_stage)
                    else "TODO_timing_results_required"
                ),
            }
        )
    return pd.DataFrame(rows)
