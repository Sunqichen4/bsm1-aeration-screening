from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .fixed_aeration import run_surrogate_fixed_aeration
from .utils import ensure_dirs, read_json, write_json


DEFAULT_THRESHOLD_OFFSETS = (-20.0, -10.0, 0.0, 10.0, 20.0)


def generate_kla_grid(kla_min: float, kla_max: float, n_candidates: int) -> np.ndarray:
    """Return a deterministic inclusive KLa grid for one scenario."""
    if int(n_candidates) < 2:
        raise ValueError("n_candidates must be at least 2.")
    if not np.isfinite(kla_min) or not np.isfinite(kla_max) or kla_max <= kla_min:
        raise ValueError("kla_max must be finite and greater than kla_min.")
    return np.linspace(float(kla_min), float(kla_max), int(n_candidates), dtype=float)


def compliance_mask(frame: pd.DataFrame, rule: str, nh4_limit: float) -> pd.Series:
    """Return the explicitly selected surrogate/BSM1 compliance rule."""
    normalized = str(rule).strip().lower()
    if normalized == "mean":
        return pd.to_numeric(frame["SNH_mean_eval"], errors="coerce") <= float(nh4_limit)
    if normalized == "zero_violation":
        return pd.to_numeric(
            frame["SNH_violation_hours_gt_limit"], errors="coerce"
        ) <= 1e-12
    raise ValueError("compliance_rule must be 'mean' or 'zero_violation'.")


def as_bool_series(values: pd.Series) -> pd.Series:
    """Read in-memory or CSV round-tripped boolean columns without ambiguity."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(["true", "false", "1", "0"])
    if invalid.any():
        raise ValueError(f"Cannot interpret boolean values: {sorted(normalized[invalid].unique())}")
    return normalized.isin(["true", "1"])


def screen_candidates(
    *,
    scenarios: Sequence[str],
    kla_values: Sequence[float],
    warmup_days: float,
    eval_days: float,
    dt_minutes: float,
    nh4_limit: float,
    oxygen_transfer_factor: float,
    compliance_rule: str,
    runner: Callable[..., dict] = run_surrogate_fixed_aeration,
) -> pd.DataFrame:
    """Evaluate every scenario/KLa candidate with the reduced surrogate."""
    rows: list[dict] = []
    candidate_number = 0
    for scenario in scenarios:
        for scenario_index, kla in enumerate(kla_values, start=1):
            candidate_number += 1
            row = runner(
                scenario_name=str(scenario),
                kla=float(kla),
                warmup_days=float(warmup_days),
                eval_days=float(eval_days),
                dt_minutes=float(dt_minutes),
                nh4_limit=float(nh4_limit),
                oxygen_transfer_factor=float(oxygen_transfer_factor),
            )
            row.update(
                {
                    "candidate_id": f"C{candidate_number:06d}",
                    "scenario_candidate_index": int(scenario_index),
                }
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No screening candidates were generated.")
    frame["mean_NH4_compliant"] = (
        pd.to_numeric(frame["SNH_mean_eval"], errors="coerce") <= float(nh4_limit)
    )
    frame["zero_violation_compliant"] = (
        pd.to_numeric(frame["SNH_violation_hours_gt_limit"], errors="coerce") <= 1e-12
    )
    frame["predicted_compliant"] = compliance_mask(frame, compliance_rule, nh4_limit)
    frame["screening_decision"] = "reject_predicted_noncompliant"

    for _, indices in frame.groupby("scenario", sort=False).groups.items():
        ordered = frame.loc[indices].sort_values("kla")
        feasible = ordered[ordered["predicted_compliant"]]
        if feasible.empty:
            continue
        first_index = feasible.index[0]
        frame.loc[feasible.index, "screening_decision"] = "reject_energy_dominated"
        frame.loc[first_index, "screening_decision"] = "retain_lowest_predicted_compliant"

    return frame.sort_values(["scenario", "kla"]).reset_index(drop=True)


def _nearest_index(group: pd.DataFrame, target_kla: float) -> int:
    distance = (pd.to_numeric(group["kla"], errors="coerce") - float(target_kla)).abs()
    return int(distance.idxmin())


def select_bsm1_shortlist(
    candidates: pd.DataFrame,
    *,
    threshold_offsets: Sequence[float] = DEFAULT_THRESHOLD_OFFSETS,
    no_transition_tail_width: float = 80.0,
    no_transition_tail_points: int = 5,
) -> pd.DataFrame:
    """Select auditable BSM1 checks around each predicted transition.

    Transition scenarios receive low/high range anchors plus fixed KLa offsets
    around the surrogate threshold.  If the surrogate predicts no transition,
    the high-KLa tail is sampled instead; this is designed to detect a
    conservative false rejection without running the complete BSM1 grid.
    """
    if candidates.empty:
        raise ValueError("Cannot select a BSM1 shortlist from an empty table.")
    if int(no_transition_tail_points) < 2 or float(no_transition_tail_width) <= 0.0:
        raise ValueError("The no-transition tail requires at least 2 points and positive width.")

    reasons: dict[int, list[str]] = defaultdict(list)
    for scenario, group in candidates.groupby("scenario", sort=False):
        group = group.sort_values("kla")
        reasons[int(group.index[0])].append("low_range_anchor")
        feasible = group[as_bool_series(group["predicted_compliant"])]

        if not feasible.empty:
            transition_kla = float(feasible.iloc[0]["kla"])
            for offset in threshold_offsets:
                target = transition_kla + float(offset)
                idx = _nearest_index(group, target)
                reasons[idx].append(f"threshold_offset_{float(offset):+g}")
            reasons[int(group.index[-1])].append("high_range_anchor")
        else:
            high = float(group["kla"].max())
            low = max(float(group["kla"].min()), high - float(no_transition_tail_width))
            for target in np.linspace(low, high, int(no_transition_tail_points)):
                idx = _nearest_index(group, float(target))
                reasons[idx].append("upper_tail_no_predicted_transition")

    selected_indices = sorted(reasons)
    shortlist = candidates.loc[selected_indices].copy()
    shortlist["bsm1_selection_reason"] = [";".join(reasons[idx]) for idx in selected_indices]
    shortlist["bsm1_recheck_order"] = np.arange(1, len(shortlist) + 1, dtype=int)
    return shortlist.sort_values(["scenario", "kla"]).reset_index(drop=True)


def build_screening_summary(
    candidates: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    """Report the exact exhaustive-vs-two-stage BSM1 run counts."""
    rows: list[dict] = []
    scenario_names = list(candidates["scenario"].drop_duplicates())
    for scenario in scenario_names + ["ALL"]:
        all_mask = pd.Series(True, index=candidates.index) if scenario == "ALL" else candidates["scenario"].eq(scenario)
        short_mask = pd.Series(True, index=shortlist.index) if scenario == "ALL" else shortlist["scenario"].eq(scenario)
        group = candidates.loc[all_mask]
        selected = shortlist.loc[short_mask]
        total = int(len(group))
        rechecks = int(len(selected))
        avoided = int(total - rechecks)
        rows.append(
            {
                "scenario": scenario,
                "candidate_cases_if_exhaustive_bsm1": total,
                "surrogate_predicted_compliant_count": int(as_bool_series(group["predicted_compliant"]).sum()),
                "surrogate_predicted_noncompliant_count": int((~as_bool_series(group["predicted_compliant"])).sum()),
                "bsm1_rechecks_planned": rechecks,
                "bsm1_runs_avoided": avoided,
                "bsm1_run_reduction_fraction": float(avoided / total) if total else np.nan,
                "bsm1_run_reduction_percent": float(100.0 * avoided / total) if total else np.nan,
                "exhaustive_to_recheck_ratio": float(total / rechecks) if rechecks else np.inf,
                "surrogate_runtime_total_seconds": float(group["runtime_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _format_number(value: float) -> str:
    return format(float(value), ".12g")


def write_bsm1_recheck_commands(
    *,
    shortlist: pd.DataFrame,
    output_dir: Path,
    warmup_days: float,
    eval_days: float,
    timestep_min: float,
    nh4_limit: float,
    oxygen_transfer_factor: float,
) -> tuple[Path, list[str]]:
    """Write one directly runnable PowerShell command per scenario."""
    commands: list[str] = []
    for scenario, group in shortlist.groupby("scenario", sort=False):
        kla_text = ",".join(_format_number(v) for v in group.sort_values("kla")["kla"])
        scenario_output = output_dir / "bsm1_rechecks" / str(scenario)
        commands.append(
            "python -B scripts/run_bsm1_validation.py "
            f"--scenarios {scenario} --kla-values \"{kla_text}\" --mode steady_const "
            f"--warmup-days {_format_number(warmup_days)} --eval-days {_format_number(eval_days)} "
            f"--timestep-min {_format_number(timestep_min)} --nh4-limit {_format_number(nh4_limit)} "
            f"--oxygen-transfer-factor {_format_number(oxygen_transfer_factor)} --skip-abac "
            f"--output-dir \"{scenario_output}\""
        )

    script_path = output_dir / "run_bsm1_rechecks.ps1"
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "# Generated by scripts/run_large_scale_screening.py",
        "# Run this file from the project root in the active PyCharm/conda terminal.",
    ]
    for command in commands:
        lines.extend(
            [
                command,
                'if ($LASTEXITCODE -ne 0) { throw "BSM1 recheck failed with exit code $LASTEXITCODE." }',
            ]
        )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    (output_dir / "bsm1_recheck_commands.txt").write_text(
        "\n\n".join(commands) + "\n", encoding="utf-8"
    )
    return script_path, commands


def run_large_scale_screening(
    *,
    output_dir: Path,
    scenarios: Sequence[str],
    candidates_per_scenario: int = 250,
    kla_min: float = 40.0,
    kla_max: float = 520.0,
    warmup_days: float = 60.0,
    eval_days: float = 7.0,
    dt_minutes: float = 15.0,
    bsm1_timestep_min: float = 1.0,
    nh4_limit: float = 4.0,
    oxygen_transfer_factor: float = 0.70,
    compliance_rule: str = "zero_violation",
    threshold_offsets: Sequence[float] = DEFAULT_THRESHOLD_OFFSETS,
    no_transition_tail_width: float = 80.0,
    no_transition_tail_points: int = 5,
    runner: Callable[..., dict] = run_surrogate_fixed_aeration,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the surrogate screen and write the BSM1 verification plan."""
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"

    kla_values = generate_kla_grid(kla_min, kla_max, candidates_per_scenario)
    candidates = screen_candidates(
        scenarios=scenarios,
        kla_values=kla_values,
        warmup_days=warmup_days,
        eval_days=eval_days,
        dt_minutes=dt_minutes,
        nh4_limit=nh4_limit,
        oxygen_transfer_factor=oxygen_transfer_factor,
        compliance_rule=compliance_rule,
        runner=runner,
    )
    shortlist = select_bsm1_shortlist(
        candidates,
        threshold_offsets=threshold_offsets,
        no_transition_tail_width=no_transition_tail_width,
        no_transition_tail_points=no_transition_tail_points,
    )
    reason_map = shortlist.set_index("candidate_id")["bsm1_selection_reason"]
    candidates["selected_for_bsm1"] = candidates["candidate_id"].isin(shortlist["candidate_id"])
    candidates["bsm1_selection_reason"] = candidates["candidate_id"].map(reason_map).fillna("")
    summary = build_screening_summary(candidates, shortlist)

    candidates.to_csv(table_dir / "large_scale_surrogate_candidates.csv", index=False)
    shortlist.to_csv(table_dir / "large_scale_bsm1_shortlist.csv", index=False)
    summary.to_csv(table_dir / "large_scale_screening_summary.csv", index=False)
    command_path, commands = write_bsm1_recheck_commands(
        shortlist=shortlist,
        output_dir=output_dir,
        warmup_days=warmup_days,
        eval_days=eval_days,
        timestep_min=bsm1_timestep_min,
        nh4_limit=nh4_limit,
        oxygen_transfer_factor=oxygen_transfer_factor,
    )

    overall = summary[summary["scenario"].eq("ALL")].iloc[0]
    write_json(
        manifest_dir / "large_scale_screening_manifest.json",
        {
            "scenarios": list(scenarios),
            "candidate_definition": "one fixed-KLa/scenario pair equals one candidate case and one exhaustive BSM1 run",
            "candidates_per_scenario": int(candidates_per_scenario),
            "kla_min": float(kla_min),
            "kla_max": float(kla_max),
            "total_candidate_cases": int(len(candidates)),
            "compliance_rule": compliance_rule,
            "nh4_limit": float(nh4_limit),
            "surrogate_warmup_days": float(warmup_days),
            "surrogate_eval_days": float(eval_days),
            "surrogate_timestep_min": float(dt_minutes),
            "bsm1_timestep_min": float(bsm1_timestep_min),
            "oxygen_transfer_factor": float(oxygen_transfer_factor),
            "threshold_offsets_kla": [float(v) for v in threshold_offsets],
            "no_transition_tail_width_kla": float(no_transition_tail_width),
            "no_transition_tail_points": int(no_transition_tail_points),
            "bsm1_rechecks_planned": int(len(shortlist)),
            "bsm1_runs_avoided": int(overall["bsm1_runs_avoided"]),
            "bsm1_run_reduction_percent": float(overall["bsm1_run_reduction_percent"]),
            "commands_file": str(command_path),
            "commands": commands,
            "note": (
                "Run counts are exact counts of candidate/scenario cases. Runtime savings are not claimed until "
                "the selected BSM1 rechecks have completed and --collect-only has measured their runtimes. "
                "Boundary offsets and range anchors are retained to audit false-safe and conservative classifications."
            ),
        },
    )
    return candidates, shortlist, summary


def _bsm1_row_for_candidate(
    bsm1_frame: pd.DataFrame,
    *,
    scenario: str,
    kla: float,
) -> pd.Series:
    fixed = bsm1_frame[bsm1_frame["experiment"].eq("fixed_kla")].copy()
    fixed = fixed[fixed["scenario"].eq(scenario)]
    distance = (pd.to_numeric(fixed["kla"], errors="coerce") - float(kla)).abs()
    if fixed.empty or not np.isfinite(distance.min()) or float(distance.min()) > 1e-6:
        raise ValueError(f"Missing BSM1 result for scenario={scenario}, KLa={kla:.12g}.")
    return fixed.loc[distance.idxmin()]


def collect_bsm1_rechecks(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect the selected BSM1 runs and quantify the realized two-stage screen."""
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    manifest_path = manifest_dir / "large_scale_screening_manifest.json"
    shortlist_path = table_dir / "large_scale_bsm1_shortlist.csv"
    candidates_path = table_dir / "large_scale_surrogate_candidates.csv"
    if not manifest_path.exists() or not shortlist_path.exists() or not candidates_path.exists():
        raise FileNotFoundError("Run the surrogate large-scale screening before collecting BSM1 rechecks.")

    manifest = read_json(manifest_path)
    shortlist = pd.read_csv(shortlist_path)
    candidates = pd.read_csv(candidates_path)
    bsm1_by_scenario: dict[str, pd.DataFrame] = {}
    for scenario in shortlist["scenario"].drop_duplicates():
        path = output_dir / "bsm1_rechecks" / str(scenario) / "tables" / "bsm1_validation_summary.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run {output_dir / 'run_bsm1_rechecks.ps1'} from the project root first."
            )
        bsm1_by_scenario[str(scenario)] = pd.read_csv(path)

    records: list[dict] = []
    metric_columns = [
        "SNH_mean_eval",
        "SNH_p95_eval",
        "SNH_violation_hours_gt_limit",
        "SNH_violation_ratio_gt_limit",
        "TN_mean_eval",
        "TN_p95_eval",
        "commanded_aeration_index",
        "bsm1_standard_aeration_energy_kWh_per_d",
        "runtime_seconds",
    ]
    for _, candidate in shortlist.iterrows():
        scenario = str(candidate["scenario"])
        bsm1_row = _bsm1_row_for_candidate(
            bsm1_by_scenario[scenario], scenario=scenario, kla=float(candidate["kla"])
        )
        record = candidate.to_dict()
        for column in metric_columns:
            record[f"bsm1_{column}"] = bsm1_row[column] if column in bsm1_row.index else np.nan
        records.append(record)

    rechecks = pd.DataFrame(records)
    bsm_rule_frame = pd.DataFrame(
        {
            "SNH_mean_eval": rechecks["bsm1_SNH_mean_eval"],
            "SNH_violation_hours_gt_limit": rechecks["bsm1_SNH_violation_hours_gt_limit"],
        }
    )
    rechecks["bsm1_compliant"] = compliance_mask(
        bsm_rule_frame, str(manifest["compliance_rule"]), float(manifest["nh4_limit"])
    )
    surrogate_ok = as_bool_series(rechecks["predicted_compliant"])
    bsm1_ok = as_bool_series(rechecks["bsm1_compliant"])
    rechecks["classification_relation"] = np.select(
        [surrogate_ok & ~bsm1_ok, ~surrogate_ok & bsm1_ok],
        ["false_safe", "conservative"],
        default="agreement",
    )
    rechecks.to_csv(table_dir / "large_scale_bsm1_recheck_results.csv", index=False)

    summary_rows: list[dict] = []
    scenario_names = list(candidates["scenario"].drop_duplicates())
    for scenario in scenario_names + ["ALL"]:
        candidate_group = candidates if scenario == "ALL" else candidates[candidates["scenario"].eq(scenario)]
        recheck_group = rechecks if scenario == "ALL" else rechecks[rechecks["scenario"].eq(scenario)]
        total = int(len(candidate_group))
        completed = int(len(recheck_group))
        runtime_total = float(pd.to_numeric(recheck_group["bsm1_runtime_seconds"], errors="coerce").sum())
        runtime_mean = float(pd.to_numeric(recheck_group["bsm1_runtime_seconds"], errors="coerce").mean())
        exhaustive_estimate = float(runtime_mean * total)
        surrogate_runtime = float(pd.to_numeric(candidate_group["runtime_seconds"], errors="coerce").sum())
        two_stage_runtime = float(surrogate_runtime + runtime_total)
        summary_rows.append(
            {
                "scenario": scenario,
                "candidate_cases_if_exhaustive_bsm1": total,
                "bsm1_rechecks_completed": completed,
                "bsm1_runs_avoided": int(total - completed),
                "bsm1_run_reduction_percent": float(100.0 * (total - completed) / total),
                "classification_agreement_count": int(recheck_group["classification_relation"].eq("agreement").sum()),
                "false_safe_count": int(recheck_group["classification_relation"].eq("false_safe").sum()),
                "conservative_count": int(recheck_group["classification_relation"].eq("conservative").sum()),
                "surrogate_runtime_total_seconds": surrogate_runtime,
                "bsm1_recheck_runtime_total_seconds": runtime_total,
                "bsm1_mean_runtime_per_recheck_seconds": runtime_mean,
                "estimated_exhaustive_bsm1_runtime_seconds": exhaustive_estimate,
                "observed_two_stage_runtime_seconds": two_stage_runtime,
                "estimated_runtime_saved_seconds": float(exhaustive_estimate - two_stage_runtime),
                "estimated_runtime_reduction_percent": float(
                    100.0 * (exhaustive_estimate - two_stage_runtime) / exhaustive_estimate
                ) if exhaustive_estimate > 0.0 else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "large_scale_screening_summary_after_bsm1.csv", index=False)
    overall = summary[summary["scenario"].eq("ALL")].iloc[0]
    write_json(
        manifest_dir / "large_scale_screening_collection_manifest.json",
        {
            "bsm1_rechecks_completed": int(len(rechecks)),
            "candidate_cases_if_exhaustive_bsm1": int(len(candidates)),
            "bsm1_runs_avoided": int(overall["bsm1_runs_avoided"]),
            "bsm1_run_reduction_percent": float(overall["bsm1_run_reduction_percent"]),
            "false_safe_count": int(overall["false_safe_count"]),
            "conservative_count": int(overall["conservative_count"]),
            "observed_surrogate_runtime_seconds": float(overall["surrogate_runtime_total_seconds"]),
            "observed_bsm1_recheck_runtime_seconds": float(overall["bsm1_recheck_runtime_total_seconds"]),
            "estimated_exhaustive_bsm1_runtime_seconds": float(overall["estimated_exhaustive_bsm1_runtime_seconds"]),
            "observed_two_stage_runtime_seconds": float(overall["observed_two_stage_runtime_seconds"]),
            "estimated_runtime_reduction_percent": float(overall["estimated_runtime_reduction_percent"]),
            "runtime_estimation_note": (
                "The exhaustive BSM1 runtime is estimated as the observed mean selected-run runtime multiplied by "
                "the exhaustive candidate count. Candidate and completed-run counts are exact."
            ),
        },
    )
    return rechecks, summary
