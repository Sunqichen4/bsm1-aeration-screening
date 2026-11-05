"""Development-only fitting and label-blind screening for new combinations.

These policies are a NEW follow-up experiment, not an implementation or
retrospective re-labelling of the paper's previous post-audit policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .classification import classify_candidate
from .steady_conditions import SteadyOperatingPoint


DEVELOPMENT_POINTS = {
    "baseline": SteadyOperatingPoint("baseline", 0., 1.),
    "high_load": SteadyOperatingPoint("high_load", 1., 1.),
    "low_oxygen_transfer": SteadyOperatingPoint("low_oxygen_transfer", 0., .7),
}
METHODS = ("frozen_surrogate", "condition_logistic")


def parse_bool(values) -> np.ndarray:
    normalized = pd.Series(values).astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Missing or invalid Boolean classification.")
    return normalized.isin(["true", "1"]).to_numpy()


def validate_protocol(p: dict) -> None:
    if p["development_scenarios"] != list(DEVELOPMENT_POINTS):
        raise ValueError("This protocol requires the three specified development scenarios.")
    for key in ("load_coordinates", "oxygen_transfer_factors", "kla_values"):
        a = np.asarray(p[key], dtype=float)
        if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
            raise ValueError(f"{key} must be finite, unique and increasing.")
    if not all(0 < x < 1 for x in p["load_coordinates"]):
        raise ValueError("New loads must be strictly between the two development loads.")
    if not all(.7 < x < 1 for x in p["oxygen_transfer_factors"]):
        raise ValueError("New transfer factors must be strictly between .7 and 1.")
    k = p["kla_values"]
    if k[0] != 40 or k[-1] != 520 or len(k) < 4:
        raise ValueError("The new grid must span 40 to 520 with at least four points.")
    for key, expected in (("warmup_days", 60.), ("eval_days", 7.),
                          ("surrogate_timestep_min", 15.), ("bsm1_timestep_min", 1.),
                          ("nh4_limit", 4.), ("tempmodel_policy", "off")):
        if p[key] != expected:
            raise ValueError(f"{key} must stay {expected!r} to match development conditions.")
    if p["minimum_compliant_tail_points"] < 3 or int(p["minimum_compliant_tail_points"]) != p["minimum_compliant_tail_points"]:
        raise ValueError("Require at least three compliant tail points.")
    for key in ("extra_effective_kla_guard", "logistic_l2_penalty"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")


def candidate_grid(p: dict) -> pd.DataFrame:
    validate_protocol(p)
    rows = []
    for i, load in enumerate(p["load_coordinates"], 1):
        for j, alpha in enumerate(p["oxygen_transfer_factors"], 1):
            point = SteadyOperatingPoint(f"new_load{i}_transfer{j}", load, alpha)
            for k, kla in enumerate(p["kla_values"], 1):
                rows.append({"candidate_id": f"H{i}{j}_{k:03d}", "scenario": point.name,
                             "kla": float(kla), "effective_kla": float(kla * alpha),
                             **point.metadata()})
    return pd.DataFrame(rows)


def development_data(frame: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Only the old data are accepted here; test targets are never arguments."""
    d = frame.loc[frame.scenario.isin(DEVELOPMENT_POINTS)].copy()
    if len(d) != 750 or d.duplicated(["scenario", "kla"]).any() or d.candidate_id.duplicated().any():
        raise ValueError("Expected 750 unique development cases (3 x 250).")
    for key, value in (("warmup_days", p["warmup_days"]), ("eval_days", p["eval_days"]),
                       ("timestep_min", p["bsm1_timestep_min"]), ("nh4_limit", p["nh4_limit"])):
        if not np.allclose(d[key].astype(float), value, rtol=0, atol=1e-10):
            raise ValueError(f"Development metadata mismatch: {key}.")
    if parse_bool(d.tempmodel).any() or not d.inflow_mode.eq("steady_const").all():
        raise ValueError("Development data must use steady inputs and fixed temperature kinetics.")
    metrics = d[["SNH_max_eval", "SNH_violation_hours_gt_limit", "runtime_seconds"]].to_numpy(float)
    if not np.isfinite(metrics).all() or np.any(metrics < 0):
        raise ValueError("Invalid development numerical results.")
    d["target"] = parse_bool(d.SNH_zero_violation)
    if not np.array_equal(d.target, d.SNH_max_eval.to_numpy(float) <= p["nh4_limit"]):
        raise ValueError("Development peak concentrations disagree with classifications.")
    if not np.array_equal(d.target, d.SNH_violation_hours_gt_limit.to_numpy(float) == 0):
        raise ValueError("Development violation durations disagree with classifications.")
    for name, point in DEVELOPMENT_POINTS.items():
        mask = d.scenario.eq(name)
        group = d.loc[mask].sort_values("kla")
        if len(group) != 250 or not np.allclose(group.kla, np.linspace(40, 520, 250)):
            raise ValueError(f"Incomplete development grid: {name}.")
        if not np.allclose(group.effective_kla, group.kla * point.oxygen_transfer_factor):
            raise ValueError("Development effective KLa mismatch.")
        for key, value in point.metadata().items():
            d.loc[mask, key] = value
    return d.sort_values(["scenario", "kla"]).reset_index(drop=True)


def threshold_info(kla, flags, tail_points: int = 3) -> dict:
    k = np.asarray(kla, dtype=float)
    y = parse_bool(flags)
    if len(k) < 2 or len(k) != len(y) or not np.isfinite(k).all() or np.any(np.diff(k) <= 0):
        raise ValueError("Threshold grid must be finite and strictly increasing.")
    out = {"status": "unresolved", "lower": None, "upper": None,
           "grid_step": float(np.max(np.diff(k)))}
    if np.any(np.diff(y.astype(int)) < 0):
        out["status"] = "nonmonotone"
    elif y.all():
        out["status"] = "below_grid"
    elif y.any():
        first = int(np.flatnonzero(y)[0])
        if len(y) - first >= tail_points:
            out.update(status="bracketed", lower=float(k[first - 1]), upper=float(k[first]))
    return out


def features(frame: pd.DataFrame) -> np.ndarray:
    # Explicit effective aeration captures transfer loss once; the load
    # coordinate jointly represents the prescribed COD and NH4 perturbation.
    x = np.column_stack([frame.effective_kla.to_numpy(float) / 240.,
                         frame.load_coordinate.to_numpy(float)])
    if not np.isfinite(x).all():
        raise ValueError("Non-finite baseline features.")
    return x


def fit_logistic(dev: pd.DataFrame, penalty: float) -> dict:
    x, y = features(dev), parse_bool(dev.target).astype(float)
    if len(np.unique(y)) != 2 or np.linalg.matrix_rank(np.c_[np.ones(len(x)), x]) != 3:
        raise ValueError("Baseline development needs both labels and identifiable features.")

    def objective(w):
        z = w[0] + x @ w[1:]
        loss = np.sum(np.logaddexp(0., z) - y * z) + penalty * (w[1:] @ w[1:])
        residual = expit(z) - y
        gradient = np.r_[np.sum(residual), x.T @ residual + 2 * penalty * w[1:]]
        return float(loss), gradient

    result = minimize(objective, np.zeros(3), jac=True, method="L-BFGS-B",
                      bounds=[(None, None), (0, None), (None, 0)],
                      options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Baseline fit failed: {result.message}")
    return {"weights": result.x.tolist(), "l2_penalty": penalty,
            "features": ["effective_kla / 240", "load_coordinate"],
            "objective": "sum_cross_entropy_plus_l2_slopes", "cutoff": .5,
            "training_cases": len(dev)}


def logistic_predictions(frame: pd.DataFrame, model: dict) -> np.ndarray:
    w = np.asarray(model["weights"], dtype=float)
    return expit(w[0] + features(frame) @ w[1:])


def calibrate_buffers(dev: pd.DataFrame, predicted, p: dict) -> dict:
    d = dev.copy()
    d["predicted"] = parse_bool(predicted)
    records = []
    for name, group in d.groupby("scenario", sort=True):
        g = group.sort_values("effective_kla")
        truth = threshold_info(g.effective_kla, g.target, p["minimum_compliant_tail_points"])
        pred = threshold_info(g.effective_kla, g.predicted, p["minimum_compliant_tail_points"])
        records.append({"scenario": name, "reference": truth, "prediction": pred})
    enabled = all(r["reference"]["status"] == r["prediction"]["status"] == "bracketed" for r in records)
    release = rejection = 0.
    if enabled:
        # Use opposite bracket ends, not point estimates, to include the old
        # grid uncertainty. The same formula applies to both methods.
        release = max(0., max(r["reference"]["upper"] - r["prediction"]["lower"] for r in records))
        rejection = max(0., max(r["prediction"]["upper"] - r["reference"]["lower"] for r in records))
    return {"enabled": enabled, "release_buffer": release + p["extra_effective_kla_guard"],
            "rejection_buffer": rejection + p["extra_effective_kla_guard"],
            "development_thresholds": records, "unit": "effective_KLa_d-1",
            "calibration_scope": "old_development_only_not_a_statistical_guarantee"}


def screening_decisions(grid: pd.DataFrame, predicted, buffers: dict, p: dict) -> pd.DataFrame:
    """Label-blind: does not accept any new-condition BSM1 result."""
    d = grid.copy()
    if any(c in d.columns for c in ("target", "bsm1_compliant", "SNH_zero_violation")):
        raise ValueError("Test target columns must not enter the screening function.")
    d["raw_compliant"] = parse_bool(predicted)
    d["recheck"] = True
    d["gate_reason"] = "uncalibrated_policy"
    for _, group in d.groupby("scenario", sort=True):
        g = group.sort_values("effective_kla")
        threshold = threshold_info(g.effective_kla, g.raw_compliant, p["minimum_compliant_tail_points"])
        d.loc[g.index, "predicted_threshold_status"] = threshold["status"]
        scope = (g.load_coordinate.between(0, 1).all()
                 and g.oxygen_transfer_factor.between(.7, 1).all()
                 and g.kla.between(40, 520).all()
                 and g.operating_point_definition.eq("steady_load_transfer_v1").all())
        if not scope:
            d.loc[g.index, "gate_reason"] = "outside_steady_domain"
            continue
        if not buffers["enabled"]:
            continue
        if threshold["status"] != "bracketed":
            d.loc[g.index, "gate_reason"] = "unresolved_or_nonmonotone_threshold"
            continue
        # New-grid bracket ends account for the coarser new KLa spacing.
        low = threshold["lower"] - buffers["rejection_buffer"]
        high = threshold["upper"] + buffers["release_buffer"]
        d.loc[g.index, "direct_reject_below_effective_kla"] = low
        d.loc[g.index, "direct_release_above_effective_kla"] = high
        reject = (~g.raw_compliant) & (g.effective_kla < low)
        retain = g.raw_compliant & (g.effective_kla > high)
        d.loc[g.index, "gate_reason"] = "threshold_and_development_error_band"
        for mask, reason in ((reject, "direct_reject"), (retain, "direct_release")):
            indices = g.index[mask]
            d.loc[indices, "recheck"] = False
            d.loc[indices, "gate_reason"] = reason
    return d


def evaluate(decisions: pd.DataFrame, reference: pd.DataFrame, costs: dict, tail_points: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use reference answers only on selected calls when reconstructing policy."""
    if reference.candidate_id.duplicated().any() or decisions.duplicated(["method", "candidate_id"]).any():
        raise ValueError("Duplicate audit cases.")
    expected = set(reference.candidate_id)
    if not expected or any(set(g.candidate_id) != expected for _, g in decisions.groupby("method")):
        raise ValueError("Every method must have the entire reference grid.")
    r = reference[["candidate_id", "scenario", "kla", "SNH_zero_violation", "elapsed_seconds", "SNH_max_eval"]].copy()
    r.columns = ["candidate_id", "reference_scenario", "reference_kla", "bsm1_compliant", "bsm1_seconds", "bsm1_max_snh"]
    d = decisions.merge(r, on="candidate_id", validate="many_to_one")
    if not d.scenario.eq(d.reference_scenario).all() or not np.allclose(d.kla, d.reference_kla):
        raise ValueError("Reference case identities disagree.")
    if not np.isfinite(d[["bsm1_seconds", "bsm1_max_snh"]].to_numpy(float)).all() or (d.bsm1_seconds <= 0).any():
        raise ValueError("Invalid reference metrics or elapsed times.")
    for c in ("raw_compliant", "recheck", "bsm1_compliant"):
        d[c] = parse_bool(d[c])
    d["final_compliant"] = np.where(d.recheck, d.bsm1_compliant, d.raw_compliant)
    for stage in ("raw", "final"):
        d[f"{stage}_classification"] = [classify_candidate(a, b) for a, b in zip(d[f"{stage}_compliant"], d.bsm1_compliant)]
    summaries, thresholds = [], []
    for method, group in d.groupby("method"):
        cost = costs[method]
        if any(not np.isfinite(v) or v < 0 for v in cost.values()):
            raise ValueError("Invalid prescreen timing.")
        for scenario, g in [("ALL", group)] + list(group.groupby("scenario")):
            summary = {"method": method, "scenario": scenario, "candidates": len(g),
                       "bsm1_compliant_candidates": int(g.bsm1_compliant.sum()),
                       "bsm1_rechecks": int(g.recheck.sum()), "bsm1_calls_avoided": int((~g.recheck).sum()),
                       "direct_releases": int(((~g.recheck) & g.raw_compliant).sum()),
                       "direct_rejections": int(((~g.recheck) & (~g.raw_compliant)).sum()),
                       "reference_case_seconds": float(g.bsm1_seconds.sum()),
                       "selected_case_seconds": float(g.loc[g.recheck, "bsm1_seconds"].sum())}
            for stage in ("raw", "final"):
                for label in ("true_safe", "true_reject", "false_safe", "conservative_reject"):
                    summary[f"{stage}_{label}"] = int(g[f"{stage}_classification"].eq(label).sum())
            summary["lost_feasible_candidates"] = summary["final_conservative_reject"]
            if scenario == "ALL":
                prescreen = sum(cost.values())
                reconstructed = prescreen + summary["selected_case_seconds"]
                summary.update(prescreen_seconds=prescreen, **cost,
                               reconstructed_workflow_seconds=reconstructed,
                               reconstructed_reduction_percent=100 * (1 - reconstructed / summary["reference_case_seconds"]),
                               timing_basis="case_time_reconstruction_not_independent_end_to_end_timing")
            else:
                ordered = g.sort_values("kla")
                ref_t = threshold_info(ordered.kla, ordered.bsm1_compliant, tail_points)
                for stage in ("raw", "final"):
                    t = threshold_info(ordered.kla, ordered[f"{stage}_compliant"], tail_points)
                    error = None if t["upper"] is None or ref_t["upper"] is None else t["upper"] - ref_t["upper"]
                    thresholds.append({"method": method, "scenario": scenario, "stage": stage,
                                       **{f"prediction_{k}": v for k, v in t.items()},
                                       **{f"reference_{k}": v for k, v in ref_t.items()},
                                       "threshold_upper_error_commanded_kla": error})
            summaries.append(summary)
    return d, pd.DataFrame(summaries), pd.DataFrame(thresholds)
