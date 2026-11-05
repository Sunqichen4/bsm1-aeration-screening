"""Controlled oxygen-demand-feedback ablation for the existing NH4 task.

The only scientific difference between ``full`` and
``no_oxygen_demand_feedback`` is deletion of -0.4*r_h -2.5*r_a from dSO/dt.
Both retain oxygen limitation of both reaction rates, all six state equations,
and the same historical biomass closure.  The ablation is a deliberate
counterfactual, not a physically validated alternative process model.

No old fitted parameter JSON or new reference labels are read.  Default
literals come from the original SurrogateParams class, whose empirical
biomass response already incorporates historical BSM1 information.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares
from scipy.special import expit

MODELS = ("full", "no_oxygen_demand_feedback")
STATE_NAMES = ("XH", "SS", "XA", "SNO", "SNH", "SO")
PARAMETER_NAMES = (
    "tau_h", "mu_h", "mu_a", "k_s", "k_nh", "k_oh", "k_oa", "hill_n",
    "so_sat", "oxygen_gain", "y_no", "ss_to_nh", "nitrification_rate_cap",
    "nh4_nitrification_factor", "biomass_tau_h", "biomass_tau_a", "xh_reference",
    "xa_reference", "bsm1_xba_asymptote", "bsm1_xba_logistic_slope",
    "bsm1_xba_logistic_midpoint", "xh_oxygen_response_weight",
    "biomass_load_exponent", "biomass_temperature_exponent",
)
FIT_NAMES = ("tau_h", "mu_a", "k_oa", "oxygen_gain", "ss_to_nh",
             "nitrification_rate_cap", "nh4_nitrification_factor")
DEFAULT_BOUNDS = {
    "tau_h": (2., 24.), "mu_a": (.2, 2.5), "k_oa": (.2, 8.),
    "oxygen_gain": (.1, 1.5), "ss_to_nh": (0., .1),
    "nitrification_rate_cap": (.2, 4.), "nh4_nitrification_factor": (1., 10.),
}
INITIAL_STATE = (1200., 75., 110., 6., 8., 2.)
UPPER_STATE = np.array((5000., 500., 1200., 80., 80., 8.))
NH4_LIMIT = 4.
PROBABILITY_TEMPERATURE = .5
EPSILON = 1e-9
PRIOR_SOURCE = (
    "Original JWPE_article_clean_package/src/aerbench/surrogate.py, "
    "SurrogateParams literal defaults; no calibrated parameter JSON loaded. "
    "The fixed normalized biomass response and scales are historical "
    "BSM1-derived empirical information shared by both process models."
)


def default_parameters() -> dict[str, float]:
    """A fresh dictionary of original class defaults, never old fitted values."""
    values = (8., .75, 1.05, 30., 1., .35, 3., 2., 8., .55, 3., .030,
              1.60, 4.8, 120., 120., 1200., 534., 152.5, .080, 123.7,
              .018, .50, 1.)
    return dict(zip(PARAMETER_NAMES, values))


def protocol_defaults() -> dict:
    return {
        "models": list(MODELS), "fit_names": list(FIT_NAMES),
        "bounds": {k: list(v) for k, v in DEFAULT_BOUNDS.items()},
        "seed": 20261014, "n_starts": 3, "max_nfev": 800,
        "initialization": "normalized_bounds_midpoint_then_two_uniform_starts",
        "random_start_interval_fraction": [.05, .95],
        "probability_temperature_mgN_L": PROBABILITY_TEMPERATURE,
        "nh4_limit_mgN_L": NH4_LIMIT,
        "objective": "unweighted_Brier_mean_squared_probability_error",
        "optimizer": {"method": "trf", "ftol": 1e-7, "xtol": 1e-8,
                      "gtol": 1e-7, "diff_step": 1e-6},
        "final_ode_timestep_min": 1., "warmup_days": 60., "eval_days": 7.,
        "equivalence_tolerance_mgN_L": .001, "near_bound_fraction": 1e-5,
        "fixed_biomass_tau_a_h": 120., "prior_source": PRIOR_SOURCE,
    }


def _protocol(protocol: dict | None) -> dict:
    p = protocol_defaults()
    supplied = {} if protocol is None else protocol.get("controlled_models", protocol)
    for key in p:
        if key in supplied:
            p[key] = supplied[key]
    if tuple(p["models"]) != MODELS or tuple(p["fit_names"]) != FIT_NAMES:
        raise ValueError("The controlled pair and seven fit names are prespecified.")
    if p["n_starts"] != 3 or p["initialization"] != protocol_defaults()["initialization"]:
        raise ValueError("Use the prespecified midpoint plus two random starts.")
    if p["nh4_limit_mgN_L"] != 4. or p["probability_temperature_mgN_L"] != .5:
        raise ValueError("The primary models are fitted once at 4 mg/L and temperature .5.")
    if p["fixed_biomass_tau_a_h"] != 120.:
        raise ValueError("The unidentifiable relaxation time remains the shared default 120 h.")
    if p["objective"] != protocol_defaults()["objective"]:
        raise ValueError("Both models must use the same unweighted Brier objective.")
    bounds = np.array([p["bounds"][name] for name in FIT_NAMES], float)
    if bounds.shape != (7, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each fit parameter needs a finite ordered pair of bounds.")
    a, b = p["random_start_interval_fraction"]
    if not 0. < a < b < 1. or int(p["max_nfev"]) != p["max_nfev"] or p["max_nfev"] < 1:
        raise ValueError("Invalid initialization interval or optimization budget.")
    if p["warmup_days"] != 60. or p["eval_days"] != 7. or p["final_ode_timestep_min"] != 1.:
        raise ValueError("Final development checks require the fixed 60+7 day, 1 minute protocol.")
    if not 0. < p["equivalence_tolerance_mgN_L"] < 1. or not 0. < p["near_bound_fraction"] < .1:
        raise ValueError("Invalid diagnostic tolerances.")
    if p["optimizer"]["method"] != "trf":
        raise ValueError("The controlled optimizer is bounded TRF least squares.")
    return p


def parameter_array(parameters) -> np.ndarray:
    if isinstance(parameters, dict) and "parameters" in parameters:
        parameters = parameters["parameters"]
    if isinstance(parameters, dict):
        values = np.array([float(parameters[name]) for name in PARAMETER_NAMES])
    else:
        values = np.asarray(parameters, dtype=float)
    if values.shape != (len(PARAMETER_NAMES),) or not np.isfinite(values).all():
        raise ValueError("All 24 process parameters must be finite.")
    positive = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17)
    if any(values[i] <= 0. for i in positive) or values[11] < 0.:
        raise ValueError("Process denominators/rates must be positive, ss_to_nh nonnegative.")
    return values


def _model_code(model: str) -> int:
    if model not in MODELS:
        raise ValueError(f"Unknown controlled model: {model}")
    return int(model == "no_oxygen_demand_feedback")


def _inputs(frame: pd.DataFrame):
    load = frame.load_coordinate.to_numpy(float)
    alpha = frame.oxygen_transfer_factor.to_numpy(float)
    kla = frame.kla.to_numpy(float)
    if not len(frame) or not np.isfinite(np.column_stack((load, alpha, kla))).all():
        raise ValueError("Nonempty, finite operating inputs required.")
    if np.any(load < 0.) or np.any(alpha <= 0.) or np.any(alpha > 1.) or np.any(kla < 0.):
        raise ValueError("Require L>=0, 0<alpha<=1, KLa>=0.")
    return load, alpha, kla


def _booleans(series) -> np.ndarray:
    text = pd.Series(series).astype(str).str.lower().str.strip()
    if not text.isin(("true", "false", "1", "0")).all():
        raise ValueError("Binary labels must be explicit true/false or 1/0.")
    return text.isin(("true", "1")).to_numpy()


def _development(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) != 750 or set(frame.scenario) != {"baseline", "high_load", "low_oxygen_transfer"}:
        raise ValueError("Training is restricted to the 750 old development cases.")
    d = frame.copy().sort_values(["scenario", "kla"]).reset_index(drop=True)
    if d.duplicated(["scenario", "kla"]).any() or not (d.groupby("scenario").size() == 250).all():
        raise ValueError("Each old scenario must contain 250 unique candidates.")
    _inputs(d)
    label_key = "SNH_zero_violation" if "SNH_zero_violation" in d else "target"
    labels = _booleans(d[label_key])
    if "target" in d and "SNH_zero_violation" in d and not np.array_equal(labels, _booleans(d.target)):
        raise ValueError("Development label columns disagree.")
    mapping = {"baseline": (0., 1.), "high_load": (1., 1.), "low_oxygen_transfer": (0., .7)}
    for name, (load, alpha) in mapping.items():
        g = d[d.scenario == name]
        if not np.allclose(g.load_coordinate, load, rtol=0., atol=1e-12) or not np.allclose(g.oxygen_transfer_factor, alpha, rtol=0., atol=1e-12):
            raise ValueError("Old development scenario mapping changed.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Development labels need both classes.")
    return d, labels


@njit(cache=True)
def _targets(p, load, alpha, kla):
    cod, nh4, activity = 260. * (1. + .20 * load), 30. * (1. + .35 * load), 1. + .05 * load
    arg = max(-60., min(60., p[19] * (alpha * kla - p[20])))
    reference_arg = max(-60., min(60., p[19] * (160. - p[20])))
    response = (1. + np.exp(-reference_arg)) / (1. + np.exp(-arg))
    xh = p[16] * (cod / 260.) ** p[22] * (1. + p[21] * (response - 1.))
    xa = p[17] * response * (nh4 / 30.) ** p[22] * activity ** p[23]
    return cod, nh4, activity, max(0., xh), max(0., xa)


def closure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed shared prior features; no fitted coefficients or labels read."""
    loads, alphas, klas = _inputs(frame)
    p = parameter_array(default_parameters())
    rows = [_targets(p, l, a, k) for l, a, k in zip(loads, alphas, klas)]
    result = pd.DataFrame(rows, columns=("cod_input", "nh4_input", "activity_factor", "XH_eq", "XA_eq"), index=frame.index)
    result.insert(0, "effective_kla", alphas * klas)
    return result


@njit(cache=True)
def _rhs(x, p, cod, nh4, activity, xheq, xaeq, alpha, kla, code):
    xh, ss, xa, sno, snh, so = x
    rh = p[1] * ss / (p[3] + ss + EPSILON) * so / (p[5] + so + EPSILON) * xh / 1000.
    oxygen_a = so ** p[7] / (p[6] ** p[7] + so ** p[7] + EPSILON)
    ra = min(p[2] * activity * snh / (p[4] + snh + EPSILON) * oxygen_a * xa / 100., p[12])
    ro = alpha * p[9] * kla / 240. * (p[8] - so)
    demand = .4 * rh + 2.5 * ra if code == 0 else 0.
    return ((xheq - xh) / p[14], (cod - ss) / p[0] - 18. * rh,
            (xaeq - xa) / p[15], (2. - sno) / p[0] + p[10] * ra - .25 * rh,
            (nh4 - snh) / p[0] + p[11] * max(0., ss) / p[0] - p[13] * ra,
            (.4 - so) / p[0] + ro - demand)


@njit(cache=True)
def _add(x, k, scale):
    return (x[0]+scale*k[0], x[1]+scale*k[1], x[2]+scale*k[2],
            x[3]+scale*k[3], x[4]+scale*k[4], x[5]+scale*k[5])


@njit(cache=True)
def _advance(x, p, cod, nh4, activity, xheq, xaeq, alpha, kla, dt, code):
    k1 = _rhs(x, p, cod, nh4, activity, xheq, xaeq, alpha, kla, code)
    k2 = _rhs(_add(x, k1, dt*.5), p, cod, nh4, activity, xheq, xaeq, alpha, kla, code)
    k3 = _rhs(_add(x, k2, dt*.5), p, cod, nh4, activity, xheq, xaeq, alpha, kla, code)
    k4 = _rhs(_add(x, k3, dt), p, cod, nh4, activity, xheq, xaeq, alpha, kla, code)
    h = dt/6.
    return (x[0]+h*(k1[0]+2.*k2[0]+2.*k3[0]+k4[0]),
            x[1]+h*(k1[1]+2.*k2[1]+2.*k3[1]+k4[1]),
            x[2]+h*(k1[2]+2.*k2[2]+2.*k3[2]+k4[2]),
            x[3]+h*(k1[3]+2.*k2[3]+2.*k3[3]+k4[3]),
            x[4]+h*(k1[4]+2.*k2[4]+2.*k3[4]+k4[4]),
            x[5]+h*(k1[5]+2.*k2[5]+2.*k3[5]+k4[5]))


@njit(cache=True)
def _simulate(p, load, alpha, kla, dt_min, warmup_d, eval_d, code):
    dt = dt_min / 60.
    nw, ne = int(round(warmup_d * 24. / dt)), int(round(eval_d * 24. / dt))
    cod, nh4, activity, xheq, xaeq = _targets(p, load, alpha, kla)
    x = INITIAL_STATE
    low, high = np.full(6, np.inf), np.full(6, -np.inf)
    lower, upper = np.zeros(6, np.int64), np.zeros(6, np.int64)
    raw_min = np.full(6, np.inf)
    snh_sum, sno_sum, so_sum, violations = 0., 0., 0., 0
    for step in range(nw + ne):
        if step >= nw:
            for j in range(6):
                low[j], high[j] = min(low[j], x[j]), max(high[j], x[j])
            snh_sum, sno_sum, so_sum = snh_sum + x[4], sno_sum + x[3], so_sum + x[5]
            violations += int(x[4] > NH4_LIMIT)
        candidate = _advance(x, p, cod, nh4, activity, xheq, xaeq, alpha, kla, dt, code)
        for j in range(6):
            raw_min[j] = min(raw_min[j], candidate[j])
            lower[j] += int(candidate[j] < 0.)
            upper[j] += int(candidate[j] > UPPER_STATE[j])
        x = (min(UPPER_STATE[0], max(0., candidate[0])), min(UPPER_STATE[1], max(0., candidate[1])),
             min(UPPER_STATE[2], max(0., candidate[2])), min(UPPER_STATE[3], max(0., candidate[3])),
             min(UPPER_STATE[4], max(0., candidate[4])), min(UPPER_STATE[5], max(0., candidate[5])))
    return snh_sum/ne, violations*dt, sno_sum/ne, so_sum/ne, low, high, lower, upper, raw_min


@njit(cache=True)
def _positive_root(b, c):
    discriminant = np.sqrt(b*b + 4.*c)
    return 2.*c/(b+discriminant) if b >= 0. else .5*(-b+discriminant)


@njit(cache=True)
def _equilibrium_given_so(so, p, cod, nh4, activity, xh, xa):
    ah = p[1] * so / (p[5] + so + EPSILON) * xh / 1000.
    ks = p[3] + EPSILON
    ss = _positive_root(ks + p[0]*18.*ah - cod, cod*ks)
    rh = ah * ss / (ks + ss)
    aa = p[2] * activity * so**p[7] / (p[6]**p[7] + so**p[7] + EPSILON) * xa / 100.
    incoming = nh4 + p[11]*ss
    kn = p[4] + EPSILON
    snh = _positive_root(kn + p[0]*p[13]*aa - incoming, incoming*kn)
    ra = aa * snh / (kn + snh)
    if ra > p[12]:
        ra, snh = p[12], incoming - p[0]*p[13]*p[12]
    return ss, max(0., snh), rh, ra


@njit(cache=True)
def _steady_values(p, load, alpha, kla, code):
    cod, nh4, activity, xh, xa = _targets(p, load, alpha, kla)
    gain = alpha * kla * p[9] / 240.
    if code == 1:
        so = (.4/p[0] + gain*p[8]) / (1./p[0] + gain)
    else:
        left, right = 0., p[8]
        for _ in range(48):
            so = .5*(left + right)
            _, _, rh, ra = _equilibrium_given_so(so, p, cod, nh4, activity, xh, xa)
            balance = (.4-so)/p[0] + gain*(p[8]-so) - .4*rh - 2.5*ra
            if balance > 0.:
                left = so
            else:
                right = so
        so = .5*(left+right)
    ss, snh, rh, ra = _equilibrium_given_so(so, p, cod, nh4, activity, xh, xa)
    sno = 2. + p[0]*(p[10]*ra - .25*rh)
    return (xh, ss, xa, sno, snh, so)


@njit(cache=True)
def _steady_batch(p, loads, alphas, klas, code):
    result = np.empty(len(loads))
    for i in range(len(loads)):
        result[i] = _steady_values(p, loads[i], alphas[i], klas[i], code)[4]
    return result


def steady_predictions(frame, parameters, model="full") -> np.ndarray:
    return _steady_batch(parameter_array(parameters), *_inputs(frame), _model_code(model))


def predict_cases(frame, parameters, model="full", dt_minutes=1., *, warmup_days=60., eval_days=7.) -> pd.DataFrame:
    """Fixed initial state, RK4, step-start evaluation [warmup,warmup+eval).

    Only operating inputs and identifiers are read; target columns are ignored.
    The output contains state summaries, not complete state trajectories.
    """
    if dt_minutes <= 0. or warmup_days < 0. or eval_days <= 0.:
        raise ValueError("Positive timestep/evaluation and nonnegative warmup required.")
    counts = np.array((warmup_days, eval_days))*1440./dt_minutes
    if not np.isfinite(counts).all() or not np.allclose(counts, np.round(counts), rtol=0., atol=1e-7):
        raise ValueError("Integration windows must align to complete timesteps.")
    p, code = parameter_array(parameters), _model_code(model)
    loads, alphas, klas = _inputs(frame)
    _simulate(p, 0., 1., 160., dt_minutes, 0., dt_minutes/1440., code)
    rows = []
    for i, (_, case) in enumerate(frame.iterrows()):
        started = time.perf_counter()
        mean, duration, sno, so, low, high, lower, upper, raw_min = _simulate(
            p, loads[i], alphas[i], klas[i], dt_minutes, warmup_days, eval_days, code)
        elapsed = time.perf_counter() - started
        if not np.isfinite([mean, duration, sno, so, *low, *high, *raw_min]).all():
            raise FloatingPointError(f"Nonfinite {model} result at row {i}.")
        row = {"candidate_id": case.get("candidate_id", f"{model}_{i:06d}"),
               "scenario": case.get("scenario", "unspecified"), "model": model,
               "kla": klas[i], "load_coordinate": loads[i], "oxygen_transfer_factor": alphas[i],
               "effective_kla": klas[i]*alphas[i], "SNH_mean_eval": mean,
               "SNH_max_eval": high[4], "SNH_zero_violation": bool(duration == 0.),
               "SNH_violation_hours_gt_limit": duration, "SNO_mean_eval": sno, "DO_mean_eval": so,
               "probability_feasible": float(expit((NH4_LIMIT-high[4])/PROBABILITY_TEMPERATURE)),
               "elapsed": elapsed, "runtime_seconds": elapsed, "integrator": "rk4",
               "dt_minutes": dt_minutes, "warmup_days": warmup_days, "eval_days": eval_days,
               "evaluation_samples": int(round(counts[1])),
               "evaluation_alignment": "step_start_[warmup,warmup+eval)",
               "lower_clips_total": int(lower.sum()), "upper_clips_total": int(upper.sum())}
        for j, name in enumerate(STATE_NAMES):
            row.update({f"{name}_min_eval": low[j], f"{name}_max_eval": high[j],
                        f"{name}_min_unbounded": raw_min[j], f"{name}_lower_clips": int(lower[j]),
                        f"{name}_upper_clips": int(upper[j])})
        rows.append(row)
    return pd.DataFrame(rows)


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _json_save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fit_pair(development_frame, output_dir, protocol=None) -> dict[str, dict]:
    """Fit both models on all 750 old labels and perform full finite-time checks.

    Returns per-model metadata including numerical suitability status. Failure
    of an audit is preserved and never triggers model/bounds/start adaptation.
    An existing training directory is not overwritten.
    """
    p = _protocol(protocol)
    d, target_bool = _development(development_frame)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"fit_design.json").exists() or any((out/m/"parameters.json").exists() for m in MODELS):
        raise FileExistsError("Existing controlled training artifacts must not be overwritten.")
    labels = target_bool.astype(float)
    loads, alphas, klas = _inputs(d)
    frozen = parameter_array(default_parameters())
    indexes = np.array([PARAMETER_NAMES.index(name) for name in FIT_NAMES])
    bounds = np.array([p["bounds"][name] for name in FIT_NAMES], float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    span = upper-lower
    rng = np.random.default_rng(p["seed"])
    starts = np.vstack((np.full(len(FIT_NAMES), .5), rng.uniform(*p["random_start_interval_fraction"], size=(2, len(FIT_NAMES)))))
    labels_payload = d[["scenario", "load_coordinate", "oxygen_transfer_factor", "kla"]].copy()
    labels_payload["label"] = target_bool
    label_hash = hashlib.sha256(labels_payload.to_csv(index=False, float_format="%.17g").encode("utf-8")).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    design = {"protocol": p, "source_sha256": source_hash, "development_label_sha256": label_hash,
              "shared_default_parameters": default_parameters(), "prior_source": PRIOR_SOURCE,
              "initial_normalized_vectors": starts.tolist(),
              "initial_parameter_vectors": (lower + starts*span).tolist(),
              "initialization_excludes_old_fitted_parameters": True,
              "conditional_comparison": "Only oxygen-demand feedback in dSO; all historical closure information shared.",
              "fixed_parameters": [name for name in PARAMETER_NAMES if name not in FIT_NAMES],
              "strictly_unidentifiable_in_steady_training": ["biomass_tau_a", "biomass_tau_h"],
              "new_reference_labels_read": False}
    design["design_sha256"] = _canonical_hash(design)
    _json_save(out/"fit_design.json", design)
    result = {}
    for model in MODELS:
        directory = out/model
        directory.mkdir(exist_ok=True)
        code = _model_code(model)
        _steady_batch(frozen, loads[:1], alphas[:1], klas[:1], code)
        evaluations, runs, candidates = [], [], []

        def residual(theta, start_index):
            parameters = frozen.copy()
            parameters[indexes] = lower + span*theta
            snh = _steady_batch(parameters, loads, alphas, klas, code)
            errors = expit((NH4_LIMIT-snh)/PROBABILITY_TEMPERATURE)-labels
            if not np.isfinite(errors).all():
                raise FloatingPointError("Nonfinite controlled training objective.")
            evaluations.append({"start_index": start_index, "call_index": len(evaluations)+1,
                                "Brier_score": float(np.mean(errors**2))})
            return errors

        fit_started = time.perf_counter()
        for start_index, initial in enumerate(starts):
            initial_loss = float(np.mean(residual(initial, start_index)**2))
            started = time.perf_counter()
            fitted = least_squares(lambda theta: residual(theta, start_index), initial,
                                   bounds=(0., 1.), max_nfev=int(p["max_nfev"]), **p["optimizer"])
            elapsed = time.perf_counter()-started
            values = lower + span*fitted.x
            parameters = frozen.copy()
            parameters[indexes] = values
            candidates.append(parameters)
            singular_values = np.linalg.svd(fitted.jac, compute_uv=False)
            run = {"start_index": start_index, "success": bool(fitted.success), "status": int(fitted.status),
                   "message": str(fitted.message), "nfev": int(fitted.nfev), "njev": int(fitted.njev or 0),
                   "elapsed_seconds": elapsed, "initial_loss": initial_loss,
                   "final_loss": float(np.mean(fitted.fun**2)), "loss_improvement": initial_loss-float(np.mean(fitted.fun**2)),
                   "optimality": float(fitted.optimality), "jacobian_rank": int(np.linalg.matrix_rank(fitted.jac)),
                   "jacobian_singular_values": json.dumps(singular_values.tolist())}
            for j, name in enumerate(FIT_NAMES):
                run.update({f"initial_{name}": float(lower[j]+span[j]*initial[j]),
                            f"final_{name}": float(values[j]), f"lower_{name}": float(lower[j]),
                            f"upper_{name}": float(upper[j]),
                            f"near_lower_{name}": bool(fitted.x[j] <= p["near_bound_fraction"]),
                            f"near_upper_{name}": bool(1.-fitted.x[j] <= p["near_bound_fraction"])})
            runs.append(run)
            pd.DataFrame(runs).to_csv(directory/"optimizer_runs.csv", index=False)
            pd.DataFrame(evaluations).to_csv(directory/"optimizer_evaluations.csv", index=False)
        fit_elapsed = time.perf_counter()-fit_started
        best = int(np.argmin([r["final_loss"] for r in runs]))
        parameters = dict(zip(PARAMETER_NAMES, candidates[best].tolist()))
        metadata = {"model": model, "parameters": parameters, "selected_start": best,
                    "fit_names": list(FIT_NAMES), "n_training_cases": 750,
                    "supervision": "old_750_binary_labels_only_for_current_fit",
                    "prior_source": PRIOR_SOURCE, "protocol": p, "design_sha256": design["design_sha256"],
                    "source_sha256": source_hash, "development_label_sha256": label_hash,
                    "fit_runtime_seconds": fit_elapsed, "selected_jacobian_rank": runs[best]["jacobian_rank"],
                    "all_starts_optimizer_success": all(r["success"] for r in runs),
                    "selected_optimizer_success": runs[best]["success"],
                    "finite_time_equivalence_status": "pending",
                    "strictly_unidentifiable_fixed_relaxation_parameters": ["biomass_tau_h", "biomass_tau_a"],
                    "historical_closure_shared_by_pair": True, "new_reference_labels_read": False}
        _json_save(directory/"parameters.json", metadata)
        steady = steady_predictions(d, parameters, model)
        started = time.perf_counter()
        finite = predict_cases(d, parameters, model, dt_minutes=p["final_ode_timestep_min"],
                               warmup_days=p["warmup_days"], eval_days=p["eval_days"])
        finite_elapsed = time.perf_counter()-started
        finite["target"] = target_bool
        finite["steady_SNH"] = steady
        finite["steady_finite_abs_difference"] = np.abs(finite.SNH_max_eval-steady)
        finite["steady_finite_label_disagreement"] = (steady <= NH4_LIMIT) != finite.SNH_zero_violation
        finite.to_csv(directory/"development_predictions.csv", index=False)
        audit = {"model": model, "cases": len(d), "dt_minutes": p["final_ode_timestep_min"],
                 "warmup_days": 60., "eval_days": 7., "finite_prediction_elapsed_seconds": finite_elapsed,
                 "max_abs_SNH_difference": float(finite.steady_finite_abs_difference.max()),
                 "mean_abs_SNH_difference": float(finite.steady_finite_abs_difference.mean()),
                 "n_steady_finite_label_disagreements": int(finite.steady_finite_label_disagreement.sum()),
                 "n_ode_label_errors": int(np.count_nonzero(finite.SNH_zero_violation.to_numpy() != target_bool)),
                 "total_lower_clips": int(finite.lower_clips_total.sum()),
                 "total_upper_clips": int(finite.upper_clips_total.sum()),
                 "state_clips": {name: {"lower": int(finite[f"{name}_lower_clips"].sum()),
                                        "upper": int(finite[f"{name}_upper_clips"].sum())} for name in STATE_NAMES}}
        equivalent = audit["max_abs_SNH_difference"] < p["equivalence_tolerance_mgN_L"] and audit["n_steady_finite_label_disagreements"] == 0
        metadata["finite_time_equivalence_status"] = "verified_on_all_750" if equivalent else "FAILED_requires_explicit_protocol_review"
        metadata["finite_time_audit"] = audit
        metadata["eligible_for_new_predictions"] = bool(equivalent and metadata["all_starts_optimizer_success"])
        _json_save(directory/"steady_finite_audit.json", audit)
        _json_save(directory/"parameters.json", metadata)
        result[model] = metadata
    _json_save(out/"fit_pair_complete.json", {"design_sha256": design["design_sha256"],
                                            "models": {m: {"eligible_for_new_predictions": result[m]["eligible_for_new_predictions"],
                                                           "fit_runtime_seconds": result[m]["fit_runtime_seconds"]} for m in MODELS}})
    return result
