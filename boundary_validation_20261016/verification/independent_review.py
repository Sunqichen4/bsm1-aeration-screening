"""Independent artifact-only M3B audit. No analyzer imports, fitting or ODE.

Run after main/half references, their raw audits and boundary analysis finish:
python verification/independent_review.py [study_directory]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODELS = ("full_calibrated", "full_fixed_ratio", "no_feedback", "poly_ridge",
          "rbf_kernel_ridge", "legacy_M2_binary")
STRATA = ("coarse", "prediction_enriched", "all")
WINDOWS = (.25, .5, 1.)
M3_FREEZE_SHA = "c5a5aa2e6ea73d2c0b143d1d011529d4d1b4b89dec29fba293dcbf0283f43965"
M3_MANIFEST_SHA = "300aacad5d3833e44af39ffde9105609e3f662cae6123fa2062d6ecd7419dfc3"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def require(value, message):
    if not value:
        raise ValueError(message)


def booleans(values):
    parsed = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, str) and value.lower().strip() in ("true", "false", "0", "1"):
            parsed.append(value.lower().strip() in ("true", "1"))
        elif isinstance(value, (int, np.integer)) and value in (0, 1):
            parsed.append(bool(value))
        else:
            raise ValueError("Invalid saved boolean: " + repr(value))
    return np.array(parsed, dtype=bool)


def finite(values, label):
    array = np.asarray(values, dtype=float)
    require(np.isfinite(array).all(), "Nonfinite values: " + label)
    return array


def json_records(frame):
    # Native JSON encodings preserve nulls for missing sides without NaN tokens.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def subset(frame, stratum):
    return frame if stratum == "all" else frame.loc[frame.grid_kind.eq(stratum)]


def compare_table(calculated, recorded, keys, label):
    require(not calculated.duplicated(keys).any(), label+": recomputed duplicate keys")
    require(not recorded.duplicated(keys).any(), label+": saved duplicate keys")
    a = calculated.set_index(keys).sort_index()
    b = recorded.set_index(keys).sort_index()
    require(a.index.equals(b.index), label+": row identities differ")
    require(set(a.columns) == set(b.columns), label+": metric columns differ")
    worst = 0.
    for column in a:
        left, right = a[column], b[column]
        missing = left.isna().to_numpy()
        require(np.array_equal(missing, right.isna().to_numpy()), label+": missing-value mismatch "+column)
        left, right = left.loc[~missing], right.loc[~missing]
        if pd.api.types.is_bool_dtype(left.dtype):
            require(np.array_equal(booleans(left), booleans(right)), label+": boolean mismatch "+column)
        elif pd.api.types.is_numeric_dtype(left.dtype):
            x, y = finite(left, column), finite(right, column)
            require(np.allclose(x, y, rtol=1e-12, atol=1e-10), label+": numeric mismatch "+column)
            worst = max(worst, float(np.max(np.abs(x-y), initial=0.)))
        else:
            require(np.array_equal(left.to_numpy(), right.to_numpy()), label+": identity mismatch "+column)
    return {"passed": True, "rows": len(a), "columns": len(a.columns),
            "maximum_absolute_numeric_difference": worst}


def origin_and_protection(study, m3_root, skip_external=False):
    require(digest(study/"source_M3_freeze.json") == M3_FREEZE_SHA, "M3 freeze identity changed")
    require(digest(study/"source_M3_manifest.json") == M3_MANIFEST_SHA, "M3 source manifest identity changed")
    source = read_json(study/"source_M3_manifest.json")
    hashes = {e["path"]: e["sha256"] for e in source["files"]}
    names = ("outputs/development/models.json", "outputs/development/gates.json", "process_model.py",
             "statistical_models.py", "run_study.py", "legacy/controlled_models.py", "legacy/M2_full_parameters.json")
    for name in names:
        require(digest(study/name) == hashes[name], "M3 model/gate/source changed: "+name)
    internal = {"passed": True, "model_gate_source_files_checked": len(names),
                "model_sha256": digest(study/names[0]), "gate_sha256": digest(study/names[1]),
                "historical_label_values_loaded": False}
    if skip_external:
        return dict(internal, external_protection_status="SKIPPED_EXPLICITLY",
                    original_M3_files_checked=None, pre_M3_protected_files_checked=None,
                    external_protection_reason="Explicit --skip-external-protection; package-internal origin snapshots and scientific sources verified only.")
    for entry in source["files"]:
        require(digest(m3_root/entry["path"]) == entry["sha256"], "Original M3 file changed: "+entry["path"])
    # Only hash historical protected files; do not load their scientific labels.
    original_manifest = read_json(m3_root/"source_manifest.json")
    checks = 0
    for entry in original_manifest["files"]:
        relative = entry.get("protected_source", entry.get("source"))
        if relative is None:
            continue
        require(digest(m3_root.parent/relative) == entry["sha256"], "Protected original file changed: "+relative)
        checks += 1
    return dict(internal, external_protection_status="VERIFIED",
                original_M3_files_checked=len(source["files"]), pre_M3_protected_files_checked=checks)


def freeze_check(study):
    freeze = read_json(study/"freeze.json")
    seen = set()
    for entry in freeze["files"]:
        relative = entry["path"]
        path = (study/relative).resolve()
        require(path.is_relative_to(study) and relative not in seen, "Unsafe or repeated frozen path")
        require(digest(path) == entry["sha256"], "Frozen artifact changed: "+relative)
        seen.add(relative)
    require({"new_candidates.csv", "frozen_predictions.csv", "half_step_candidates.csv",
             "outputs/development/models.json", "outputs/development/gates.json"} <= seen,
            "Essential scientific artifacts are not frozen")
    return freeze, {"passed": True, "frozen_file_count": len(seen), "sha256": digest(study/"freeze.json")}


def join_from_raw_tables(candidates, predictions, references, gates):
    require(candidates.candidate_id.is_unique and references.candidate_id.is_unique, "Repeated candidate/reference identities")
    require(not predictions.duplicated(["candidate_id", "model"]).any(), "Repeated prediction identities")
    ids = set(candidates.candidate_id)
    require(set(references.candidate_id) == ids, "Reference coverage differs")
    require(set(predictions.model) == set(MODELS), "Six-model identity mismatch")
    for name in MODELS:
        require(set(predictions.loc[predictions.model.eq(name), "candidate_id"]) == ids, "Incomplete model: "+name)
    require(references.status.eq("success").all(), "Failed references present")
    require(not candidates.duplicated(["scenario", "effective_kla"]).any(), "Repeated physical inputs")
    canonical = candidates.set_index("candidate_id")
    for label, frame in (("prediction", predictions), ("reference", references)):
        for column in ("scenario", "kla", "effective_kla", "grid_kind", "load_coordinate", "oxygen_transfer_factor"):
            if column in frame:
                actual, expected = frame[column].to_numpy(), frame.candidate_id.map(canonical[column]).to_numpy()
                # The native wrapper reports effective z as alpha*KLa. Only
                # this derived reference coordinate can differ by roundoff;
                # commanded KLa and every frozen prediction coordinate are exact.
                same = np.allclose(actual, expected, rtol=0., atol=1e-12) if label == "reference" and column == "effective_kla" else np.array_equal(actual, expected)
                require(same, "Saved coordinate mismatch: "+label+"."+column)
    values = references[["candidate_id", "SNH_max_eval", "SNH_zero_violation"]].copy()
    truth = finite(values.SNH_max_eval, "reference peaks") <= 4.
    require(np.array_equal(truth, booleans(values.SNH_zero_violation)), "Reference threshold classification mismatch")
    values["truth"] = truth
    joined = predictions.merge(values, on="candidate_id", validate="many_to_one")
    peak = finite(joined.predicted_peak, "model peaks")
    require((peak >= 0).all(), "Negative model peak")
    raw = peak <= 4.
    require(np.array_equal(raw, booleans(joined.raw_compliant)), "Raw threshold mismatch")
    buffer = joined.model.map({k: v["absolute_peak_buffer_mgN_L"] for k,v in gates.items()}).to_numpy()
    require(np.isfinite(buffer).all() and (buffer >= 0).all(), "Invalid model buffers")
    require(np.array_equal(buffer, joined.buffer_mgN_L.to_numpy()), "Frozen buffer mismatch")
    load = joined.load_coordinate.to_numpy()
    alpha = joined.oxygen_transfer_factor.to_numpy()
    z = joined.effective_kla.to_numpy()
    joint = (load >= 0)&(load <= 1)&(alpha >= .7+.3*load-1e-12)&(alpha <= 1.)
    feature = (z >= 28.+12.*load-1e-12)&(z <= 520.)
    release = (peak+buffer < 4.)&joint&feature
    reject = (peak-buffer > 4.)&joint&feature
    route = ~(release|reject)
    for name, array in (("in_joint_domain", joint), ("in_feature_domain", feature),
                        ("direct_release", release), ("direct_reject", reject), ("recheck", route)):
        require(np.array_equal(array, booleans(joined[name])), "Frozen Gate route mismatch: "+name)
    joined["raw"] = raw
    joined["check"] = route
    joined["final"] = np.where(route, joined.truth.to_numpy(), raw)
    return joined


def summary_row(group):
    error = group.predicted_peak.to_numpy()-group.SNH_max_eval.to_numpy()
    truth, raw, check, final = (group[k].to_numpy(dtype=bool) for k in ("truth", "raw", "check", "final"))
    false_release, false_reject = raw&~truth, ~raw&truth
    wrong = raw != truth
    return dict(n_candidates=len(group), reference_compliant_count=int(truth.sum()),
        rmse=float(np.sqrt(np.dot(error, error)/len(error))), mae=float(np.abs(error).mean()),
        mean_error=float(error.mean()), raw_false_releases=int(false_release.sum()),
        raw_false_rejections=int(false_reject.sum()), raw_errors=int(wrong.sum()),
        gate_selected=int(check.sum()), gate_selected_fraction=float(check.sum()/len(group)),
        gate_corrected_false_releases=int((check&false_release).sum()),
        gate_corrected_false_rejections=int((check&false_reject).sum()),
        gate_corrected_errors=int((check&wrong).sum()), gate_missed_false_releases=int((~check&false_release).sum()),
        gate_missed_false_rejections=int((~check&false_reject).sum()), gate_missed_errors=int((~check&wrong).sum()),
        gate_rechecks_of_correct_raw_predictions=int((check&~wrong).sum()),
        final_false_releases=int((final&~truth).sum()), final_false_rejections=int((~final&truth).sum()),
        final_errors=int((final != truth).sum()))


def metrics_tables(joined):
    overall, conditions = [], []
    for stratum in STRATA:
        part = subset(joined, stratum)
        if part.empty:
            continue
        for model in MODELS:
            group = part.loc[part.model.eq(model)]
            overall.append(dict(stratum=stratum, model=model, **summary_row(group)))
            for scenario, condition in group.groupby("scenario"):
                conditions.append(dict(stratum=stratum, model=model, scenario=scenario, **summary_row(condition)))
    return pd.DataFrame(overall), pd.DataFrame(conditions)


def coverage_table(candidates, references):
    truth = candidates.merge(references[["candidate_id", "SNH_max_eval"]], on="candidate_id", validate="one_to_one")
    rows = []
    for stratum in STRATA:
        for scenario in sorted(truth.scenario.unique()):
            part = subset(truth, stratum)
            group = part.loc[part.scenario.eq(scenario)].sort_values(["effective_kla", "candidate_id"])
            peaks = group.SNH_max_eval.to_numpy()
            safe = peaks <= 4.
            row = dict(stratum=stratum, scenario=scenario, n_candidates=len(group),
                compliant_count=int(safe.sum()), noncompliant_count=int((~safe).sum()),
                has_both_sides=bool(safe.any() and (~safe).any()))
            for label, mask in (("compliant", safe), ("noncompliant", ~safe)):
                positions = np.flatnonzero(mask)
                nearest = group.iloc[positions[np.argmin(np.abs(peaks[positions]-4.))]] if len(positions) else None
                row[f"nearest_{label}_distance_mgN_L"] = abs(float(nearest.SNH_max_eval)-4.) if nearest is not None else None
                row[f"nearest_{label}_candidate_id"] = str(nearest.candidate_id) if nearest is not None else None
                row[f"nearest_{label}_peak_mgN_L"] = float(nearest.SNH_max_eval) if nearest is not None else None
                row[f"nearest_{label}_effective_kla"] = float(nearest.effective_kla) if nearest is not None else None
            for window in WINDOWS:
                suffix = str(window).replace(".", "p")
                near = np.abs(peaks-4.) <= window
                row[f"within_{suffix}_count"] = int(near.sum())
                row[f"within_{suffix}_compliant_count"] = int((near&safe).sum())
                row[f"within_{suffix}_noncompliant_count"] = int((near&~safe).sum())
            change = safe[:-1] != safe[1:]
            row["adjacent_label_change_count"] = int(change.sum())
            row["adjacent_strict_sign_change_count"] = int((((peaks[:-1]-4.)*(peaks[1:]-4.)) < 0.).sum())
            row["noncompliant_to_compliant_count"] = int((~safe[:-1]&safe[1:]).sum())
            row["compliant_to_noncompliant_count"] = int((safe[:-1]&~safe[1:]).sum())
            row["exact_threshold_count"] = int((peaks == 4.).sum())
            gaps = np.diff(group.effective_kla.to_numpy())[change]
            row["smallest_observed_label_change_z_gap"] = float(gaps.min()) if len(gaps) else None
            rows.append(row)
    return pd.DataFrame(rows)


def half_review(study, references, frozen_sha):
    selected = read_csv(study/"half_step_candidates.csv")
    half = read_csv(study/"numerical/half_step_summary.csv")
    require(selected.candidate_id.is_unique and half.candidate_id.is_unique, "Repeated half-step identities")
    require(set(half.candidate_id) == set(selected.candidate_id), "Half-step coverage differs")
    require(half.status.eq("success").all(), "Failed half-step reference")
    main = references.set_index("candidate_id")
    primary_peaks = half.candidate_id.map(main.SNH_max_eval).to_numpy()
    require(np.array_equal(primary_peaks, half.main_SNH_max_eval.to_numpy()), "Half-step main peaks mismatch")
    half_peaks = finite(half.SNH_max_eval, "half-step peaks")
    difference = np.abs(half_peaks-primary_peaks)
    flipped = (half_peaks <= 4.) != (primary_peaks <= 4.)
    require(np.allclose(difference, half.absolute_peak_difference.to_numpy(), rtol=0., atol=1e-12), "Saved half-step difference mismatch")
    require(np.array_equal(flipped, booleans(half.label_changed)), "Saved half-step flips mismatch")
    require(np.array_equal(half_peaks <= 4., booleans(half.SNH_zero_violation)), "Half-step labels mismatch")
    require(np.array_equal(primary_peaks <= 4., booleans(half.main_SNH_zero_violation)), "Saved main half-subset labels mismatch")
    audit = read_json(study/"numerical/half_step_audit.json")
    require(audit["passed"] is True and audit["freeze_sha256"] == frozen_sha, "Half-step raw audit failed or stale")
    require(audit["expected_count"] == len(half) == audit["successful_count"] and audit["failed_count"] == 0, "Half-step audit counts mismatch")
    require(math.isclose(float(difference.max()), audit["maximum_absolute_peak_difference"], rel_tol=0., abs_tol=1e-12), "Half-step audited maximum mismatch")
    require(int(flipped.sum()) == audit["label_change_count"], "Half-step audited flip count mismatch")
    return {"passed": True, "n_candidates": len(half), "maximum_absolute_peak_difference": float(difference.max()),
            "label_change_count": int(flipped.sum()),
            "flipped_candidate_ids": half.loc[flipped, "candidate_id"].tolist(),
            "selection_is_pre_reference_frozen": True,
            "main_labels_replaced": False}


def raw_peak_confirmation(study, table, frozen_sha, half=False):
    """Recheck the target peak directly from saved effluent arrays, no native code."""
    base = study/"numerical" if half else study
    samples = 20160 if half else 10080
    dt = .5 if half else 1.
    for row in table.itertuples(index=False):
        identity = row.candidate_id
        require(identity.replace("_", "").isalnum(), "Unexpected candidate filename identity")
        metadata = read_json(base/"reference"/(identity+".json"))
        raw = base/"reference_trajectories"/(identity+"_evaluation.npz")
        require(metadata["candidate_id"] == identity and metadata["freeze_sha256"] == frozen_sha,
                "Raw metadata identity mismatch: "+identity)
        require(digest(raw) == metadata["raw_file_sha256"] == row.raw_file_sha256,
                "Raw trajectory hash mismatch: "+identity)
        with np.load(raw, allow_pickle=False) as arrays:
            columns = arrays["state_columns"].tolist()
            require(columns.count("SNH") == 1, "NH4 column identity mismatch")
            peaks = finite(arrays["effluent_states"][:,columns.index("SNH")], "raw effluent NH4")
            times = finite(arrays["time_day"], "raw endpoint times")
            require(len(peaks) == len(times) == samples, "Raw reporting window sample count mismatch")
            expected = 60.+np.arange(samples)*dt/1440.
            require(np.allclose(times, expected, rtol=0., atol=1e-10), "Raw reporting endpoints mismatch")
            require((times >= 60.-1e-10).all() and (times < 67.).all(), "Raw reporting window mismatch")
            peak = float(peaks.max())
            require(math.isclose(peak, row.SNH_max_eval, rel_tol=0., abs_tol=1e-12), "Raw/CSV peak mismatch")
            require(math.isclose(peak, metadata["SNH_max_eval"], rel_tol=0., abs_tol=1e-12), "Raw/JSON peak mismatch")
            require(bool(peak <= 4.) == booleans([row.SNH_zero_violation])[0], "Raw/CSV classification mismatch")
    return {"passed": True, "count": len(table), "samples_per_case": samples,
            "effluent_column": "SNH", "native_code_imported": False}


def markdown(result):
    lines = ["# M3B结果独立复核", "", "**复核通过。** 本脚本直接读取冻结预测、输入及参考汇总，独立计算分类、Gate与指标；未导入正式分析器，未拟合或积分。", "",
        f"覆盖{result['candidate_count']}个候选、6模型，共{result['prediction_count']}条预测。总体/分层表、工况表与双侧覆盖表均匹配正式分析，容差为绝对1e-10及相对1e-12。所有科学CSV采用round_trip解析。", "",
        "| 分层 | 模型 | RMSE | MAE | 原始误放/误拒 | 核验数 | 最终误放/误拒 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in result["metrics"]:
        lines.append(f"| {r['stratum']} | {r['model']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['raw_false_releases']}/{r['raw_false_rejections']} | {r['gate_selected']}/{r['n_candidates']} | {r['final_false_releases']}/{r['final_false_rejections']} |")
    lines += ["", "浓度单位为mg N/L。108点固定网格为主比较；legacy预测交叉引导加密层单列，混合总体不表示自然分布。", "",
        f"固定网格双侧覆盖：{result['coarse_conditions_with_both_sides']}/6工况；距阈值0.25、0.5及1.0范围的数量与逐工况最近两侧距离见JSON。", "",
        f"半步长{result['half_step']['n_candidates']}点的最大峰值差为{result['half_step']['maximum_absolute_peak_difference']:.9g}，分类翻转{result['half_step']['label_change_count']}点。主参考标签保留。", "",
        (f"原M3保护清单核对{result['origin_protection']['original_M3_files_checked']}个文件；更早原稿/程序保护清单核对{result['origin_protection']['pre_M3_protected_files_checked']}个文件，均一致。" if result['origin_protection']['external_protection_status'] == 'VERIFIED' else
         "本次按显式选项跳过包外原M3及更早工作树保护核查，仅核查包内来源快照与7项模型/Gate/科学源码字节。包外文件是否保持一致未验证。"), "",
        "六个工况为设计条件，点位共享工况与模型，不能将候选视为独立同分布总体样本。采样双侧不证明全局单调或连续最优；零观察错误不构成总体风险保证。匹配反馈结论使用两个五参数模型；六参数完整模型及历史二元监督M2分别保留其比较身份。", "",
        "本复核不覆盖新的文献、真实厂测量或未采样工况。本脚本还独立从主/半步长NPZ效流SNH列重算峰值、检查报告时刻及轨迹哈希；原生依赖、初始化和输入构造的完整审计由绑定相同冻结身份的原始参考审计提供。", ""]
    return "\n".join(lines)


def review(study=ROOT, m3_root=None, skip_external_protection=False):
    study = Path(study).resolve()
    m3_root = Path(m3_root).resolve() if m3_root else study.parent/"model_revision_20261016"
    inputs = ("freeze.json", "source_M3_freeze.json", "source_M3_manifest.json", "new_candidates.csv",
              "frozen_predictions.csv", "reference_summary.csv", "reference_audit.json", "metrics_overall.csv",
              "metrics_by_condition.csv", "coverage_by_condition.csv", "half_step_candidates.csv",
              "numerical/half_step_summary.csv", "numerical/half_step_audit.json",
              "outputs/development/models.json", "outputs/development/gates.json")
    before = {name: digest(study/name) for name in inputs}
    freeze, freeze_status = freeze_check(study)
    protection = origin_and_protection(study, m3_root, skip_external=skip_external_protection)
    candidates, predictions, references = (read_csv(study/name) for name in inputs[3:6])
    require(len(candidates) == freeze["n_candidates"] and candidates.scenario.nunique() == 6, "Frozen design counts mismatch")
    require(set(read_json(study/"outputs/development/models.json")) == set(MODELS), "Frozen model identities mismatch")
    audit = read_json(study/"reference_audit.json")
    require(audit["passed"] is True and audit["freeze_sha256"] == before["freeze.json"], "Main raw audit failed or stale")
    require(audit["expected_count"] == len(candidates) == audit["successful_count"] and audit["failed_count"] == 0, "Main audit counts mismatch")
    joined = join_from_raw_tables(candidates, predictions, references, read_json(study/"outputs/development/gates.json"))
    overall, condition_metrics = metrics_tables(joined)
    coverage = coverage_table(candidates, references)
    checks = {"metrics_overall": compare_table(overall, read_csv(study/"metrics_overall.csv"), ["stratum", "model"], "overall"),
              "metrics_by_condition": compare_table(condition_metrics, read_csv(study/"metrics_by_condition.csv"), ["stratum", "model", "scenario"], "condition metrics"),
              "coverage_by_condition": compare_table(coverage, read_csv(study/"coverage_by_condition.csv"), ["stratum", "scenario"], "condition coverage")}
    half = half_review(study, references, before["freeze.json"])
    raw = {"main": raw_peak_confirmation(study, references, before["freeze.json"]),
           "half": raw_peak_confirmation(study, read_csv(study/"numerical/half_step_summary.csv"), before["freeze.json"], half=True)}
    require(before == {name: digest(study/name) for name in inputs}, "Inputs changed during independent review")
    result = dict(passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
        review_source_sha256=digest(Path(__file__)), source_sha256=before, freeze=freeze_status,
        candidate_count=len(candidates), prediction_count=len(predictions), condition_count=6,
        origin_protection=protection, table_checks=checks, half_step=half, raw_peak_confirmation=raw,
        coarse_conditions_with_both_sides=int(coverage.loc[coverage.stratum.eq("coarse"), "has_both_sides"].sum()),
        metrics=json_records(overall), condition_metrics=json_records(condition_metrics),
        coverage=json_records(coverage), native_ODE_executed=False, models_refit=False,
        formal_analyzer_imported=False, scope="Descriptive frozen conditions; six conditions are not iid plant samples.")
    (study/"independent_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (study/"independent_review.md").write_text(markdown(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_directory", nargs="?", type=Path, default=ROOT)
    parser.add_argument("--m3-root", type=Path)
    parser.add_argument("--skip-external-protection", action="store_true",
                        help="Explicitly skip absent external historical worktrees; still verify all package-internal scientific artifacts and metrics.")
    args = parser.parse_args()
    try:
        result = review(args.study_directory, args.m3_root, args.skip_external_protection)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"passed": True, "n_candidates": result["candidate_count"],
                      "coarse_dual_side_conditions": result["coarse_conditions_with_both_sides"],
                      "half_label_changes": result["half_step"]["label_change_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
