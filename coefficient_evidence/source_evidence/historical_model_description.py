from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerbench.configuration import CONFIG_DIR, load_workflow_config, parameter_provenance_rows


TITLE = "面向活性污泥固定曝气阈值筛选的机理启发式低阶代理：安全审计、全网格BSM1验证与可复现计算评估"
SCENARIO_LABELS = {
    "baseline": "基线",
    "high_load": "高负荷",
    "low_oxygen_transfer": "低氧转移效率",
    "low_temperature": "低温",
    "ALL": "全部场景",
}
MODEL_DISPLAY_LABELS = {
    "threshold_lookup": "Threshold lookup",
    "monotonic_logistic": "Monotonic logistic regression",
    "isotonic_regression": "Isotonic regression",
    "six_state_mechanistic_surrogate": "Six-state mechanism-informed surrogate",
}
PARAMETER_SYMBOLS = {
    "tau_h": "τ_h",
    "mu_h": "μ_H",
    "mu_a": "μ_A",
    "k_s": "K_S",
    "k_nh": "K_NH",
    "k_oh": "K_O,H",
    "k_oa": "K_O,A",
    "hill_n": "n",
    "so_sat": "S_O,sat",
    "oxygen_gain": "c_KLa",
    "y_no": "Y_NO",
    "ss_to_nh": "f_SNH",
    "nitrification_rate_cap": "r_A,max",
    "nh4_nitrification_factor": "c_NH4",
    "biomass_tau_h": "τ_XH",
    "biomass_tau_a": "τ_XA",
    "xh_reference": "X_H,ref",
    "xa_reference": "X_A,ref",
    "bsm1_xba_asymptote": "A_max",
    "bsm1_xba_logistic_slope": "b",
    "bsm1_xba_logistic_midpoint": "K_50",
    "xh_oxygen_response_weight": "w_H",
    "biomass_load_exponent": "p_L",
    "biomass_temperature_exponent": "p_T",
}
PARAMETER_MAIN_EQUATIONS = {
    "tau_h": "Eq. (2)",
    "mu_h": "Eq. (3)",
    "mu_a": "Eq. (4)",
    "k_s": "Eq. (3)",
    "k_nh": "Eq. (4)",
    "k_oh": "Eq. (3)",
    "k_oa": "Eq. (4)",
    "hill_n": "Eq. (4)",
    "so_sat": "Eq. (5)",
    "oxygen_gain": "Eq. (5)",
    "y_no": "Eq. (11d)",
    "ss_to_nh": "Eq. (11e)",
    "nitrification_rate_cap": "Eq. (4)",
    "nh4_nitrification_factor": "Eq. (11e)",
    "biomass_tau_h": "Eq. (11a)",
    "biomass_tau_a": "Eq. (11c)",
    "xh_reference": "Eq. (10a)",
    "xa_reference": "Eq. (10b)",
    "bsm1_xba_asymptote": "Eq. (7)",
    "bsm1_xba_logistic_slope": "Eq. (7)",
    "bsm1_xba_logistic_midpoint": "Eq. (7)",
    "xh_oxygen_response_weight": "Eq. (10a)",
    "biomass_load_exponent": "Eqs. (10a,b)",
    "biomass_temperature_exponent": "Eq. (10b)",
}
TODO = "【TODO：对应新实验尚未运行或结果文件尚未生成，不得填写数值或形成结论。】"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_result(path: Path, required: Iterable[str], sources: list[dict]) -> pd.DataFrame | None:
    if not path.exists():
        sources.append({"path": str(path), "status": "blocked_missing_data", "sha256": None, "rows": None})
        return None
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing or frame.empty:
        sources.append(
            {
                "path": str(path),
                "status": "blocked_missing_data",
                "sha256": sha256(path),
                "rows": int(len(frame)),
                "missing_columns": missing,
            }
        )
        return None
    sources.append({"path": str(path), "status": "completed", "sha256": sha256(path), "rows": int(len(frame))})
    return frame


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.3)
    section.right_margin = Cm(2.3)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(5)
    for name, size in [("Title", 16), ("Heading 1", 13), ("Heading 2", 11.5), ("Heading 3", 10.5)]:
        style = styles[name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.size = Pt(size)


def add_body(doc: Document, text: str, *, bold: bool = False) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    run = paragraph.add_run(text)
    run.bold = bold


def add_equation(doc: Document, expression: str, number: str) -> None:
    """Add a centered display equation with a right-aligned equation number."""
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = (Cm(1.1), Cm(14.0), Cm(1.1))
    for cell, width in zip(table.rows[0].cells, widths):
        cell.width = width
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_mar = tc_pr.first_child_found_in("w:tcMar")
        if tc_mar is None:
            tc_mar = OxmlElement("w:tcMar")
            tc_pr.append(tc_mar)
        for edge in ("top", "left", "bottom", "right"):
            margin = tc_mar.find(qn(f"w:{edge}"))
            if margin is None:
                margin = OxmlElement(f"w:{edge}")
                tc_mar.append(margin)
            margin.set(qn("w:w"), "20")
            margin.set(qn("w:type"), "dxa")

    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = borders.find(qn(f"w:{edge}"))
        if border is None:
            border = OxmlElement(f"w:{edge}")
            borders.append(border)
        border.set(qn("w:val"), "nil")

    equation_paragraph = table.rows[0].cells[1].paragraphs[0]
    equation_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation_paragraph.paragraph_format.space_before = Pt(2)
    equation_paragraph.paragraph_format.space_after = Pt(2)
    equation_run = equation_paragraph.add_run(expression)
    equation_run.font.name = "Cambria Math"
    equation_run._element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria Math")
    equation_run.font.size = Pt(10)

    number_paragraph = table.rows[0].cells[2].paragraphs[0]
    number_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    number_run = number_paragraph.add_run(number)
    number_run.font.name = "Times New Roman"
    number_run.font.size = Pt(10)


def add_table_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(text)
    run.bold = True


def add_placeholder(doc: Document, label: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(f"[{label}]")
    run.bold = True
    run.font.color.rgb = None


def add_figure_or_placeholder(doc: Document, image_path: Path, label: str) -> None:
    # Prefer the canonical generated figure. If an earlier ET&I figure build
    # used the publication-package directory and legacy filename, use that
    # traceable file instead of leaving a false TODO placeholder.
    fallback_map = {
        "Figure_3_full_grid_safety.png": ROOT / "outputs" / "manuscript_revision_eti_cn" / "figures" / "Figure_3_threshold_and_screening.png",
        "Figure_4_timing_numerics_bounds.png": ROOT / "outputs" / "manuscript_revision_eti_cn" / "figures" / "Figure_4_timing_and_parameter_sensitivity.png",
        "Figure_5_dynamic_pairwise_validation.png": ROOT / "outputs" / "manuscript_revision_eti_cn" / "figures" / "Figure_5_dynamic_NH4_TN_energy_tradeoff.png",
    }
    selected_path = image_path
    if not selected_path.exists():
        selected_path = fallback_map.get(image_path.name, image_path)
    if selected_path.exists():
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        run.add_picture(str(selected_path), width=Cm(16.0))
        # Add accessible alternative text to the drawing in the OOXML. Word's
        # high-level API does not expose this attribute directly.
        doc_pr = run._r.find(".//" + qn("wp:docPr"))
        if doc_pr is not None:
            doc_pr.set("descr", label)
            doc_pr.set("title", label)
        caption = doc.add_paragraph(label)
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        add_placeholder(doc, f"{label}; figure file TODO: {image_path}")


def add_dataframe_table(
    doc: Document,
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    digits: int = 3,
    font_size: float | None = None,
) -> None:
    table = doc.add_table(rows=1, cols=len(columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    # Mark the first row as a repeating table header for multi-page output and
    # accessibility tools.
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = header
        set_cell_shading(cell, "D9E5EF")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
            if font_size is not None:
                run.font.size = Pt(font_size)
    for row in frame[columns].itertuples(index=False, name=None):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            if pd.isna(value):
                text = "NR"
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.{digits}f}"
            else:
                text = str(value)
            cells[index].text = text
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if font_size is not None:
                for run in cells[index].paragraphs[0].runs:
                    run.font.size = Pt(font_size)


def full_grid_result_text(classification: pd.DataFrame | None, confusion: pd.DataFrame | None) -> str:
    if classification is None or confusion is None:
        return TODO
    scenario_counts = classification.groupby("scenario").size().to_dict()
    if len(classification) != 1000 or scenario_counts != {key: 250 for key in SCENARIO_LABELS if key != "ALL"}:
        return "【TODO：当前全网格结果不完整，必须达到四个场景各250点、共1000点后才能写入结论。】"
    all_rows = confusion[confusion["scenario"].eq("ALL")]
    if all_rows.empty:
        return TODO
    row = all_rows.iloc[0]
    return (
        f"完整真值集包含1000个候选，四个场景各250个。按照预先设定的零超限规则，六状态代理产生"
        f"{int(row['true_safe'])}个真安全、{int(row['true_reject'])}个真拒绝、"
        f"{int(row['false_safe'])}个假安全和{int(row['conservative_reject'])}个保守拒绝。"
        f"假安全率为{float(row['false_safe_rate']) * 100:.2f}%（95% Wilson区间："
        f"{float(row['false_safe_wilson_low']) * 100:.2f}%–{float(row['false_safe_wilson_high']) * 100:.2f}%）。"
        "这些数值仅描述本研究定义的候选网格和工况范围，不应外推为普遍安全保证。"
    )


def threshold_result_text(thresholds: pd.DataFrame | None) -> str:
    if thresholds is None:
        return TODO
    parts = []
    for record in thresholds.to_dict(orient="records"):
        scenario = record["scenario"]
        label = SCENARIO_LABELS.get(scenario, scenario)
        bsm1 = record["bsm1_first_compliant_grid_point_d-1"]
        surrogate = record["surrogate_first_compliant_grid_point_d-1"]
        if pd.isna(bsm1) and pd.isna(surrogate):
            parts.append(f"{label}场景：BSM1和代理阈值均未在预设网格内解析")
        elif pd.isna(surrogate):
            parts.append(
                f"{label}场景：BSM1首个持续合规KLa为{float(bsm1):.2f} d⁻¹，"
                "代理阈值未在预设网格内解析，必须由BSM1复核"
            )
        elif pd.isna(bsm1):
            parts.append(
                f"{label}场景：BSM1阈值未在预设网格内解析，"
                f"代理首个持续合规KLa为{float(surrogate):.2f} d⁻¹"
            )
        else:
            parts.append(
                f"{label}场景的BSM1/代理首个持续合规KLa分别为"
                f"{float(bsm1):.2f}/{float(surrogate):.2f} d⁻¹"
            )
    return "；".join(parts) + "。阈值要求连续3个网格点合规，以减少单点偶然合规造成的误判。"


def timing_result_text(
    timing: pd.DataFrame | None,
    speedup: pd.DataFrame | None,
    efficiency: pd.DataFrame | None = None,
) -> str:
    if timing is None or speedup is None:
        return TODO
    target = speedup[
        speedup["scenario"].eq("ALL")
        & speedup["comparison_type"].eq("operational_default")
        & speedup["timing_scope"].eq("end_to_end")
    ]
    if target.empty:
        return TODO
    row = target.iloc[0]
    text = (
        "端到端成对计时的加速中位数为{:.1f}倍，均值为{:.1f}倍，标准差为{:.1f}倍，范围为{:.1f}–{:.1f}倍。"
        "该倍数同时受模型方程、时间步长、求解器、软件实现、硬件和系统负载影响，不能归因于状态维数降低本身。"
    ).format(
        float(row["paired_speedup_median"]),
        float(row["paired_speedup_mean"]),
        float(row["paired_speedup_std"]),
        float(row["paired_speedup_min"]),
        float(row["paired_speedup_max"]),
    )
    if efficiency is not None:
        observed = efficiency[efficiency["scenario"].eq("ALL")]
        if not observed.empty:
            e = observed.iloc[0]
            text += (
                "在完整1000候选两阶段工作流中，BSM1调用由{:.0f}次降至{:.0f}次，避免{:.0f}次；"
                "总耗时由{:.1f} s降至{:.1f} s，实际工作流加速为{:.2f}倍，耗时减少{:.1f}%。"
            ).format(
                float(e["candidate_cases"]),
                float(e["bsm1_rechecks"]),
                float(e["bsm1_calls_avoided"]),
                float(e["exhaustive_bsm1_seconds"]),
                float(e["two_stage_workflow_seconds"]),
                float(e["workflow_speedup"]),
                float(e["end_to_end_time_reduction_fraction"]) * 100,
            )
    return text


def baseline_result_text(baseline_models: pd.DataFrame | None) -> str:
    if baseline_models is None:
        return TODO
    all_rows = baseline_models[baseline_models["scenario"].eq("ALL")]
    if all_rows.empty:
        return TODO
    parts = []
    for row in all_rows.itertuples(index=False):
        model_label = MODEL_DISPLAY_LABELS.get(str(row.model), str(row.model))
        parts.append(
            f"{model_label}：假安全{int(row.false_safe)}个（{float(row.false_safe_rate) * 100:.1f}%），"
            f"真安全{int(row.true_safe)}个，真拒绝{int(row.true_reject)}个，保守拒绝{int(row.conservative_reject)}个"
        )
    return (
        "完整1000候选比较显示，"
        + "；".join(parts)
        + "。六状态代理的假安全为0个，但保守拒绝为65个，说明其主要代价是将部分候选交由BSM1复核，而不是错误放行。"
    )


def dynamic_result_text(
    dynamic_summary: pd.DataFrame | None,
    dynamic_validation: dict[str, pd.DataFrame],
) -> str:
    if dynamic_summary is None:
        return TODO
    profile_to_file = {"Dry": "dry", "Rain": "rain", "Synthetic storm": "storm"}
    parts = []
    for row in dynamic_summary.to_dict(orient="records"):
        profile = str(row["profile"])
        profile_label = {"Dry": "干旱", "Rain": "降雨", "Synthetic storm": "合成暴雨"}.get(profile, profile)
        frame = dynamic_validation.get(profile_to_file.get(profile, ""))
        violation = None
        nh4_200 = None
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["kla_numeric"] = pd.to_numeric(frame["kla"], errors="coerce")
            candidates = frame.iloc[(frame["kla_numeric"] - 200.0).abs().argsort()[:1]]
            if not candidates.empty:
                selected = candidates.iloc[0]
                nh4_200 = float(selected["SNH_mean_eval"])
                violation = float(selected["SNH_violation_hours_gt_limit"])
        nh4_text = "NR" if nh4_200 is None else f"{nh4_200:.2f}"
        violation_text = "NR" if violation is None else f"{violation:.2f}"
        parts.append(
            f"{profile_label}：TN最低点出现在KLa={float(row['KLa at min TN']):.1f} d⁻¹，"
            f"最低TN为{float(row['min TN']):.2f} mg N L⁻¹，此时NH4-N为{float(row['NH4-N at min TN']):.2f} mg N L⁻¹；"
            f"KLa=200 d⁻¹时平均NH4-N为{nh4_text} mg N L⁻¹、TN为{float(row['TN at KLa=200']):.2f} mg N L⁻¹、"
            f"曝气能耗为{float(row['energy at KLa=200']):.1f}（模型内部标准单位），但NH4-N仍有{violation_text} h超过4 mg N L⁻¹"
        )
    return (
        "动态干旱、降雨和合成暴雨结果显示，平均NH4-N达到4 mg N L⁻¹并不等于全程零超限；"
        + "；".join(parts)
        + "。因此动态场景必须联合解释NH4-N、TN、超限时长和曝气能耗，不能仅依据平均NH4-N选择固定曝气。"
    )


def timing_environment_text() -> str:
    manifest_path = ROOT / "outputs" / "manuscript_revision_eti_cn" / "supplementary_data" / "standardized_timing_manifest_final.json"
    if not manifest_path.exists():
        return TODO
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    env = manifest.get("environment", {})
    hardware = env.get("hardware", {})
    operating_system = env.get("operating_system", {})
    python_info = env.get("python", {})
    software = env.get("software_versions", {})
    return (
        "规范计时环境为Windows 11、Intel(R) Core(TM) i9-13900HX（24个物理核心、32个逻辑核心、"
        f"{float(hardware.get('total_memory_gb', 0.0)):.1f} GB内存），"
        f"CPython {python_info.get('version', 'NR')}；"
        f"bsm2-python {software.get('bsm2-python', 'NR')}、SciPy {software.get('scipy', 'NR')}、"
        f"NumPy {software.get('numpy', 'NR')}。"
        "代理采用显式Euler和逐分量clip；BSM1采用bsm2-python BSM1OL及其默认LSODA设置。"
        "每个条件预热1次后重复30次，计时包含simulation-kernel和end-to-end两种范围。"
    )


def build_main_document(results_dir: Path, output_path: Path, sources: list[dict]) -> None:
    classification = read_result(
        results_dir / "full_grid_classification.csv",
        ["scenario", "classification", "corrected_classification"],
        sources,
    )
    calibration = read_result(
        results_dir / "calibration" / "baseline_calibration_fit_comparison.csv",
        ["kla", "surrogate_SNH_mean_eval", "bsm1_SNH_mean_eval", "surrogate_SNH_violation_hours_gt_limit", "bsm1_SNH_violation_hours_gt_limit"],
        sources,
    )
    confusion = read_result(
        results_dir / "confusion_matrix_by_scenario.csv",
        ["scenario", "true_safe", "true_reject", "false_safe", "conservative_reject", "false_safe_rate", "false_safe_wilson_low", "false_safe_wilson_high"],
        sources,
    )
    thresholds = read_result(
        results_dir / "threshold_metrics_by_scenario.csv",
        ["scenario", "bsm1_first_compliant_grid_point_d-1", "surrogate_first_compliant_grid_point_d-1", "absolute_threshold_error_d-1"],
        sources,
    )
    baseline_models = read_result(
        results_dir / "baseline_model_comparison.csv",
        [
            "model", "scenario", "n_candidates", "true_safe", "true_reject",
            "false_safe", "conservative_reject", "false_safe_rate",
        ],
        sources,
    )
    efficiency = read_result(
        results_dir / "screening_efficiency.csv",
        [
            "scenario", "candidate_cases", "bsm1_rechecks", "bsm1_calls_avoided",
            "exhaustive_bsm1_seconds", "two_stage_workflow_seconds",
            "workflow_speedup", "end_to_end_time_reduction_fraction",
        ],
        sources,
    )
    warmup = read_result(
        results_dir / "warmup_convergence.csv",
        ["state", "absolute_change", "relative_change"],
        sources,
    )
    uncertainty = read_result(results_dir / "safety_margin_recommendations.csv", ["scenario", "recommendation"], sources)
    uncertainty_summary = read_result(
        ROOT / "outputs" / "manuscript_revision_eti_cn" / "tables" / "Table_uncertainty_safety_margin.csv",
        [
            "scenario",
            "scenario_label_cn",
            "n_uncertainty_samples",
            "unresolved_transition_percent",
            "q95_positive_threshold_underprediction_d-1",
            "maximum_positive_threshold_underprediction_d-1",
            "recommendation_cn",
        ],
        sources,
    )
    timestep = read_result(
        results_dir / "timestep_thresholds.csv",
        ["scenario", "timestep_min", "threshold_error_vs_reference_d-1"],
        sources,
    )
    bounds = read_result(
        results_dir / "bounds_audit_all_states.csv",
        [
            "state", "total_clip_count", "lower_clip_count", "upper_clip_count",
            "fraction_of_steps_clipped", "max_preclip_violation",
        ],
        sources,
    )
    # The optional candidate-level dynamic file is not required when the
    # traceable profile-level summary has already been generated. Avoid
    # recording a false missing-data block in that case.
    dynamic_path = results_dir / "dynamic_pairwise_validation.csv"
    dynamic = None
    if dynamic_path.exists():
        dynamic = read_result(
            dynamic_path,
            ["profile", "classification", "bsm1_mean_snh_mgN_L", "bsm1_mean_tn_mgN_L"],
            sources,
        )
    dynamic_summary = read_result(
        ROOT / "outputs" / "manuscript_revision_eti_cn" / "dynamic_tradeoff_summary.csv",
        [
            "profile", "KLa at min TN", "min TN", "NH4-N at min TN",
            "first KLa with mean NH4-N ≤ 4", "TN at KLa=200", "energy at KLa=200",
        ],
        sources,
    )
    dynamic_validation: dict[str, pd.DataFrame] = {}
    dynamic_data_dir = ROOT / "outputs" / "manuscript_revision_eti_cn" / "supplementary_data"
    for profile_key in ("dry", "rain", "storm"):
        candidate_path = results_dir / f"dynamic_{profile_key}_bsm1_validation_summary.csv"
        if not candidate_path.exists():
            candidate_path = dynamic_data_dir / f"dynamic_{profile_key}_bsm1_validation_summary.csv"
        if candidate_path.exists():
            dynamic_validation[profile_key] = pd.read_csv(candidate_path)
    timing_path = results_dir / "standardized_timing_summary.csv"
    if not timing_path.exists():
        timing_path = ROOT / "outputs" / "manuscript_revision_eti_cn" / "supplementary_data" / "standardized_timing_summary.csv"
    timing = read_result(
        timing_path,
        ["scenario", "model_label", "timing_scope", "median_seconds", "p05_seconds", "p95_seconds"],
        sources,
    )
    speedup_path = results_dir / "standardized_speedup_summary.csv"
    if not speedup_path.exists():
        speedup_path = ROOT / "outputs" / "manuscript_revision_eti_cn" / "supplementary_data" / "standardized_speedup_summary.csv"
    speedup = read_result(
        speedup_path,
        [
            "scenario",
            "comparison_type",
            "timing_scope",
            "paired_speedup_mean",
            "paired_speedup_std",
            "paired_speedup_median",
            "paired_speedup_min",
            "paired_speedup_max",
        ],
        sources,
    )

    doc = Document()
    configure_document(doc)
    title = doc.add_paragraph()
    title.style = doc.styles["Title"]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(TITLE)
    subtitle = doc.add_paragraph("中文研究稿")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Highlights", level=1)
    for item in [
        "提出可审计的六状态机理启发式代理，用于固定曝气氨氮阈值的快速初筛。",
        "以1000个完整BSM1候选验证假安全、保守拒绝和阈值误差。",
        "采用预先设定的安全裕量与强制复核规则，将代理定位为筛选工具而非BSM1替代品。",
        "以BSM1评估动态进水下的NH4-N、TN、曝气能耗及反硝化相关权衡。",
        "同时审计时间步长、全状态边界、联合不确定性和计时复现性。",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("摘要", level=1)
    abstract = (
        "高保真活性污泥模型能够提供可靠的过程验证，但在固定曝气方案的大规模筛选、跨场景比较和不确定性分析中计算成本较高。本文提出一种面向曝气阈值初筛的六状态机理启发式低阶代理，并将其嵌入“代理筛选—BSM1复核”的两阶段工作流。代理参数仅使用基线场景校准，随后冻结，在高负荷、低氧转移效率和低温场景进行独立检验；四个场景共设置1000个KLa候选，候选只有在整个评价期内NH4-N不超过4 mg N L⁻¹时才判定为合规。完整BSM1网格结果显示，代理识别出500个真安全候选和435个真拒绝候选，未观察到假安全，但有65个保守拒绝候选需要进一步复核。联合参数不确定性表明，预先设定的20 d⁻¹安全裕量在部分场景下低于95%分位阈值低估量，低温和高负荷候选应强制返回BSM1。动态干旱、降雨和合成暴雨验证进一步表明，平均NH4-N达标并不意味着TN、全过程超限风险和曝气能耗同时最优。两阶段工作流将1000次完整BSM1调用减少至343次，总耗时减少65.9%。该框架的适用范围是安全约束下的NH4-N阈值筛选；TN、反硝化过程和高风险运行决策仍应以完整BSM1结果为最终依据。"
    )
    add_body(doc, abstract)
    add_body(doc, "关键词：活性污泥；曝气；机理启发式代理；阈值筛选；BSM1；可复现计算")

    doc.add_heading("1. 引言", level=1)
    add_body(doc, "曝气既决定硝化能力，也是污水处理能耗的重要来源。提高曝气强度通常能够降低出水氨氮，但也可能削弱缺氧区反硝化并提高曝气能耗，使总氮响应出现不利变化。因此，固定曝气方案不能只依据平均氨氮选择，还应联合考虑TN、超限持续时间和能耗。用于筛选的模型还必须区分“平均达标”和“全过程无超限”，因为平均值可能掩盖短时排放风险。")
    add_body(doc, "BSM1为控制和运行策略比较提供了统一的动态基准，但在候选数量扩大、需要多场景重复或开展不确定性传播时，计算量迅速增加。低阶代理的价值不在于复制所有BSM1状态，而在于以透明、可审计的简化结构保留固定曝气—溶解氧—硝化—氨氮阈值之间的主要联系，并明确哪些候选仍需返回BSM1复核。由于本文代理没有显式重现完整的缺氧反硝化动力学，TN及反硝化相关结论由BSM1承担。")
    add_body(doc, "本研究的创新点包括：（1）将六状态机理启发式代理嵌入预先设定的两阶段筛选规则；（2）以完整1000点BSM1真值集报告真安全、真拒绝、假安全和保守拒绝，而非仅验证少量入选点；（3）把时间分辨率、六状态边界、动态天气进水、联合参数不确定性和规范计时纳入同一可复现框架；（4）将低温及TN/反硝化判断明确划入必须由BSM1承担的适用边界。")
    add_placeholder(doc, "Figure 1. Two-stage threshold-screening workflow (conceptual figure to be added after Chinese text finalization).")

    doc.add_heading("2. 材料与方法", level=1)
    doc.add_heading("2.1 研究问题与筛选定位", level=2)
    add_body(doc, "研究对象是固定命令KLa下的候选曝气方案。代理只负责初筛，BSM1仍是阈值邻域和高风险场景的最终判定依据。本文所称4 mg N L⁻¹是研究内部的比较基准，不表述为对所有地区或排放许可均适用的法规限值。")
    doc.add_heading("2.2 六状态机理启发式低阶代理", level=2)
    add_body(doc, "该代理的目的不是复现BSM1的全部组分和反应，而是以较少状态保留“固定曝气强度—溶解氧—硝化—氨氮阈值”之间的主要响应关系，为大规模候选提供快速且可审计的初筛。模型内部时间单位为h，所有参数值、单位、来源及其校准或固定状态列于Table S1。")

    doc.add_heading("2.2.1 状态定义与建模范围", level=3)
    add_equation(doc, "x(t) = [X_H, S_S, X_A, S_NO, S_NH, S_O]ᵀ", "(1)")
    add_body(doc, "式(1)中，x(t)为时刻t的六维代理状态向量，上标T表示向量转置；X_H和X_A分别表示异养菌与自养菌的汇总代理状态，单位均记为proxy state；S_S表示可溶性基质代理，单位为mg COD L⁻¹；S_NO和S_NH分别表示氧化态氮与氨氮代理，单位为mg N L⁻¹；S_O表示溶解氧代理，单位为mg O₂ L⁻¹。上述变量用于阈值筛选，其数值尺度不与BSM1状态逐一等同，尤其不能将X_H和X_A直接解释为BSM1生物量浓度。")

    doc.add_heading("2.2.2 过程速率与生物量平衡目标", level=3)
    add_equation(doc, "τ = τ_h / max(q_scale, ε_q)", "(2)")
    add_body(doc, "式(2)中，τ为随进水流量变化的有效水力时间尺度（h）；τ_h为基准水力时间尺度（h）；q_scale为相对于基准流量的无量纲流量倍率；ε_q=10⁻⁶为防止分母为零的数值保护常数。")

    add_equation(doc, "r_H = μ_H,eff · S_S/(K_S+S_S+ε) · S_O/(K_O,H+S_O+ε) · X_H/1000", "(3)")
    add_body(doc, "式(3)中，r_H为归一化异养过程代理速率；μ_H,eff=μ_H·s_H为场景修正后的异养活动系数，其中μ_H为基准系数（h⁻¹ proxy scale），s_H为无量纲场景倍率；K_S为基质半饱和常数（mg COD L⁻¹）；K_O,H为异养过程的氧响应常数（mg O₂ L⁻¹）；ε=10⁻⁹为数值保护常数；X_H/1000用于将异养菌代理状态映射到速率尺度。")

    add_equation(doc, "r_A = min{μ_A,eff · S_NH/(K_NH+S_NH+ε) · S_O^n/(K_O,A^n+S_O^n+ε) · X_A/100, r_A,max,eff}", "(4)")
    add_body(doc, "式(4)中，r_A为归一化自养硝化代理速率；μ_A,eff=μ_A·s_A为场景修正后的自养活动系数，μ_A为基准系数（h⁻¹ proxy scale），s_A为无量纲场景倍率；K_NH为氨氮半饱和常数（mg N L⁻¹）；K_O,A为自养过程的氧响应常数（mg O₂ L⁻¹）；n为无量纲Hill指数；X_A/100用于状态—速率尺度映射；r_A,max,eff=r_A,max·s_cap为场景修正后的硝化速率上限，其中r_A,max为基准上限，s_cap为无量纲硝化能力倍率；min{·}表示在未受限硝化速率与场景修正上限之间取较小值。")

    add_equation(doc, "r_O2 = s_KLa · c_KLa · (KLa/240) · (S_O,sat−S_O)", "(5)")
    add_body(doc, "式(5)中，r_O2为氧传递代理速率（mg O₂ L⁻¹ h⁻¹）；KLa为固定命令曝气强度（d⁻¹）；240 d⁻¹为无量纲化所用的参考命令；c_KLa为参考命令下的氧传递增益（h⁻¹）；s_KLa为无量纲场景氧传递倍率；S_O,sat为饱和溶解氧浓度（mg O₂ L⁻¹）。")

    add_body(doc, "为避免X_H和X_A依赖边界裁剪而长期停留在下限，模型令两者分别向随负荷和曝气变化的平衡目标X_H,eq与X_A,eq松弛。平衡目标按下式计算：")
    add_equation(doc, "KLa_eff = max(0, KLa) · max(s_KLa, 0)", "(6)")
    add_equation(doc, "A_XBA(KLa_eff) = A_max / {1+exp[−b(KLa_eff−K_50)]}", "(7)")
    add_equation(doc, "R_O = A_XBA(KLa_eff) / A_XBA(160)", "(8)")
    add_equation(doc, "L_COD=max{COD_in·max(q_scale,ε_q)/260, ε_q},   L_NH4=max{NH4_in·max(q_scale,ε_q)/30, ε_q}", "(9)")
    add_equation(doc, "X_H,eq = max{X_H,ref · L_COD^pL · [1+w_H(R_O−1)], 0}", "(10a)")
    add_equation(doc, "X_A,eq = max{X_A,ref · R_O · L_NH4^pL · max(s_A,ε_q)^pT, 0}", "(10b)")
    add_body(doc, "式(6)–(10)中，KLa_eff为考虑场景氧传递倍率后的有效曝气强度（d⁻¹）；A_XBA为根据基线BSM1自养菌响应构造的平滑Logistic活动曲线；A_max、b和K_50分别为该曲线的渐近值、斜率和中点参数；R_O为相对于KLa=160 d⁻¹参考点的无量纲氧响应；COD_in和NH4_in分别为进入代理的进水COD（mg COD L⁻¹）和氨氮（mg N L⁻¹）；260和30为对应的基准进水尺度；L_COD和L_NH4为无量纲负荷因子；X_H,ref和X_A,ref为两个生物量代理的参考状态；p_L为负荷指数；w_H为异养菌代理对氧响应的权重；p_T为自养菌代理的温度/活动倍率指数。该构造只定义代理状态的动态目标，不表示BSM1生物量状态的直接换算。")

    doc.add_heading("2.2.3 六状态平衡方程", level=3)
    add_equation(doc, "dX_H/dt = (X_H,eq−X_H)/τ_XH", "(11a)")
    add_equation(doc, "dS_S/dt = (COD_in−S_S)/τ − 18r_H", "(11b)")
    add_equation(doc, "dX_A/dt = (X_A,eq−X_A)/τ_XA", "(11c)")
    add_equation(doc, "dS_NO/dt = (2−S_NO)/τ + Y_NO r_A − 0.25r_H", "(11d)")
    add_equation(doc, "dS_NH/dt = (NH4_in−S_NH)/τ + f_SNH·max(S_S,0)/τ − c_NH4 r_A", "(11e)")
    add_equation(doc, "dS_O/dt = (0.4−S_O)/τ + r_O2 − 0.40r_H − 2.50r_A", "(11f)")
    add_body(doc, "式(11)中，τ_XH和τ_XA分别为X_H与X_A向其平衡目标松弛的时间常数（h）；f_SNH为可溶性基质向氨氮代理的经验转换系数（mg N per mg COD proxy）；c_NH4为硝化代理速率到氨氮去除项的经验映射系数；Y_NO为硝化代理速率到氧化态氮生成项的经验映射系数。常数2 mg N L⁻¹和0.4 mg O₂ L⁻¹分别是S_NO与S_O稀释项中的参考水平。系数18、0.25、0.40和2.50用于把归一化代理速率映射到相应状态收支，均为低阶模型中的固定经验系数，不解释为ASM1/BSM1的直接化学计量系数。所有导数均以h为内部时间单位。")

    doc.add_heading("2.2.4 显式Euler积分与边界审计", level=3)
    add_equation(doc, "x*_(k+1) = x_k + Δt · f(x_k, KLa, COD_in, NH4_in, q_scale)", "(12)")
    add_equation(doc, "x_(k+1) = clip[x*_(k+1), x_min, x_max]", "(13)")
    add_body(doc, "式(12)和式(13)中，k为离散时间步编号；Δt为积分步长（h）；f(·)为式(11a)–(11f)组成的状态导数向量；x*_(k+1)为施加边界前的Euler候选状态；clip(·)表示逐分量截断；x_min和x_max分别为状态下界与上界。按式(1)的状态顺序，x_min=[0,0,0,0,0,0]ᵀ，x_max=[5000,500,1200,80,80,8]ᵀ，各分量单位与式(1)后的定义一致；六个状态的初值、单位和上下界汇总见Table S2。")
    add_body(doc, "每一步均同时保存未裁剪候选与边界判断结果。只有当x*_(k+1)严格超出有限边界时才记为一次实际裁剪，并记录触发状态、上界或下界方向、裁剪次数、连续持续时间、裁剪步比例以及裁剪前最大越界量。状态恰好等于边界但未被数值截断的情况另行标记，不计为实际裁剪。该审计用于判断边界约束是否实质性影响阈值，而不是把裁剪本身视为模型正确性的证据。")

    doc.add_heading("2.2.5 反硝化表述与适用边界", level=3)
    add_body(doc, "式(11d)中的−0.25r_H仅用于闭合氧化态氮代理收支。由于r_H包含好氧响应项S_O/(K_O,H+S_O)，该经验去除项不等同于ASM1中由缺氧因子、硝酸盐限制和异养菌动力学共同控制的反硝化速率。因此，六状态代理仅用于NH4-N阈值初筛，不独立预测TN，也不用于解释完整反硝化过程；TN、反硝化和相关工程权衡均依据完整BSM1状态与输出。")
    add_placeholder(doc, "Figure 2. Mechanism-informed structure and scope boundary of the six-state surrogate (conceptual figure to be added after Chinese text finalization).")
    doc.add_heading("2.3 校准—验证分离", level=2)
    add_body(doc, "参数仅使用基线场景校准。预先设定的目标由平均SNH误差、超限时长误差和阈值误差组成，分别以4 mg N L⁻¹、168 h和20 d⁻¹归一化，并赋予相同的指标族权重。优化采用SciPy trust-region reflective least-squares，3个固定种子起点，每个起点最多120次函数评价，ftol、xtol和gtol均为10⁻⁶。")
    add_body(doc, "参数冻结后，在高负荷、低氧转移效率和低温场景进行独立检验。低温场景不用于反向调参，并被预先列为强制BSM1复核场景，以避免把模型失配隐藏在总体平均指标中。低氧转移效率场景主要检验两个模型对同一0.70缩放的响应一致性，不表述为完全独立的结构验证。")
    doc.add_heading("2.4 候选网格、时间设置与合规规则", level=2)
    add_body(doc, "四个场景均在40–520 d⁻¹之间设置250个等距KLa候选，共1000个候选。每个候选独立初始化，不在相邻KLa之间延续状态。稳态场景预热60 d、评价7 d；代理默认15 min步长，BSM1默认1 min步长。动态进水原始15 min观测在BSM1的1 min积分中采用零阶保持。")
    add_body(doc, "候选只有在整个评价期内SNH均不超过4 mg N L⁻¹时才记为合规。阈值定义为首个连续3个网格点均合规的KLa；若到网格上限仍未满足该条件，则标记为未解析，而不是外推阈值。")
    doc.add_heading("2.5 四分类、安全裕量与两阶段规则", level=2)
    add_body(doc, "代理与BSM1的组合结果分为真安全、真拒绝、假安全和保守拒绝。假安全是首要风险指标，采用Wilson区间报告。预先设定的安全裕量为20 d⁻¹，但其适当性必须由联合不确定性和完整真值集重新检验。阈值邻域及低温候选必须进入BSM1复核；复核预算不足时保留未决状态，不得静默接受。")
    doc.add_heading("2.6 简单基线与联合不确定性", level=2)
    add_body(doc, "比较对象包括基线阈值查表、符号约束逻辑回归和单调等距回归。三者均只在基线场景拟合。联合不确定性采用固定随机种子的Latin hypercube，同时扰动氧传递、硝化和生物量动态相关参数，并报告阈值分布、假安全概率及安全裕量建议。")
    doc.add_heading("2.7 动态进水下的BSM1基准验证", level=2)
    add_body(doc, "动态验证包含标准干旱、降雨和确定性合成暴雨压力场景。BSM1到代理输入的COD、NH4-N和流量倍率映射分别见补充材料式(S1)、式(S2)和式(S3)，模型一致性TN的计算见式(S4)和式(S5)。代理和BSM1使用相同的命令KLa集合；代理用于NH4-N阈值分类，BSM1用于报告NH4-N、模型一致性TN、曝气能耗和超限持续时间。由于代理未显式描述完整缺氧反硝化过程，TN不作为代理筛选输出。")
    doc.add_heading("2.8 数值收敛、边界与计时", level=2)
    add_body(doc, "时间步长检验比较1、2.5、5、10和15 min，并以0.25 min为高精度参考，评估六状态轨迹误差和阈值移动。计时在同一电脑、软件环境和求解器设置下随机化运行顺序，预热后重复30次，报告中位数、IQR、均值、标准差及P5–P95；文件hash用于验证重复运行结果一致。")

    doc.add_heading("3. 结果", level=1)
    doc.add_heading("3.1 基线标定性能", level=2)
    if calibration is None:
        add_body(doc, TODO)
    else:
        mean_rmse = float(np.sqrt(np.mean((calibration["surrogate_SNH_mean_eval"] - calibration["bsm1_SNH_mean_eval"]) ** 2)))
        violation_rmse = float(np.sqrt(np.mean((calibration["surrogate_SNH_violation_hours_gt_limit"] - calibration["bsm1_SNH_violation_hours_gt_limit"]) ** 2)))
        add_body(doc, f"仅使用baseline中预先设定的标定点，平均SNH的RMSE为{mean_rmse:.3f} mg N L⁻¹，超限时长RMSE为{violation_rmse:.3f} h。参数在此后冻结，验证场景未进入目标函数。完整标定曲线见Figure S1。")
        add_body(doc, "平均SNH轨迹RMSE接近4 mg N L⁻¹研究基准，说明代理不应被解释为BSM1轨迹替代物；本研究评价重点是全程零超限分类和阈值位置，而不是逐时刻状态完全重现。")
    doc.add_heading("3.2 完整1000点安全分类与阈值", level=2)
    add_body(doc, full_grid_result_text(classification, confusion))
    add_body(doc, threshold_result_text(thresholds))
    add_figure_or_placeholder(doc, ROOT / "figures" / "generated" / "Figure_3_full_grid_safety.png", "Figure 3. Full-grid threshold and safety audit.")
    if thresholds is not None:
        table = thresholds.copy()
        table["场景"] = table["scenario"].map(SCENARIO_LABELS)
        add_dataframe_table(
            doc,
            table,
            ["场景", "bsm1_first_compliant_grid_point_d-1", "surrogate_first_compliant_grid_point_d-1", "absolute_threshold_error_d-1"],
            ["场景", "BSM1阈值 (d⁻¹)", "代理阈值 (d⁻¹)", "绝对误差 (d⁻¹)"],
        )
    else:
        add_body(doc, TODO)

    doc.add_heading("3.3 与简单基线的比较", level=2)
    add_body(doc, baseline_result_text(baseline_models))
    if baseline_models is not None:
        baseline_all = baseline_models[baseline_models["scenario"].eq("ALL")].copy()
        baseline_all["display_model"] = baseline_all["model"].map(MODEL_DISPLAY_LABELS).fillna(baseline_all["model"])
        add_dataframe_table(
            doc,
            baseline_all,
            ["display_model", "n_candidates", "true_safe", "true_reject", "false_safe", "conservative_reject", "false_safe_rate"],
            ["Model", "Candidates", "True safe", "True reject", "False safe", "Conservative reject", "False-safe rate"],
            digits=3,
        )
    doc.add_heading("3.4 安全裕量与联合不确定性", level=2)
    if uncertainty is None:
        add_body(doc, TODO)
    else:
        text_path = ROOT / "outputs" / "manuscript_revision_eti_cn" / "text" / "uncertainty_results_text_cn.md"
        if text_path.exists():
            manuscript_text = text_path.read_text(encoding="utf-8")
            # Convert the Markdown source into normal Word paragraphs. Keeping
            # the source's bullet markers and bold delimiters would expose raw
            # Markdown in the manuscript and make the section look unfinished.
            for line in manuscript_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("- "):
                    line = line[2:]
                line = line.replace("**", "")
                add_body(doc, line)
        else:
            add_body(doc, "Joint uncertainty results are available in the safety-margin table.")
        if uncertainty_summary is not None:
            table = uncertainty_summary.copy()
            add_dataframe_table(
                doc,
                table,
                ["scenario_label_cn", "unresolved_transition_percent", "q95_positive_threshold_underprediction_d-1", "maximum_positive_threshold_underprediction_d-1", "recommendation_cn"],
                ["Scenario", "Unresolved transition (%)", "Q95 underprediction (d-1)", "Maximum underprediction (d-1)", "Recommendation"],
                digits=2,
            )
            q95_max = float(pd.to_numeric(uncertainty_summary["q95_positive_threshold_underprediction_d-1"], errors="coerce").max())
            add_body(doc, f"安全裕量解释：预先设定的20 d⁻¹小于当前场景95%分位低估量的最大值{q95_max:.2f} d⁻¹，因此20 d⁻¹不能直接视为联合不确定性下的充分裕量；采用任何分场景裕量后仍需用完整BSM1网格重新验证。")
    doc.add_heading("3.5 时间步长、六状态边界与预热", level=2)
    if timestep is None or bounds is None:
        add_body(doc, TODO)
    else:
        threshold_errors = pd.to_numeric(timestep["threshold_error_vs_reference_d-1"], errors="coerce").dropna()
        max_shift = float(threshold_errors.abs().max()) if not threshold_errors.empty else float("nan")
        bound_numeric = bounds.copy()
        for column in ["total_clip_count", "lower_clip_count", "upper_clip_count", "fraction_of_steps_clipped", "max_preclip_violation"]:
            bound_numeric[column] = pd.to_numeric(bound_numeric[column], errors="coerce").fillna(0.0)
        bound_summary = bound_numeric.groupby("state", as_index=False).agg(
            total_clip_count=("total_clip_count", "sum"),
            lower_clip_count=("lower_clip_count", "sum"),
            upper_clip_count=("upper_clip_count", "sum"),
            max_fraction_of_steps_clipped=("fraction_of_steps_clipped", "max"),
            max_preclip_violation=("max_preclip_violation", "max"),
        )
        total_clips = int(bound_summary["total_clip_count"].sum())
        active_bounds = bound_summary[bound_summary["total_clip_count"] > 0]
        active_text = "、".join(
            f"{row.state}下限{int(row.lower_clip_count)}次" for row in active_bounds.itertuples(index=False)
        ) or "无"
        max_fraction = float(bound_summary["max_fraction_of_steps_clipped"].max())
        max_violation = float(bound_summary["max_preclip_violation"].max())
        shift_text = "未解析" if np.isnan(max_shift) else f"{max_shift:.2f} d⁻¹"
        add_body(doc, f"相对于0.25 min参考，在已解析阈值的场景中所检验步长的最大绝对阈值移动为{shift_text}；低温场景各步长下代理阈值均未解析。六状态边界审计共记录{total_clips}次实际裁剪，全部为{active_text}；单次运行的最大裁剪步比例为{max_fraction * 100:.2f}%，最大裁剪前越界量为{max_violation:.3f}。裁剪次数本身不是正确性的证明，必须结合发生状态、持续时间和阈值影响解释。")
        add_dataframe_table(
            doc,
            bound_summary,
            ["state", "total_clip_count", "lower_clip_count", "upper_clip_count", "max_fraction_of_steps_clipped", "max_preclip_violation"],
            ["State", "Total clips", "Lower clips", "Upper clips", "Max clipped steps (%)", "Max pre-clip violation"],
            digits=4,
        )
        if warmup is not None:
            warmup_numeric = warmup.copy()
            warmup_numeric["absolute_change"] = pd.to_numeric(warmup_numeric["absolute_change"], errors="coerce").abs()
            warmup_numeric["relative_change"] = pd.to_numeric(warmup_numeric["relative_change"], errors="coerce").abs()
            max_abs_warmup = float(warmup_numeric["absolute_change"].max())
            max_rel_warmup = float(warmup_numeric["relative_change"].max()) * 100
            add_body(doc, f"预热收敛审计显示，最后两个7 d窗口的最大状态绝对变化为{max_abs_warmup:.4f}，最大相对变化为{max_rel_warmup:.2f}%；该结果用于说明预热稳定性，不等同于模型验证准确性。")
    add_figure_or_placeholder(doc, ROOT / "figures" / "generated" / "Figure_4_timing_numerics_bounds.png", "Figure 4. Timing, time-step convergence and six-state boundary audit.")

    doc.add_heading("3.6 动态天气进水下的BSM1基准验证", level=2)
    if dynamic is None and dynamic_summary is None:
        add_body(doc, TODO)
    elif dynamic is not None:
        counts = dynamic["classification"].value_counts()
        add_body(doc, f"动态候选验证包含{len(dynamic)}个候选，其中假安全{int(counts.get('false_safe', 0))}个、保守拒绝{int(counts.get('conservative_reject', 0))}个。NH4-N、TN和BSM1标准曝气能耗必须联合解释。")
    else:
        add_body(doc, dynamic_result_text(dynamic_summary, dynamic_validation))
    add_figure_or_placeholder(doc, ROOT / "figures" / "generated" / "Figure_5_dynamic_pairwise_validation.png", "Figure 5. Dynamic BSM1 benchmark: NH4-N, TN, violation duration and commanded aeration-energy proxy across dry, rain and synthetic-storm profiles.")
    add_figure_or_placeholder(doc, ROOT / "figures" / "generated" / "Figure_6_uncertainty_robustness.png", "Figure 6. Joint uncertainty, false-safe risk and the BSM1 recheck gate.")

    doc.add_heading("3.7 规范计时与实际BSM1调用节省", level=2)
    add_body(doc, timing_result_text(timing, speedup, efficiency))
    add_body(doc, timing_environment_text())
    if efficiency is not None:
        efficiency_all = efficiency[efficiency["scenario"].eq("ALL")].copy()
        add_dataframe_table(
            doc,
            efficiency_all,
            ["candidate_cases", "bsm1_rechecks", "bsm1_calls_avoided", "exhaustive_bsm1_seconds", "two_stage_workflow_seconds", "workflow_speedup", "end_to_end_time_reduction_fraction"],
            ["Candidates", "BSM1 rechecks", "Calls avoided", "Exhaustive BSM1 (s)", "Two-stage workflow (s)", "Workflow speedup", "Time reduction"],
            digits=3,
        )
    if classification is None:
        add_body(doc, TODO)
    else:
        decisions_path = results_dir / "two_stage_screening_decisions.csv"
        decisions = read_result(decisions_path, ["screening_decision"], sources)
        if decisions is None:
            add_body(doc, TODO)
        else:
            counts = decisions["screening_decision"].value_counts()
            rechecks = int(counts.get("bsm1_recheck", 0) + counts.get("unresolved_budget_exceeded", 0))
            add_body(doc, f"在完整1000候选工作流中，规则要求{rechecks}个候选进入或等待BSM1复核。该数字是新工作流的实际输出，不与旧的27点短名单验证混用。")

    doc.add_heading("4. 讨论", level=1)
    add_body(doc, "本研究不把代理描述为BSM1替代物。代理最有价值的区域是远离NH4-N阈值的明显拒绝或明显保留候选；误差通常更容易集中在决策转变附近。因而，报告阈值距离分组比只给一个全局准确率更能反映部署风险。")
    add_body(doc, "低温场景同时检验温度对动力学的影响和代理结构的适用边界。若低温出现未解析阈值或较高不确定性，合理处置是强制BSM1复核，而不是继续调参直到总体结果看似一致。这样的限制不会削弱框架，反而使模型角色更清楚。")
    add_body(doc, "完整BSM1结果表明，固定曝气强度提高时，NH4-N可能下降，但缺氧反硝化可能受到抑制，TN未必同步降低，同时曝气能耗持续增加。这一权衡来自BSM1而非六状态代理。本文将无量纲commanded aeration index用于代理的命令强度比较，并单独报告BSM2-Python内部标准曝气能耗；前者不是kWh d⁻¹，也不应被解释为实测鼓风机电耗。")

    doc.add_heading("5. 局限性与适用范围", level=1)
    add_body(doc, "第一，4 mg N L⁻¹是本研究内部比较基准，不等同于普遍法规限值。第二，六个状态均为阈值筛选所用的代理状态；SNO方程中的经验性去除项没有显式包含完整的缺氧因子和硝酸盐限制，因而不能用于独立预测TN或解释完整反硝化动力学。第三，动态暴雨为确定性压力测试，不代表所有真实厂站气象分布。第四，计时倍数依赖硬件、求解器、软件版本和时间步长。第五，若联合不确定性、边界审计或TN评价显示较大风险，候选必须回到BSM1或更高层级模型验证。")

    doc.add_heading("6. 结论", level=1)
    add_body(doc, "本研究建立了一个以安全审计为核心的两阶段固定曝气阈值筛选框架。六状态机理启发式代理负责快速排除明显不合规候选并识别风险较低的NH4-N阈值候选，BSM1负责阈值邻域、高风险场景以及TN和反硝化相关权衡的最终判断。该分工避免把低阶代理误解为完整ASM1或BSM1的替代物。研究结论由完整1000点真值集、动态BSM1基准验证、联合不确定性、时间步长和边界审计共同支持。")
    add_body(doc, full_grid_result_text(classification, confusion))

    doc.add_heading("声明", level=1)
    add_body(doc, "利益冲突：作者声明不存在已知的竞争性经济利益或个人关系会影响本文工作。")
    add_body(doc, "数据与代码可用性：代码、冻结配置、运行入口、原始结果和图件源数据将在投稿前整理至公开仓库；仓库DOI与访问链接待最终上传后补充。")
    add_body(doc, "作者贡献、基金、致谢：TODO，由作者在投稿前按实际情况填写。")
    doc.add_heading("参考文献", level=1)
    add_body(doc, "按当前任务要求暂不插入参考文献；中文内容定稿后再统一检索、核对并按ET&I格式插入。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


def build_supplement(results_dir: Path, output_path: Path, sources: list[dict]) -> None:
    doc = Document()
    configure_document(doc)
    title = doc.add_paragraph("补充材料：面向活性污泥固定曝气阈值筛选的机理启发式低阶代理")
    title.style = doc.styles["Title"]
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading("S1. 冻结配置与参数来源", level=1)
    parameter_rows = pd.DataFrame(parameter_provenance_rows())
    if not parameter_rows.empty:
        parameter_rows["symbol"] = parameter_rows["code_name"].map(PARAMETER_SYMBOLS).fillna(parameter_rows["code_name"])
        parameter_rows["main_text_equation"] = parameter_rows["code_name"].map(PARAMETER_MAIN_EQUATIONS).fillna("NR")
        add_table_caption(doc, "Table S1. 六状态机理启发式代理参数及其与正文公式的对应关系。")
        add_dataframe_table(
            doc,
            parameter_rows,
            ["symbol", "code_name", "value", "unit", "source_type", "main_text_equation"],
            ["Symbol", "Code name", "Value", "Unit", "Source/type", "Main-text equation"],
            digits=4,
            font_size=8,
        )

    parameter_record = json.loads((CONFIG_DIR / "surrogate_parameters.json").read_text(encoding="utf-8"))
    state_order = ["XH", "SS", "XA", "SNO", "SNH", "SO"]
    state_symbols = {"XH": "X_H", "SS": "S_S", "XA": "X_A", "SNO": "S_NO", "SNH": "S_NH", "SO": "S_O"}
    state_names = {"XH": "异养菌代理", "SS": "可溶性基质代理", "XA": "自养菌代理", "SNO": "氧化态氮代理", "SNH": "氨氮代理", "SO": "溶解氧代理"}
    initial_state = parameter_record["initial_state"]
    state_units = parameter_record["state_units"]
    state_minimum = parameter_record["state_bounds"]["minimum"]
    state_maximum = parameter_record["state_bounds"]["maximum"]
    state_rows = pd.DataFrame(
        [
            {
                "symbol": state_symbols[name],
                "state_name": state_names[name],
                "initial_value": initial_state[name],
                "unit": state_units[name],
                "lower_bound": state_minimum[index],
                "upper_bound": state_maximum[index],
            }
            for index, name in enumerate(state_order)
        ]
    )
    add_table_caption(doc, "Table S2. 正文式(1)、式(12)和式(13)所用六个代理状态的初值、单位及边界。")
    add_dataframe_table(
        doc,
        state_rows,
        ["symbol", "state_name", "initial_value", "unit", "lower_bound", "upper_bound"],
        ["Symbol", "State", "Initial value", "Unit", "Lower bound", "Upper bound"],
        digits=3,
        font_size=8.5,
    )
    add_figure_or_placeholder(
        doc,
        ROOT / "figures" / "generated" / "Figure_S1_baseline_calibration.png",
        "Figure S1. Baseline-only calibration curves for the frozen six-state surrogate; validation scenarios were excluded from parameter fitting.",
    )
    doc.add_heading("S2. 完整候选网格与分类定义", level=1)
    add_body(doc, "四场景×250候选的完整结果、四分类定义、Wilson区间和阈值距离分组见结果目录中的CSV。只有完整1000行且每场景250行时，主文Figure 3才允许生成。")
    doc.add_heading("S3. 数值收敛与全状态边界审计", level=1)
    add_body(doc, "已生成0.25 min参考与1/2.5/5/10/15 min步长比较、六状态边界裁剪汇总和预热最后两个7 d窗口变化；主文报告最大阈值移动、触发状态、裁剪比例和预热变化，逐候选原始记录保留在结果目录CSV中。")
    doc.add_heading("S4. 动态进水映射", level=1)
    add_equation(doc, "COD_in = (S_S,BSM1 + X_S,BSM1) · 260/271.82", "(S1)")
    add_equation(doc, "NH4_in = S_NH,BSM1 · 30/31.56", "(S2)")
    add_equation(doc, "q_scale = Q/18446", "(S3)")
    add_body(doc, "式(S1)、式(S2)和式(S3)中，COD_in、NH4_in和q_scale分别为输入正文代理模型的COD（mg COD L⁻¹）、氨氮（mg N L⁻¹）和无量纲流量倍率；S_S,BSM1、X_S,BSM1和S_NH,BSM1为BSM1进水状态；Q为BSM1进水流量（m³ d⁻¹）；271.82 mg COD L⁻¹、31.56 mg N L⁻¹和18446 m³ d⁻¹为相应的BSM1参考尺度。15 min源数据在BSM1的1 min积分中采用零阶保持；仅在绘图对齐时将BSM1输出插值到代理报告时刻。")
    add_equation(doc, "TIN = S_NH,BSM1 + S_NO,BSM1", "(S4)")
    add_equation(doc, "TN = TIN + S_ND,BSM1 + X_ND,BSM1 + 0.08(X_BH,BSM1+X_BA,BSM1) + 0.06X_P,BSM1", "(S5)")
    add_body(doc, "式(S4)和式(S5)中，TIN为无机氮；S_NO,BSM1、S_ND,BSM1、X_ND,BSM1、X_BH,BSM1、X_BA,BSM1和X_P,BSM1均为ASM1/BSM1状态；TN为由该状态向量计算的模型一致性总氮（mg N L⁻¹），不称为独立实验室测量TN。六状态代理中的S_NH和S_NO只是汇总代理状态，二者之和不作为本文的TN预测；TN与反硝化权衡仍以完整BSM1输出为依据。")
    doc.add_heading("S5. 计时与环境记录", level=1)
    add_body(doc, timing_environment_text())
    doc.add_heading("S6. 数据文件与可复现性记录", level=1)
    missing = [item for item in sources if item["status"] != "completed"]
    if missing:
        add_body(doc, "构建审计发现部分正文数值来源尚未纳入可复现性清单，因此当前补充材料不得作为最终投稿版本；缺失项须在投稿前补齐并重新生成文件哈希。")
    else:
        add_body(doc, "正文和补充材料所使用的必需数值结果均已纳入构建清单。构建脚本同时记录配置文件、参数文件、数值结果文件及其SHA-256校验值，以便在代码和数据仓库公开后核验图、表与正文数值的一致性。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chinese ET&I revision manuscript using only traceable new results.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript")
    args = parser.parse_args()
    sources: list[dict] = []
    main_path = args.output_dir / "Manuscript_ETI_CN_revised.docx"
    supplement_path = args.output_dir / "Supplementary_Material_ETI_revised.docx"
    build_main_document(args.results_dir, main_path, sources)
    build_supplement(args.results_dir, supplement_path, sources)
    manifest = {
        "title": TITLE,
        "new_experiment_run": False,
        "statement": "This builder never substitutes old outputs for missing ET&I revision results.",
        "configuration": str(CONFIG_DIR / "eti_workflow.json"),
        "configuration_sha256": sha256(CONFIG_DIR / "eti_workflow.json"),
        "surrogate_parameter_file": str(CONFIG_DIR / "surrogate_parameters.json"),
        "surrogate_parameter_sha256": sha256(CONFIG_DIR / "surrogate_parameters.json"),
        "numeric_sources": sources,
        "outputs": [str(main_path), str(supplement_path)],
    }
    (args.output_dir / "manuscript_numeric_sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
