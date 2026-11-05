"""Data-faithful static figures from independently audited M3B analysis only.

Reference points are raw sampled peaks.  Lines connect model predictions at
frozen candidates as visual guides; they are not new integrations or fitted
interpolants.  No curve is extended beyond observed candidate coordinates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ("full_calibrated", "full_fixed_ratio", "no_feedback", "poly_ridge",
          "rbf_kernel_ridge", "legacy_M2_binary")
LABELS = {"full_calibrated": "Full, calibrated ratio (6p)",
          "full_fixed_ratio": "Full, fixed ratio (5p)",
          "no_feedback": "No feedback (5p)", "poly_ridge": "Polynomial ridge",
          "rbf_kernel_ridge": "RBF kernel ridge", "legacy_M2_binary": "Historical M2 (binary fit)"}
SHORT = ("Full\n6p", "Full\n5p", "No\nfeedback", "Poly\nridge", "RBF\nridge", "Historical\nM2")
COLORS = {name: color for name, color in zip(MODELS, ("#193B67", "#1A8A88", "#DF7549", "#B69322", "#805EA4", "#4E9DBC"))}
STYLES = {name: style for name, style in zip(MODELS, ("-", "--", "-.", ":", (0, (5, 2, 1, 2)), (0, (8, 3))))}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 13, "axes.labelsize": 13, "axes.titlesize": 14,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                         "figure.facecolor": "#F7F6F2", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
                         "grid.color": "#D7DADB", "grid.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "savefig.facecolor": "#F7F6F2"})


def panel_label(ax, letter: str) -> None:
    ax.text(-0.12, 1.065, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)


def export(fig, out: Path, stem: str) -> list[Path]:
    paths = []
    for ext, dpi in (("png", 260), ("tiff", 600), ("svg", 300), ("pdf", 300)):
        path = out / f"{stem}.{ext}"
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def load_checked(study: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    summary = json.loads((study / "boundary_analysis_summary.json").read_text(encoding="utf-8"))
    if sha(study / "freeze.json") != summary["freeze_sha256"]:
        raise ValueError("Boundary analysis belongs to another freeze")
    for name, digest in summary["source_sha256"].items():
        if sha(study / name) != digest:
            raise ValueError("Analyzed source changed: " + name)
    for name, digest in summary["output_sha256"].items():
        if sha(study / name) != digest:
            raise ValueError("Boundary analysis output changed: " + name)
    files = ("evaluation_joined.csv", "boundary_candidate_truth.csv", "metrics_overall.csv", "near_threshold_metrics.csv")
    joined, truth, overall, near = (read_csv(study / name) for name in files)
    if set(joined.model) != set(MODELS) or truth.scenario.nunique() != 6:
        raise ValueError("Expected all six models and six conditions")
    return joined, truth, overall, near, summary


def response_figure(joined: pd.DataFrame, truth: pd.DataFrame, out: Path) -> list[Path]:
    """Testable statement: the sampled curves show actual threshold-side coverage."""
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.6), sharex=True, sharey=True)
    ymax = max(5., float(np.ceil(max(joined.predicted_peak.max(), truth.SNH_max_eval.max()) / 5.) * 5.))
    for i, (scenario, ax) in enumerate(zip(sorted(truth.scenario.unique()), axes.flat)):
        local = joined.loc[joined.scenario.eq(scenario)]
        true = truth.loc[truth.scenario.eq(scenario)].sort_values("effective_kla")
        panel_label(ax, "abcdef"[i])
        inset = ax.inset_axes([.51, .54, .46, .39])
        inset.set_facecolor("#FAFAF8")
        for name in MODELS:
            group = local.loc[local.model.eq(name)].sort_values("effective_kla")
            ax.plot(group.effective_kla, group.predicted_peak, color=COLORS[name], linestyle=STYLES[name],
                    lw=1.8, alpha=.95, zorder=2)
            inset.plot(group.effective_kla, group.predicted_peak, color=COLORS[name], linestyle=STYLES[name],
                       lw=.9, alpha=.9)
        coarse = true.loc[true.grid_kind.eq("coarse")]
        enrich = true.loc[true.grid_kind.eq("prediction_enriched")]
        ax.scatter(coarse.effective_kla, coarse.SNH_max_eval, marker="o", s=37, c="#222222",
                   edgecolors="white", linewidths=.5, zorder=5)
        ax.scatter(enrich.effective_kla, enrich.SNH_max_eval, marker="D", s=53, facecolors="white",
                   edgecolors="#9A2F75", linewidths=1.6, zorder=6)
        ax.axhline(4., color="#777777", lw=1., linestyle=(0, (3, 2)), zorder=1)
        inset.scatter(coarse.effective_kla, coarse.SNH_max_eval, marker="o", s=8, c="#222222", zorder=4)
        inset.scatter(enrich.effective_kla, enrich.SNH_max_eval, marker="D", s=11, facecolors="white",
                      edgecolors="#9A2F75", linewidths=.7, zorder=5)
        inset.axhline(4., color="#777777", lw=.6, linestyle=(0, (3, 2)))
        inset.set_xlim(float(truth.effective_kla.min()) - 3, float(truth.effective_kla.max()) + 3)
        inset.set_ylim(0, ymax * 1.035)
        inset.set_xticks([112, 220]); inset.set_yticks([0, ymax])
        inset.tick_params(axis="both", labelsize=8, pad=1, length=2)
        inset.set_title("Full sampled range", fontsize=9, pad=2)
        L = float(true.load_coordinate.iloc[0]); alpha = float(true.oxygen_transfer_factor.iloc[0])
        ax.set_title(f"L = {L:.2f}, α = {alpha:.4f}", pad=10)
        ax.set_xlim(136., 196.)
        ax.set_ylim(0., 12.)
        ax.set_xticks([140, 160, 180])
        if i // 3 == 1:
            ax.set_xlabel("Effective KLa, αKLa (d⁻¹)")
        if i % 3 == 0:
            ax.set_ylabel("Effluent NH₄ peak (mg N/L)")
        safe = int((true.SNH_max_eval <= 4.).sum())
        ax.text(.02, .98, f"BSM1: {safe} compliant\n{len(true) - safe} noncompliant", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
    handles = [Line2D([0], [0], color=COLORS[name], linestyle=STYLES[name], lw=2., label=LABELS[name]) for name in MODELS]
    handles.extend([Line2D([0], [0], marker="o", linestyle="none", color="#222222", markersize=6, label="BSM1: fixed grid"),
                    Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", color="#9A2F75", markersize=6, label="BSM1: enriched"),
                    Line2D([0], [0], color="#777777", linestyle=(0, (3, 2)), label="Compliance limit: 4 mg N/L")])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=3, frameon=False,
               handlelength=2.8, columnspacing=2.6, fontsize=11)
    fig.text(.5, .006, "Main panels: αKLa 136–196 d⁻¹, peak 0–12 mg N/L; insets include all samples. Lines join frozen predictions only.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.065, right=.99, top=.94, bottom=.20, wspace=.16, hspace=.34)
    joined[["model", "candidate_id", "scenario", "grid_kind", "effective_kla", "predicted_peak", "SNH_max_eval"]].to_csv(
        out / "boundary_response_plot_data.csv", index=False, encoding="utf-8-sig")
    return export(fig, out, "M3B_threshold_response")


def performance_figure(overall: pd.DataFrame, near: pd.DataFrame, out: Path) -> list[Path]:
    """Testable statement: precision, decision error and Gate cost differ."""
    primary = overall.loc[overall.stratum.eq("coarse")].set_index("model").loc[list(MODELS)].reset_index()
    local = near.loc[near.stratum.eq("coarse") & near.window_mgN_L.eq(.5)].set_index("model").loc[list(MODELS)].reset_index()
    if primary.n_candidates.nunique() != 1 or local.n_candidates.nunique() != 1:
        raise ValueError("Model denominators differ in the same stratum")
    n, n_near = int(primary.n_candidates.iloc[0]), int(local.n_candidates.iloc[0])
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3))
    for i, ax in enumerate(axes.flat):
        panel_label(ax, "abcd"[i])
        ax.set_xticks(x, SHORT)
        ax.margins(x=.06)
    ax = axes[0, 0]
    bars = ax.bar(x, primary.rmse, color=[COLORS[name] for name in MODELS], edgecolor="#FFFFFF", linewidth=.8, width=.68)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in primary.rmse], padding=3, fontsize=11)
    ax.set_ylim(0, max(float(primary.rmse.max()) * 1.22, .5))
    ax.set_ylabel("Peak RMSE (mg N/L)")
    ax.set_title(f"Fixed-grid concentration error (n = {n})", pad=10)
    ax = axes[0, 1]
    a = ax.bar(x - .18, primary.raw_false_releases, width=.36, color="#BC4C48", label="False release", edgecolor="white")
    b = ax.bar(x + .18, primary.raw_false_rejections, width=.36, color="#447EAB", label="False rejection", edgecolor="white")
    ax.bar_label(a, padding=2, fontsize=10); ax.bar_label(b, padding=2, fontsize=10)
    ax.set_ylim(0, max(float(max(primary.raw_false_releases.max(), primary.raw_false_rejections.max())) * 1.28, 2.))
    ax.set_ylabel("Raw decision error (count)")
    ax.set_title("Fixed-grid threshold decisions", pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax = axes[1, 0]
    bars = ax.bar(x, 100. * primary.gate_selected_fraction, color=[COLORS[name] for name in MODELS], edgecolor="white", width=.68)
    ax.bar_label(bars, labels=[f"{int(r.gate_selected)}/{n}" for r in primary.itertuples()], padding=3, fontsize=10)
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Gate reference fraction (%)")
    ax.set_title("Fixed-grid reference cost", pad=10)
    for xpos, row in enumerate(primary.itertuples()):
        ax.text(xpos, 3., f"Final errors: {int(row.final_errors)}", rotation=90, va="bottom", ha="center", fontsize=9, color="white")
    ax = axes[1, 1]
    finite = local.rmse.notna()
    bars = ax.bar(x[finite], local.loc[finite, "rmse"], color=[COLORS[MODELS[i]] for i in np.flatnonzero(finite)], edgecolor="white", width=.68)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in local.loc[finite, "rmse"]], padding=3, fontsize=11)
    ax.set_ylim(0, max(float(local.rmse.max()) * 1.22, .5) if finite.any() else 1.)
    ax.set_ylabel("Peak RMSE (mg N/L)")
    ax.set_title(f"Near threshold: |BSM1 − 4| ≤ 0.5 (n = {n_near})", pad=10)
    if not finite.any():
        ax.text(.5, .5, "No fixed-grid references in this window", transform=ax.transAxes, ha="center", va="center")
    fig.text(.5, .019, "All panels use the preselected fixed grid. Historical M2 used binary supervision; other models used continuous targets.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.075, right=.99, top=.94, bottom=.10, wspace=.26, hspace=.44)
    primary.to_csv(out / "boundary_performance_plot_data.csv", index=False, encoding="utf-8-sig")
    local.to_csv(out / "boundary_near_threshold_plot_data.csv", index=False, encoding="utf-8-sig")
    return export(fig, out, "M3B_fixed_grid_performance")


def make_figures(study_dir: Path = ROOT) -> dict:
    study = Path(study_dir).resolve()
    joined, truth, overall, near, summary = load_checked(study)
    style()
    out = study / "figures"
    out.mkdir(exist_ok=True)
    paths = response_figure(joined, truth, out) + performance_figure(overall, near, out)
    paths.extend(out / name for name in ("boundary_response_plot_data.csv", "boundary_performance_plot_data.csv",
                                        "boundary_near_threshold_plot_data.csv"))
    manifest = dict(version="M3B_boundary_figures_v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=sha(Path(__file__)), freeze_sha256=summary["freeze_sha256"],
                    boundary_analysis_summary_sha256=sha(study / "boundary_analysis_summary.json"),
                    primary_stratum="coarse", peak_units="mg N/L", effective_kla_units="d^-1",
                    reference_policy="Raw independent sampled peaks; no smoothing or added response points.",
                    line_policy="Straight joins of frozen sampled model predictions as visual guides only.",
                    response_zoom_policy="Main axes: effective KLa 136–196 d^-1 and peak 0–12 mg N/L; full-range insets include all frozen samples and all concentration values.",
                    formats="PNG 260 dpi; TIFF 600 dpi LZW; editable SVG/PDF text",
                    files={path.relative_to(study).as_posix(): sha(path) for path in paths})
    (out / "boundary_figures_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(make_figures(args.study_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
