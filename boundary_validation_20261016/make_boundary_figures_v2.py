"""Submission-ready M3B figures using the frozen, audited analysis tables.

The figure design follows the supplied reference library: faceted interaction
panels, restrained categorical colors, and explicit tabular-style annotations.
No numerical data are changed, smoothed, interpolated, or extended.
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
LABELS = {
    "full_calibrated": "Full, calibrated ratio (6p)",
    "full_fixed_ratio": "Full, fixed ratio (5p)",
    "no_feedback": "No feedback (5p)",
    "poly_ridge": "Polynomial ridge",
    "rbf_kernel_ridge": "RBF kernel ridge",
    "legacy_M2_binary": "Historical M2 (binary fit)",
}
SHORT = ("Full\n6p", "Full\n5p", "No\nfeedback", "Poly\nridge", "RBF\nridge", "Historical\nM2")
COLORS = {
    # Keep the palette used by the existing M1/M2/M3 figures so readers can
    # identify a model consistently across the manuscript.
    "full_calibrated": "#193B67",
    "full_fixed_ratio": "#1A8A88",
    "no_feedback": "#DF7549",
    "poly_ridge": "#B69322",
    "rbf_kernel_ridge": "#805EA4",
    "legacy_M2_binary": "#4E9DBC",
}
STYLES = {
    "full_calibrated": "-",
    "full_fixed_ratio": "--",
    "no_feedback": "-.",
    "poly_ridge": ":",
    "rbf_kernel_ridge": (0, (6, 2, 1.5, 2)),
    "legacy_M2_binary": (0, (8, 3)),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.facecolor": "#F5F4F0",
        "axes.facecolor": "#FFFFFF",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#44515A",
        "axes.linewidth": 0.75,
        "grid.color": "#D7DEE1",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "#F5F4F0",
    })


def panel_label(ax, letter: str) -> None:
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", color="#25313B")
    ax.set_axisbelow(True)
    ax.grid(axis="y")


def export(fig, out: Path, stem: str) -> list[Path]:
    paths = []
    for ext, dpi in (("png", 320), ("tiff", 600), ("svg", 300), ("pdf", 300)):
        path = out / f"{stem}.{ext}"
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.10, **kwargs)
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
            raise ValueError("Analyzed source output changed: " + name)
    files = ("evaluation_joined.csv", "boundary_candidate_truth.csv",
             "metrics_overall.csv", "near_threshold_metrics.csv")
    joined, truth, overall, near = (read_csv(study / name) for name in files)
    if set(joined.model) != set(MODELS) or truth.scenario.nunique() != 6:
        raise ValueError("Expected all six models and six conditions")
    return joined, truth, overall, near, summary


def response_figure(joined: pd.DataFrame, truth: pd.DataFrame, out: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(13.7, 7.7), sharex=True, sharey=True)
    ymax = max(5.0, float(np.ceil(max(joined.predicted_peak.max(), truth.SNH_max_eval.max()) / 5.0) * 5.0))
    for i, (scenario, ax) in enumerate(zip(sorted(truth.scenario.unique()), axes.flat)):
        local = joined.loc[joined.scenario.eq(scenario)]
        true = truth.loc[truth.scenario.eq(scenario)].sort_values("effective_kla")
        panel_label(ax, "abcdef"[i])
        # The green band is a visual encoding of the actual compliance region.
        ax.axhspan(0, 4, facecolor="#EAF5EF", alpha=0.60, zorder=0)
        for name in MODELS:
            group = local.loc[local.model.eq(name)].sort_values("effective_kla")
            ax.plot(group.effective_kla, group.predicted_peak, color=COLORS[name],
                    linestyle=STYLES[name], lw=1.65, alpha=.97, zorder=2)
        coarse = true.loc[true.grid_kind.eq("coarse")]
        enrich = true.loc[true.grid_kind.eq("prediction_enriched")]
        ax.scatter(coarse.effective_kla, coarse.SNH_max_eval, marker="o", s=31,
                   c="#20252A", edgecolors="white", linewidths=.45, zorder=5)
        ax.scatter(enrich.effective_kla, enrich.SNH_max_eval, marker="D", s=47,
                   facecolors="white", edgecolors="#9D3679", linewidths=1.35, zorder=6)
        ax.axhline(4., color="#626B70", lw=.9, linestyle=(0, (3, 2)), zorder=1)
        # inset: all samples, retaining the true sampled peaks and frozen lines.
        inset = ax.inset_axes([.535, .54, .445, .37])
        inset.set_facecolor("#FAFAF8")
        for name in MODELS:
            group = local.loc[local.model.eq(name)].sort_values("effective_kla")
            inset.plot(group.effective_kla, group.predicted_peak, color=COLORS[name],
                       linestyle=STYLES[name], lw=.76, alpha=.90)
        inset.scatter(coarse.effective_kla, coarse.SNH_max_eval, marker="o", s=7,
                      c="#20252A", zorder=4)
        inset.scatter(enrich.effective_kla, enrich.SNH_max_eval, marker="D", s=10,
                      facecolors="white", edgecolors="#9D3679", linewidths=.65, zorder=5)
        inset.axhline(4., color="#626B70", lw=.55, linestyle=(0, (3, 2)))
        inset.set_xlim(float(truth.effective_kla.min()) - 3, float(truth.effective_kla.max()) + 3)
        inset.set_ylim(0, ymax * 1.035)
        inset.set_xticks([112, 220]); inset.set_yticks([0, ymax])
        inset.tick_params(axis="both", labelsize=7, pad=1, length=2)
        inset.set_title("full sampled range", fontsize=8, pad=1.5, color="#4A555B")
        for spine in inset.spines.values():
            spine.set_edgecolor("#9AA4A9"); spine.set_linewidth(.55)
        L = float(true.load_coordinate.iloc[0]); alpha = float(true.oxygen_transfer_factor.iloc[0])
        ax.set_title(f"L = {L:.2f}, α = {alpha:.4f}", pad=7, color="#25313B", fontweight="bold")
        ax.set_xlim(136., 196.); ax.set_ylim(0., 12.); ax.set_xticks([140, 160, 180])
        if i // 3 == 1: ax.set_xlabel("Effective KLa, αKLa (d$^{-1}$)")
        if i % 3 == 0: ax.set_ylabel("Effluent NH$_4$ peak (mg N/L)")
        safe = int((true.SNH_max_eval <= 4.).sum())
        ax.text(.025, .965, f"n$_C$ = {safe}  |  n$_{{NC}}$ = {len(true) - safe}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.1,
                color="#324047", bbox=dict(facecolor="white", edgecolor="#D4DADB",
                                            linewidth=.45, alpha=.90, pad=2.2))
    handles = [Line2D([0], [0], color=COLORS[name], linestyle=STYLES[name], lw=1.9,
                      label=LABELS[name]) for name in MODELS]
    handles += [Line2D([0], [0], marker="o", linestyle="none", color="#20252A",
                        markersize=5.5, label="BSM1: fixed grid"),
                Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
                        markeredgecolor="#9D3679", color="#9D3679", markersize=5.7,
                        label="BSM1: enriched"),
                Line2D([0], [0], color="#626B70", linestyle=(0, (3, 2)), lw=1.0,
                        label="Compliance limit: 4 mg N/L")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .017), ncol=3,
               frameon=False, handlelength=2.6, columnspacing=1.9, fontsize=9.4)
    fig.text(.5, .001, "Green band denotes the compliant region (BSM1 peak ≤ 4 mg N/L). Lines join frozen predictions only.",
             ha="center", fontsize=8.8, color="#566269")
    fig.subplots_adjust(left=.065, right=.995, top=.94, bottom=.19, wspace=.16, hspace=.34)
    joined[["model", "candidate_id", "scenario", "grid_kind", "effective_kla",
            "predicted_peak", "SNH_max_eval"]].to_csv(out / "boundary_response_plot_data_v2.csv",
                                                        index=False, encoding="utf-8-sig")
    return export(fig, out, "M3B_threshold_response_v2")


def performance_figure(overall: pd.DataFrame, near: pd.DataFrame, out: Path) -> list[Path]:
    primary = overall.loc[overall.stratum.eq("coarse")].set_index("model").loc[list(MODELS)].reset_index()
    local = near.loc[near.stratum.eq("coarse") & near.window_mgN_L.eq(.5)].set_index("model").loc[list(MODELS)].reset_index()
    if primary.n_candidates.nunique() != 1 or local.n_candidates.nunique() != 1:
        raise ValueError("Model denominators differ in the same stratum")
    n, n_near = int(primary.n_candidates.iloc[0]), int(local.n_candidates.iloc[0])
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.05))
    for i, ax in enumerate(axes.flat):
        panel_label(ax, "abcd"[i]); ax.set_xticks(x, SHORT); ax.margins(x=.06)
    # a: concentration accuracy
    ax = axes[0, 0]
    bars = ax.bar(x, primary.rmse, color=[COLORS[name] for name in MODELS],
                  edgecolor="#FFFFFF", linewidth=.75, width=.68)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in primary.rmse], padding=2.5, fontsize=9.8)
    ax.set_ylim(0, max(float(primary.rmse.max()) * 1.22, .5)); ax.set_ylabel("Peak RMSE (mg N/L)")
    ax.set_title(f"Concentration error (n = {n})", pad=8, fontweight="bold")
    # b: raw decision errors, including zero false release markers
    ax = axes[0, 1]
    a = ax.bar(x - .17, primary.raw_false_releases, width=.34, color="#C34E4B",
               label="False release", edgecolor="white", linewidth=.6)
    b = ax.bar(x + .17, primary.raw_false_rejections, width=.34, color="#4B82AE",
               label="False rejection", edgecolor="white", linewidth=.6)
    ax.scatter(x - .17, primary.raw_false_releases, s=20, facecolors="none",
               edgecolors="#C34E4B", linewidths=1.1, zorder=4)
    ax.bar_label(a, labels=[str(int(v)) for v in primary.raw_false_releases], padding=2, fontsize=9)
    ax.bar_label(b, labels=[str(int(v)) for v in primary.raw_false_rejections], padding=2, fontsize=9)
    ax.set_ylim(-1.0, max(float(primary.raw_false_rejections.max()) * 1.25, 3.))
    ax.set_ylabel("Raw decision error (count)"); ax.set_title("Threshold decisions", pad=8, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=8.7)
    # c: gate reference cost; top labels carry the count, so no dense vertical text
    ax = axes[1, 0]
    bars = ax.bar(x, 100. * primary.gate_selected_fraction, color=[COLORS[name] for name in MODELS],
                  edgecolor="white", linewidth=.75, width=.68)
    ax.bar_label(bars, labels=[f"{int(r.gate_selected)}/{n}" for r in primary.itertuples()],
                 padding=2.5, fontsize=9.2)
    ax.set_ylim(0, 114); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Gate reference fraction (%)")
    ax.set_title("Reference cost", pad=8, fontweight="bold")
    ax.text(.01, -.25, "Final errors = 0 for all six models on the fixed grid", transform=ax.transAxes,
            fontsize=8.5, color="#566269", ha="left")
    # d: near-threshold performance, with the two most accurate methods called out
    ax = axes[1, 1]
    finite = local.rmse.notna()
    bars = ax.bar(x[finite], local.loc[finite, "rmse"],
                  color=[COLORS[MODELS[i]] for i in np.flatnonzero(finite)],
                  edgecolor="white", linewidth=.75, width=.68)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in local.loc[finite, "rmse"]], padding=2.5, fontsize=9.6)
    ax.set_ylim(0, max(float(local.rmse.max()) * 1.22, .5) if finite.any() else 1.)
    ax.set_ylabel("Peak RMSE (mg N/L)"); ax.set_title(f"Near threshold: |BSM1 − 4| ≤ 0.5 (n = {n_near})",
                                                         pad=8, fontweight="bold")
    if finite.any():
        best_i = int(np.nanargmin(local.rmse.to_numpy()))
        ax.get_xticklabels()[best_i].set_fontweight("bold")
    fig.text(.5, .018, "All panels use the preselected fixed grid; the historical M2 model used binary supervision.",
             ha="center", fontsize=8.8, color="#566269")
    fig.subplots_adjust(left=.075, right=.99, top=.94, bottom=.105, wspace=.25, hspace=.48)
    primary.to_csv(out / "boundary_performance_plot_data_v2.csv", index=False, encoding="utf-8-sig")
    local.to_csv(out / "boundary_near_threshold_plot_data_v2.csv", index=False, encoding="utf-8-sig")
    return export(fig, out, "M3B_fixed_grid_performance_v2")


def make_figures(study_dir: Path = ROOT) -> dict:
    study = Path(study_dir).resolve(); joined, truth, overall, near, summary = load_checked(study)
    style(); out = study / "figures_v2"; out.mkdir(exist_ok=True)
    paths = response_figure(joined, truth, out) + performance_figure(overall, near, out)
    paths.extend(out / name for name in ("boundary_response_plot_data_v2.csv",
                                         "boundary_performance_plot_data_v2.csv",
                                         "boundary_near_threshold_plot_data_v2.csv"))
    manifest = dict(version="M3B_boundary_figures_v2", created_at_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=sha(Path(__file__)), freeze_sha256=summary["freeze_sha256"],
                    boundary_analysis_summary_sha256=sha(study / "boundary_analysis_summary.json"),
                    primary_stratum="coarse", peak_units="mg N/L", effective_kla_units="d^-1",
                    reference_policy="Raw independent sampled peaks; no smoothing or added response points.",
                    line_policy="Straight joins of frozen sampled model predictions as visual guides only.",
                    design_note="Faceted interaction plot and restrained categorical palette based on the supplied reference library.",
                    formats="PNG 320 dpi; TIFF 600 dpi LZW; editable SVG/PDF text",
                    files={path.relative_to(study).as_posix(): sha(path) for path in paths})
    (out / "boundary_figures_manifest_v2.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", nargs="?", type=Path, default=ROOT)
    args = parser.parse_args(); print(json.dumps(make_figures(args.study_dir), indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
