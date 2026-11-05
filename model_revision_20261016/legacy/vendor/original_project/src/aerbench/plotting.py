from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_pareto(summary_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for scenario, g in summary_df.groupby("scenario"):
        fig = plt.figure(figsize=(6, 4))
        for _, row in g.iterrows():
            xerr = row.get("energy_kwh_ci95", row.get("energy_kwh_std", 0.0))
            yerr = row.get("violation_hours_ci95", row.get("violation_hours_std", 0.0))
            plt.errorbar(
                row["energy_kwh_mean"],
                row["violation_hours_mean"],
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                capsize=3,
                label=row["controller"],
            )
        plt.xlabel("Aeration energy proxy (kWh)")
        plt.ylabel("NH4 violation duration (h)")
        plt.title(f"Energy-NH4 trade-off: {scenario}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out_path.parent / f"pareto_{scenario}.png", dpi=300)
        plt.close(fig)


def plot_warmup_bias(warmup_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = warmup_df.groupby("scenario", as_index=False).agg(
        energy_bias=("warmup_bias_energy_kwh", "mean"),
        violation_bias=("warmup_bias_violation_hours", "mean"),
    )
    fig = plt.figure(figsize=(7, 4))
    plt.bar(g["scenario"], g["energy_bias"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Warm-up mixed energy bias (kWh)")
    plt.tight_layout()
    fig.savefig(out_path.parent / "warmup_energy_bias.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.bar(g["scenario"], g["violation_bias"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Warm-up mixed NH4 violation bias (h)")
    plt.tight_layout()
    fig.savefig(out_path.parent / "warmup_violation_bias.png", dpi=300)
    plt.close(fig)


def plot_tuning_pareto(tuning_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fam, g in tuning_df.groupby("controller_family"):
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(g["energy_kwh_mean"], g["violation_hours_mean"], s=18, alpha=0.7)
        if "pareto_efficient" in g.columns:
            p = g[g["pareto_efficient"].astype(bool)]
            if not p.empty:
                plt.scatter(p["energy_kwh_mean"], p["violation_hours_mean"], s=45, marker="x", label="Pareto-efficient")
                plt.legend(fontsize=8)
        plt.xlabel("Mean aeration energy proxy (kWh)")
        plt.ylabel("Mean NH4 violation duration (h)")
        plt.title(f"Baseline tuning Pareto front: {fam}")
        plt.tight_layout()
        fig.savefig(out_dir / f"tuning_pareto_{fam.replace('/', '_')}.png", dpi=300)
        plt.close(fig)


def plot_riskcue_marginal(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "delta_violation_hours_mean" not in df.columns:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 4))
    plt.bar(df["scenario"], -df["delta_violation_hours_mean"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Violation hours reduced by Risk-cue-MPC vs MPC")
    plt.tight_layout()
    fig.savefig(out_dir / "riskcue_violation_reduction.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.bar(df["scenario"], df["delta_energy_kwh_mean"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Additional energy of Risk-cue-MPC vs MPC (kWh)")
    plt.tight_layout()
    fig.savefig(out_dir / "riskcue_energy_cost.png", dpi=300)
    plt.close(fig)


def plot_bsm1_kla_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty or "experiment" not in df.columns:
        return
    fixed = df[df["experiment"].eq("fixed_kla")].copy()
    if fixed.empty:
        return
    ycol = "SNH_violation_hours_gt_limit" if "SNH_violation_hours_gt_limit" in fixed.columns else "SNH_violation_days_gt_4"
    mode_col = "inflow_mode" if "inflow_mode" in fixed.columns else None
    out_dir.mkdir(parents=True, exist_ok=True)

    group_cols = ["scenario"]
    if mode_col:
        group_cols = [mode_col, "scenario"]

    for keys, g in fixed.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = ("", keys)
        if mode_col:
            mode, scenario = keys
            suffix = f"{mode}_{scenario}"
            title_prefix = f"BSM1 {mode}: {scenario}"
        else:
            scenario = keys[-1]
            suffix = str(scenario)
            title_prefix = f"BSM1: {scenario}"

        g = g.sort_values("kla")

        fig = plt.figure(figsize=(6, 4))
        plt.plot(g["kla"], g[ycol], marker="o")
        plt.xlabel("Nominal KLa")
        plt.ylabel("Effluent NH4 violation duration (h)")
        plt.title(f"{title_prefix} fixed-KLa violations")
        plt.tight_layout()
        fig.savefig(out_dir / f"bsm1_kla_violation_{suffix}.png", dpi=300)
        plt.close(fig)

        if "SNH_mean_eval" in g.columns:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(g["kla"], g["SNH_mean_eval"], marker="o")
            plt.xlabel("Nominal KLa")
            plt.ylabel("Mean effluent NH4 during evaluation (mgN/L)")
            plt.title(f"{title_prefix} fixed-KLa NH4 response")
            plt.tight_layout()
            fig.savefig(out_dir / f"bsm1_kla_snh_mean_{suffix}.png", dpi=300)
            plt.close(fig)

        if "TN_mean_eval" in g.columns:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(g["kla"], g["TN_mean_eval"], marker="o")
            plt.xlabel("Nominal KLa")
            plt.ylabel("Mean model-consistent effluent TN (mgN/L)")
            plt.title(f"{title_prefix} fixed-KLa TN response")
            plt.tight_layout()
            fig.savefig(out_dir / f"bsm1_kla_tn_mean_{suffix}.png", dpi=300)
            plt.close(fig)

        if "commanded_aeration_index" in g.columns:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(g["kla"], g["commanded_aeration_index"], marker="o")
            plt.xlabel("Nominal/commanded KLa")
            plt.ylabel("Commanded aeration index (-)")
            plt.title(f"{title_prefix} fixed-KLa commanded aeration index")
            plt.tight_layout()
            fig.savefig(out_dir / f"bsm1_kla_commanded_aeration_index_{suffix}.png", dpi=300)
            plt.close(fig)

        if "effective_aeration_kWh_per_d" in g.columns:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(g["kla"], g["effective_aeration_kWh_per_d"], marker="o")
            plt.xlabel("Nominal/commanded KLa")
            plt.ylabel("Effective-KLa internal aeration energy (kWh/d)")
            plt.title(f"{title_prefix} fixed-KLa effective energy")
            plt.tight_layout()
            fig.savefig(out_dir / f"bsm1_kla_effective_energy_{suffix}.png", dpi=300)
            plt.close(fig)


def make_all_plots(output_dir: Path = Path("outputs")) -> None:
    fig_dir = output_dir / "figures"
    summary_path = output_dir / "tables" / "summary_metrics.csv"
    if summary_path.exists():
        plot_pareto(pd.read_csv(summary_path), fig_dir / "pareto.png")
    warmup_path = output_dir / "tables" / "warmup_contamination_bias.csv"
    if warmup_path.exists():
        plot_warmup_bias(pd.read_csv(warmup_path), fig_dir / "warmup_bias.png")
    tuning_path = output_dir / "tables" / "baseline_tuning_results.csv"
    if tuning_path.exists():
        plot_tuning_pareto(pd.read_csv(tuning_path), fig_dir)
    riskcue_path = output_dir / "tables" / "riskcue_marginal_value.csv"
    if riskcue_path.exists():
        plot_riskcue_marginal(pd.read_csv(riskcue_path), fig_dir)
    bsm1_path = output_dir / "bsm1_validation" / "bsm1_validation_summary.csv"
    if bsm1_path.exists():
        plot_bsm1_kla_sweep(pd.read_csv(bsm1_path), fig_dir)
