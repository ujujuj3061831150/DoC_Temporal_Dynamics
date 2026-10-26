"""Test zero-collision script for Fig 2 and Fig 3."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("v3")
RESULTS_DIR = ROOT / "results/temporal_dynamics_v1"

def generate_fig2(output_path: Path):
    metrics_file = RESULTS_DIR / "metrics_summary.csv"
    metrics_df = pd.read_csv(metrics_file).set_index("model")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(3.55, 1.36), dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.95], "wspace": 0.35}
    )

    # ── (a) 2x2 Factorial Decomposition ──────────────────────────────────────────
    models = ["canonical__C", "canonical__D", "plus_lag1__C", "plus_lag1__D"]
    aucs = [metrics_df.loc[m, "macro_auroc"] for m in models]
    colors1 = ["#94A3B8", "#64748B", "#10B981", "#34D399"]

    x_a = np.arange(4)
    y_min_a = 0.70
    bars1 = ax1.bar(x_a, [val - y_min_a for val in aucs], bottom=y_min_a,
                    color=colors1, width=0.55, edgecolor="none", zorder=3)
    ax1.set_title("(a) $2 \\times 2$ Factorial", fontweight="bold", fontsize=7.8, pad=3)
    ax1.set_ylabel("AUROC", fontsize=7.2, fontweight="bold")
    ax1.set_ylim(0.70, 0.95)
    ax1.set_xlim(-0.70, 3.70)
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(["Lin.", "Rank", "Lin.", "Rank"], fontsize=6.0)
    ax1.tick_params(axis="x", length=0, pad=4)
    ax1.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    
    # Sub-category group labels below x-axis: "Static" (6 chars) and "+Order" (6 chars)
    ax1.text(0.5, -0.22, "Static", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", fontsize=5.8, fontweight="bold", color="#475569")
    ax1.text(2.5, -0.22, "+Order", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", fontsize=5.8, fontweight="bold", color="#047857")
    
    # Reference line at Static baseline (0.820)
    ax1.axhline(aucs[0], color="#94A3B8", ls="--", lw=0.7, zorder=2)
    
    for bar, val in zip(bars1, aucs):
        y_pos = val + 0.008 if val >= aucs[0] else val + 0.006
        ax1.annotate(
            f"{val:.3f}",
            (bar.get_x() + bar.get_width()/2, y_pos),
            ha="center", va="bottom",
            fontweight="bold", fontsize=5.6, color="#0F172A"
        )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.6)
    ax1.spines["bottom"].set_linewidth(0.6)

    # ── (b) Representation Ablations ─────────────────────────────────────────────
    # Use vertical labels (rotation=90) to eliminate ANY adjacent text collision!
    ablations = [
        "Static",
        r"Repl $\rho_1$",
        r"$+\rho_1$",
        r"$+D_1$",
        r"$r_{\mathrm{res}}$",
        "5-Feat",
        "4-Feat",
        "T-GECF",
    ]
    abl_aucs = [0.8200, 0.8851, 0.8911, 0.8611, 0.8433, 0.9149, 0.9324, 0.9116]

    abl_colors = [
        "#94A3B8",  # Static baseline
        "#4ADE80",  # Repl rho1
        "#16A34A",  # +rho1
        "#34D399",  # +D1
        "#38BDF8",  # r_res
        "#047857",  # 5-feat combined (rho1 + r)
        "#1D4ED8",  # 4-feat orthogonal HERO!
        "#F59E0B",  # T-GECF safety wrapper
    ]
    
    x_b = np.arange(len(ablations))
    y_min_b = 0.72
    bars2 = ax2.bar(
        x_b, [val - y_min_b for val in abl_aucs], bottom=y_min_b,
        color=abl_colors, width=0.58, edgecolor="none", zorder=3
    )
    ax2.set_title("(b) Representation Ablations", fontweight="bold", fontsize=7.8, pad=3)
    ax2.set_xlim(-0.70, len(ablations) - 0.30)
    ax2.set_xticks(x_b)
    # Vertical rotation with ha="center", va="top" ensures ZERO adjacent overlap
    ax2.set_xticklabels(ablations, rotation=90, ha="center", va="top", fontsize=5.8)
    ax2.tick_params(axis="x", length=0, pad=4)
    ax2.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    ax2.set_ylim(0.72, 0.97)
    
    # Reference line at Static baseline (0.820)
    ax2.axhline(abl_aucs[0], color="#94A3B8", ls="--", lw=0.7, zorder=2)
    
    # Clean milestone callouts: Baseline (0.820), Pure Temporal (0.891), and Hero 4-Feat (0.932)
    key_annots = [
        (0, 0.8200, "0.820", "#475569"),
        (2, 0.8911, "0.891", "#16A34A"),
        (6, 0.9324, r"\textbf{0.932}", "#1D4ED8"),
    ]
    for idx, val, txt, col in key_annots:
        bar = bars2[idx]
        ax2.annotate(
            txt,
            (bar.get_x() + bar.get_width()/2, val + 0.007),
            ha="center", va="bottom",
            fontweight="bold", fontsize=5.6, color=col
        )
        
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(0.6)
    ax2.spines["bottom"].set_linewidth(0.6)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.27, wspace=0.32)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated Fig 2: {output_path}")


def generate_fig3(output_path: Path):
    shuff_file = RESULTS_DIR / "shuffle_summary.csv"
    shuff_df = pd.read_csv(shuff_file)
    
    real_lag_c = 0.8911
    real_lag_d = 0.8591
    real_inc_c = 0.8611
    baseline_c = 0.8200
    
    # Figure height 1.45 in, tight margins
    fig, ax = plt.subplots(figsize=(3.55, 1.45), dpi=300)
    
    data_to_plot = [
        shuff_df["plus_lag1__C"].values,
        shuff_df["plus_lag1__D"].values,
        shuff_df["plus_increment__C"].values,
    ]
    x_positions = [1, 2, 3]
    labels = [
        "Plus $\\rho_1$\n(Linear)",
        "Plus $\\rho_1$\n(Rank)",
        "Plus $D_1$\n(Linear)",
    ]
    
    parts = ax.violinplot(data_to_plot, positions=x_positions, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#CBD5E1")
        pc.set_edgecolor("#64748B")
        pc.set_alpha(0.75)
        pc.set_linewidth(0.6)
    parts["cmeans"].set_color("#0F172A")
    parts["cmeans"].set_linewidth(1.5)
    parts["cbars"].set_color("#64748B")
    parts["cbars"].set_linewidth(0.6)
    parts["cmaxes"].set_color("#64748B")
    parts["cmaxes"].set_linewidth(0.6)
    parts["cmins"].set_color("#64748B")
    parts["cmins"].set_linewidth(0.6)
    
    real_points = [real_lag_c, real_lag_d, real_inc_c]
    ax.scatter(
        x_positions, real_points,
        color="#2563EB", edgecolors="#0F172A", lw=0.6, s=38, zorder=5,
        label="Intact Order"
    )
    
    ax.axhline(
        baseline_c, color="#64748B", ls="--", lw=0.8, zorder=2,
        label=f"Static Base ({baseline_c:.3f})"
    )
    
    ax.set_title("10,000 Synchronous Surrogates vs. Intact Order", fontweight="bold", fontsize=7.8, pad=4)
    ax.set_ylabel("Macro-AUROC (25 Folds)", fontsize=7.0, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    ax.set_xlim(0.4, 3.6)
    # y-axis from 0.58 to 0.95 gives plenty of lower-left headroom for legend
    ax.set_ylim(0.58, 0.95)
    
    # Legend at lower left: in y in [0.58, 0.74], well BELOW 0.820 line and clear of data
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.02, 0.04),
        frameon=True, facecolor="white", edgecolor="#E2E8F0", framealpha=0.92,
        fontsize=6.0, handletextpad=0.3, borderpad=0.3
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    
    fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.22)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated Fig 3: {output_path}")

if __name__ == "__main__":
    t_dir = Path("v3/manuscript_temporal/figures")
    generate_fig2(t_dir / "test_fig2.pdf")
    generate_fig3(t_dir / "test_fig3.pdf")
