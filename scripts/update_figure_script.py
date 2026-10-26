"""Update generate_temporal_figures.py in both locations with the verified zero-collision implementation."""
from pathlib import Path

code_fig2_fig3 = '''def plot_figure2_violin(results_dir: Path, output_path: Path):
    """Figure 2 (or 3): Performance Collapse Violin Plot under 10,000 Synchronous Permutations."""
    shuff_file = results_dir / "shuffle_summary.csv"
    metrics_file = results_dir / "metrics_summary.csv"
    if not shuff_file.exists() or not metrics_file.exists():
        print(f"Waiting for benchmark files in {results_dir}")
        return
        
    shuff_df = pd.read_csv(shuff_file)
    metrics_df = pd.read_csv(metrics_file).set_index("model")
    
    # Native single-column canvas: 3.55 inches wide, 1.45 inches high
    fig, ax = plt.subplots(figsize=(3.55, 1.45), dpi=300)
    
    real_lag_c = metrics_df.loc["plus_lag1__C", "macro_auroc"]
    real_lag_d = metrics_df.loc["plus_lag1__D", "macro_auroc"]
    real_inc_c = metrics_df.loc["plus_increment__C", "macro_auroc"]
    baseline_c = metrics_df.loc["canonical__C", "macro_auroc"]
    
    data_to_plot = [
        shuff_df["plus_lag1__C"].values,
        shuff_df["plus_lag1__D"].values,
        shuff_df["plus_increment__C"].values,
    ]
    x_positions = [1, 2, 3]
    labels = [
        "Plus $\\\\rho_1$\\n(Linear)",
        "Plus $\\\\rho_1$\\n(Rank)",
        "Plus $D_1$\\n(Linear)",
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
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path} and {png_path}")


def plot_figure3_factorial_and_ablations(results_dir: Path, output_path: Path):
    """Figure 3 (or 2): 2x2 Orthogonal Factorial Matrix & Feature Ablations."""
    metrics_file = results_dir / "metrics_summary.csv"
    contrasts_file = results_dir / "contrasts.csv"
    if not metrics_file.exists() or not contrasts_file.exists():
        return
        
    metrics_df = pd.read_csv(metrics_file).set_index("model")
    
    # Native single-column canvas: 3.55 inches wide, 1.36 inches high
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
    bars1 = ax1.bar(
        x_a, [val - y_min_a for val in aucs], bottom=y_min_a,
        color=colors1, width=0.55, edgecolor="none", zorder=3
    )
    ax1.set_title("(a) $2 \\\\times 2$ Factorial", fontweight="bold", fontsize=7.8, pad=3)
    ax1.set_ylabel("AUROC", fontsize=7.2, fontweight="bold")
    ax1.set_ylim(0.70, 0.95)
    ax1.set_xlim(-0.70, 3.70)
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(["Lin.", "Rank", "Lin.", "Rank"], fontsize=6.0)
    ax1.tick_params(axis="x", length=0, pad=2)
    ax1.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    
    # Sub-category group labels below x-axis
    ax1.text(0.5, -0.22, "Static", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", fontsize=5.8, fontweight="bold", color="#475569")
    ax1.text(2.5, -0.22, "+Order", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", fontsize=5.8, fontweight="bold", color="#047857")
    
    # Static baseline reference line
    ax1.axhline(aucs[0], color="#94A3B8", ls="--", lw=0.7, zorder=2)
    
    # Annotate bar values clearly above the bars
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
    ablations = [
        "Static",
        r"Repl $\\\\rho_1$",
        r"$+\\\\rho_1$",
        r"$+D_1$",
        r"$r_{\\\\mathrm{res}}$",
        "5-Feat",
        "4-Feat",
        "T-GECF",
    ]
    abl_models = [
        "canonical__C",
        "replace_cv_lag1__C",
        "plus_lag1__C",
        "plus_increment__C",
        "residual_corr__C",
        "plus_lag1_residual__C",
        "drop_c_spatial",
        "t_gecf_adaptive",
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
    ax2.set_xticklabels(ablations, rotation=90, ha="center", va="top", fontsize=5.8)
    ax2.tick_params(axis="x", length=0, pad=2)
    ax2.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    ax2.set_ylim(0.72, 0.97)
    
    # Reference line at Static baseline (0.820)
    ax2.axhline(abl_aucs[0], color="#94A3B8", ls="--", lw=0.7, zorder=2)
    
    # Clean milestone callouts: Baseline (0.820), Pure Temporal (0.891), and Hero 4-Feat (0.932)
    key_annots = [
        (0, 0.8200, "0.820", "#475569"),
        (2, 0.8911, "0.891", "#16A34A"),
        (6, 0.9324, r"\\\\textbf{0.932}", "#1D4ED8"),
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
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path} and {png_path}")
'''

import re

for file_path in ["v3/scripts/generate_temporal_figures.py", "scripts/generate_temporal_figures.py"]:
    p = Path(file_path)
    if not p.exists():
        continue
    content = p.read_text(encoding="utf-8")
    
    # Replace from def plot_figure2_violin to def export_latex_table1
    pattern = r"def plot_figure2_violin\(.*?\n(?=def export_latex_table1)"
    new_content, count = re.subn(pattern, code_fig2_fig3 + "\n\n", content, flags=re.DOTALL)
    if count == 1:
        p.write_text(new_content, encoding="utf-8")
        print(f"Successfully updated {file_path}")
    else:
        print(f"Could not update {file_path}, pattern matched {count} times")
