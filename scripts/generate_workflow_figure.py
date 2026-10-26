"""
Generate Supplementary Figure S1: Bedside CDSS Operational Workflow & Architecture.
Publication-grade IEEE Transactions / Nature Biomedical Engineering styling:
- Unified deep slate header banners across all four stages for authoritative cohesion.
- Pure white stage containers with crisp neutral borders, eliminating nested card syndrome.
- Streamlined visual schematics and mathematical operator definitions without bullet points.
- Scalp 10-20 schematic with 6 active telemetry electrodes (F3, F4, C3, C4, O1, O2).
- Discrete epoch timeline with physical artifact gap masking (I_adj = 0/1).
- Parallel orthogonal feature extraction (Temporal Memory & Spatial OLS Decoupling).
- 4-domain feature suite with domain badges and T-GECF continuous volatility gating.
- ICU clinical triage cutoff gate with bifurcated action pathways (Up: CMD/fMRI, Down: Longitudinal EEG).
- STIX mathematical typography with zero overlap or clipping.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]


def build_workflow_figure(output_paths: list[Path]):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(15.2, 5.3), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    # =========================================================================
    # RESTRAINED, PUBLICATION-GRADE ACADEMIC PALETTE
    # =========================================================================
    C_HEADER_BG = "#1E293B"      # Deep Slate 900: unified, authoritative headers
    C_STAGE_BG = "#FFFFFF"       # Clean pure white stage background
    C_BORDER = "#CBD5E1"         # Slate 300 neutral border (0.8 pt)
    C_DIVIDER = "#E2E8F0"        # Slate 200 horizontal divider rule
    C_BOX_BG = "#F8FAFC"         # Slate 50 clean neutral background
    C_BOX_BORDER = "#CBD5E1"     # Neutral formula border
    C_MATH_BG = "#F0F9FF"        # Sky 50 light tint for key mathematical extraction
    C_MATH_BORDER = "#BAE6FD"    # Sky 200 border
    C_TEXT = "#0F172A"           # Slate 900 primary text
    C_MUTED = "#475569"          # Slate 600 secondary text
    C_BLUE = "#2563EB"           # Royal Blue for dataflow & linear fusion
    C_BLUE_BG = "#EFF6FF"        # Blue 50
    C_BLUE_BORDER = "#93C5FD"    # Blue 300
    C_GREEN = "#16A34A"          # Emerald Green for positive triage (CMD candidate)
    C_GREEN_BG = "#F0FDF4"       # Emerald 50 fill
    C_GREEN_BORDER = "#86EFAC"   # Emerald 300 border
    C_AMBER = "#B45309"          # Amber 700 for high volatility / Rank Consensus
    C_AMBER_BG = "#FFFBEB"       # Amber 50 fill
    C_AMBER_BORDER = "#FDE68A"   # Amber 200 border
    C_RED = "#DC2626"            # Subtle Crimson for artifact gap masking
    C_RED_BG = "#FEF2F2"         # Crimson 50 fill
    C_RED_BORDER = "#FCA5A5"     # Crimson 300 border

    STAGE_TITLES = [
        "Stage 1: Bedside Acquisition & Masking",
        "Stage 2: Dual Orthogonal Extraction",
        "Stage 3: Fault-Tolerant Fusion",
        "Stage 4: ICU Clinical Triage Funnel",
    ]

    col_w = 3.20
    col_gap = 0.52
    x_starts = [0.42 + i * (col_w + col_gap) for i in range(4)]
    y_bot = 0.22
    y_top = 5.08
    box_h = y_top - y_bot  # 4.86
    h_h = 0.44

    # 1. Main Stage Containers & Unified Header Banners
    for i, (xs, title) in enumerate(zip(x_starts, STAGE_TITLES)):
        # Primary container (clean white, crisp neutral border)
        container = patches.FancyBboxPatch(
            (xs, y_bot), col_w, box_h,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=C_STAGE_BG, edgecolor=C_BORDER, linewidth=0.8, zorder=1
        )
        ax.add_patch(container)

        # Unified Header Banner across all 4 stages
        header = patches.FancyBboxPatch(
            (xs, y_top - h_h), col_w, h_h,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=C_HEADER_BG, edgecolor=C_HEADER_BG, linewidth=0, zorder=2
        )
        ax.add_patch(header)
        header_sq = patches.Rectangle(
            (xs, y_top - h_h), col_w, h_h / 2,
            facecolor=C_HEADER_BG, edgecolor=C_HEADER_BG, linewidth=0, zorder=2
        )
        ax.add_patch(header_sq)

        ax.text(
            xs + col_w / 2, y_top - h_h / 2, title,
            color="#FFFFFF", fontsize=9.2, fontweight="bold",
            ha="center", va="center", zorder=3
        )

    # Common vertical alignment coordinates across all stages
    y_subhead1 = 4.38
    y_divider = 2.50
    y_subhead2 = 2.26
    y_bottom_tag = 0.44
    h_bottom_tag = 0.36

    # =========================================================================
    # STAGE 1: Bedside Acquisition & Masking
    # =========================================================================
    x1 = x_starts[0]
    cx1 = x1 + col_w / 2

    # Top Section Label
    ax.text(x1 + 0.20, y_subhead1, "Sparse 6-Channel Telemetry", fontsize=8.6, fontweight="bold", color=C_TEXT, zorder=3)

    # Scalp schematic (compact, clean, positioned with breathing room)
    head_cx, head_cy, head_r = x1 + 0.76, 3.48, 0.58
    head_circ = patches.Circle((head_cx, head_cy), head_r, facecolor="#F8FAFC", edgecolor="#64748B", linewidth=1.1, zorder=3)
    ax.add_patch(head_circ)
    # Nose (smooth arc/polygon)
    nose = patches.Polygon([
        (head_cx - 0.07, head_cy + head_r - 0.01),
        (head_cx, head_cy + head_r + 0.11),
        (head_cx + 0.07, head_cy + head_r - 0.01)
    ], facecolor="#F8FAFC", edgecolor="#64748B", linewidth=1.1, zorder=3)
    ax.add_patch(nose)
    # Ears (smooth curved arcs)
    ear_l = patches.Arc((head_cx - head_r, head_cy), 0.10, 0.26, angle=0, theta1=90, theta2=270, color="#64748B", lw=1.1, zorder=3)
    ear_r = patches.Arc((head_cx + head_r, head_cy), 0.10, 0.26, angle=0, theta1=270, theta2=90, color="#64748B", lw=1.1, zorder=3)
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)

    # 6 Electrodes (F3, F4, C3, C4, O1, O2)
    electrodes = [
        ("F3", head_cx - 0.25, head_cy + 0.23),
        ("F4", head_cx + 0.25, head_cy + 0.23),
        ("C3", head_cx - 0.35, head_cy - 0.02),
        ("C4", head_cx + 0.35, head_cy - 0.02),
        ("O1", head_cx - 0.21, head_cy - 0.34),
        ("O2", head_cx + 0.21, head_cy - 0.34),
    ]
    for name, ex, ey in electrodes:
        dot = patches.Circle((ex, ey), 0.11, facecolor=C_HEADER_BG, edgecolor="#0F172A", linewidth=0.7, zorder=4)
        ax.add_patch(dot)
        ax.text(ex, ey, name, color="#FFFFFF", fontsize=5.8, fontweight="bold", ha="center", va="center", zorder=5)

    # Clean Specs Badges right of scalp
    specs = [
        ("10-20 Sparse", 3.94),
        ("CAR Reference", 3.48),
        ("1,000 Hz / 4-s Epochs", 3.02),
    ]
    for spec_text, sy in specs:
        spec_box = patches.FancyBboxPatch(
            (x1 + 1.60, sy - 0.15), 1.40, 0.30,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            facecolor="#F1F5F9", edgecolor=C_BORDER, lw=0.6, zorder=3
        )
        ax.add_patch(spec_box)
        ax.text(x1 + 2.30, sy, spec_text, fontsize=6.8, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=4)

    # Divider rule between Top & Bottom of Stage 1
    ax.plot([x1 + 0.15, x1 + col_w - 0.15], [y_divider, y_divider], color=C_DIVIDER, lw=0.8, zorder=2)

    # Bottom Section Label
    ax.text(x1 + 0.20, y_subhead2, "Physical Adjacency Masking", fontsize=8.6, fontweight="bold", color=C_TEXT, zorder=3)

    # Discrete time strip: t_1 -> t_2 (Intact) -> Artifact Gap -> t_3
    bx_y = 1.62
    bx_h = 0.40
    bx_w = 0.46
    gap_w = 0.82

    b1_x = cx1 - 1.35
    ax.add_patch(patches.Rectangle((b1_x, bx_y), bx_w, bx_h, facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8, zorder=3))
    ax.text(b1_x + bx_w/2, bx_y + bx_h/2, r"$t_1$", fontsize=7.4, color=C_TEXT, ha="center", va="center", zorder=4)

    # t1 -> t2 (Intact, I_adj = 1)
    ax.annotate(
        "", xy=(b1_x + bx_w + 0.22, bx_y + bx_h/2), xytext=(b1_x + bx_w, bx_y + bx_h/2),
        arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.8), zorder=3
    )
    ax.text(b1_x + bx_w + 0.11, bx_y + bx_h + 0.07, r"$I_{\mathrm{adj}}=1$", fontsize=6.2, color=C_GREEN, fontweight="bold", ha="center", zorder=4)

    # t2
    b2_x = b1_x + bx_w + 0.22
    ax.add_patch(patches.Rectangle((b2_x, bx_y), bx_w, bx_h, facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8, zorder=3))
    ax.text(b2_x + bx_w/2, bx_y + bx_h/2, r"$t_2$", fontsize=7.4, color=C_TEXT, ha="center", va="center", zorder=4)

    # t2 -> Gap (Dotted transition)
    ax.plot([b2_x + bx_w, b2_x + bx_w + 0.14], [bx_y + bx_h/2, bx_y + bx_h/2], color=C_RED, lw=1.6, linestyle=":", zorder=3)

    # Artifact Gap Block (Clipped > 500 uV, masked I_adj = 0)
    bgap_x = b2_x + bx_w + 0.14
    ax.add_patch(patches.Rectangle((bgap_x, bx_y), gap_w, bx_h, facecolor=C_RED_BG, edgecolor=C_RED_BORDER, lw=0.8, hatch="//", zorder=3))
    ax.text(bgap_x + gap_w/2, bx_y + bx_h/2 + 0.08, "Artifact Gap", fontsize=5.8, color="#991B1B", fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(bgap_x + gap_w/2, bx_y + bx_h/2 - 0.09, r"$(>500\,\mu\mathrm{V})$", fontsize=5.4, color="#991B1B", ha="center", va="center", zorder=4)
    ax.text(bgap_x + gap_w/2, bx_y + bx_h + 0.07, r"$I_{\mathrm{adj}}=0$", fontsize=6.2, color=C_RED, fontweight="bold", ha="center", zorder=4)

    # Gap -> t3 (Dotted transition)
    ax.plot([bgap_x + gap_w, bgap_x + gap_w + 0.14], [bx_y + bx_h/2, bx_y + bx_h/2], color=C_RED, lw=1.6, linestyle=":", zorder=3)

    # t3
    b3_x = bgap_x + gap_w + 0.14
    ax.add_patch(patches.Rectangle((b3_x, bx_y), bx_w, bx_h, facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8, zorder=3))
    ax.text(b3_x + bx_w/2, bx_y + bx_h/2, r"$t_3$", fontsize=7.4, color=C_TEXT, ha="center", va="center", zorder=4)

    # Mathematical definition box
    f_box1 = patches.FancyBboxPatch(
        (x1 + 0.20, 0.98), col_w - 0.40, 0.46,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=C_BOX_BG, edgecolor=C_BOX_BORDER, lw=0.7, zorder=3
    )
    ax.add_patch(f_box1)
    ax.text(
        cx1, 1.21,
        r"$I_{\mathrm{adj}}(t_i, t_{i+1}) = \mathbf{1}_{\{t_{i+1} - t_i = 1\}}$",
        fontsize=8.4, color=C_TEXT, ha="center", va="center", zorder=4
    )

    # Bottom tag: Zero cross-gap leakage
    leakage_box = patches.FancyBboxPatch(
        (x1 + 0.20, y_bottom_tag), col_w - 0.40, h_bottom_tag,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F1F5F9", edgecolor=C_BORDER, lw=0.6, zorder=3
    )
    ax.add_patch(leakage_box)
    ax.text(cx1, y_bottom_tag + h_bottom_tag/2, "Zero Cross-Gap Leakage", fontsize=7.2, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=4)

    # =========================================================================
    # STAGE 2: Dual Orthogonal Extraction
    # =========================================================================
    x2 = x_starts[1]
    cx2 = x2 + col_w / 2

    # Pathway A: Temporal Sequence Memory (Upper)
    ax.text(x2 + 0.20, y_subhead1, "Temporal Sequence Memory", fontsize=8.6, fontweight="bold", color=C_TEXT, zorder=3)

    f_box2a = patches.FancyBboxPatch(
        (x2 + 0.20, 3.26), col_w - 0.40, 0.86,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=C_MATH_BG, edgecolor=C_MATH_BORDER, lw=0.7, zorder=3
    )
    ax.add_patch(f_box2a)
    ax.text(cx2, 3.78, r"$\rho_1 = \mathrm{Corr}(G_{t_i}, G_{t_{i+1}} \mid I_{\mathrm{adj}}=1)$", fontsize=7.6, color=C_TEXT, ha="center", va="center", zorder=4)
    ax.text(cx2, 3.44, r"$D_1 = -\frac{1}{M}\sum |\Delta G| / (\bar{G} + \epsilon)$", fontsize=7.6, color=C_TEXT, ha="center", va="center", zorder=4)

    tag_2a = patches.FancyBboxPatch(
        (x2 + 0.20, 2.74), col_w - 0.40, 0.36,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F8FAFC", edgecolor=C_BORDER, lw=0.6, zorder=3
    )
    ax.add_patch(tag_2a)
    ax.text(cx2, 2.92, "Autoregressive Mesocircuit Continuity", fontsize=7.2, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=4)

    # Divider rule between Pathway A & Pathway B
    ax.plot([x2 + 0.15, x2 + col_w - 0.15], [y_divider, y_divider], color=C_DIVIDER, lw=0.8, zorder=2)

    # Pathway B: Spatial Decoupling via OLS (Lower)
    ax.text(x2 + 0.20, y_subhead2, "Spatial Decoupling via OLS", fontsize=8.6, fontweight="bold", color=C_TEXT, zorder=3)

    f_box2b = patches.FancyBboxPatch(
        (x2 + 0.20, 1.08), col_w - 0.40, 0.98,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=C_MATH_BG, edgecolor=C_MATH_BORDER, lw=0.7, zorder=3
    )
    ax.add_patch(f_box2b)
    ax.text(cx2, 1.68, r"$\mathbf{P}^\perp = \mathbf{I} - \mathbf{X}_c(\mathbf{X}_c^\top \mathbf{X}_c)^{-1}\mathbf{X}_c^\top$", fontsize=7.6, color=C_TEXT, ha="center", va="center", zorder=4)
    ax.text(cx2, 1.34, r"$r_{\mathrm{residual}} = \frac{1}{3}\sum_{(L,R)} \mathrm{Corr}(\boldsymbol{\epsilon}_L, \boldsymbol{\epsilon}_R)$", fontsize=7.4, color=C_TEXT, ha="center", va="center", zorder=4)

    tag_2b = patches.FancyBboxPatch(
        (x2 + 0.20, y_bottom_tag), col_w - 0.40, h_bottom_tag,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F8FAFC", edgecolor=C_BORDER, lw=0.6, zorder=3
    )
    ax.add_patch(tag_2b)
    ax.text(cx2, y_bottom_tag + h_bottom_tag/2, r"OLS Decoupling of Volume Conduction ($R^2=47\%$)", fontsize=6.8, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=4)

    # =========================================================================
    # STAGE 3: Fault-Tolerant Fusion
    # =========================================================================
    x3 = x_starts[2]
    cx3 = x3 + col_w / 2

    # Top: 4-Feature Orthogonal Suite
    ax.text(x3 + 0.20, y_subhead1, "4-Feature Orthogonal Suite", fontsize=8.6, fontweight="bold", color=C_TEXT, zorder=3)

    ax.text(
        cx3, 3.96,
        r"$\mathbf{x} = [c_{\mathrm{global}},\, -\mathrm{CV}_\theta,\, \rho_1,\, r_{\mathrm{residual}}]^\top$",
        fontsize=8.0, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=4
    )

    # 4 Domain Badges
    badge_w = 0.64
    badge_gap = 0.08
    badge_start = cx3 - 1.40
    domains = [
        ("Spectral", r"$c_{\mathrm{glob}}$"),
        ("Variability", r"$-\mathrm{CV}_\theta$"),
        ("Temporal", r"$\rho_1$"),
        ("Spatial", r"$r_{\mathrm{res}}$")
    ]
    for k, (dom_name, sym) in enumerate(domains):
        bx = badge_start + k * (badge_w + badge_gap)
        b_patch = patches.FancyBboxPatch(
            (bx, 3.16), badge_w, 0.52,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.6, zorder=3
        )
        ax.add_patch(b_patch)
        ax.text(bx + badge_w/2, 3.48, dom_name, fontsize=5.8, color="#1D4ED8", fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(bx + badge_w/2, 3.29, sym, fontsize=6.4, color=C_TEXT, ha="center", va="center", zorder=4)

    # Domain representation tag above divider
    suite_sub = patches.FancyBboxPatch(
        (x3 + 0.20, 2.74), col_w - 0.40, 0.36,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F8FAFC", edgecolor=C_BORDER, lw=0.6, zorder=3
    )
    ax.add_patch(suite_sub)
    ax.text(cx3, 2.92, "Domain-Decoupled Orthogonal Suite", fontsize=7.2, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=4)

    # Divider rule at y = 2.50 (perfectly aligned across all panels)
    ax.plot([x3 + 0.15, x3 + col_w - 0.15], [y_divider, y_divider], color=C_DIVIDER, lw=0.8, zorder=2)

    # Bottom: T-GECF Adaptive Volatility Wrapper
    ax.text(x3 + 0.20, y_subhead2, "T-GECF Volatility Wrapper", fontsize=8.6, fontweight="bold", color=C_TEXT, zorder=3)

    f_box3 = patches.FancyBboxPatch(
        (x3 + 0.20, 1.34), col_w - 0.40, 0.76,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=C_BOX_BG, edgecolor=C_BOX_BORDER, lw=0.7, zorder=3
    )
    ax.add_patch(f_box3)
    ax.text(cx3, 1.84, r"$\beta(z_v) = [1 + \exp(-\kappa(z_v - \theta_0))]^{-1}$", fontsize=7.6, color=C_TEXT, ha="center", va="center", zorder=4)
    ax.text(cx3, 1.50, r"$S_{\mathrm{T-GECF}} = \beta \tilde{S}_{\mathrm{rank}} + (1 - \beta) \tilde{S}_{\mathrm{linear}}$", fontsize=7.6, color=C_TEXT, ha="center", va="center", zorder=4)

    # Dual-state indicator pill (Harmonious Amber for High Volatility, NO discordant crimson)
    bar_y = 0.88
    pill_w = (col_w - 0.40) / 2
    # Left pill: beta -> 0 Linear
    ax.add_patch(patches.FancyBboxPatch(
        (x3 + 0.20, bar_y), pill_w - 0.03, 0.38,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=C_BLUE_BG, edgecolor=C_BLUE_BORDER, lw=0.7, zorder=3
    ))
    ax.text(x3 + 0.20 + (pill_w - 0.03)/2, bar_y + 0.24, r"Low Volatility ($\beta \to 0$)", fontsize=5.8, color="#1E40AF", fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(x3 + 0.20 + (pill_w - 0.03)/2, bar_y + 0.09, "Linear Consensus (0.9324)", fontsize=5.6, color=C_TEXT, ha="center", va="center", zorder=4)

    # Right pill: beta -> 1 Rank MDCO (Warm Amber)
    ax.add_patch(patches.FancyBboxPatch(
        (x3 + 0.20 + pill_w + 0.03, bar_y), pill_w - 0.03, 0.38,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor=C_AMBER_BG, edgecolor=C_AMBER_BORDER, lw=0.7, zorder=3
    ))
    ax.text(x3 + 0.20 + pill_w + 0.03 + (pill_w - 0.03)/2, bar_y + 0.24, r"High Volatility ($\beta \to 1$)", fontsize=5.8, color=C_AMBER, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(x3 + 0.20 + pill_w + 0.03 + (pill_w - 0.03)/2, bar_y + 0.09, "Rank Consensus (Robust)", fontsize=5.6, color=C_TEXT, ha="center", va="center", zorder=4)

    # Bottom tag (aligned with Stages 1, 2, and 4)
    prune_box = patches.FancyBboxPatch(
        (x3 + 0.20, y_bottom_tag), col_w - 0.40, h_bottom_tag,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F1F5F9", edgecolor=C_BORDER, lw=0.6, zorder=3
    )
    ax.add_patch(prune_box)
    ax.text(cx3, y_bottom_tag + h_bottom_tag/2, r"Unsupervised Collinearity Pruned ($|r| > 0.90$)", fontsize=6.8, color=C_MUTED, fontweight="bold", ha="center", va="center", zorder=4)

    # =========================================================================
    # STAGE 4: ICU Clinical Triage Funnel
    # =========================================================================
    x4 = x_starts[3]
    cx4 = x4 + col_w / 2

    # Top Section Label (Matching Stages 1, 2, and 3)
    ax.text(x4 + 0.20, y_subhead1, "Actionable Bedside Triage", fontsize=8.6, fontweight="bold", color=C_TEXT, zorder=3)

    # Central Decision Gate / Diamond centered at (cx4, y_divider)
    dcy = y_divider  # 2.50
    dw, dh = 2.04, 0.62
    diamond = patches.Polygon([
        (cx4, dcy + dh/2),
        (cx4 + dw/2, dcy),
        (cx4, dcy - dh/2),
        (cx4 - dw/2, dcy)
    ], facecolor="#FFFFFF", edgecolor=C_HEADER_BG, linewidth=1.2, zorder=4)
    ax.add_patch(diamond)
    ax.text(cx4, dcy + 0.09, "Triage Cutoff Gate", fontsize=7.4, fontweight="bold", color=C_TEXT, ha="center", va="center", zorder=5)
    ax.text(cx4, dcy - 0.09, "Sensitivity > 90%", fontsize=6.8, color=C_BLUE, fontweight="bold", ha="center", va="center", zorder=5)

    # Internal clean entry flow from Stage 4 left edge into Decision Gate
    ax.annotate(
        "", xy=(cx4 - dw/2, dcy), xytext=(x4 + 0.06, dcy),
        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.5), zorder=5
    )

    # Branch UP: S >= tau into Stratum A (Clean Emerald Card)
    card_a_y = 2.92
    card_a_h = 1.28
    ax.annotate(
        "", xy=(cx4, card_a_y), xytext=(cx4, dcy + dh/2),
        arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.6), zorder=4
    )
    ax.text(cx4 + 0.12, (dcy + dh/2 + card_a_y) / 2, r"$\mathbf{S \geq \tau}$", fontsize=7.4, color=C_GREEN, fontweight="bold", ha="left", va="center", zorder=5)

    # Stratum A: Preserved Reverberation (Upper Emerald Card)
    card_a = patches.FancyBboxPatch(
        (x4 + 0.20, card_a_y), col_w - 0.40, card_a_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=C_GREEN_BG, edgecolor=C_GREEN_BORDER, linewidth=0.9, zorder=3
    )
    ax.add_patch(card_a)
    ax.text(cx4, card_a_y + 0.95, "Preserved Reverberation (CMD Candidate)", fontsize=8.0, fontweight="bold", color="#15803D", ha="center", va="center", zorder=4)
    ax.text(cx4, card_a_y + 0.58, r"$\to$ Priority Confirmatory fMRI / $^{18}\mathrm{F}$-FDG PET", fontsize=7.4, color="#166534", fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(cx4, card_a_y + 0.24, r"Halves Secondary Transport Risks ($\approx 50\%$)", fontsize=6.8, color="#15803D", ha="center", va="center", zorder=4)

    # Branch DOWN: S < tau into Stratum B (Clean Slate Card)
    card_b_y = 0.88
    card_b_h = 1.28
    ax.annotate(
        "", xy=(cx4, card_b_y + card_b_h), xytext=(cx4, dcy - dh/2),
        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.6), zorder=4
    )
    ax.text(cx4 + 0.12, (dcy - dh/2 + card_b_y + card_b_h) / 2, r"$\mathbf{S < \tau}$", fontsize=7.4, color=C_MUTED, fontweight="bold", ha="left", va="center", zorder=5)

    # Stratum B: Depressed Dynamics (Lower Slate Card)
    card_b = patches.FancyBboxPatch(
        (x4 + 0.20, card_b_y), col_w - 0.40, card_b_h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor="#F8FAFC", edgecolor=C_BORDER, linewidth=0.9, zorder=3
    )
    ax.add_patch(card_b)
    ax.text(cx4, card_b_y + 0.95, "Depressed Dynamics", fontsize=8.0, fontweight="bold", color="#334155", ha="center", va="center", zorder=4)
    ax.text(cx4, card_b_y + 0.58, r"$\to$ Longitudinal Bedside EEG Monitoring", fontsize=7.4, color="#475569", fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(cx4, card_b_y + 0.24, "Safeguards Against Premature WLST", fontsize=6.8, color=C_MUTED, ha="center", va="center", zorder=4)

    # Bottom tag (matching Stages 1, 2, and 3)
    triage_tag = patches.FancyBboxPatch(
        (x4 + 0.20, y_bottom_tag), col_w - 0.40, h_bottom_tag,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F1F5F9", edgecolor=C_BORDER, lw=0.6, zorder=3
    )
    ax.add_patch(triage_tag)
    ax.text(cx4, y_bottom_tag + h_bottom_tag/2, "Prospective Screening & Safe Triage Protocol", fontsize=6.8, color=C_MUTED, fontweight="bold", ha="center", va="center", zorder=4)

    # =========================================================================
    # INTER-STAGE CONNECTORS (Pipeline at y = 2.50)
    # =========================================================================
    h_y = y_divider  # 2.50

    # 1 -> 2: Clean arrow & badge
    x1_out = x1 + col_w
    x2_in = x2
    mid_12 = (x1_out + x2_in) / 2
    ax.annotate(
        "", xy=(x2_in - 0.03, h_y), xytext=(x1_out + 0.03, h_y),
        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.5), zorder=5
    )
    b12 = patches.FancyBboxPatch(
        (mid_12 - 0.24, h_y + 0.12), 0.48, 0.36,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F1F5F9", edgecolor=C_BORDER, lw=0.6, zorder=6
    )
    ax.add_patch(b12)
    ax.text(mid_12, h_y + 0.30, "Clean Epochs\n& Masks", fontsize=6.0, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=7)

    # 2 -> 3: Clean arrow & badge
    x2_out = x2 + col_w
    x3_in = x3
    mid_23 = (x2_out + x3_in) / 2
    ax.annotate(
        "", xy=(x3_in - 0.03, h_y), xytext=(x2_out + 0.03, h_y),
        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.5), zorder=5
    )
    b23 = patches.FancyBboxPatch(
        (mid_23 - 0.24, h_y + 0.12), 0.48, 0.36,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#F1F5F9", edgecolor=C_BORDER, lw=0.6, zorder=6
    )
    ax.add_patch(b23)
    ax.text(mid_23, h_y + 0.30, "Orthogonal\nFeatures", fontsize=6.0, color=C_TEXT, fontweight="bold", ha="center", va="center", zorder=7)

    # 3 -> 4: Clean arrow connecting Stage 3 to Stage 4 container
    x3_out = x3 + col_w
    mid_34 = (x3_out + x4) / 2
    ax.annotate(
        "", xy=(x4 - 0.03, h_y), xytext=(x3_out + 0.03, h_y),
        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.5), zorder=5
    )
    b34 = patches.FancyBboxPatch(
        (mid_34 - 0.24, h_y + 0.12), 0.48, 0.36,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.6, zorder=6
    )
    ax.add_patch(b34)
    ax.text(mid_34, h_y + 0.30, "Fused Score\n$S_{\mathrm{T-GECF}}$", fontsize=6.0, color="#1D4ED8", fontweight="bold", ha="center", va="center", zorder=7)

    # Save outputs
    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, format="pdf", bbox_inches="tight", pad_inches=0.04)
        print(f"Saved: {p}")
        if p.suffix == ".pdf":
            png_p = p.with_suffix(".png")
            fig.savefig(png_p, format="png", dpi=250, bbox_inches="tight", pad_inches=0.04)
            print(f"Saved PNG preview: {png_p}")

    plt.close(fig)


if __name__ == "__main__":
    targets = [
        ROOT / "manuscript_temporal/figures/fig_s1_cdss_workflow.pdf",
        ROOT / "figures/fig_s1_cdss_workflow.pdf",
        ROOT.parent / "manuscript_temporal/figures/fig_s1_cdss_workflow.pdf",
    ]
    build_workflow_figure(targets)
