"""Surgically eliminate all MDCO references from LaTeX manuscript, tables, and scripts."""
from pathlib import Path

def replace_in_file(path_str, replacements):
    p = Path(path_str)
    if not p.exists():
        print(f"Skipping non-existent: {path_str}")
        return
    text = p.read_text(encoding="utf-8")
    orig = text
    for old, new in replacements:
        if old not in text:
            print(f"WARNING: '{old}' not found in {path_str}")
        else:
            text = text.replace(old, new)
            print(f"Replaced in {path_str}: '{old}' -> '{new}'")
    if text != orig:
        p.write_text(text, encoding="utf-8")
        print(f"Successfully updated: {path_str}")
    else:
        print(f"No changes made to {path_str}")

tex_replacements = [
    (r"\textbf{Multiscale Dynamic Consensus Operator (MDCO)}:", r"\textbf{Non-Parametric Rank Consensus ($S_{\text{rank}}$)}:"),
    (r"Adapted from multi-criteria decision frameworks~\cite{grabisch2009aggregation}, MDCO maps signed features", r"Adapted from multi-criteria decision frameworks~\cite{grabisch2009aggregation}, rank consensus maps signed features"),
    (r"S_{\text{MDCO}}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^K \tilde{F}_k(s_k x_k),", r"S_{\text{rank}}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^K \tilde{F}_k(s_k x_k),"),
    (r"S_{\text{T-GECF}}(\mathbf{x}) = \beta(\mathbf{x}) \tilde{S}_{\text{MDCO}}(\mathbf{x}) + (1 - \beta(\mathbf{x})) \tilde{S}_{\text{linear}}(\mathbf{x}),", r"S_{\text{T-GECF}}(\mathbf{x}) = \beta(\mathbf{x}) \tilde{S}_{\text{rank}}(\mathbf{x}) + (1 - \beta(\mathbf{x})) \tilde{S}_{\text{linear}}(\mathbf{x}),"),
    (r"where $\tilde{S}_{\text{MDCO}}$ and $\tilde{S}_{\text{linear}}$ are standardized", r"where $\tilde{S}_{\text{rank}}$ and $\tilde{S}_{\text{linear}}$ are standardized"),
    (r"bounded rank consensus $\tilde{S}_{\text{MDCO}}(\mathbf{x})$", r"bounded rank consensus $\tilde{S}_{\text{rank}}(\mathbf{x})$"),
    (r"Cell B (Static Baseline, MDCO)", r"Cell B (Static Baseline, Rank)"),
    (r"Cell D (Plus Lag-1 $\rho_1$, MDCO)", r"Cell D (Plus Lag-1 $\rho_1$, Rank)"),
    (r"In MDCO, this suite elevated", r"In rank consensus, this suite elevated"),
    (r"Paired Bayesian bootstrap intervals for MDCO,", r"Paired Bayesian bootstrap intervals for rank consensus,"),
    (r"and raised MDCO to \textbf{0.9076}", r"and raised rank consensus to \textbf{0.9076}"),
    (r"and 0.9069 in MDCO.", r"and 0.9069 in rank consensus."),
    (r"Plus Lag-1 (MDCO) fell from 0.8591", r"Plus Lag-1 (Rank) fell from 0.8591"),
    (r"shifted weight to bounded rank MDCO", r"shifted weight to bounded rank consensus"),
    (r"while non-parametric rank mapping (MDCO) guarantees", r"while non-parametric rank mapping ($S_{\text{rank}}$) guarantees"),
    (r"shifting to bounded rank MDCO", r"shifting to bounded rank consensus"),
]

for p in ["v3/manuscript_temporal/manuscript.tex", "manuscript_temporal/manuscript.tex"]:
    replace_in_file(p, tex_replacements)

t1_replacements = [
    (r"\quad Canonical Static (4 Feats, MDCO)", r"\quad Canonical Static (4 Feats, Rank Consensus)"),
    (r"\quad Plus Lag-1 $\rho_1$ (5 Feats, MDCO)", r"\quad Plus Lag-1 $\rho_1$ (5 Feats, Rank Consensus)"),
    (r"\quad Plus $\rho_1 + r_{\text{res}}$ (5 Feats, MDCO)", r"\quad Plus $\rho_1 + r_{\text{res}}$ (5 Feats, Rank Consensus)"),
    (r"\quad Plus Jump $D_1$ (5 Feats, MDCO)", r"\quad Plus Jump $D_1$ (5 Feats, Rank Consensus)"),
    (r"\quad Orthogonal 4-Feat (Drop $c_{\text{spatial}}$, MDCO)", r"\quad Orthogonal 4-Feat (Drop $c_{\text{spatial}}$, Rank Consensus)"),
]

for p in ["v3/manuscript_temporal/table1_results.tex", "manuscript_temporal/table1_results.tex"]:
    replace_in_file(p, t1_replacements)

t2_replacements = [
    (r"\textbf{Cohort / Duration} & \textbf{Linear} & \textbf{MDCO} \\", r"\textbf{Cohort / Duration} & \textbf{Linear} & \textbf{Rank Consensus} \\"),
]

for p in ["v3/manuscript_temporal/table2_multiscale_hyperparam.tex", "manuscript_temporal/table2_multiscale_hyperparam.tex"]:
    replace_in_file(p, t2_replacements)

fig_replacements = [
    (r'"Plus $\rho_1$\n(MDCO)"', r'"Plus $\rho_1$\n(Rank)"'),
    (r'cells = ["Static\nLin.", "Static\nMDCO", r"Temp.\nLin.", r"Temp.\nMDCO"]', r'cells = ["Static\nLin.", "Static\nRank", r"Temp.\nLin.", r"Temp.\nRank"]'),
]

for p in ["v3/scripts/generate_temporal_figures.py", "scripts/generate_temporal_figures.py"]:
    replace_in_file(p, fig_replacements)

workflow_replacements = [
    (r"C_AMBER = '#B45309'          # Amber 700 for high volatility / Rank MDCO", r"C_AMBER = '#B45309'          # Amber 700 for high volatility / Rank Consensus"),
    (r'C_AMBER = "#B45309"          # Amber 700 for high volatility / Rank MDCO', r'C_AMBER = "#B45309"          # Amber 700 for high volatility / Rank Consensus'),
    (r"\tilde{S}_{\mathrm{MDCO}}", r"\tilde{S}_{\mathrm{rank}}"),
    (r"Rank MDCO (Robust)", r"Rank Consensus (Robust)"),
]

for p in ["v3/scripts/generate_workflow_figure.py", "scripts/generate_workflow_figure.py"]:
    replace_in_file(p, workflow_replacements)
