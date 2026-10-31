# A Bedside Decision Support System for Disorders of Consciousness: Multi-Scale Temporal Dynamics and Multicenter Validation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](LICENSE.md)
[![Package](https://img.shields.io/badge/Package-v1.0.6-blue.svg)](https://github.com/ujujuj3061831150/DoC_Temporal_Dynamics)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-37%2F37%20Passed-brightgreen)](tests/)
[![Paper Page Budget](https://img.shields.io/badge/Manuscript-9%20Pages%20IEEEtran-blue)](manuscript/manuscript.pdf)
[![Reproducibility](https://img.shields.io/badge/Artifact-Checks%20Available-blue)](verify_artifact.py)

Official open-science research artifact and replication codebase accompanying the manuscript:
> **A Bedside Decision Support System for Disorders of Consciousness: Multi-Scale Temporal Dynamics and Multicenter Validation**  
> *Hongjie Chen, Jianming Chen, Guanyu Xiong, Fang Li, and Fang Duan*, October 2026 (Package version: `v1.0.6`).

---

## 1. Executive Scientific Summary

Bedside electroencephalography (EEG) is the clinical cornerstone for evaluating residual awareness in disorders of consciousness (DoC). However, standard bedside markers predominantly rely on marginal spectral dispersion (such as theta coefficient of variation, $-\mathrm{CV}_\theta$) and unadjusted bilateral channel correlations. Because sample variance and Pearson correlations are symmetric functions of observation sets, they exhibit strict **permutation invariance**, mathematically discarding chronological sequence continuity.

Grounded in Nicholas Schiff's **anterior forebrain mesocircuit hypothesis**, which associates conscious recovery with sustained reciprocal thalamocortical reverberation, this codebase provides:
1. **Physical Order-Aware Dynamical Operators**:
   - Lag-1 autocorrelation persistence ($\rho_1 = \mathrm{Corr}(G_t, G_{t+1})$) respecting physical continuous epoch boundaries ($\mathbb{I}_{\text{adj}} \iff \Delta\text{index} == 1$).
   - Normalized adjacent jump rate ($D_1$).
   - Common-mode-decoupled residual bilateral resonance ($r_{\text{residual}}$) via ordinary least squares (OLS) orthogonal projection ($\mathbf{P}^\perp = \mathbf{I} - \mathbf{X}_c(\mathbf{X}_c^\top \mathbf{X}_c)^{-1}\mathbf{X}_c^\top$).
2. **Zero-Leakage Consensus & Adaptive Fusion**:
   - Linear standardized soft-truncated consensus (Cell C).
   - Empirical non-parametric rank fusion (Temporal-MDCO, Cell D).
   - Continuous volatility-gated hybrid consensus (**T-GECF**) dynamically balancing linear representation and rank stability.
3. **Causal Counterfactual Verification**:
   - 10,000 synchronous whole-channel time-scrambling surrogate permutations preserving instantaneous spatial covariance while destroying chronological sequence order under Frozen Intact Scales.
4. **Multicenter Independent Multi-Cohort Validation ($N=129$ total)**:
   - Prospective inpatient hospital cohort from China Rehabilitation Research Center (CRRC, $N=15$, NeuSen NSM2 / EEGCap32-pony 1000 Hz, AUROC 0.7778 after target-batch standardization and common direction alignment; fixed c=1.5 clipping yields 0.7963).
   - Cross-center transfer onto the 62-channel continuous Figshare DoC cohort ($N=50$, AUROC 0.7691).
   - Macro-dynamical sleep stage persistence across 24-hour PSG in the Wielek cohort ($N=23$, transition AUC 0.7348).

### Empirical Performance Summary (Mendeley Cohort, N=41)

| Model Configuration | Representation | Fusion Scheme | AUROC | Bal. Acc. | Sensitivity | Specificity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Cell A (Static Baseline)** | Permutation-Invariant ($-\mathrm{CV}_\theta$) | Linear Standardized | **0.8200** | 0.7201 | 0.8333 | 0.6069 |
| **Cell B (Static MDCO)** | Permutation-Invariant ($-\mathrm{CV}_\theta$) | Non-Parametric Rank | **0.7893** | 0.7270 | 0.8333 | 0.6207 |
| **Cell C (Temporal Linear)** | Order-Aware Dynamics ($\rho_1$) | Linear Standardized | **0.8911** | 0.7997 | 0.9167 | 0.6828 |
| **Cell D (Temporal MDCO)** | Order-Aware Dynamics ($\rho_1$) | Non-Parametric Rank | **0.8591** | 0.7187 | 0.8167 | 0.6207 |
| **Peak Model ($\rho_1 + r_{\text{residual}}$)** | Dynamics + Decoupled Cov | Soft-Truncated Linear ($c=1.5\sigma$) | **0.9149** | 0.8399 | 0.9833 | 0.6966 |
| **Adaptive Gated (T-GECF)** | Volatility-Gated Hybrid | Continuous Sigmoid | **0.9116** | 0.8399 | 0.9833 | 0.6966 |
| **Recommended Deployment (4-Feat)** | $[c_{\text{global}}, -\mathrm{CV}_\theta, \rho_1, r_{\text{residual}}]$ | Equal-Weight Linear | **0.9324** | 0.8101 | 0.9167 | 0.7034 |
| **10,000-Permutation Null** | Scrambled Surrogates | Null Baseline | **0.8182** | 0.7391 | 0.8175 | 0.6607 |

- $\Delta_{\text{rep}} = \text{Cell C} - \text{Cell A} = \mathbf{+0.0711}$ (95% CI $[+0.0011, +0.1534]$, $p = 0.0471$)
- $\Delta_{\text{fuse}} = \text{Cell D} - \text{Cell C} = \mathbf{-0.0320}$ (Rank tail compression penalty)
- Collinearity-Ablated Orthogonal Model: Eliminating static spectral slowing redundancy ($r=0.9813$) elevates AUROC to **0.9324** (PR-AUC = 0.8998, net gain $+0.1124$, 95% CI $[+0.0174, +0.2186]$, positive gain probability 99.16\%), recommended for clinical deployment.
- Permutation causal null collapse: $p < 0.0001$ (0 out of 10,000 surrogate iterations exceeded intact AUROC).

---

## 2. Directory Layout

```
DoC_Temporal_Dynamics_v1.0.0/
├── README.md                      # This comprehensive documentation
├── LICENSE.md                     # Dual MIT / CC-BY 4.0 license scope
├── CITATION.cff                   # Machine-readable citation metadata
├── zenodo.json                    # Zenodo deposition metadata
├── requirements.txt               # Pinned Python package dependencies
├── pytest.ini                     # Pytest configuration
├── verify_artifact.py             # 1-Click automated verification script
│
├── manuscript/                    # Submission-ready paper assets
│   ├── manuscript.pdf             # Verified 9-page IEEEtran double-column paper
│   ├── manuscript.tex             # LaTeX source file
│   ├── table1_results.tex         # Main results LaTeX table
│   ├── table2_multiscale_hyperparam.tex # Validation & sensitivity LaTeX table
│   ├── references.bib             # BibTeX reference library
│   └── figures/                   # Publication vector PDFs (Figs 1, 2, 3)
│
├── src/temporal_dynamics/         # Core scientific library
│   ├── __init__.py                # Module initialization
│   ├── operators.py               # Order-aware operators, OLS projection, gap protection
│   ├── consensus.py               # Zero-leakage estimators (Linear, MDCO, T-GECF, ML)
│   ├── surrogates.py              # Whole-channel synchronous permutation generator
│   ├── adapters.py                # Mendeley bedside & Wielek PSG cohort adapters
│   └── figshare_adapter.py        # Figshare independent pDoC external validation adapter
│
├── scripts/                       # Reproduction & execution entrypoints
│   ├── run_temporal_mdco_benchmark.py       # Full 25-fold CV & bootstrap benchmark
│   ├── run_10000_shuffles_and_export.py     # 10,000-permutation causal test
│   ├── run_figshare_external_validation.py  # Figshare cross-center zero-shot validation
│   ├── run_extended_benchmarks.py           # 100-fold stability & sensitivity benchmarks
│   └── generate_temporal_figures.py         # Generates Figs 1, 2, 3 PDFs
│
├── tests/                         # Full automated test suite
│   ├── test_temporal_dynamics.py  # Core 9 property tests (gaps, surrogates, leakage)
│   ├── test_figshare_adapter.py   # Figshare adapter tests (no spectral circularity)
│   └── test_extended_audit.py     # Extreme boundary & numerical stability stress tests
│
├── results/                       # Precomputed empirical results & hashes
│   ├── temporal_dynamics_v1/      # Primary benchmark tables & bootstrap intervals
│   ├── multiscale_matched_v2/     # Canonical benchmark feature records
│   └── target_profiles_v11/       # 41-patient pre-extracted continuous theta series
│
├── data/                          # Public benchmark datasets (Wielek sleep transitions)
├── figshare_23552964_meta.json    # Figshare cohort upload metadata
├── figshare_6ch_cache/            # Pre-extracted 6-channel arrays for Figshare validation
│
└── docs/                          # Audit reports and compliance logs
    ├── TEMPORAL_DYNAMICS_RESEARCH_PLAN.md
    ├── TEMPORAL_DYNAMICS_AUDIT_REPORT.md
    └── Author_Response_to_Reviewers.md
```

---

## 3. Installation & Quickstart

### 3.1 Environment Setup
Python 3.9+ (tested on Python 3.10 and 3.11) is required.

```bash
git clone https://github.com/ujujuj3061831150/DoC_Temporal_Dynamics.git
cd DoC_Temporal_Dynamics
pip install -r requirements.txt
```

### 3.2 One-Click Automated Verification
To check archived result hashes, run the tests, and recompute the selected cached-data metrics:

```bash
python verify_artifact.py
```

The verifier reports the hashes checked, the collected test count, PDF page counts, and selected recomputed metrics. This check does not rerun the full 10,000-surrogate experiment.

### 3.3 Optional Bedside Power-Block Replay

```bash
python scripts/run_bedside_playback.py --subject Nekrasova_P36 --speed 0
python scripts/run_bedside_playback.py --subject Nekrasova_P25 --speed 0
python scripts/build_bedside_playback_demo.py
python scripts/render_playback_animation.py
```

Open `docs/bedside_cdss_demo.html` for the standalone dashboard. It reads actual cached power/QC/score trajectories with archived source means, scales, signs and training-median thresholds. The source-training examples verify engineering consistency and are not held-out clinical validation. Event-level timing and numerical tolerances are documented in `results/bedside_playback_demo/BEDSIDE_PLAYBACK_VERIFICATION_REPORT.md`.

---

## 4. Reproducing Paper Results

### 4.1 Run 25-Fold Cross-Validation & Bootstrap Contrasts
```bash
python scripts/run_temporal_mdco_benchmark.py --output results/temporal_dynamics_v1
```
Computes the 2x2 factorial decomposition (Cells A--D), supervised ML baselines, 10,000-iteration Dirichlet Bayesian bootstrap confidence intervals, and 32-repeat surrogate tests.

### 4.2 Run 10,000 Synchronous Permutation Null Benchmark
```bash
python scripts/run_10000_shuffles_and_export.py
```
Executes the full 10,000 synchronous whole-channel time-scrambling surrogate permutations preserving instantaneous spatial covariance while destroying chronological sequence order under Frozen Intact Scales ($p < 0.0001$).

### 4.3 Matched-Input Model Comparisons (Table 1 and Supplementary Table S2)
```bash
python scripts/run_matched_model_comparison.py --output results/matched_model_comparison_v1
```
This comparison uses the same feature columns and coordinate transforms for equal-mean consensus, source-tuned Ridge/Logistic, and shrinkage-LDA. Discovery hyperparameters and preprocessing are fitted within the nested training folds. The fixed c=1.5 clipped mean is a separate scoring-rule variant.

- Mendeley: raw equal mean 0.9324 versus matched Logistic 0.8964; paired difference +0.0360 (95% Bayesian interval [0.0090, 0.0728]). Clipped mean: 0.9291.
- CRRC 15, common target alignment: raw mean 0.7778; clipped mean 0.7963; Ridge/Logistic/LDA 0.7407/0.7593/0.7593.
- Figshare 50, short-increment branch, frozen source: raw mean 0.7691 versus Ridge/Logistic/LDA 0.7759/0.7674/0.7742.

Primary classification thresholds are source-training medians. Unlabeled target-batch median operating points are saved separately as auxiliary results. Target standardization/direction alignment are batch adaptation protocols, distinct from fully frozen-source transfer. `summary.json`, subject-level score CSVs, fitted states, selection traces and `manifest.json` provide the comparison record. Earlier benchmark JSON files retain their original model configurations; the matched comparison is the source for the revised matched-model claims.

### Domain Baseline Operating Metrics
```bash
python scripts/recompute_domain_baseline_metrics.py
```
Recomputes LZC and spectral-entropy metrics from the included unchanged feature cache and the same 25 frozen subject partitions. Signs and median thresholds are fitted on outer-training subjects only. AUROC/AP use fold means; operating metrics pool the repeated outer-test decisions. Subject predictions, fitted thresholds, paired Dirichlet intervals and input hashes are in `results/domain_baseline_metrics_v1/`.

### 4.4 Run CRRC Hospital Inpatient Cohort Evaluation
```bash
python scripts/evaluate_crrc_cohort.py
```
Evaluates the prospective inpatient hospital cohort from China Rehabilitation Research Center ($N=15$, NeuSen NSM2 / EEGCap32-pony 1000 Hz) under Protocol 1 (Strict 6-channel clean subset, $N=13$, AUROC 0.7143) and Protocol 2 (Full cohort with target alignment and fixed $c=1.5$ clipping, $N=15$, AUROC 0.7963).

### 4.5 Run Figshare Independent Zero-Shot External Validation
```bash
python scripts/run_figshare_external_validation.py
```
Applies models trained exclusively on the Mendeley 6-channel cohort onto the independent Figshare 62-channel cohort without fine-tuning, strictly evaluating on training-set median thresholds.

### 4.6 Re-Generate Manuscript Vector Figures
```bash
python scripts/generate_temporal_figures.py --input results/temporal_dynamics_v1 --output manuscript/figures
```
Recreates `fig1_mesocircuit_memory.pdf`, `fig2_factorial_ablations.pdf`, and `fig3_permutation_collapse.pdf`.

---

## 5. Ethical Statement & Data Availability

The release combines three public datasets with de-identified CRRC features. Cohort-specific access and ethics information is listed below:
1. **Mendeley Bedside Cohort ($N=41$)**: Open-access under CC-BY 4.0; Federal Research and Clinical Center of Intensive Care Medicine and Rehabilitation, Moscow ethics approval. DOI: [10.17632/6wx4n25h4v.1](https://doi.org/10.17632/6wx4n25h4v.1).
2. **Figshare Continuous Cohort ($N=50$)**: Open-access under CC-BY 4.0; institutional ethics approval from Hangzhou Mingzhou Brain Rehabilitation Hospital (institutional ethics approval cited in Ling et al., 2024; Xu et al., 2021). DOI: [10.6084/m9.figshare.23552964.v1](https://doi.org/10.6084/m9.figshare.23552964.v1).
3. **Wielek Polysomnography Cohort ($N=23$)**: Open-access; Salzburg Ethics Commission (No. 415-E/1517/14-2013). DOI: [10.6084/m9.figshare.5669641.v1](https://doi.org/10.6084/m9.figshare.5669641.v1).
4. **CRRC Inpatient Hospital Cohort ($N=15$)**: Medical Ethics Committee of China Rehabilitation Research Center (Approval No. 2025-118-01; Duan et al., 2026). Fully de-identified tabular feature matrices are open-source in `results/crrc_validation/crrc_deidentified_features.csv`. Raw clinical telemetry is restricted by institutional patient confidentiality regulations.

No identifiable protected health information (PHI) is present in this release.

---

## 6. License & Citation

- **Code**: Licensed under the [MIT License](LICENSE.md).
- **Documentation & Aggregate Results**: Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE.md).

If you use this codebase or methodology in your academic work, please cite:
```bibtex
@article{chen2026bedside,
  title={A Bedside Decision Support System for Disorders of Consciousness: Multi-Scale Temporal Dynamics and Multicenter Validation},
  author={Chen, Hongjie and Chen, Jianming and Xiong, Guanyu and Li, Fang and Duan, Fang},
  journal={IEEE Journal of Biomedical and Health Informatics (Under Review)},
  year={2026},
  url={https://github.com/ujujuj3061831150/DoC_Temporal_Dynamics}
}
```
