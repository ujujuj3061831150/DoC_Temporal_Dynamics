# Cover Letter — IEEE JBHI Submission

**Date**: 2026-10-28
**To**: Editor-in-Chief, IEEE Journal of Biomedical and Health Informatics
**From**: Fang Duan (Corresponding Author), College of Information Science and Engineering, Huaqiao University, Xiamen 361021, China (nkfetsh@gmail.com)
**Re**: Submission of regular research article — "A Bedside Decision Support System for Disorders of Consciousness: Multi-Scale Temporal Dynamics and Multicenter Validation"

---

Dear Editor-in-Chief,

We are pleased to submit our manuscript for consideration as a regular research article in the IEEE Journal of Biomedical and Health Informatics.

**Summary of contributions.** Misdiagnosis rates near 40% and 15–35% covert-awareness prevalence make bedside differentiation of minimally conscious state (MCS) from vegetative state/unresponsive wakefulness syndrome (VS/UWS) a critical unmet need. Conventional bedside EEG markers are mathematically permutation-invariant — arbitrary temporal shuffling leaves them unchanged — discarding the multi-second sequence memory that may relate to the anterior forebrain mesocircuit hypothesis. Our work delivers:

1. **Order-aware physical operators** combining lag-1 persistence ρ₁, normalized jump rate D₁, and common-mode-decoupled residual bilateral correlation r_residual, with temporal operators computed strictly on physically adjacent epochs.
2. **A 4-feature orthogonal deployment suite attaining 0.9324 AUROC** (PR-AUC 0.8998) on the N=41 discovery cohort, with a direct paired Dirichlet bootstrap margin over the strongest supervised baseline (+0.0360 (95% interval [+0.0090, +0.0728])).
3. **Multicenter evaluation across four cohorts (N=129 records)**, including independent CRRC inpatient telemetry (N=15, 1000 Hz), where shared unlabeled target standardization and direction alignment yield 0.7778 AUROC, increasing to 0.7963 with fixed truncation. Matched supervised baselines yield 0.7407–0.7593. The exploratory short-record Figshare D₁ branch achieves 0.7691 AUROC (N=50), comparable to matched baselines; file-level diagnoses and intersegment continuity require independent confirmation, while Wielek sleep macro-dynamics (N=23) provide complementary evidence at a different timescale.
4. **Sequence-dependence verification**: 10,000 synchronous permutation surrogates reduce the lag-1 extension from 0.8911 to 0.8182 ± 0.0049 AUROC (p < 0.0001). An additional EEGNet-style baseline yields 0.5702 AUROC under the recorded protocol.
5. **Lightweight scoring** illustrated by cached-power replay in two source-training records; raw EEG end-to-end latency and clinical utility remain to be evaluated.
6. **An auditable public artifact** with source code, de-identified features, subject-level predictions, fitted states, and SHA-256 manifests (available at https://github.com/ujujuj3061831150/DoC_Temporal_Dynamics). The artifact includes automated tests and checks selected cached-data metrics. AI assistance with code implementation and manuscript preparation is disclosed in the manuscript Acknowledgment and replication repository documentation.

**Fit to JBHI.** The manuscript connects biomedical signal processing, interpretable small-sample learning, and clinical EEG assessment through a lightweight framework evaluated across recording regimes and centers.

**Related published work.** The CRRC cohort is described in Duan et al., *Frontiers in Neurology* (2026), doi:10.3389/fneur.2026.1927313. That study examined auditory-paradigm-dependent EEG features. The present manuscript evaluates order-aware EEG descriptors and consensus scoring across multiple cohorts, including transfer to CRRC.

**Ethics and data.** All cohorts were acquired under institutional ethical approvals (CRRC Ethics No. 2025-118-01; Salzburg Commission No. 415-E/1517/14-2013, with written surrogate consent); public repositories are cited with DOIs.

**Declarations.** This manuscript is original, is not under consideration elsewhere, and all authors have approved the submission. The authors declare no competing interests.

Thank you for your consideration.

Sincerely,

Fang Duan (on behalf of all authors)
Hongjie Chen, Jianming Chen, Guanyu Xiong, Fang Li, and Fang Duan
