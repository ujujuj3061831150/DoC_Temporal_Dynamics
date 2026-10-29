# Longitudinal Follow-up Availability & Linkage Audit: Standalone Draft

**Document**: Standalone Methods, Results, and Discussion Draft for Longitudinal Audit & Prognostic Verification  
**Target Manuscript**: IEEE Journal of Biomedical and Health Informatics (JBHI) Revision / Extension  
**Protocol Registration**: `DOC-PROGNOSTIC-v1.0.0` (SHA-256: `e0d608e05f23bdabd7a1f45610adefad10785b03898f1b8daa126e9c7140fb6a`)  
**Note**: This document is stored separately in `manuscript_temporal/` and does **not** modify or overwrite the frozen 9-page main text.

---

## 1. Methods Section (Draft)

### 2.X Longitudinal Follow-up Availability, Eligibility Criteria, and Linkage Verification
To examine whether baseline resting-state EEG temporal dynamics contain prognostic information regarding future functional recovery beyond concurrent behavioral assessment, we systematically investigated public DoC cohorts with documented longitudinal outcomes.

1. **Cohort Identification**:
   - **Figshare Cohort (Ling et al., 2024)**: Evaluated 50 prolonged DoC patients (19 MCS, 31 VS/UWS) admitted to Hangzhou Mingzhou Brain Rehabilitation Hospital with 6-channel resting-state EEG (250 Hz, 120 s). The study protocol specified a structured 6-month telephone follow-up evaluating the Glasgow Outcome Scale (GOS, 1–5), categorizing favorable recovery as GOS $\ge 4$ (moderate disability or good recovery; $n = 17$, comprising 7 MCS and 10 VS) and unfavorable recovery as GOS $< 4$ (severe disability, vegetative state, or death; $n = 33$, comprising 12 MCS and 21 VS).
   - **Wielek Cohort (Wielek et al., 2018)**: Evaluated 23 DoC patients (11 MCS, 12 UWS) with documented long-term clinical diagnoses over variable follow-up horizons (1 to 150 months; 13 documented outcomes, 10 missing).
   - **Mendeley (Nekrasova et al., 2019) & CRRC (Beijing 2024) Cohorts**: Evaluated as purely cross-sectional baseline cohorts with admission CRS-R; neither collected post-discharge longitudinal recovery outcomes.

2. **Pre-Registered Linkage Verification Protocol**:
   To prevent confirmation bias and maintain data integrity (pre-registered protocol `DOC-PROGNOSTIC-v1.0.0`), patient inclusion in prognostic modeling required:
   - (i) Availability of artifact-screened continuous multichannel EEG ($\ge 120\text{ s}$, valid retention $\ge 50\%$);
   - (ii) Verified 1-to-1 linkage between raw EEG recording files and individual longitudinal outcome records;
   - (iii) Absolute prohibition against inferring outcome labels from EEG features, spectra, or model scores.
   A strict **fail-closed gate** was embedded: only confirmed, unambiguous pairings were permitted into statistical inference.

3. **Frozen Model Specification**:
   Prognostic evaluations applied the frozen source-domain parameters derived from the 4-feature orthogonal suite $\mathbf{x} = [c_{\text{global}}, -\mathrm{CV}_\theta, \rho_1, r_{\text{residual}}]^T$:
   $$S(\mathbf{x}) = \frac{1}{4} \sum_{k=1}^4 s_k \frac{x_k - \mu_k}{\sigma_k}$$
   with orientation signs $\mathbf{s} = [+1, +1, +1, -1]^T$, frozen means $\boldsymbol{\mu} = [0.451, -1.662, 0.335, 0.334]^T$, scales $\boldsymbol{\sigma} = [0.228, 1.565, 0.211, 0.177]^T$, and fixed population operating threshold $\theta_0 = +0.0451$.

4. **Statistical Analysis Framework**:
   Pre-registered statistical hypotheses comprised:
   - Two-sided Mann-Whitney $U$ test evaluating score distribution differences between favorable and unfavorable recovery groups;
   - Univariate logistic regression reporting Odds Ratios (OR) per 1-SD increase in $S(\mathbf{x})$ with 95% Wald confidence intervals;
   - Multivariable logistic regression adjusting for baseline CRS-R total scores (for sample sizes $N \ge 30$), testing incremental prognostic value via Likelihood Ratio Test (LRT) $\Delta \chi^2$ and change in AUC ($\Delta\text{AUC}$);
   - Diagnostic sensitivity, specificity, PPV, and NPV evaluated strictly at the pre-specified fixed threshold $\theta_0 = +0.0451$ without ROC post-hoc tuning.

---

## 2. Results Section (Draft)

### 3.X Longitudinal Linkage Audit & Structural Availability Status
A comprehensive forensic audit of all 129 patients across the 4 cohorts revealed distinct structural boundaries between cross-sectional availability and longitudinal traceability:

1. **Figshare 23552964 Linkage Audit**:
   While the 50 raw BrainVision resting-state EEG recordings are publicly accessible on Figshare, the deposited files are cataloged under anonymized pinyin initials (`dgx`, `hj`, `cgr`, `pjq`, etc.). Although the published manuscript reports exact cohort-level 6-month GOS recovery numbers (17 favorable vs 33 unfavorable; Table S5), **neither the repository metadata nor the supplementary documents provide a de-identified subject-level lookup table linking individual EEG file names to their respective GOS scores**. In accordance with our pre-registered protocol, these 50 patients are classified as *Missing Information Pending Author Linkage*.

2. **Wielek 2018 Linkage & Modality Audit**:
   Cross-referencing the published S1 demographic table (listing patient numbers `Nb 1` to `Nb 23`) against the local dataset (`VP01` to `VP23`) revealed that 19 of 23 patients collide on identical `(Diagnosis, CRS-R)` pairs (e.g., five UWS patients share a CRS-R of 4, with disparate outcomes ranging from death at 1 month to MCS emergence at 14 months). Only 4 patients possess unique CRS-R values, of which 2 have missing outcomes. Furthermore, the local Wielek repository provides post-processed 30-second sleep stage predictions rather than raw continuous EEG waveforms, precluding derivation of the 4-feature orthogonal suite. Consequently, Wielek yielded 0 confirmed continuous EEG prognostic pairs.

3. **Enforcement of the Fail-Closed Verification Gate**:
   Across the 129 screened patients, confirmed public 1-to-1 longitudinal pairings totaled 0 (50 unlinked in Figshare, 4 ambiguous and 19 collided in Wielek, 41 cross-sectional in Mendeley, 15 cross-sectional in CRRC).
   Pursuant to the pre-registered protocol, the analysis engine strictly enforced the fail-closed gate: **no synthetic or guessed labels were evaluated, and no unverified clinical conclusions were drawn**.

4. **Statistical Execution Readiness Verification**:
   The statistical testing engine was calibrated on a pre-registered simulation cohort matching Figshare's exact marginals (17 favorable, 33 unfavorable), confirming numerical stability for the Mann-Whitney test, Hessian-based logistic confidence intervals, and LRT incremental value statistics ($p_{\text{LRT}} = 3.97 \times 10^{-4}$). The complete data ingestion interface is deployed and pre-tested to execute instantaneously upon receipt of author-verified linkage tables.

---

## 3. Discussion Section (Draft)

### 4.X Longitudinal Prognostic Potential vs Cross-Sectional Diagnostic Triage
A critical conceptual boundary in neurorehabilitation is the distinction between concurrent behavioral diagnosis and long-term recovery potential:

1. **Physiological Foundations of Recovery Potential**:
   The four orthogonal temporal dynamic features captured in $S(\mathbf{x})$ reflect underlying neurophysiological integrity:
   - Global spatial coordination ($c_{\text{global}}$) indexes large-scale corticocortical and thalamocortical functional connectivity;
   - Regularity of theta power ($-\mathrm{CV}_\theta$) reflects the stabilization of intrinsic pacemaker activity against diffuse metabolic exhaustion;
   - Directional persistence ($\rho_1$) captures long-range temporal autocorrelation in cortical power fluctuations;
   - Residual interhemispheric coordination ($r_{\text{residual}}$) reflects callosal and bilateral subcortical synchrony independent of global volume conduction.
   Preservation of these dynamical properties in a behaviorally unresponsive patient may indicate that the underlying neural substrate possesses the physiological scaffolding necessary for neuroplastic recovery.

2. **Rigorous Differentiation from Cognitive Motor Dissociation (CMD)**:
   It is vital to state clearly that **a high resting-state EEG temporal score does not equate to a diagnosis of Cognitive Motor Dissociation (CMD)**. As established in recent multicenter studies (e.g., Bodien et al., *New England Journal of Medicine* 2024), CMD formally requires evidence of volitional brain activation in response to active command-following paradigms (such as motor imagery fMRI or EEG during mental tennis or hand-opening tasks). A high resting-state temporal dynamic score reflects preserved intrinsic network organization, not active task performance. In clinical decision support, such scores serve as candidate prioritization flags for escalating patients to active cognitive paradigms, functional MRI, or targeted neuromodulation, rather than definitive evidence of covert consciousness.

3. **Open Science Imperative for Neurotrauma Repositories**:
   Our linkage audit highlights an urgent structural need in computational neurotrauma research. While data sharing mandates increasingly lead to the deposition of raw electrophysiological recordings, clinical follow-up metadata are frequently severed or reported solely as cohort-level summary statistics. This creates an unbridgeable disconnect where machine learning models can be trained cross-sectionally, but cannot be independently audited for longitudinal clinical translation. We have initiated formal, non-identifiable data linkage requests to corresponding authors, establishing an auditable framework for transparent validation.
