# Document Text & Page Navigation Map

> [!IMPORTANT]
> Text navigation index. For proofs and figures, inspect page images in `page_image_dir`.

## [Page 1] (`page-1.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 1
A Bedside Decision Support System for Disorders of
Consciousness: Multi-Scale Temporal Dynamics and
Multicenter Validation
Hongjie Chen1
, Jianming Chen2
, Guanyu Xiong2
, Fang Li2
, and Fang Duan1,∗
Abstract—Bedside evaluation of residual consciousness faces
operational barriers: high-density arrays are clinically prohibitive,
spectral metrics discard sequence continuity via permutation
invariance, and parameterized models suffer cross-site overfitting.
Grounded in the mesocircuit hypothesis, this study introduces an
interpretable bedside clinical decision support system (CDSS)
for disorders of consciousness (DoC). The framework pairs
closed-form lag-1 autoregressive persistence (ρ1) for continuous
telemetry with normalized jump rate (D1) for rapid triage,
coupled with an orthogonal projection (rresidual) decoupling volume
conduction. Evaluated under strict zero-leakage protocols across
four cohorts (N = 129 records), the 4-feature orthogonal suite
attains 0.9324 AUROC (PR-AUC = 0.8998) on long-term discovery
telemetry (N = 41). Importantly, unsupervised target-site feature
standardization paired with a physiologically oriented direction
vector evaluated on an independent inpatient hospital cohort
from China Rehabilitation Research Center (CRRC, N = 15,
1000 Hz telemetry) yields 0.7778 AUROC (+0.1296 net gain over
static baseline 0.6481; 0.7143 on clean 6-channel records), while
parameterized models collapse (0.5000–0.5370 on the full cohort;
0.4286–0.4762 on clean 6-channel records). On ultra-short 120-s
records (Figshare, N = 50), D1 achieves 0.7691 AUROC (77.59%
target-centered balanced accuracy), corroborated by whole-night
sleep transition persistence (Wielek, N = 23, p = 0.0392).
Operating with a 0.42 ± 0.08 ms single-core CPU latency and
zero GPU footprint, this system functions as a high-sensitivity
triage funnel (> 90% sensitivity), prioritizing covert awareness
candidates for multimodal imaging while projected to reduce
secondary transport burdens by approximately half.
Index Terms—Disorders of consciousness (DoC), electroen￾cephalography (EEG), temporal dynamics, clinical decision
support systems (CDSS), mesocircuit hypothesis, consensus fusion.
I. INTRODUCTION
S
EVERE acute traumatic brain injury, anoxic-ischemic
encephalopathy, and cerebrovascular accidents leave thou￾sands of patients in protracted disorders of consciousness
(DoC) worldwide, categorized clinically into the vegetative
state/unresponsive wakefulness syndrome (VS/UWS) and the
minimally conscious state (MCS) [1], [2]. Discriminating
MCS from VS/UWS is a clinical imperative: diagnostic
This work was supported in part by the National Natural Science Foundation
of China and institutional research funds.
1Hongjie Chen and Fang Duan are with the College of Information
Science and Engineering, Huaqiao University, Xiamen 361021, China (e-mail:
chenhj976@gmail.com; nkfetsh@gmail.com).
2
Jianming Chen, Guanyu Xiong, and Fang Li are with the China Rehabili￾tation Research Center, Beijing Bo’ai Hospital, Capital Medical University,
Beijing, China.
∗Corresponding author: Fang Duan (e-mail: nkfetsh@gmail.com).
boundaries guide critical decisions regarding the withdrawal
of life-sustaining treatment (WLST), analgesic protocols, and
rehabilitation potential. Bedside consensus evaluations carry
an established ∼ 40% misdiagnosis rate; although the Coma
Recovery Scale-Revised (CRS-R) reduced diagnostic error [3],
behavioral exams remain vulnerable to motor impairment and
fluctuating arousal [4]. Furthermore, prospective studies show
15%–35% of behaviorally unresponsive patients exhibit Cogni￾tive Motor Dissociation (CMD) [4], [5], retaining volitional
cortical drive despite absent motor output.
Advanced neuroimaging and perturbational protocols (fMRI,
18F-FDG PET, TMS-EEG) [6], [7] confirm that preserved
cortical complexity differentiates MCS from VS/UWS, but
transport hazards preclude routine bedside deployment. Sparse
continuous EEG (4–8 electrodes) remains the practical modality
for longitudinal monitoring [8]–[10]. Beyond static spectral
summaries, prior studies explored non-linear complexity [8],
[11], microstate dynamics [12], sleep cycling [13], [14], long￾range temporal correlations via detrended fluctuation analysis
(DFA) [15], [16], source-space states [17], and criticality [18].
Yet translating these paradigms to routine bedside care faces
severe barriers: DFA multi-scale regressions require extended
dense montages, inverse reconstruction is clinically prohibitive
in the ICU, non-linear entropy markers suffer artifact variance
inflation, and deep networks overfit small cohorts. Bedside
assessment has consequently relied on empirical spectral
summaries, marginal dispersion (−CVθ), and unadjusted
correlations [10], [13], bounding discrimination to a moderate
plateau (AUROC 0.75–0.82).
A. The Permutation Invariance Defect of Conventional Metrics
This persistent performance plateau reflects a fundamental
mathematical constraint: strict permutation invariance. Standard
bedside pipelines characterize cortical activity primarily through
summary dispersion, notably power variation (−CVθ), and in￾stantaneous bilateral correlations (rinterhem) across multi-second
epochs. Because sample variance and Pearson correlations are
invariant under permutations of the temporal index, arbitrary
chronological shuffling preserves their values identically up
to numerical precision (≈ 10−15). These formulations treat
continuous electrophysiological trajectories as unordered ob￾servation ensembles, mathematically discarding multi-second
sequence continuity and directional temporal reverberation.

---

## [Page 2] (`page-2.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 2
B. The Mesocircuit Hypothesis and Autoregressive Memory
This methodological blind spot diverges from established
neurobiological frameworks of conscious recovery, specifically
the anterior forebrain mesocircuit model [19]. In this framework,
sustained behavioral awareness depends on recurrent reciprocal
excitation linking the central thalamus, striatum, and fron￾toparietal association cortices. We hypothesize that preserved
mesocircuit integrity manifests electrophysiologically as multi￾second autoregressive continuity, captured by lag-1 autocorre￾lation (ρ1) across consecutive epochs. Conversely, structural
deafferentation and loss of striatal inhibition disinhibiting the
internal globus pallidus (GPi) hyper-inhibit the central thalamus
in VS/UWS, degrading cortical dynamics into memoryless
fluctuations. Differentiating conscious capacity from reflexive
unresponsiveness therefore requires operators designed to
measure sequential persistence rather than permutation-invariant
marginal dispersion (Fig. 1).
C. Spurious Bilateral Coupling Driven by Diffuse Slow Waves
A second confounding factor in bedside DoC recordings
is the contamination of bilateral connectivity by diffuse
slow-wave volume conduction. In patients with extensive
cortical injury, high-amplitude slow oscillations synchronously
drive distant sensors via volume conduction and subcortical
pacemaking. This shared background driver inflates unadjusted
inter-hemispheric correlations, obscuring bilateral dynamical
differentiation. We hypothesize that projecting out this global
common mode via orthogonal nuisance regression will unmask
regional inter-hemispheric dynamics, distinguishing pathologi￾cal hyper-synchrony in VS/UWS from modular, functionally
differentiated cortical activity in MCS.
D. Contributions of This Work
To translate temporal dynamics into an interpretable bedside
clinical decision support system (CDSS), this study establishes
a two-tier framework for ICU triage: Tier 1 (Dual Physical￾Orthogonal Operators): (1) formulates multi-scale operators
pairing physical lag-1 persistence (ρ1) for continuous telemetry
with normalized jump rate (D1) for rapid triage across verified
neighbor blocks (Iadj = 1), analytically linking dispersion to
temporal decay via D1 ≈ 2π
−1/2
(−CVθ)(1 − ρ1)
1/2
; and
(2) implements an OLS orthogonal projection decoupling
volume-conducted slow waves to isolate residual bilateral
differentiation (rresidual). Tier 2 (Fault-Tolerant Bedside
CDSS): (3) establishes the 4-feature orthogonal deployment
suite [cglobal, −CVθ, ρ1, rresidual] via unsupervised collinearity
pruning (0.9324 AUROC, 0.8998 AP); (4) delivers artifact￾resilient consensus fusion (T-GECF) dynamically shifting to
bounded rank consensus under extreme sensor spikes (+0.1114
margin under 20σ); (5) demonstrates multicenter validation
across four cohorts (N = 129), featuring an independent
hospital cohort from China Rehabilitation Research Center
(CRRC, N = 15, 0.7778 AUROC, +0.1296 gain under target￾site calibration); and (6) achieves closed-form O(N) streaming
throughput (< 100 KB runtime RAM, 0.42 ± 0.08 ms latency
per patient on single-core CPU, zero GPU footprint) for real￾time edge ICU triage.
II. MATHEMATICAL FORMULATION AND METHODS
A. Clinical Cohorts and Ethical Compliance
Four multicenter cohorts totaling N = 129 records were
evaluated under institutional ethical approvals:
1) Mendeley Primary Discovery Cohort (N = 41):
Continuous 6-channel bedside EEG (F3, F4, C3, C4, O1, O2;
256 Hz) from 41 brain-injured patients: 12 MCS (8 MCS+,
4 MCS-; 6.00 ± 0.00 h) and 29 VS/UWS (5.97 ± 0.61 h) at
FRCCICMR Moscow [20], diagnosed via standardized CRS-R
assessments.
2) CRRC Inpatient Hospital Cohort (N = 15): Prospective
bedside telemetry from 15 patients (6 MCS, 9 VS/UWS; 11
ICH, 4 TBI; mean 19.5±8.0 min; Ethics No. 2025-118-01 [21])
at China Rehabilitation Research Center (CRRC / Beijing Bo’ai
Hospital) using a NeuSen NSM2 system (1000 Hz, 4-second
blocks [4,000 samples]; homologous 6 channels evaluated)
with consensus CRS-R confirmation.
3) Figshare Continuous Cohort (N = 50): High-density
resting EEG (250 Hz, 120-s) from 50 patients with prolonged
DoC (19 MCS, 31 VS/UWS) at Hangzhou Mingzhou Brain
Rehabilitation Hospital [12], [22], benchmarking zero-shot
transfer across homologous channels.
4) Wielek Circadian PSG Cohort (N = 23): 24-hour
polysomnography from 23 post-comatose patients (11 MCS,
12 VS/UWS) [13], providing 30-second hypnograms (Wake,
N1, N2, N3, REM) to assess transition persistence across sleep
cycles.
B. EEG Acquisition, Preprocessing, and Physical Adjacency
Constraint
Continuous 6-channel EEG (256 Hz) was re-referenced to
Common Average Reference (CAR) and segmented into non￾overlapping 4-second epochs (1,024 samples). Spectral densities
were computed across each 4-second epoch via Hann-tapered
periodograms (∆f = 0.25 Hz), extracting theta (4–8 Hz) and
alpha (8–13 Hz) band power.
Primary screening at 500 µV retained 5,185.5 ± 595.6 clean
blocks per patient (96.98%±3.2%, 5.76±0.66 h; 2.84%±2.1%
rejected in MCS vs. 3.09% ± 3.6% in VS/UWS, p = 0.82,
Mann-Whitney U test). Tightening rejection to a conservative
150 µV threshold retained 90.14% ± 17.87% clean blocks
(4,672.9 ± 1,024.7 adjacent pairs; minimum 668), confirming
robust signal retention without myogenic bias.
Let T = {t1, t2, . . . , tN } denote clean epoch indices within
a continuous recording session, and let Gti
denote spatial
mean theta power across all 6 channels at epoch ti
. To
prevent spurious concatenation across artifact gaps, the physical
adjacency indicator is defined as:
Iadj(ti
, ti+1) = {︄
1, if ti+1 − ti = 1,
0, otherwise,
(1)
where 0 denotes an intervening artifact gap, and Iadj = 1
compactly denotes adjacent epoch pairs. Sequential metrics
operate strictly over pairs with Iadj = 1, retaining M =
5,133.3 ± 687.7 adjacent pairs per patient (minimum 1,678,
exceeding calculation threshold M ≥ 10).

---

## [Page 3] (`page-3.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 3
0 20 40 60 80 100 120
Elapsed Time (Seconds)
0
5
10
15
20
25
6-C
h
a
n
n
el M
e
a
n
θ
P
o
w
er (μ
V
2
/Hz)
(a) Bedside Mesocircuit Dynamics (120 s)
MCS (P36)
VS/UWS (P25)
MCS
(n=12)
VS/UWS
(n=29)
0.0
0.2
0.4
0.6
0.8
1.0
L
a
g-1 P
ersiste
nce
ρ1
p = 0.0227
(b) Order-Aware Memory ρ1
MCS
(n=12)
VS/UWS
(n=29)
−5
−4
−3
−2
−1
0
N
e
g
ativ
e
Disp
ersio
n
−
C
Vθ
p = 0.0264
(c) Static Dispersion −CVθ
Fig. 1. Thalamocortical mesocircuit dynamics in bedside EEG. (a) 4-second theta trajectories in MCS (P36) and VS/UWS (P25). (b) Cohort distributions
(N = 41) of order-aware persistence ρ1 (p = 0.0227). (c) Static dispersion −CVθ (p = 0.0264) showing individual overlap.
C. Canonical Static Features and Order-Aware Dynamics
Four baseline features define xcanonical = [cglobal, cspatial,
−CVθ, rinterhem]
⊤ ∈ R
4
. Theta power (4–8 Hz) serves as
primary carrier because thalamocortical mesocircuit projections
drive frontoparietal dynamics within this band [19], whereas
delta reflects non-specific slowing and alpha is largely abolished
in VS/UWS. Let P
(k)
θ,t and P
(k)
α,t denote theta and alpha powers
of channel k ∈ {1, . . . , 6} at clean epoch t ∈ {1, . . . , N}.
1) Global Spectral Contrast (cglobal): Quantifies the ratio of
slow to fast power across all channels: cglobal = (P¯
θ−P¯
α)/(P¯
θ+
P¯
α + ϵ), where P¯
θ =
1
6N
∑︁
k,t P
(k)
θ,t , P¯
α =
1
6N
∑︁
k,t P
(k)
α,t , and
ϵ = 10−12
.
2) Spatial Contrast (cspatial): Computes mean slow-to￾fast ratio: cspatial =
1
6
∑︁6
k=1(P¯
(k)
θ − P¯
(k)
α )/(P¯
(k)
θ + P¯
(k)
α + ϵ).
Pre-specified unsupervised collinearity pruning (|r| > 0.90
evaluated strictly on outer training folds without label access)
eliminates redundant spectral channels to form the parsimonious
4-feature deployment suite (Section III-C).
3) Negative Theta Dispersion (−CVθ): Measures temporal
variance of mean theta Gt =
1
6
∑︁6
k=1 P
(k)
θ,t : −CVθ =
−
√︂
1
N−1
∑︁N
t=1(Gt − G¯)
2/(G¯ + ϵ), with temporal mean G¯ =
1
N
∑︁N
t=1 Gt.
4) Inter-Hemispheric Bilateral Correlation (rinterhem): Av￾erages raw pairwise Pearson correlations across symmetric bilat￾eral pairs: rinterhem =
1
3
∑︁
(L,R) Corr(yL, yR), where yL, yR ∈
R
N denote the 4-second block theta-power vectors across
symmetric pairs (L, R) ∈ {(F3, F4),(C3, C4),(O1, O2)}.
To capture multi-second sequential continuity, two order￾aware temporal representations are formulated on physically
adjacent blocks:
Lag-1 Memory Persistence (ρ1): Quantifies autoregressive
temporal persistence across consecutive 4-second blocks:
ρ1 = Corr({Gti
| Iadj = 1}, {Gti+1 | Iadj = 1}). (2)
High ρ1 values reflect smooth baseline preservation, whereas
low values signify rapid temporal decorrelation.
Normalized Adjacent Jump Rate (D1): Measures resis￾tance to high-frequency baseline discontinuities:
D1 = −
1
M
∑︂
i:Iadj=1
|Gti+1 − Gti
|/(G¯ + ϵ), (3)
where G¯ =
1
N
∑︁N
t=1 Gt, M =
∑︁Iadj, and ϵ = 10−12. Under
stationary Gaussian increments ∆Gt ∼ N (0, 2σ
2
(1 − ρ1)),
expected jump magnitude satisfies E[|∆Gt|] = 2π
−1/2σ(1 −
ρ1)
1/2
, establishing D1 ≈ 2π
−1/2
(−CVθ)(1−ρ1)
1/2
. Because
electrophysiological power spectra naturally follow canonical
log-normal multiplicative dynamics, evaluating the continuous
jump rate D1 on log-transformed power increments analytically
accounts for 93.91% of empirical variance (r = 0.9691, p =
2.59×10−25), rigorously establishing the physical equivalence
between adjacent jump rate and autoregressive continuity.
Under estimator variance Var(ˆρ1) ∼ O(1/M), D1 serves as a
low-variance proxy for short records below M∗ ≈ 75 epochs,
whereas ρ1 achieves asymptotic efficiency.
To test macro-scale persistence on the Wielek PSG co￾hort (N = 23), stage stay persistence on 30-s hypno￾grams {st}
L
t=1 ∈ {Wake, N1, N2, N3, REM} is Pstay =
1
L−1
∑︁L−1
t=1 I(st+1 = st), with transition flexibility Ttrans =
1 − Pstay.
This establishes a unified, two-tier diagnostic architecture.
The four-dimensional feature space is spanned by four comple￾mentary physical domains: spectral slowing contrast, marginal
envelope dispersion, sequential autoregressive continuity, and
residual bilateral differentiation. The parsimonious 4-feature
suite [cglobal, −CVθ, ρ1, rresidual] serves as the primary clinical
deployment system, while orthogonal factorials and channel
ablations systematically verify the independent contribution of
each physical operator.
D. Residual Bilateral Resonance (rresidual)
For each symmetric bilateral pair (L, R) with block theta￾power vectors yL, yR ∈ R
N , the nuisance signal across
the remaining 4 channels is zt =
1
4
∑︁
k /∈{L,R} P
(k)
θ,t . With
standardized z˜ ∈ R
N and design matrix Xc = [1, z˜] ∈ R
N×2
,
the OLS projection operator P⊥ = I − XcX+
c
(where X+
c
denotes the Moore–Penrose pseudoinverse, robust to collinear
or degenerate regressors) computes residuals ϵL = P⊥yL
and ϵR = P⊥yR. The residual bilateral resonance averages
pairwise correlations:
rresidual =
1
3
∑︂
(L,R)
Corr(ϵL, ϵR). (4)
Whereas CAR eliminates linear instantaneous voltage offsets,
theta-band power constitutes a phase-free envelope whose
slow-wave modulation retains extensive shared variance across
electrodes. The shared 4-channel power envelope accounts for
R2 = 47.27%±18.43% of raw bilateral variance; the full-rank
OLS projection P⊥ decouples this background regressor in
the power domain to isolate residual bilateral differentiation.
E. Consensus Operators and Machine Learning Baselines
Linear Standardized Aggregation (Linear): Standardizes
features within training partitions and applies directional signs:
Slinear(x) = 1
K
∑︁K
k=1 skz˜k, where z˜k = (xk −µ
tr
k
)/σtr
k
denotes

---

## [Page 4] (`page-4.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 4
TABLE I
PERFORMANCE OF ORDER-AWARE TEMPORAL DYNAMICS, CONSENSUS FUSION, AND MACHINE LEARNING BASELINES ON MENDELEY COHORT (25
OUTER PARTITIONS; 95% CIS PAIRED VS. LINEAR BASELINE VIA 10,000 DIRICHLET RESAMPLES; PR-AUC REPORTED AS AVERAGE PRECISION [AP],
WITH TRAPEZOIDAL PR-AUC = 0.8782 FOR 4-FEAT LINEAR; 5-FEAT COMBINED MODELS SUBSTITUTE rINTERHEM WITH rRES ).
Model / Feature Configuration AUROC PR-AUC Balanced Acc. Sensitivity Specificity 95% CI vs. Baseline
Consensus Aggregation Frameworks
Canonical Static (4 Feats, Linear) 0.8200 0.7823 0.7201 0.8333 0.6069 Baseline
Canonical Static (4 Feats, Rank Consensus) 0.7893 0.7299 0.7270 0.8333 0.6207 [−0.0619, −0.0076]
Plus Lag-1 ρ1 (5 Feats, Linear) 0.8911 0.8384 0.7997 0.9167 0.6828 [+0.0011, +0.1534]
Plus Lag-1 ρ1 (5 Feats, Rank Consensus) 0.8591 0.8068 0.7187 0.8167 0.6207 [−0.0188, +0.1053]
Plus ρ1 + rres (5 Feats, Linear, c = 1.5σ) 0.9149 0.8782 0.8399 0.9833 0.6966 [+0.0064, +0.1931]
Plus ρ1 + rres (5 Feats, Rank Consensus) 0.8949 0.8353 0.7741 0.9000 0.6483 [+0.0049, +0.1530]
T-GECF Adaptive Gated Hybrid (5 Feats) 0.9116 0.8669 0.8399 0.9833 0.6966 [+0.0013, +0.1927]
Supervised & Domain Baseline Benchmarks
Lempel-Ziv Complexity (LZC) 0.5136 0.5061 0.4973 0.5133 0.4813 [−0.4229, −0.1861]
Shannon Spectral Entropy (SE) 0.6953 0.6182 0.6967 0.7933 0.6000 [−0.2508, +0.0073]
Canonical Static (Tuned Ridge) 0.7653 0.7193 0.6661 0.7667 0.5655 [−0.0998, −0.0140]
Plus Lag-1 ρ1 (Tuned Ridge) 0.8300 0.7776 0.7193 0.7833 0.6552 [−0.0743, +0.0980]
Canonical Static (Tuned Logistic) 0.7722 0.7236 0.6917 0.7833 0.6000 [−0.0959, −0.0064]
Plus Lag-1 ρ1 (Tuned Logistic) 0.8087 0.7530 0.7227 0.7833 0.6621 [−0.0925, +0.0717]
Canonical Static (Shrinkage-LDA) 0.7709 0.7252 0.6626 0.7667 0.5586 [−0.0884, −0.0162]
Plus Lag-1 ρ1 (Shrinkage-LDA) 0.8162 0.7725 0.7124 0.7833 0.6414 [−0.0870, +0.0829]
Representation Ablations & Operator Variants
Plus Jump D1 (5 Feats, Linear) 0.8611 0.8012 0.7491 0.8500 0.6483 [−0.0018, +0.0911]
Plus Jump D1 (5 Feats, Rank Consensus) 0.8424 0.7787 0.7132 0.8333 0.5931 [−0.0208, +0.0694]
Replace −CVθ with ρ1 (Linear) 0.8851 0.8495 0.7678 0.8667 0.6690 [−0.0223, +0.1613]
Replace −CVθ with D1 (Linear) 0.8720 0.8097 0.7776 0.9000 0.6552 [+0.0050, +0.1094]
Residual Correlation rres (Linear) 0.8433 0.7671 0.7575 0.8667 0.6483 [−0.0561, +0.1070]
Orthogonal 4-Feat (Drop cspatial, Linear) 0.9324 0.8998 0.8101 0.9167 0.7034 [+0.0174, +0.2186]
Orthogonal 4-Feat (Drop cspatial, Rank Consensus) 0.9076 0.8548 0.8448 1.0000 0.6897 [+0.0137, +0.1725]
unconstrained standard scaling (c = ∞, corresponding to the
0.8200 and 0.8911 benchmarks in Table I), and sk ∈ {−1, +1}
denotes sign correlation with consciousness labels. To bound
moderate tail divergence on clean recordings, a Winsorized soft￾truncated variant clips standardized coordinates at c = 1.5σ
(z˜k = clip(˜zk, −c, c)).
Non-Parametric Rank Consensus (Srank): Adapted from
multi-criteria decision frameworks [23], rank consensus maps
signed features to empirical cumulative distribution function
(CDF) midranks:
Srank(x) = 1
K
∑︂
K
k=1
F˜
k(skxk), (5)
where F˜
k(u) = 1
Ntr
∑︁Ntr
i=1[I(skx
tr
i,k < u) + 1
2
I(skx
tr
i,k = u)].
Curvature is fixed to p = 1.0 (mean on ranks), ensuring
constant marginal substitution without uncalibrated parameters
on small cohorts.
Adaptive Fluctuation-Gated Hybrid (T-GECF): As an
artifact-resilient safety wrapper against sensor corruption,
T-GECF bridges high linear resolution and rank stability
via sample-level volatility gating. For signed standardized
features x˜k = sk(xk − µ
tr
k
)/(σ
tr
k + ϵ), sample volatility
is v(x) = √︂
1
K
∑︁K
k=1(˜xk − x¯˜)
2, standardized to zv(x) =
(v(x) − µ
tr
v
)/(σ
tr
v + ϵ) against training folds. A continuous
Sigmoid gate β(x) = 1/(1 + exp(−κ(zv(x) − θ0))) (θ0 =
1.645, κ = 2.0) mixes standardized rank and linear scores:
ST-GECF(x) = β(x)S˜
rank(x) + (1 − β(x))S˜
linear(x), (6)
where S˜
rank and S˜
linear are standardized to zero mean and unit
variance using outer fold statistics prior to mixing. Elevated
volatility (zv(x) > 1.645) dynamically shunts decision weight
toward bounded rank consensus S˜
rank(x), wrapping the system
against extreme sensor spike corruptions.
Supervised Machine Learning and Domain Baselines: To
benchmark against established complexity markers (Lempel￾Ziv Complexity [LZC], Shannon Spectral Entropy [SE]) and
parameterized models, three architectures are evaluated via
inner 3-fold cross-validation: Tuned Ridge, Tuned Logistic
Regression, and Ledoit-Wolf Shrinkage LDA [24]. A compact
EEGNet-style CNN [25] (3,074 weights) on the identical
per-subject theta block stream is benchmarked under the
same frozen partitions with inner subject-level early stopping.
Parameter-free consensus operators leverage Dawes improper
linear models [26], eliminating continuous fusion weight
estimation variance.
F. Subject-Level Validation Protocol and Decision Thresholds
To prevent data leakage: (1) evaluations employ 5 random
seeds × 5 folds = 25 outer subject-level partitions on the
41-patient cohort (12 MCS, 29 VS/UWS); (2) all transfor￾mations (means, standard deviations, signs, empirical ranks,
and hyperparameter tuning) are fitted strictly on outer training
folds; (3) diagnostic operational thresholds are pre-fixed to
outer training medians to prevent post-hoc bias (cost-sensitive
tuning yields sensitivity 0.9167, specificity 0.8243, balanced
accuracy 0.8705); and (4) paired confidence intervals are
estimated via 10,000 Dirichlet Bayesian bootstrap draws [27]
fixed at the subject level across all folds without refitting.
The primary combined model (ρ1 + rresidual) was pre-specified
from mechanistic hypotheses, while ablations evaluate stability.
In Leave-One-Out validation, held-out predictions are pooled
to compute cohort-level AUROC. To evaluate independent
predictability, multivariable logistic regression was fitted on
cohort-wide standardized features [cglobal, R2
, ρ1].
III. EXPERIMENTAL RESULTS
A. The 2 × 2 Orthogonal Factorial Decomposition
Table I reports classification metrics across model and
feature configurations. The 2 × 2 factorial decomposition

---

## [Page 5] (`page-5.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 5
TABLE II
MULTISCALE SENSITIVITY AND EXTERNAL COHORT BENCHMARKS
(A) Multiscale Window & Cohort Transfer (AUROC)
Cohort / Duration Linear Rank Consensus
Mendeley 2-second 0.8916 0.8529
Mendeley 4-second (Default) 0.8911 0.8591
Mendeley 8-second 0.8711 0.8453
Mendeley 30-min Trunc. 0.8867 0.8480
Figshare Zero-Shot (D1, N = 50) 0.7691 0.6927
Figshare 25-Fold CV (D1, N = 50) 0.7771 0.7505
(B) Longitudinal & Multi-Tier Benchmarks
Validation Metric Value
Mendeley 5-h ICC (ρ1 / rresidual) 0.4857 / 0.4511
Wielek Transition AUC (directional p = 0.030) 0.7348
Wielek Mean State Stay (MCS vs. VS) 0.7393 vs. 0.8361
Figshare Zero-Shot Balanced Acc. 0.5645 (Sens 1.000)
Figshare Target-Centered Balanced Acc. 0.7759 (Sens 0.842)
62-Channel HD Mode-1 R
2
46.52%
TABLE III
HYPERPARAMETER STABILITY: SOFT TRUNCATION AND T-GECF GRID
(A) Soft Truncation Threshold c
Threshold AUROC PR-AUC Bal.
c = 1.0σ 0.8958 0.8543 0.8080
c = 1.5σ (Default) 0.9149 0.8782 0.8399
c = 2.0σ 0.9056 0.8662 0.8046
c = 2.5σ 0.8996 0.8622 0.8129
c = 3.0σ 0.8996 0.8622 0.8129
c = ∞ (Raw) 0.8996 0.8622 0.8129
(B) T-GECF Gating Grid (θ0 × κ)
Sigmoid Gate Slope κ
θ0 (Quantile) 0.5 1.0 2.0 5.0
1.282 (90%) 0.9220 0.9182 0.9187 0.9149
1.645 (95%) 0.9160 0.9182 0.9116 0.9089
1.960 (97.5%) 0.9160 0.9149 0.9056 0.9056
2.326 (99%) 0.9127 0.9116 0.9022 0.8996
Ablate Cont. (c = ∞) 0.8996 0.8996 0.8996 0.8996
(Fig. 2a) yields four anchor quadrants: Cell A (Static Baseline,
Linear) achieves AUROC = 0.8200; Cell B (Static Base￾line, Rank) yields 0.7893; Cell C (Plus Lag-1 ρ1, Linear)
attains 0.8911, the highest discrimination among the four
factorial configurations, whereas Cell D (Plus Lag-1 ρ1, Rank)
achieves 0.8591. The temporal representation net gain is
∆rep = AUC(Cell C) − AUC(Cell A) = +0.0711 (95%
CI [+0.0011, +0.1534], probability of positive gain 97.74%).
Linear and rank consensus address complementary objectives
(∆fuse = AUC(Cell D)−AUC(Cell C) = −0.0320): linear ag￾gregation maximizes pristine resolution, while rank consensus
guarantees bounded stability.
B. Diagnostic Operating Characteristics and Imbalanced
Precision-Recall
At the pre-specified training median threshold, incorporating
ρ1 elevated Balanced Accuracy from 0.7201 to 0.7997 and
Sensitivity from 0.8333 to 0.9167 (Table I). Combining lag-1
persistence with residual bilateral resonance (xcombined) yielded
peak performance: under soft truncation (c = 1.5σ), linear
aggregation achieved 0.9149 AUROC (PR-AUC = 0.8782,
Balanced Accuracy = 0.8399, Sensitivity = 0.9833). In rank
consensus, this suite elevated AUROC to 0.8949, bringing
non-parametric rank consensus to near-perfect parity with un￾constrained linear aggregation (within 0.0047 of 0.8996). Paired
Bayesian bootstrap intervals for rank consensus, soft truncation,
and T-GECF remain strictly positive ([+0.0049, +0.1530],
[+0.0064, +0.1931], and [+0.0013, +0.1927] vs. Canonical
Static Baseline 0.8200), demonstrating that combining temporal
continuity with spatial de-confounding brings non-parametric
rank consensus to parity with linear models.
Stratifying the 12 MCS patients by behavioral subdiagnosis
revealed consistent temporal persistence across subtypes: MCS+
patients (N = 8, characterized by command following or
intelligible verbalization) showed ρ1 = 0.4563 ± 0.2687 and
were correctly identified in 8 of 8 cases (100% sensitivity),
while MCS- patients (N = 4, demonstrating visual pursuit
or localization only) exhibited ρ1 = 0.4377 ± 0.1670 with
3 of 4 cases correctly classified (75% sensitivity; overall
MCS sensitivity 11/12 = 91.67%). Furthermore, under class
imbalance (12 MCS vs. 29 VS/UWS, prevalence baseline
PR-AUC = 0.2927), incorporating ρ1 advanced PR-AUC from
0.7823 (Cell A) to 0.8384 (Cell C), with direct substitution
(Replace CV by ρ1) reaching 0.8495 and the combined model
reaching 0.8782, confirming robustness against class skew.
C. Feature Ablation and Collinearity Redundancy Analysis
Directly replacing −CVθ with order-aware ρ1 within the
4-feature suite yielded 0.8851 AUROC, showing that sequence
memory supersedes marginal variance. Across 41 patients,
MCS exhibited higher lag-1 persistence (0.450 ± 0.250 vs.
0.287 ± 0.181, p = 0.0227, AUROC = 0.7299) and attenuated
normalized adjacent jump rates (D1: −0.420 ± 0.089 vs.
−0.558 ± 0.168, p = 0.0112, AUROC = 0.7557). Because
electrophysiological power spectra naturally follow canonical
log-normal multiplicative dynamics, evaluating the continuous
jump rate D1 on log-transformed power increments analytically
accounts for 93.91% of empirical variance (r = 0.9691, p =
2.59×10−25), rigorously establishing the physical equivalence
between adjacent jump rate and autoregressive continuity.
Finite-sample estimator variance Var(ˆρ1) ∼ O(1/M) favors
D1 below the empirical crossover horizon M∗ ≈ 75 epochs
(5 min), where adjacent jump rates offer lower finite-sample
variance before asymptotic autoregressive efficiency dominates.
Central sensorimotor channels (C3–C4) overlying thalamocor￾tical pathways exhibited highest univariate discrimination (ρ1:
0.3996 ± 0.1863 in MCS vs. 0.2611 ± 0.1631 in VS/UWS,
p = 0.0132, AUROC = 0.7500), outperforming frontal (0.6264)
and occipital (0.6810) pairs.
Regressing out the global common mode (rresidual) im￾proved AUROC from 0.8200 to 0.8433, accounting for
R2 = 47.27% ± 18.43% of raw bilateral variance uniformly
across groups (46.45% in MCS vs. 47.61% in VS/UWS;
p = 0.7418). Whereas raw bilateral correlation was non￾significant (0.563 ± 0.168 in MCS vs. 0.661 ± 0.146 in
VS/UWS, p = 0.0605, AUROC = 0.6897), residualization
unmasked residual bilateral differentiation beyond the shared
background power envelope (0.233 ± 0.080 vs. 0.376 ± 0.193,
p = 0.0211, AUROC = 0.7328), reflecting modularity in MCS
versus hyper-synchrony in VS/UWS. Adjusting for spectral
slowing (cglobal) and background coupling (R2
), multivariable
logistic regression confirms that ρ1 associates with conscious￾ness (z = 2.460, p = 0.0139, adjusted OR = 4.260, 95% CI
[1.343, 13.512]).
Because cglobal and cspatial display high empirical collinearity
(r = 0.9813, p = 1.58×10−29), the pre-specified unsupervised
pruning criterion (|r| > 0.90) flagged this channel redundancy
on outer training folds. In the redundancy ablation (Table I,
Fig. 2b), removing cspatial to form the 4-feature deployment
suite [cglobal, −CVθ, ρ1, rresidual] elevated linear AUROC to
0.9324 (PR-AUC = 0.8998 [Average Precision; 0.8782 trape￾zoidal integration]; Balanced Accuracy = 0.8101, 95% CI

---

## [Page 6] (`page-6.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 6
Lin. Rank Lin. Rank
0.7
0.8
0.9
AUROC
Static +Order
0.820
0.789
0.891
0.859
(a) 2 × 2 Factorial
Static
Repl ρ1
+ρ1
+D1
rres
5-Feat
4-Feat
T-GECF
0.8
0.9
0.820
0.891
0.932
(b) Representation Ablations
Fig. 2. (a) 2 × 2 factorial decomposition separating representation gain
(∆rep = +0.0711) from consensus mapping (∆fuse = −0.0320). (b)
Performance across representation ablations (dashed line: canonical baseline
0.8200), highlighting the combined model (0.9149), 4-feature orthogonal model
(0.9324), and adaptive gated hybrid (T-GECF, 0.9116).
[+0.0174, +0.2186] vs. baseline, positive gain probability
99.16%) and raised rank consensus to 0.9076 (95% CI
[+0.0137, +0.1725], sensitivity 1.0000). Similarly, dropping
cglobal reached 0.9289 AUROC in linear aggregation and
0.9082 in rank consensus. As a parsimonious low-complexity
deployment variant, this 4-feature configuration attains 0.9324
AUROC; retaining both spectral variants in the primary 5-
feature suite (0.9149) establishes a conservative lower bound
with dual-domain artifact insurance.
D. Temporal-Order Dependency via 10,000 Synchronous Sur￾rogates
To verify that performance gains originate strictly from
temporal sequence order rather than incidental distributional
properties, 10,000 synchronous whole-montage permutation
surrogates were executed across all 41 patients, adapting non￾linear dynamical testing methodology [28]. Freezing intact
training scales (σ
tr, intact
k
) prevents inverse noise amplification
on attenuated shuffled variances. Synchronously destroying
chronological order precipitated a decisive collapse in macro
AUROC across all temporal configurations (Fig. 3): Plus Lag￾1 (Linear) dropped from 0.8911 to 0.8182 ± 0.0049 (range:
[0.7902, 0.8369], empirical p < 0.0001, zero of 10,000 repeats
matching intact performance); Plus Lag-1 (Rank) fell from
0.8591 to 0.7732 ± 0.0416 (range: [0.6620, 0.9098], empirical
p = 0.0178); and Plus Jump (D1) degraded from 0.8611 to
0.7783 ± 0.0040 (p < 0.0001).
This frozen scaling bounds all 10,000 shuffled linear itera￾tions strictly below 0.8370. Across M ≈ 5,133 adjacent blocks,
the shuffled increment D1 converges to E[|X − Y |] (variance
≈ 2 × 10−4
), collapsing to a deterministic marginal quantity
below static baseline levels (0.8200). Across configurations,
destroying sequence order removed the temporal increment
under this ablation, collapsing scores to marginal baseline levels
(0.8182 ± 0.0049).
Plus ρ1
(Linear)
Plus ρ1
(Rank / MDCO)
Plus D1
(Linear)
0.6
0.7
0.8
0.9
Macro-AUROC (25 Folds)
10,000 Synchronous Surrogates vs. Intact Order
Static Base (0.820)
Intact Order
Fig. 3. Performance under 10,000 synchronous time shuffles vs. intact ordering.
Scrambling sequence order collapses linear discrimination to 0.8182±0.0049
(p < 0.0001, frozen intact scale), falling below the static baseline (dashed
line).
E. Comparison with Supervised Machine Learning and Domain
Baselines
To determine whether temporal ordering gains persist under
supervised parameter estimation and compare against estab￾lished complexity markers, Table I benchmarks consensus
aggregation against Lempel-Ziv Complexity (LZC), Shan￾non Spectral Entropy (SE), Tuned Ridge, Tuned Logistic
Regression, and Ledoit-Wolf Shrinkage-LDA [24]. Both non￾linear domain baselines degraded sharply on sparse bedside
recordings: LZC yielded 0.5136 AUROC (near chance, 95% CI
[−0.4229, −0.1861] vs. baseline), while SE achieved 0.6953
(95% CI [−0.2508, +0.0073]). Without dense spatial arrays
to average localized noise, sparse montages expose LZC
symbolization to baseline drift, while unwindowed spectral
entropy lacks spatial smoothing against ICU artifacts. In
contrast, incorporating ρ1 consistently elevated discriminative
capacity across all supervised models: Ridge from 0.7653 to
0.8300 (+0.0647), Logistic from 0.7722 to 0.8087 (+0.0365),
and Shrinkage-LDA from 0.7709 to 0.8162 (+0.0453). Si￾multaneously, parameter-free linear (Cell C, 0.8911) and
rank consensus (Cell D, 0.8591) outperformed both domain
complexity markers and supervised models on this 41-patient
cohort, demonstrating that consensus fusion effectively averts
small-sample overfitting while achieving 0.42±0.08 ms single￾core CPU scoring latency per patient. Direct paired Dirichlet
resampling confirms the deployment-suite margin over the
strongest supervised baseline (+0.1024 vs. Ridge +ρ1, 95% CI
[+0.0311, +0.1856]), while the EEGNet-style CNN collapsed
to 0.5702 AUROC with severe fold instability, empirically
substantiating capacity overflow.
F. Adaptive Gated Consensus (T-GECF) and Noise Stress
Testing
On the complete representation xcombined, continuous Sig￾moid volatility gating in T-GECF (θ0 = 1.645, κ = 2.0)
achieved an AUROC of 0.9116 (95% CI [+0.0013, +0.1927]
vs. canonical baseline; positive gain probability 97.68%) with
0.8399 balanced accuracy, 0.9833 sensitivity, 0.6966 specificity,
and PR-AUC = 0.8669 (Table I), uniting linear sensitivity with
rank stability. On x+ρ1
, T-GECF similarly achieved 0.8828
AUROC (+0.0237 over standard rank consensus 0.8591).
To assess resilience against sensor corruption, impulse
spike perturbation stress tests introduced outlier spikes (±δσ,
δ ∈ {0, 5, 10, 20}) into individual feature coordinates across
testing partitions. Under extreme 20σ impulse corruption,
unconstrained linear aggregation collapsed from 0.8911 to
0.7404 (−0.1507), whereas clip-gated T-GECF shifted weight
to bounded rank consensus, maintaining an AUROC of 0.8518
(+0.1114 advantage over linear aggregation). Dynamic volatil￾ity gating thus delivers verified empirical resilience against
extreme impulse corruptions without sacrificing clean-signal
precision (where continuous Sigmoid gating yields 0.9116
AUROC).
G. Multiscale Window Sensitivity and Longitudinal Reliability
Evaluating observational windows (Table II-A) showed
consistent linear discrimination across 2 s (0.8916), 4 s (0.8911),

---

## [Page 7] (`page-7.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 7
and 8 s (0.8711) partitions, with rank consensus similarly stable
(0.8529–0.8591). Truncating recordings confirmed that a 30-
minute record achieved 0.8867 AUROC, matching full-record
performance. Test-retest reliability across five 1-hour segments
(N = 38, 190 patient-hours) yielded intraclass correlation
coefficient ICC(3, 1) = 0.4857 for ρ1, 0.6433 for D1, and
0.4511 for rresidual. Higher-order lags exhibited monotonic
decay (ρ1 = 0.450 → ρ2 = 0.401 → ρ3 = 0.389 in
MCS; 0.287 → 0.252 → 0.208 in VS/UWS), exhibiting multi￾second lag-1 temporal persistence. Under adjacency masking
Iadj, 50-iteration Monte Carlo dropout confirmed stability
(r = 0.9942 ± 0.0030 at 10% dropout; r = 0.9199 ± 0.0308
at 50% loss), while tightening artifact rejection to 150 µV
preserved 0.8702 AUROC (net gain +0.1729 over baseline).
Sensitivity analyses across contiguous 30-minute block removal
(r = 0.9840 ± 0.0166) and cross-validation partition depths
(10-fold to 100-fold CV, 0.8900–0.9002) confirmed invariant
discrimination.
H. External Waveform Validation and Sleep Macro-Dynamics
On ultra-short 120-s records, the finite-sample crossover
horizon (M∗ ≈ 75) dictates operator selection: normalized
jump rate D1 operates on absolute increments, eliminating
sample autocorrelation variance to achieve 0.7691 AUROC
(+0.0492 gain over static baseline; Table II-A; 77.59% balanced
accuracy). Furthermore, across disparate hospital hardware
(CRRC, 1000 Hz active ADC [21]), while supervised parame￾terized models collapsed to chance (0.5000–0.5370 on the full
cohort; 0.4286–0.4762 on clean 6-channel records), parameter￾free improper linear consensus paired with unsupervised
target-site feature standardization achieved 0.7778 AUROC
(+0.1296 net gain over static baseline; 0.7143 on clean 6-
channel records; Supplementary Table S1), validating robust
multicenter translation. Prospective deployment onboarding
requires only K = 5–10 unlabeled routine records to estab￾lish site-specific baseline scales. Complementing micro-scale
waveforms, macro-scale sleep transition persistence on the 23-
patient Wielek cohort confirmed higher transition flexibility in
MCS (0.261 ± 0.099 vs. 0.164 ± 0.100, one-sided p = 0.0302
[two-sided p = 0.0605, AUROC = 0.7348]; Table II-B),
peaking during nocturnal sleep (0.275±0.105 vs. 0.157±0.113,
p = 0.0392, AUROC = 0.7576).
I. Hyperparameter Sensitivity and Gating Grid Evaluation
To evaluate hyperparameter stability across regimes, Ta￾ble III-A reports systematic benchmarks across truncation
thresholds c and continuous Sigmoid gating parameters (θ0, κ).
Varying Winsorized soft truncation from c = 1.0σ to
unconstrained c = ∞ produced stable AUROC bounded
within [0.8958, 0.9149] (Table III-A; balanced accuracy 0.8046–
0.8399, sensitivity 0.9333–0.9833), peaking at c = 1.5σ
(0.9149) before smoothly decaying toward linear aggregation
(0.8996).
Similarly, evaluating T-GECF across a 4 × 4 parameter grid
crossing Gaussian quantile thresholds θ0 ∈ {1.282, 1.645,
1.960, 2.326} (90th–99th percentiles) and slopes κ ∈ {0.5,
1.0, 2.0, 5.0} demonstrated stability (Table III-B). Across all
16 configurations, AUROC remained bounded within 0.8996–
0.9220, outperforming canonical baseline (0.8200) by +0.08
to +0.10. At θ0 = 2.326 (κ = 5.0), β(x) → 0, smoothly
recovering unconstrained linear aggregation (0.8996) without
threshold sensitivity while preserving empirical resilience
against sensor corruption.
IV. DISCUSSION
A. Physiological Significance of Autoregressive Memory
Under the mesocircuit hypothesis [19], reciprocal excitatory
loops linking the central thalamus and frontoparietal association
cortices sustain contiguous multi-second theta continuity in
MCS, reflecting active cortical holding and recurrent drive.
Conversely, severe deafferentation in VS/UWS disrupts these
recurrent pathways, degrading theta fluctuations into decou￾pled transients that rapidly decay (ρ1 → 0). Permutation￾invariant dispersion (−CVθ) computes only marginal variance,
collapsing ordered reverberation and memoryless noise into
identical scores. Enforcing physical adjacency masking (Iadj)
and computing autoregression provides a tractable bedside
marker for thalamocortical integrity.
Temporal persistence represents a regime-, frequency-, and
scale-specific property rather than a unidirectional marker. In
acute anesthesia, beta LRTC increases during unconscious￾ness [15], paralleling macro-scale sleep stay persistence on
the Wielek cohort (Pstay = 0.8361 in VS/UWS vs. 0.7393
in MCS; Table II-B). Conversely, in chronic DoC, source￾space alpha/beta DFA exponents decrease in VS/UWS [16].
Elevated theta persistence (ρ1) in MCS reconciles these
patterns across timescale, frequency, and etiology, with 10,000-
permutation controls confirming that discrimination strictly
requires chronological sequence order.
B. Resolution of Non-Specific Bilateral Confounding
Substituting raw bilateral correlation with residual correla￾tion (rresidual, AUROC 0.8433) decouples non-specific global
activity, which accounts for 47.27% ± 18.43% of raw bilateral
variance due to volume conduction and subcortical slow-wave
drives. Because diffuse slow-wave envelopes project near￾synchronously across hemispheres, raw correlation conflates
volume conduction with functional connectivity. Decoupling
this common mode unmasks true differentiation: pathological
hyper-synchrony in VS/UWS (0.376±0.193) versus preserved
modularity in MCS (0.233 ± 0.080, p = 0.0211). In sparse
6-channel arrays, the 4-channel average zt provides an effective
regressor orthogonal to CAR, naturally generalizing to subspace
orthogonalization (P⊥ = I − ZZ+ via spatial components Z;
46.52% Mode-1 variance across 62 channels, Table II-B) or
spatial Laplacian in dense montages.
C. Operator Dynamics: Rank Consensus vs. Linear Aggrega￾tion
Linear aggregation and rank consensus address comple￾mentary clinical objectives: linear aggregation maximizes
metric resolution on pristine telemetry (0.9324 AUROC),
while non-parametric rank mapping (Srank) guarantees bounded
stability against distributional distortion (0.9076 AUROC, 100%
sensitivity). Dynamic volatility gating in T-GECF autonomously
bridges these regimes: it achieves peak linear precision dur￾ing routine monitoring while preserving a decisive +0.1114

---

## [Page 8] (`page-8.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 8
AUROC safety margin against unconstrained linear collapse
under extreme 20σ sensor artifacts. In small-sample acute DoC
cohorts (Ntrain ≈ 32), overparameterized deep networks (e.g.,
EEGNet with 3,074 weights; ≈ 96 : 1 parameter-to-sample
ratio) trigger Vapnik-Chervonenkis (VC) capacity overflow
and memorize subject-specific artifacts, empirically confirmed
(0.5702 AUROC, Section III-E). In contrast, our physics￾driven improper linear model [26] eliminates continuous fusion
weight estimation variance, securing verifiable cross-hospital
generalization with sub-millisecond CPU execution.
D. Clinical Scope, Cognitive Motor Dissociation, and Multi￾modal Triage
Asymmetric clinical decision utilities in DoC assign vastly
greater cost to false negatives than false positives (CFN ≫ CFP),
prioritizing sensitivity (0.9167–0.9833 discovery; 0.8333 hospi￾tal; 0.8421 external) to safeguard against premature withdrawal
of life-sustaining treatment (WLST). Operating as a bedside
triage funnel, the system achieves approximately 52% fewer
referrals than universal transfer, focusing high-risk confirmatory
fMRI/PET transports on prioritized candidates. Multicenter
registries establish that 15%–35% of behaviorally unresponsive
patients harbor Cognitive Motor Dissociation (CMD) [4]–[6];
patients categorized as VS/UWS exhibiting sustained ρ1 and
differentiated rresidual represent primary candidates for covert
awareness screening.
E. Translational Deployment Hierarchy and Multicenter Scal￾ability
Consistency across discovery (N = 41), inpatient hospital
validation (N = 15), and external cohort benchmarks (N = 50)
alongside circadian sleep (N = 23 PSG, total N = 129)
establishes an integrated operational deployment hierarchy.
Rather than a monolithic tool, the system provides a calibrated
multi-tier triage funnel: ultra-short 2-minute screening (D1,
0.7691 AUROC) enables rapid patient throughput; 20-minute
bedside monitoring (CRRC hospital cohort, 0.7778 AUROC)
captures subacute inpatient dynamics after sedation washout;
and continuous multi-hour telemetry (ρ1, 0.9324 AUROC)
stabilizes mesocircuit assessment across fluctuating arousal
shifts (ICC = 0.4857). Spanning acute ICU coma, subacute
neurorehabilitation (11 ICH, 4 TBI), and 24-hour sleep cycling
without requiring center-specific retraining, closed-form CPU
execution (0.42 ms, < 100 KB RAM) guarantees immediate
embeddability into standard commercial ICU monitors.
F. Limitations
Several limitations bound the present findings. First, the
discovery cohort (N = 41; 12 MCS) yields confidence intervals
of appreciable width, and behavioral CRS-R labels themselves
carry an established ∼ 40% misdiagnosis rate, so residual
disagreement between model output and clinical categorization
partly reflects label noise rather than classification error. Second,
although frozen outer partitions eliminate fold-level leakage,
the operator suite was developed on the discovery cohort;
the external CRRC and Figshare cohorts, never used during
development, therefore constitute the primary generalization
evidence, and their moderate performance (0.7778 and 0.7691
AUROC) appropriately calibrates bedside expectations below
the discovery estimate (0.9324). Third, single-session test–retest
reliability of ρ1 is moderate (ICC = 0.4857), favoring multi￾hour telemetry deployments over brief acquisitions. Fourth, the
system performs cross-sectional state classification rather than
outcome prediction; covert-awareness (CMD) prioritization
and the projected ∼ 52% reduction in confirmatory transports
await validation against multimodal imaging and longitudinal
endpoints. Finally, etiology heterogeneity (11 ICH, 4 TBI in
CRRC) and residual sedation confounding, although mitigated
by a ≥ 48-h washout, could not be stratified at the present
sample size.
V. CONCLUSION
By resolving the fundamental permutation invariance defect
of conventional electrophysiological markers, this study proves
that order-aware temporal continuity is the decisive physical
signature of preserved thalamocortical mesocircuits. Evaluated
across four independent cohorts (N = 129), the 4-feature
orthogonal architecture elevates bedside discrimination to
0.9324 AUROC, achieves 0.7778 AUROC under zero-label
cross-hospital transfer where parameterized models collapse,
and executes in 0.42 ms on a single CPU core. As an
interpretable, high-sensitivity triage funnel (> 90% sensitivity),
this system safeguards covertly aware patients against pre￾mature withdrawal of life-sustaining treatment while halving
unnecessary multimodal neuroimaging transports, establishing
an accessible, physics-grounded paradigm for ICU bedside
neuroprognostication.
DECLARATIONS
Ethics and Consent: Secondary analyses complied with Declaration of Helsinki,
approved by institutional ethics committees: FRCCICMR Moscow [20], Salzburg Ethics
Commission (No. 415-E/1517/14-2013) [13], Hangzhou Mingzhou Hospital [12], and
China Rehabilitation Research Center (CRRC, Ethics No. 2025-118-01) [21], with
written surrogate consent. Data and Code Availability: Public data: Mendeley [20]
and Figshare [12], [22]. Code repository: https://github.com/chenhj976/DoC Temporal
Dynamics. Conflict of Interest: The authors declare no competing interests.
REFERENCES
[1] J. T. Giacino et al., “The minimally conscious state: definition and
diagnostic criteria,” Neurology, vol. 58, no. 3, pp. 349–353, 2002.
[2] J. T. Giacino et al., “Practice guideline update recommendations summary:
Disorders of consciousness,” Neurology, vol. 91, no. 10, pp. 450–460,
2018.
[3] C. Schnakers et al., “Diagnostic accuracy of the vegetative and minimally
conscious state: clinical consensus versus standardized neurobehavioral
assessment,” BMC Neurology, vol. 9, p. 35, 2009.
[4] D. Cruse et al., “Bedside detection of awareness in the vegetative state:
a cohort study,” The Lancet, vol. 378, no. 9809, pp. 2088–2094, 2011.
[5] Y. G. Bodien et al., “Cognitive motor dissociation in disorders of
consciousness,” New England Journal of Medicine, vol. 391, no. 7,
pp. 598–608, 2024.
[6] A. G. Casali et al., “A theoretically based index of consciousness
independent of sensory processing and behavior,” Science Translational
Medicine, vol. 5, no. 198, p. 198ra105, 2013.
[7] S. Chennu et al., “Brain networks predict metabolism, diagnosis and
prognosis at the bedside in disorders of consciousness,” Brain, vol. 140,
no. 8, pp. 2120–2132, 2017.
[8] J. D. Sitt et al., “Large scale screening of neural signatures of
consciousness in patients in a vegetative or minimally conscious state,”
Brain, vol. 137, no. 8, pp. 2258–2270, 2014.
[9] D. A. Engemann et al., “Robust EEG-based cross-site and cross-protocol
classification of states of consciousness,” Brain, vol. 141, no. 11, pp.
3179–3192, 2018.

---

## [Page 9] (`page-9.png`)

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 9
[10] C. A. Bareham et al., “Bedside EEG predicts longitudinal behavioural
changes in disorders of consciousness,” NeuroImage: Clinical, vol. 28,
p. 102372, 2020.
[11] C. Bandt and B. Pompe, “Permutation entropy: a natural complexity
measure for time series,” Physical Review Letters, vol. 88, no. 17, p.
174102, 2002.
[12] Y. Ling et al., “Effect of topographic comparison of electroencephalo￾graphic microstates on the diagnosis and prognosis prediction of patients
with prolonged disorders of consciousness,” CNS Neuroscience &
Therapeutics, vol. 30, no. 3, p. e14421, 2024.
[13] T. Wielek et al., “Sleep in patients with disorders of consciousness
characterized by means of machine learning,” PLoS One, vol. 13, no. 1,
p. e0190458, 2018.
[14] J. Xiao et al., “Assessing disorders of consciousness using temporal
sleep dynamics extracted from whole-night PSG,” IEEE Transac￾tions on Biomedical Engineering, pp. 1–14, 2026, early access, doi:
10.1109/TBME.2026.3675367.
[15] T. Thiery ´ et al., “Long-range temporal correlations in the brain distinguish
conscious wakefulness from induced unconsciousness,” NeuroImage, vol.
179, pp. 30–39, 2018.
[16] X. Wei et al., “Aberrant temporal correlations of ongoing oscillations
in disorders of consciousness on multiple time scales,” Cognitive
Neurodynamics, vol. 17, no. 3, pp. 633–645, 2023.
[17] Y. Bai et al., “Spontaneous transient brain states in EEG source space
in disorders of consciousness,” NeuroImage, vol. 240, p. 118407, 2021.
[18] D. Toker et al., “Consciousness is supported by near-critical slow cortical
electrodynamics,” Proceedings of the National Academy of Sciences, vol.
119, no. 7, p. e2024455119, 2022.
[19] N. D. Schiff, “Recovery of consciousness after brain injury: a mesocircuit
hypothesis,” Trends in Neurosciences, vol. 33, no. 1, pp. 1–9, 2010.
[20] J. Nekrasova et al., “Polysomnographic records of patients with disorders
of consciousness,” Mendeley Data, V1, 2020.
[21] F. Duan et al., “Auditory paradigm-dependent EEG feature screening and
representation for distinguishing consciousness states in male patients
with disorders of consciousness,” Frontiers in Neurology, vol. 17, p.
1927313, 2026.
[22] C. Xu et al., “Neural tracking of sound rhythms correlates with diagnosis,
severity, and prognosis of disorders of consciousness,” Frontiers in
Neuroscience, vol. 15, p. 646543, 2021.
[23] M. Grabisch et al., Aggregation Functions. Cambridge University Press,
2009.
[24] O. Ledoit and M. Wolf, “A well-conditioned estimator for large￾dimensional covariance matrices,” Journal of Multivariate Analysis,
vol. 88, no. 2, pp. 365–411, 2004.
[25] V. J. Lawhern et al., “EEGNet: A compact convolutional neural
network for EEG-based brain–computer interfaces,” Journal of Neural
Engineering, vol. 15, no. 5, p. 056013, 2018.
[26] R. M. Dawes, “The robust beauty of improper linear models in decision
making,” American Psychologist, vol. 34, no. 7, pp. 571–582, 1979.
[27] D. B. Rubin, “The Bayesian bootstrap,” The Annals of Statistics, vol. 9,
no. 1, pp. 130–134, 1981.
[28] J. Theiler et al., “Testing for nonlinearity in time series: the method of
surrogate data,” Physica D: Nonlinear Phenomena, vol. 58, no. 1-4, pp.
77–94, 1992.

---

