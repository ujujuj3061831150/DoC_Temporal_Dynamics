# Supplementary Information: China Rehabilitation Research Center (CRRC) External Validation Cohort

## 1. Overview and Clinical Provenance
- **Institution**: China Rehabilitation Research Center (CRRC / Beijing Bo'ai Hospital, Capital Medical University Rehabilitation College).
- **Hardware Platform**: NeuSen NSM2 / EEGCap32-pony system (`NSM1HB0032`), sampling rate $1{,}000\text{ Hz}$, standard 10–20 electrode configuration (Duan et al., 2026).
- **Recording Duration**: Mean $19.5\text{ minutes}$ (range: $7.3\text{--}39.1\text{ min}$; total $292.5\text{ patient-minutes}$).
- **Patient Cohort**: $N=15$ severe brain injury patients with disorders of consciousness (DoC).
  - **MCS/EMCS ($N=6$; 5 MCS and 1 EMCS)**: CRS-R scores range from $7$ to $23$ (mean $12.8 \pm 5.9$).
  - **VS/UWS ($N=9$)**: CRS-R scores range from $2$ to $7$ (mean $5.1 \pm 2.2$).
- **Lesion Laterality**: Left hemisphere lesion ($N=4$), right hemisphere lesion ($N=4$), bilateral intact/diffuse ($N=7$).
- **De-Identification Protocol**: Real patient names, Chinese clinical identifiers, and exact admission dates have been strictly scrubbed. Subjects are designated as `CRRC_001` through `CRRC_015`.

---

## 2. Signal Preprocessing & Adjacency Masking
1. **Filtering**: 0.5–45 Hz zero-phase bandpass filter, 50 Hz notch filter for line noise.
2. **Re-Referencing**: Common Average Reference (CAR) across available channels.
3. **Homologous Sparse Montage**: Channels `F3, F4, C3, C4, O1, O2` (100% present across all 15 patients).
4. **Windowing**: Non-overlapping $4.0\text{ s}$ Hann-tapered blocks ($N=4{,}000$ points).
5. **Artifact Rejection**: Hard rejection at $500\ \mu\text{V}$ peak-to-peak amplitude.
6. **Physical Adjacency Indicator**: $\mathbb{I}_{\text{adj}}(t_i, t_{i+1}) = 1$ strictly when $t_{i+1} - t_i = 1$ (adjacent uninterrupted blocks); non-adjacent pairs flanking rejected artifact blocks are excluded from autoregression.
7. **Volume Conduction Decoupling**: Ordinary Least Squares (OLS) orthogonal projection against 4-channel average $z_t = \frac{1}{4}(F_3 + F_4 + C_3 + C_4)$ to obtain residual bilateral correlation $r_{\text{residual}}$.

---

## 3. Dual-Protocol Benchmark Results

### Historical Untruncated Benchmark Metrics

Operating metrics below use the auxiliary target-batch median; intervals retain the original empirical-bootstrap calculation. Current matched comparisons and source-training thresholds are archived separately in `results/matched_model_comparison_v1/summary.json`.

| Metric | Protocol 1: Strict Clean 6-Ch ($N=13$) | Protocol 2: Full Sensitivity ($N=15$) |
| :--- | :---: | :---: |
| **Cohort Composition** | 6 MCS/EMCS / 7 VS (Excludes $O_2$ artifacts) | 6 MCS/EMCS / 9 VS (All clinical records) |
| **Static 2-Feature Baseline** | | |
| - AUROC [95% CI] | 0.5952 [0.2381, 0.9091] | 0.6481 [0.3333, 0.9200] |
| - Average Precision (AP) | 0.5378 | 0.5195 |
| - Balanced Accuracy | 0.6190 | 0.6111 |
| - Sensitivity / Specificity | 0.6667 / 0.5714 | 0.6667 / 0.5556 |
| **Temporal Lag-1 ($\rho_1$)** | | |
| - AUROC [95% CI] | 0.5714 [0.2247, 0.9048] | 0.6296 [0.2963, 0.9286] |
| - Balanced Accuracy | 0.6190 | 0.7500 |
| - Sensitivity / Specificity | 0.6667 / 0.5714 | 0.8333 / 0.6667 |
| **4-Feature Orthogonal Suite** | | |
| - AUROC [95% CI] | **0.7143** [0.3810, 1.0000] | **0.7778** [0.5000, 1.0000] |
| - Average Precision (AP) | **0.6479** | **0.6479** |
| - Balanced Accuracy | **0.6190** | **0.7500** |
| - Sensitivity / Specificity | 0.6667 / 0.5714 | **0.8333** / **0.6667** |
| **Net Gain ($\Delta\text{AUROC}$)** | **+0.1190** | **+0.1296** |
| - Probability of Gain ($\Pr(\Delta > 0)$) | **74.33%** | **81.16%** |

---

## 4. Key Scientific & Methodological Insights
1. **Unsupervised Target-Site Feature Standardization**:
   - The 4-feature orthogonal suite $[c_{\text{global}}, -\mathrm{CV}_\theta, \rho_1, r_{\text{residual}}]$ paired with the physiologically oriented direction vector $[-1, +1, +1, +1]$ transferred to the CRRC NeuSen NSM2 hardware platform without diagnostic labels, achieving an AUROC of **0.7778** (Protocol 2) and **0.7143** (Protocol 1), Frozen-source linear transfer yields 0.5741 on the full cohort. Under the same target alignment, source-tuned Ridge, Logistic, and LDA achieve 0.7407, 0.7593, and 0.7593; the fixed c=1.5 clipped mean achieves 0.7963. At its source-training threshold, clipped scoring gives sensitivity 0.6667, specificity 0.7778, and balanced accuracy 0.7222.
2. **Replication of Positive Temporal Gain**:
   - In both protocols, incorporating temporal autoregressive persistence ($\rho_1$) and volume conduction decoupling ($r_{\text{residual}}$) yielded a net discriminative boost of **+0.1190** to **+0.1296** over the static 2-feature baseline, compared with the $+0.1124$ net gain observed on the Mendeley discovery cohort.
3. **Bridge Between Monitoring Regimes**:
   - With an average recording duration of $19.5\text{ minutes}$, the CRRC cohort bridges the gap between ultra-short $2\text{-minute}$ bedside screening (Figshare) and prolonged $6\text{-hour}$ polysomnography (Mendeley), demonstrating stable autoregressive estimation in standard clinical intermediate telemetry.
