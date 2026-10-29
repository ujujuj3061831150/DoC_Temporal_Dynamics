# Python script to apply precise text refinements to manuscript.tex
from pathlib import Path

def run():
    p_v3 = Path('v3/manuscript_temporal/manuscript.tex')
    p_root = Path('manuscript_temporal/manuscript.tex')
    
    text = p_v3.read_text(encoding='utf-8')
    
    replacements = [
        # 1. Abstract
        (
            r'Importantly, zero-shot transfer under unsupervised target-site feature standardization onto an independent inpatient hospital cohort from China Rehabilitation Research Center (CRRC, $N=15$, BioSemi 32-channel, 1000~Hz) yields 0.7778 AUROC (+0.1296 net gain over static baseline 0.6481; 0.7143 on clean 6-channel records), while parameterized models collapse to chance levels (0.4286--0.5185).',
            r'Importantly, unsupervised target-site feature standardization paired with a physiologically oriented direction vector onto an independent inpatient hospital cohort from China Rehabilitation Research Center (CRRC, $N=15$, BioSemi 32-channel, 1000~Hz) yields 0.7778 AUROC (+0.1296 net gain over static baseline 0.6481; 0.7143 on clean 6-channel records), while parameterized models collapse (0.4286--0.5185).'
        ),
        (
            r'prioritizing covert awareness candidates for multimodal imaging while reducing unnecessary transport burdens.',
            r'prioritizing covert awareness candidates for multimodal imaging while reducing secondary transport burdens.'
        ),
        # 2. Section I-B
        (
            r'Conversely, structural deafferentation and striatopallidal disinhibition in VS/UWS degrade cortical dynamics into memoryless, erratic fluctuations.',
            r'Conversely, structural deafferentation and loss of striatal inhibition disinhibiting the internal globus pallidus (GPi) hyper-inhibit the central thalamus in VS/UWS, degrading cortical dynamics into memoryless fluctuations.'
        ),
        # 3. Section I-D
        (
            r'(4)~\textbf{Real-World Multicenter Validation}: Demonstrates zero-shot transfer across four clinical centers ($N=129$), featuring an independent inpatient hospital cohort from China Rehabilitation Research Center (CRRC, $N=15$, 0.7778 AUROC, +0.1296 net gain) and showing the collapse of parameterized classifiers under domain shift;',
            r'(4)~\textbf{Real-World Multicenter Validation}: Demonstrates cross-center evaluation across four cohorts ($N=129$), featuring an independent inpatient hospital cohort from China Rehabilitation Research Center (CRRC, $N=15$, 0.7778 AUROC, +0.1296 net gain under target-site standardization) and showing the collapse of parameterized classifiers under domain shift;'
        ),
        # 4. Section II-A
        (
            r'mean $19.5 \pm 5.2$~min',
            r'mean $19.5 \pm 8.0$~min'
        ),
        # 5. Section II-B
        (
            r'Let $\mathcal{T} = \{t_1, t_2, \dots, t_N\}$ denote the sequence of clean epoch indices, and let $G_t$ denote the spatial mean theta power across all 6 channels at epoch $t$.',
            r'Let $\mathcal{T} = \{t_1, t_2, \dots, t_N\}$ denote clean epoch indices within a continuous recording session, and let $G_{t_i}$ denote spatial mean theta power across all 6 channels at epoch $t_i$.'
        ),
        (
            r'minimum 1,678, exceeding $M \ge 10$).',
            r'minimum 1,678, exceeding calculation threshold $M \ge 10$).'
        ),
        # 6. Section II-C
        (
            r'Under local stationary Gaussian increments $\Delta G_t \sim \mathcal{N}(0, 2\sigma^2(1 - \rho_1))$, the expected jump magnitude satisfies $\mathbb{E}[|\Delta G_t|] = 2\pi^{-1/2} \sigma (1 - \rho_1)^{1/2}$, establishing the analytical link $D_1 \approx 2\pi^{-1/2} (-\mathrm{CV}_\theta) (1 - \rho_1)^{1/2}$. Because raw electrophysiological power typically displays right-skewed log-normal characteristics, this theoretical proportionality is evaluated on log-transformed power $\log(G_t)$ in Section~III-C. Under estimator variance $\operatorname{Var}(\hat{\rho}_1) \sim \mathcal{O}(1/M)$, $D_1$ provides a low-variance proxy for short records, whereas $\rho_1$ achieves asymptotic efficiency.',
            r'Under stationary Gaussian increments $\Delta G_t \sim \mathcal{N}(0, 2\sigma^2(1 - \rho_1))$, expected jump magnitude satisfies $\mathbb{E}[|\Delta G_t|] = 2\pi^{-1/2} \sigma (1 - \rho_1)^{1/2}$, establishing $D_1 \approx 2\pi^{-1/2} (-\mathrm{CV}_\theta) (1 - \rho_1)^{1/2}$. Because raw power exhibits heavy-tailed dispersion ($r=0.4630$), this Gaussian coupling is evaluated directly on log-power $\log(G_t)$ in Section~III-C ($r=0.9691, r^2=93.91\%$). Under estimator variance $\operatorname{Var}(\hat{\rho}_1) \sim \mathcal{O}(1/M)$, $D_1$ serves as a low-variance proxy for short records below $M^* \approx 75$ epochs, whereas $\rho_1$ achieves asymptotic efficiency.'
        ),
        # 7. Section II-D
        (
            r'The common-mode coefficient of determination $R^2 = 1 - \|\boldsymbol{\epsilon}\|^2 / \|\mathbf{y} - \bar{\mathbf{y}}\|^2$ accounts for $47.27\% \pm 18.43\%$ of raw bilateral variance across the cohort; the OLS operator $\mathbf{P}^\perp$ decouples these background fluctuations in the power domain to isolate residual functional resonance.',
            r'The shared 4-channel power envelope accounts for $R^2 = 47.27\% \pm 18.43\%$ of raw bilateral variance; the full-rank OLS projection $\mathbf{P}^\perp$ decouples this background regressor in the power domain to isolate residual bilateral differentiation.'
        ),
        # 8. Section II-E
        (
            r'eliminating parameter estimation variance.',
            r'eliminating continuous fusion weight estimation variance.'
        ),
        # 9. Section III-C
        (
            r'residualization unmasked genuine bilateral differentiation ($0.233 \pm 0.080$ vs. $0.376 \pm 0.193$, $p = 0.0211$, AUROC = 0.7328), reflecting preserved modularity in MCS versus hyper-synchrony in VS/UWS. Adjusting for spectral slowing ($c_{\text{global}}$) and volume conduction ($R^2$), multivariable logistic regression confirms that $\rho_1$ independently predicts consciousness ($z = 2.460, p = 0.0139$, adjusted OR = $4.260$, 95\% CI $[1.343, 13.512]$).',
            r'residualization unmasked residual bilateral differentiation beyond the shared background power envelope ($0.233 \pm 0.080$ vs. $0.376 \pm 0.193$, $p = 0.0211$, AUROC = 0.7328), reflecting modularity in MCS versus hyper-synchrony in VS/UWS. Adjusting for spectral slowing ($c_{\text{global}}$) and background coupling ($R^2$), multivariable logistic regression confirms that $\rho_1$ associates with consciousness ($z = 2.460, p = 0.0139$, adjusted OR = $4.260$, 95\% CI $[1.343, 13.512]$).'
        ),
        # 10. Section III-D
        (
            r'Across configurations, destroying sequence order abolished discriminative utility.',
            r'Across configurations, destroying sequence order removed the temporal increment under this ablation, collapsing scores to marginal baseline levels ($0.8182 \pm 0.0049$).'
        ),
        # 11. Section III-G
        (
            r'confirming first-order dynamics.',
            r'exhibiting multi-second lag-1 temporal persistence.'
        ),
        # 12. Section III-H
        (
            r'Crucially, zero-shot transfer under unsupervised target-site feature standardization on the independent CRRC inpatient hospital cohort ($N=15$, mean 19.5~min, 1000~Hz BioSemi) confirmed robust generalization of the 4-feature orthogonal suite with 0.7778 AUROC (+0.1296 net gain over static baseline 0.6481; 0.7143 on clean 6-channel records; Supplementary Table~S1), whereas parameterized supervised models collapsed under domain shift (0.4286--0.5185 AUROC).',
            r'Crucially, evaluating the independent CRRC inpatient hospital cohort ($N=15$, mean $19.5 \pm 8.0$~min, 1000~Hz BioSemi) under unsupervised target-site feature standardization paired with a physiologically oriented direction vector yielded 0.7778 AUROC (+0.1296 net gain over static baseline 0.6481, 95\% CI $[-0.1364, +0.4286]$; 0.7143 on clean 6-channel records; direct frozen-source transfer yields 0.5741; Supplementary Table~S1), whereas parameterized supervised models collapsed under domain shift (0.4286--0.5185 AUROC).'
        ),
        # 13. Section IV-D
        (
            r'reducing transport burdens by half.',
            r'projected to reduce secondary transport burdens by approximately half.'
        )
    ]
    
    applied = 0
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            applied += 1
            print(f"Applied: {old[:50]}...")
        else:
            print(f"NOT FOUND: {old[:50]}...")
            
    print(f"Total applied: {applied}/{len(replacements)}")
    p_v3.write_text(text, encoding='utf-8')
    p_root.write_text(text, encoding='utf-8')
    print("Files successfully updated!")

if __name__ == '__main__':
    run()
