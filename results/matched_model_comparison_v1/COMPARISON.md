# Matched model comparison

All models use the same feature columns, source-only selection, common transforms and source-training median thresholds.
Target-batch adaptation is reported separately from frozen-source transfer. External labels are used only for evaluation.

The clipped equal-mean variant uses fixed c=1.5 in its scoring rule. It does not change the input coordinates or model-selection conditions for other methods.

| Evaluation | Equal mean | Equal mean c=1.5 | Ridge tuned | Logistic tuned | Shrinkage-LDA |
|---|---:|---:|---:|---:|---:|
| mendeley/canonical | 0.8200 | 0.8018 | 0.7653 | 0.7722 | 0.7709 |
| mendeley/orthogonal_lag1 | 0.9324 | 0.9291 | 0.8813 | 0.8964 | 0.8769 |
| mendeley/short_increment | 0.8720 | 0.8518 | 0.8144 | 0.8218 | 0.8311 |
| crrc_full15/orthogonal_lag1/source_frozen | 0.5741 | 0.5741 | 0.5556 | 0.5556 | 0.5556 |
| crrc_full15/orthogonal_lag1/target_standardized | 0.5000 | 0.5000 | 0.5185 | 0.5185 | 0.5370 |
| crrc_full15/orthogonal_lag1/target_aligned | 0.7778 | 0.7963 | 0.7407 | 0.7593 | 0.7593 |
| crrc_clean13/orthogonal_lag1/source_frozen | 0.5238 | 0.5238 | 0.5238 | 0.5238 | 0.5238 |
| crrc_clean13/orthogonal_lag1/target_standardized | 0.4286 | 0.4286 | 0.4762 | 0.4762 | 0.4762 |
| crrc_clean13/orthogonal_lag1/target_aligned | 0.7143 | 0.7143 | 0.7143 | 0.6905 | 0.6905 |
| figshare50/canonical/source_frozen | 0.7199 | 0.6961 | 0.7691 | 0.7402 | 0.7385 |
| figshare50/canonical/target_standardized | 0.7284 | 0.7284 | 0.7301 | 0.7351 | 0.7233 |
| figshare50/orthogonal_lag1/source_frozen | 0.5161 | 0.5433 | 0.5127 | 0.5161 | 0.5144 |
| figshare50/orthogonal_lag1/target_standardized | 0.6027 | 0.5891 | 0.5314 | 0.5772 | 0.5772 |
| figshare50/short_increment/source_frozen | 0.7691 | 0.7606 | 0.7759 | 0.7674 | 0.7742 |
| figshare50/short_increment/target_standardized | 0.7657 | 0.7606 | 0.7708 | 0.7589 | 0.7691 |

## Direct paired contrasts

| Evaluation | Reference | Comparator | Delta AUROC | 95% interval |
|---|---|---|---:|---|
| mendeley/canonical | equal_mean | equal_mean_c15 | +0.0182 | [+0.0046, +0.0369] |
| mendeley/canonical | equal_mean | ridge | +0.0547 | [+0.0137, +0.0992] |
| mendeley/canonical | equal_mean | logistic | +0.0478 | [+0.0066, +0.0954] |
| mendeley/canonical | equal_mean | lda_shrinkage | +0.0491 | [+0.0163, +0.0880] |
| mendeley/canonical | equal_mean_c15 | equal_mean | -0.0182 | [-0.0369, -0.0046] |
| mendeley/canonical | equal_mean_c15 | ridge | +0.0364 | [-0.0089, +0.0828] |
| mendeley/canonical | equal_mean_c15 | logistic | +0.0296 | [-0.0140, +0.0778] |
| mendeley/canonical | equal_mean_c15 | lda_shrinkage | +0.0309 | [-0.0067, +0.0706] |
| mendeley/orthogonal_lag1 | equal_mean | equal_mean_c15 | +0.0033 | [-0.0185, +0.0320] |
| mendeley/orthogonal_lag1 | equal_mean | ridge | +0.0511 | [+0.0184, +0.0900] |
| mendeley/orthogonal_lag1 | equal_mean | logistic | +0.0360 | [+0.0090, +0.0728] |
| mendeley/orthogonal_lag1 | equal_mean | lda_shrinkage | +0.0556 | [+0.0202, +0.0975] |
| mendeley/orthogonal_lag1 | equal_mean_c15 | equal_mean | -0.0033 | [-0.0320, +0.0185] |
| mendeley/orthogonal_lag1 | equal_mean_c15 | ridge | +0.0478 | [+0.0002, +0.0979] |
| mendeley/orthogonal_lag1 | equal_mean_c15 | logistic | +0.0327 | [-0.0059, +0.0767] |
| mendeley/orthogonal_lag1 | equal_mean_c15 | lda_shrinkage | +0.0522 | [+0.0078, +0.1025] |
| mendeley/short_increment | equal_mean | equal_mean_c15 | +0.0202 | [+0.0042, +0.0477] |
| mendeley/short_increment | equal_mean | ridge | +0.0576 | [+0.0252, +0.0971] |
| mendeley/short_increment | equal_mean | logistic | +0.0502 | [+0.0202, +0.0883] |
| mendeley/short_increment | equal_mean | lda_shrinkage | +0.0409 | [+0.0151, +0.0754] |
| mendeley/short_increment | equal_mean_c15 | equal_mean | -0.0202 | [-0.0477, -0.0042] |
| mendeley/short_increment | equal_mean_c15 | ridge | +0.0373 | [+0.0072, +0.0730] |
| mendeley/short_increment | equal_mean_c15 | logistic | +0.0300 | [+0.0045, +0.0622] |
| mendeley/short_increment | equal_mean_c15 | lda_shrinkage | +0.0207 | [-0.0024, +0.0514] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean | equal_mean_c15 | +0.0000 | [+0.0000, +0.0000] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean | ridge | +0.0185 | [+0.0001, +0.0963] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean | logistic | +0.0185 | [+0.0001, +0.0963] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean | lda_shrinkage | +0.0185 | [+0.0001, +0.0963] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean_c15 | equal_mean | +0.0000 | [+0.0000, +0.0000] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean_c15 | ridge | +0.0185 | [+0.0001, +0.0963] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean_c15 | logistic | +0.0185 | [+0.0001, +0.0963] |
| crrc_full15/orthogonal_lag1/source_frozen | equal_mean_c15 | lda_shrinkage | +0.0185 | [+0.0001, +0.0963] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean | equal_mean_c15 | +0.0000 | [+0.0000, +0.0000] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean | ridge | -0.0185 | [-0.1454, +0.0837] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean | logistic | -0.0185 | [-0.0942, -0.0001] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean | lda_shrinkage | -0.0370 | [-0.1545, -0.0006] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean_c15 | equal_mean | +0.0000 | [+0.0000, +0.0000] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean_c15 | ridge | -0.0185 | [-0.1454, +0.0837] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean_c15 | logistic | -0.0185 | [-0.0942, -0.0001] |
| crrc_full15/orthogonal_lag1/target_standardized | equal_mean_c15 | lda_shrinkage | -0.0370 | [-0.1545, -0.0006] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean | equal_mean_c15 | -0.0185 | [-0.0923, -0.0001] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean | ridge | +0.0370 | [-0.0738, +0.2062] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean | logistic | +0.0185 | [+0.0001, +0.0978] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean | lda_shrinkage | +0.0185 | [+0.0001, +0.0978] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean_c15 | equal_mean | +0.0185 | [+0.0001, +0.0923] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean_c15 | ridge | +0.0556 | [-0.0620, +0.2246] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean_c15 | logistic | +0.0370 | [+0.0021, +0.1344] |
| crrc_full15/orthogonal_lag1/target_aligned | equal_mean_c15 | lda_shrinkage | +0.0370 | [+0.0021, +0.1344] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean | equal_mean_c15 | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean | ridge | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean | logistic | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean | lda_shrinkage | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean_c15 | equal_mean | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean_c15 | ridge | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean_c15 | logistic | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/source_frozen | equal_mean_c15 | lda_shrinkage | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean | equal_mean_c15 | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean | ridge | -0.0476 | [-0.1968, -0.0010] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean | logistic | -0.0476 | [-0.1968, -0.0010] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean | lda_shrinkage | -0.0476 | [-0.1968, -0.0010] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean_c15 | equal_mean | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean_c15 | ridge | -0.0476 | [-0.1968, -0.0010] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean_c15 | logistic | -0.0476 | [-0.1968, -0.0010] |
| crrc_clean13/orthogonal_lag1/target_standardized | equal_mean_c15 | lda_shrinkage | -0.0476 | [-0.1968, -0.0010] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean | equal_mean_c15 | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean | ridge | +0.0000 | [-0.1121, +0.1140] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean | logistic | +0.0238 | [+0.0002, +0.1228] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean | lda_shrinkage | +0.0238 | [+0.0002, +0.1228] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean_c15 | equal_mean | +0.0000 | [+0.0000, +0.0000] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean_c15 | ridge | +0.0000 | [-0.1121, +0.1140] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean_c15 | logistic | +0.0238 | [+0.0002, +0.1228] |
| crrc_clean13/orthogonal_lag1/target_aligned | equal_mean_c15 | lda_shrinkage | +0.0238 | [+0.0002, +0.1228] |
| figshare50/canonical/source_frozen | equal_mean | equal_mean_c15 | +0.0238 | [-0.0109, +0.0695] |
| figshare50/canonical/source_frozen | equal_mean | ridge | -0.0492 | [-0.1031, -0.0019] |
| figshare50/canonical/source_frozen | equal_mean | logistic | -0.0204 | [-0.0510, +0.0089] |
| figshare50/canonical/source_frozen | equal_mean | lda_shrinkage | -0.0187 | [-0.0491, +0.0080] |
| figshare50/canonical/source_frozen | equal_mean_c15 | equal_mean | -0.0238 | [-0.0695, +0.0109] |
| figshare50/canonical/source_frozen | equal_mean_c15 | ridge | -0.0730 | [-0.1392, -0.0149] |
| figshare50/canonical/source_frozen | equal_mean_c15 | logistic | -0.0441 | [-0.0948, -0.0042] |
| figshare50/canonical/source_frozen | equal_mean_c15 | lda_shrinkage | -0.0424 | [-0.0961, -0.0000] |
| figshare50/canonical/target_standardized | equal_mean | equal_mean_c15 | +0.0000 | [-0.0136, +0.0115] |
| figshare50/canonical/target_standardized | equal_mean | ridge | -0.0017 | [-0.0827, +0.0665] |
| figshare50/canonical/target_standardized | equal_mean | logistic | -0.0068 | [-0.0536, +0.0301] |
| figshare50/canonical/target_standardized | equal_mean | lda_shrinkage | +0.0051 | [-0.0581, +0.0586] |
| figshare50/canonical/target_standardized | equal_mean_c15 | equal_mean | +0.0000 | [-0.0115, +0.0136] |
| figshare50/canonical/target_standardized | equal_mean_c15 | ridge | -0.0017 | [-0.0833, +0.0665] |
| figshare50/canonical/target_standardized | equal_mean_c15 | logistic | -0.0068 | [-0.0553, +0.0316] |
| figshare50/canonical/target_standardized | equal_mean_c15 | lda_shrinkage | +0.0051 | [-0.0583, +0.0593] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean | equal_mean_c15 | -0.0272 | [-0.0884, +0.0347] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean | ridge | +0.0034 | [-0.0059, +0.0165] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean | logistic | +0.0000 | [-0.0081, +0.0083] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean | lda_shrinkage | +0.0017 | [-0.0072, +0.0132] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean_c15 | equal_mean | +0.0272 | [-0.0347, +0.0884] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean_c15 | ridge | +0.0306 | [-0.0292, +0.0884] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean_c15 | logistic | +0.0272 | [-0.0329, +0.0854] |
| figshare50/orthogonal_lag1/source_frozen | equal_mean_c15 | lda_shrinkage | +0.0289 | [-0.0316, +0.0872] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean | equal_mean_c15 | +0.0136 | [-0.0172, +0.0466] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean | ridge | +0.0713 | [-0.0067, +0.1568] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean | logistic | +0.0255 | [-0.0121, +0.0696] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean | lda_shrinkage | +0.0255 | [-0.0139, +0.0716] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean_c15 | equal_mean | -0.0136 | [-0.0466, +0.0172] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean_c15 | ridge | +0.0577 | [-0.0228, +0.1428] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean_c15 | logistic | +0.0119 | [-0.0337, +0.0579] |
| figshare50/orthogonal_lag1/target_standardized | equal_mean_c15 | lda_shrinkage | +0.0119 | [-0.0351, +0.0588] |
| figshare50/short_increment/source_frozen | equal_mean | equal_mean_c15 | +0.0085 | [-0.0106, +0.0382] |
| figshare50/short_increment/source_frozen | equal_mean | ridge | -0.0068 | [-0.0319, +0.0125] |
| figshare50/short_increment/source_frozen | equal_mean | logistic | +0.0017 | [-0.0136, +0.0214] |
| figshare50/short_increment/source_frozen | equal_mean | lda_shrinkage | -0.0051 | [-0.0290, +0.0134] |
| figshare50/short_increment/source_frozen | equal_mean_c15 | equal_mean | -0.0085 | [-0.0382, +0.0106] |
| figshare50/short_increment/source_frozen | equal_mean_c15 | ridge | -0.0153 | [-0.0496, +0.0080] |
| figshare50/short_increment/source_frozen | equal_mean_c15 | logistic | -0.0068 | [-0.0373, +0.0163] |
| figshare50/short_increment/source_frozen | equal_mean_c15 | lda_shrinkage | -0.0136 | [-0.0464, +0.0089] |
| figshare50/short_increment/target_standardized | equal_mean | equal_mean_c15 | +0.0051 | [-0.0108, +0.0236] |
| figshare50/short_increment/target_standardized | equal_mean | ridge | -0.0051 | [-0.0505, +0.0344] |
| figshare50/short_increment/target_standardized | equal_mean | logistic | +0.0068 | [-0.0187, +0.0354] |
| figshare50/short_increment/target_standardized | equal_mean | lda_shrinkage | -0.0034 | [-0.0406, +0.0298] |
| figshare50/short_increment/target_standardized | equal_mean_c15 | equal_mean | -0.0051 | [-0.0236, +0.0108] |
| figshare50/short_increment/target_standardized | equal_mean_c15 | ridge | -0.0102 | [-0.0593, +0.0309] |
| figshare50/short_increment/target_standardized | equal_mean_c15 | logistic | +0.0017 | [-0.0324, +0.0328] |
| figshare50/short_increment/target_standardized | equal_mean_c15 | lda_shrinkage | -0.0085 | [-0.0502, +0.0261] |
