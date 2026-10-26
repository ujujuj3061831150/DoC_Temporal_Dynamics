import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, percentileofscore
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "v3" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))
from temporal_dynamics.adapters import load_mendeley_temporal_dataset, extract_clean_theta_series
from temporal_dynamics.consensus import SignedStandardizedMean, TemporalMDCO, AdaptiveGatedConsensus

print("==========================================================")
print("INVESTIGATING ALL REVIEWER CONCERNS RIGOROUSLY")
print("==========================================================")

# 1. Load dataset & outer folds
dataset = load_mendeley_temporal_dataset(REPO_ROOT)
subject_ids = dataset["subject_ids"]
labels = dataset["labels"]
variants = dataset["feature_matrix_variants"]

all_states = json.loads((REPO_ROOT / "results/multiscale_matched_v2/fitted_states.json").read_text())
folds = [r for r in all_states if r.get("cohort") == "mendeley" and r.get("method") == "C"]

# --- POINT 3: FULL MODEL COMBINATION ABLATION ---
print("\n--- POINT 3: FULL MODEL COMBINATIONS ---")
# Build combined feature matrices
# canonical: [c_glob, c_spat, -CV, r_inter]
# residual_corr replaces col 3 (r_inter) with r_res
# replace_cv_lag1 replaces col 2 (-CV) with rho1
# So:
# Combo 1 (4 feats): [c_glob, c_spat, rho1, r_res]
# Combo 2 (5 feats): [c_glob, c_spat, -CV, rho1, r_res]
c_glob = variants["canonical"][:, 0]
c_spat = variants["canonical"][:, 1]
neg_cv = variants["canonical"][:, 2]
r_inter = variants["canonical"][:, 3]
r_res = variants["residual_corr"][:, 3]
rho1 = variants["plus_lag1"][:, 4]
d1 = variants["plus_increment"][:, 4]

combo_4 = np.column_stack([c_glob, c_spat, rho1, r_res]) # replace both static
combo_5 = np.column_stack([c_glob, c_spat, neg_cv, rho1, r_res]) # plus rho1 and r_res
combo_6 = np.column_stack([c_glob, c_spat, neg_cv, r_inter, rho1, r_res]) # all 6

for name, mat in [("Combo_4 (rho1 + r_res, replace both)", combo_4),
                  ("Combo_5 (Canonical + rho1 + r_res)", combo_5),
                  ("Combo_6 (All 6 features)", combo_6)]:
    aucs_c, aucs_d, prs_c = [], [], []
    for f in folds:
        tr_idx = [subject_ids.index(s) for s in f["train_ids"]]
        te_idx = [subject_ids.index(s) for s in f["test_ids"]]
        X_tr, y_tr = mat[tr_idx], labels[tr_idx]
        X_te, y_te = mat[te_idx], labels[te_idx]
        
        m_c = SignedStandardizedMean().fit(X_tr, y_tr)
        p_c = m_c.predict(X_te)
        aucs_c.append(roc_auc_score(y_te, p_c))
        
        m_d = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
        p_d = m_d.predict(X_te)
        aucs_d.append(roc_auc_score(y_te, p_d))
        
        prec, rec, _ = precision_recall_curve(y_te, p_c)
        prs_c.append(auc(rec, prec))
    print(f"{name}: Linear C AUROC = {np.mean(aucs_c):.4f}, PR-AUC = {np.mean(prs_c):.4f}, MDCO D AUROC = {np.mean(aucs_d):.4f}")

# --- POINT 2 & 3: MATHEMATICAL COUPLING D1 ~ (-CV) * sqrt(1 - rho1) & c_glob vs c_spat ---
print("\n--- POINT 2 & 3: MATHEMATICAL COUPLINGS & COLLINEARITY ---")
corr_glob_spat, p_glob_spat = pearsonr(c_glob, c_spat)
print(f"Correlation between c_global and c_spatial: r = {corr_glob_spat:.4f} (p = {p_glob_spat:.4e})")

# D1 analytic approximation: D1_approx = (2/sqrt(pi)) * (-CV) * sqrt(1 - rho1)
# Note: CV = sigma/mu, -CV = -sigma/mu.
# Since rho1 can be slightly negative or bounded, clip 1 - rho1 >= 0
d1_approx = (2.0 / np.sqrt(np.pi)) * neg_cv * np.sqrt(np.clip(1.0 - rho1, 0, None))
corr_d1_analytic, p_d1 = pearsonr(d1, d1_approx)
print(f"Correlation between empirical D1 and analytic coupling (2/sqrt(pi))*(-CV)*sqrt(1-rho1): r = {corr_d1_analytic:.4f} (p = {p_d1:.4e})")

# --- POINT 4: 47% COMMON-MODE VARIANCE CALCULATION ---
print("\n--- POINT 4: OLS COMMON-MODE EXPLAINED VARIANCE (R^2) ---")
r2_list_all, r2_list_mcs, r2_list_vs = [], [], []
for i, sid in enumerate(subject_ids):
    npz_path = REPO_ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
    theta, indices = extract_clean_theta_series(npz_path) # theta is (N, 6)
    # Target pairs: (0, 1) [F3, F4], (2, 3) [C3, C4], (4, 5) [O1, O2]
    pairs = [(0, 1), (2, 3), (4, 5)]
    r2_pairs = []
    for (L, R) in pairs:
        bg_chs = [c for c in range(6) if c not in (L, R)]
        z = np.mean(theta[:, bg_chs], axis=1)
        z_norm = (z - np.mean(z)) / (np.std(z) + 1e-12)
        X_c = np.column_stack([np.ones(len(z_norm)), z_norm])
        
        # OLS on L and R
        beta_L = np.linalg.lstsq(X_c, theta[:, L], rcond=None)[0]
        y_L_pred = X_c @ beta_L
        r2_L = 1.0 - np.var(theta[:, L] - y_L_pred) / (np.var(theta[:, L]) + 1e-12)
        
        beta_R = np.linalg.lstsq(X_c, theta[:, R], rcond=None)[0]
        y_R_pred = X_c @ beta_R
        r2_R = 1.0 - np.var(theta[:, R] - y_R_pred) / (np.var(theta[:, R]) + 1e-12)
        
        r2_pairs.append(0.5 * (r2_L + r2_R))
    mean_r2 = float(np.mean(r2_pairs))
    r2_list_all.append(mean_r2)
    if labels[i] == 1:
        r2_list_mcs.append(mean_r2)
    else:
        r2_list_vs.append(mean_r2)

print(f"Full Cohort Common-Mode R^2: {np.mean(r2_list_all)*100:.2f}% ± {np.std(r2_list_all)*100:.2f}% (range: {np.min(r2_list_all)*100:.2f}% - {np.max(r2_list_all)*100:.2f}%)")
print(f"MCS Common-Mode R^2: {np.mean(r2_list_mcs)*100:.2f}% ± {np.std(r2_list_mcs)*100:.2f}%")
print(f"VS/UWS Common-Mode R^2: {np.mean(r2_list_vs)*100:.2f}% ± {np.std(r2_list_vs)*100:.2f}%")
u_stat, u_p = mannwhitneyu(r2_list_mcs, r2_list_vs)
print(f"Group difference in Common-Mode R^2: p = {u_p:.4f}")

# --- POINT 5: RECORDING DURATION IN HOURS & BLOCKS ---
print("\n--- POINT 5: EXACT RECORDING DURATION AUDIT ---")
dur_hours = []
block_counts_raw = []
block_counts_clean = []
for sid in subject_ids:
    npz_path = REPO_ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
    with np.load(npz_path) as z:
        pk = z["peaks"]
        block_counts_raw.append(len(pk))
        dur_hours.append(len(pk) * 4.0 / 3600.0)
        clean_cnt = int(np.sum(pk < 500))
        block_counts_clean.append(clean_cnt)

mcs_mask = (labels == 1)
vs_mask = (labels == 0)
dur_arr = np.array(dur_hours)
clean_arr = np.array(block_counts_clean)
raw_arr = np.array(block_counts_raw)

print(f"All Patients Duration (hours): {np.mean(dur_arr):.2f} ± {np.std(dur_arr):.2f} (range: {np.min(dur_arr):.2f} - {np.max(dur_arr):.2f})")
print(f"MCS Duration (hours): {np.mean(dur_arr[mcs_mask]):.2f} ± {np.std(dur_arr[mcs_mask]):.2f} (range: {np.min(dur_arr[mcs_mask]):.2f} - {np.max(dur_arr[mcs_mask]):.2f})")
print(f"VS/UWS Duration (hours): {np.mean(dur_arr[vs_mask]):.2f} ± {np.std(dur_arr[vs_mask]):.2f} (range: {np.min(dur_arr[vs_mask]):.2f} - {np.max(dur_arr[vs_mask]):.2f})")
print(f"All Patients Raw Blocks: {np.mean(raw_arr):.1f} ± {np.std(raw_arr):.1f}")
print(f"All Patients Clean Blocks: {np.mean(clean_arr):.1f} ± {np.std(clean_arr):.1f}")
print(f"MCS Clean Blocks: {np.mean(clean_arr[mcs_mask]):.1f} ± {np.std(clean_arr[mcs_mask]):.1f}")
print(f"VS Clean Blocks: {np.mean(clean_arr[vs_mask]):.1f} ± {np.std(clean_arr[vs_mask]):.1f}")

# --- POINT 7: LOCAL WINDOWED RHO1 (DETRENDING TEST) ---
print("\n--- POINT 7: LOCAL WINDOWED RHO1 (2-MIN & 5-MIN DETRENDING) ---")
# Compute rho1 in 2-min (30 blocks) and 5-min (75 blocks) local windows
for win_len_blocks, win_name in [(30, "2-min"), (75, "5-min")]:
    local_rho1s = []
    for sid in subject_ids:
        npz_path = REPO_ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
        theta, indices = extract_clean_theta_series(npz_path) # (N, 6)
        mean_theta = np.mean(theta, axis=1)
        
        # Split into consecutive windows
        w_rhos = []
        n_win = len(indices) // win_len_blocks
        for w in range(n_win):
            w_th = mean_theta[w*win_len_blocks : (w+1)*win_len_blocks]
            w_idx = indices[w*win_len_blocks : (w+1)*win_len_blocks]
            # within-window mean centering (detrending local DC)
            w_th_detrend = w_th - np.mean(w_th)
            # compute lag1 on adjacent pairs
            adj_mask = (w_idx[1:] - w_idx[:-1] == 1)
            if np.sum(adj_mask) >= 5:
                x = w_th_detrend[:-1][adj_mask]
                y_w = w_th_detrend[1:][adj_mask]
                denom = np.std(x) * np.std(y_w)
                if denom > 1e-12:
                    r = np.mean(x * y_w) / denom
                    w_rhos.append(r)
        local_rho1s.append(float(np.mean(w_rhos)) if len(w_rhos) > 0 else 0.0)
    
    local_rho1s = np.array(local_rho1s)
    mcs_loc = local_rho1s[mcs_mask]
    vs_loc = local_rho1s[vs_mask]
    auc_loc = roc_auc_score(labels, local_rho1s)
    u_loc, p_loc = mannwhitneyu(mcs_loc, vs_loc, alternative="greater")
    print(f"Local {win_name} Detrended rho1: MCS = {np.mean(mcs_loc):.4f} ± {np.std(mcs_loc):.4f}, VS = {np.mean(vs_loc):.4f} ± {np.std(vs_loc):.4f}, AUROC = {auc_loc:.4f}, p = {p_loc:.4f}")

# --- POINT 1 & 8: NOISE INJECTION STRESS TEST (LINEAR C vs T-GECF vs MDCO) ---
print("\n--- POINT 1 & 8: NOISE INJECTION STRESS TEST ---")
# Evaluate Linear C, Inverted T-GECF, and MDCO D when test features are corrupted by impulsive / heavy-tailed noise
noise_levels = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0] # in units of feature std
# We add an impulsive spike to 1 random feature in 20% of test samples
np.random.seed(20261014)
for noise_std in noise_levels:
    aucs_lin = []
    aucs_gecf = []
    aucs_mdco = []
    
    for f in folds:
        tr_idx = [subject_ids.index(s) for s in f["train_ids"]]
        te_idx = [subject_ids.index(s) for s in f["test_ids"]]
        X_tr = variants["plus_lag1"][tr_idx].copy()
        y_tr = labels[tr_idx]
        X_te = variants["plus_lag1"][te_idx].copy()
        y_te = labels[te_idx]
        
        # Fit models on clean training data
        m_lin = SignedStandardizedMean().fit(X_tr, y_tr)
        m_mdco = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
        
        # Inner tune T-GECF (using inverted gating: high vol -> MDCO)
        best_b, best_g = 0.5, 0.1
        m_gecf = AdaptiveGatedConsensus(beta_base=best_b, gamma=best_g).fit(X_tr, y_tr)
        
        # Inject impulsive noise on test set
        if noise_std > 0:
            for i_sample in range(len(X_te)):
                if np.random.rand() < 0.25: # 25% chance of severe electrode pop / outlier
                    corrupt_col = np.random.randint(0, X_te.shape[1])
                    feat_std = np.std(X_tr[:, corrupt_col]) + 1e-12
                    # Large outlier spike (positive or negative)
                    spike = np.random.choice([-1, 1]) * noise_std * feat_std
                    X_te[i_sample, corrupt_col] += spike
                    
        p_lin = m_lin.predict(X_te)
        p_mdco = m_mdco.predict(X_te)
        
        # Inverted T-GECF prediction:
        lin_norm = (m_gecf.linear_model.predict(X_te) - m_gecf.train_score_stats_["lin_mean"]) / m_gecf.train_score_stats_["lin_std"]
        mdco_norm = (m_gecf.mdco_model.predict(X_te) - m_gecf.train_score_stats_["mdco_mean"]) / m_gecf.train_score_stats_["mdco_std"]
        vols = m_gecf._compute_local_volatility(X_te)
        vol_z = (vols - m_gecf.train_volatility_stats_["vol_mean"]) / m_gecf.train_volatility_stats_["vol_std"]
        betas = np.clip(best_b + best_g * vol_z, 0.0, 1.0) # high vol -> higher beta -> weight on MDCO
        p_gecf = betas * mdco_norm + (1.0 - betas) * lin_norm
        
        aucs_lin.append(roc_auc_score(y_te, p_lin))
        aucs_gecf.append(roc_auc_score(y_te, p_gecf))
        aucs_mdco.append(roc_auc_score(y_te, p_mdco))
        
    print(f"Noise {noise_std:4.1f} std: Linear C = {np.mean(aucs_lin):.4f} | Inverted T-GECF = {np.mean(aucs_gecf):.4f} | MDCO D = {np.mean(aucs_mdco):.4f}")

# --- POINT 9: CONTIGUOUS BURST DROPOUT TEST ---
print("\n--- POINT 9: CONTIGUOUS BURST DROPOUT TEST ---")
for burst_min in [5, 10, 15, 30]:
    burst_blocks = burst_min * 15 # 4s blocks per minute: 60/4 = 15
    r_corrs = []
    aucs_burst = []
    for rep in range(10):
        np.random.seed(42 + rep)
        burst_rho1s = []
        for sid in subject_ids:
            npz_path = REPO_ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
            theta, indices = extract_clean_theta_series(npz_path)
            mean_theta = np.mean(theta, axis=1)
            
            # Select random burst start
            if len(mean_theta) > burst_blocks + 20:
                start = np.random.randint(0, len(mean_theta) - burst_blocks)
                mask = np.ones(len(mean_theta), dtype=bool)
                mask[start : start + burst_blocks] = False # drop burst
                
                th_sub = mean_theta[mask]
                idx_sub = indices[mask]
                
                adj = (idx_sub[1:] - idx_sub[:-1] == 1)
                x = th_sub[:-1][adj]
                y_s = th_sub[1:][adj]
                x_c = x - np.mean(x)
                y_c = y_s - np.mean(y_s)
                r_val = np.mean(x_c * y_c) / (np.std(x_c) * np.std(y_c) + 1e-12)
                burst_rho1s.append(r_val)
            else:
                burst_rho1s.append(0.0)
        r_corrs.append(pearsonr(rho1, burst_rho1s)[0])
        aucs_burst.append(roc_auc_score(labels, burst_rho1s))
    print(f"Continuous Burst Dropout {burst_min:2d} min: correlation r = {np.mean(r_corrs):.4f} ± {np.std(r_corrs):.4f}, Univariate AUROC = {np.mean(aucs_burst):.4f} ± {np.std(aucs_burst):.4f}")
