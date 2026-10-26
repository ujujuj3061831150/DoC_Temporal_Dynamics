"""
Run 10,000 Synchronous Joint Time-Shuffles with Frozen Intact Scale for Causal Counter-Example Testing.
Guarantees exact mathematical equivalence to full 6-channel synchronous shuffling while vectorizing the 1D spatial mean series.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
import time

CURRENT = Path(__file__).resolve()
for parent in [CURRENT] + list(CURRENT.parents):
    if (parent / "results" / "multiscale_matched_v2").exists() or (parent / ".git").exists():
        REPO_ROOT = parent
        break
else:
    REPO_ROOT = CURRENT.parents[1]

V3_ROOT = REPO_ROOT / "v3" if (REPO_ROOT / "v3").exists() else REPO_ROOT
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(REPO_ROOT))
if (V3_ROOT / "src").exists():
    sys.path.insert(0, str(V3_ROOT / "src"))
elif (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.temporal_dynamics.operators import extract_clean_theta_series, safe_corr
from src.temporal_dynamics.consensus import SignedStandardizedMean, TemporalMDCO
from sklearn.metrics import roc_auc_score

print("[1/4] Loading Mendeley dataset and fitted folds...")
fitted_states_file = REPO_ROOT / "results/multiscale_matched_v2/fitted_states.json"
all_states = json.loads(fitted_states_file.read_text(encoding="utf-8"))
folds = [r for r in all_states if r.get("cohort") == "mendeley" and r.get("method") == "C"]

features_json = REPO_ROOT / "results/multiscale_matched_v2/features.json"
records = [r for r in json.loads(features_json.read_text(encoding="utf-8")) if r["cohort"] == "mendeley"]

subject_ids = [r["subject_id"] for r in records]
labels_dict = {r["subject_id"]: int(r["label"]) for r in records}
canonical_dict = {r["subject_id"]: np.array(r["new"], dtype=float) for r in records}

# Load raw series and precompute 1D spatial mean g and adjacency
print("[2/4] Loading raw EEG series across 41 patients...")
series_data = {}
intact_features = {}

for sid in subject_ids:
    npz_path = REPO_ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
    theta, indices = extract_clean_theta_series(npz_path)
    g = theta.mean(axis=1)
    adjacent = (np.diff(indices) == 1)
    mu = float(g.mean())
    series_data[sid] = {
        "g": g,
        "n_blocks": len(g),
        "adjacent": adjacent,
        "mu": mu,
    }
    
    # Intact lag1 and jump
    g_curr = g[:-1][adjacent]
    g_next = g[1:][adjacent]
    lag1 = safe_corr(g_curr, g_next)
    jump = -float(np.abs(np.diff(g)[adjacent]).mean() / (mu + 1e-12))
    
    c_glob, c_spat, cv, r_inter = canonical_dict[sid]
    intact_features[sid] = {
        "canonical": np.array([c_glob, c_spat, cv, r_inter]),
        "plus_lag1": np.array([c_glob, c_spat, cv, r_inter, lag1]),
        "plus_jump": np.array([c_glob, c_spat, cv, r_inter, jump]),
    }

# Compute intact AUROCs & fit intact scalers
intact_aucs = {"plus_lag1_C": [], "plus_lag1_D": [], "plus_jump_C": [], "plus_jump_D": []}
intact_scalers_l1 = {}
intact_scalers_jump = {}

for r in folds:
    seed, fold = r["seed"], r["fold"]
    tr_sids, te_sids = r["train_ids"], r["test_ids"]
    y_tr = np.array([labels_dict[sid] for sid in tr_sids])
    y_te = np.array([labels_dict[sid] for sid in te_sids])

    # Lag1
    X_tr_l1 = np.array([intact_features[s]["plus_lag1"] for s in tr_sids])
    X_te_l1 = np.array([intact_features[s]["plus_lag1"] for s in te_sids])
    m_c_l1 = SignedStandardizedMean().fit(X_tr_l1, y_tr)
    intact_aucs["plus_lag1_C"].append(roc_auc_score(y_te, m_c_l1.predict(X_te_l1)))
    intact_scalers_l1[(seed, fold)] = (m_c_l1.scaler.mean_, m_c_l1.scaler.scale_, m_c_l1.signs_)

    m_d_l1 = TemporalMDCO(p=1.0).fit(X_tr_l1, y_tr)
    intact_aucs["plus_lag1_D"].append(roc_auc_score(y_te, m_d_l1.predict(X_te_l1)))

    # Jump
    X_tr_j = np.array([intact_features[s]["plus_jump"] for s in tr_sids])
    X_te_j = np.array([intact_features[s]["plus_jump"] for s in te_sids])
    m_c_j = SignedStandardizedMean().fit(X_tr_j, y_tr)
    intact_aucs["plus_jump_C"].append(roc_auc_score(y_te, m_c_j.predict(X_te_j)))
    intact_scalers_jump[(seed, fold)] = (m_c_j.scaler.mean_, m_c_j.scaler.scale_, m_c_j.signs_)

    m_d_j = TemporalMDCO(p=1.0).fit(X_tr_j, y_tr)
    intact_aucs["plus_jump_D"].append(roc_auc_score(y_te, m_d_j.predict(X_te_j)))

c_real = np.mean(intact_aucs["plus_lag1_C"])
d_real = np.mean(intact_aucs["plus_lag1_D"])
j_real = np.mean(intact_aucs["plus_jump_C"])

print(f"Intact plus_lag1 C: {c_real:.4f}")
print(f"Intact plus_lag1 D: {d_real:.4f}")
print(f"Intact plus_jump C: {j_real:.4f}")

# Run 10,000 synchronous permutations with frozen intact scale
rng = np.random.default_rng(20261014)
N_PERM = 10000
print(f"\n[3/4] Running {N_PERM} synchronous joint time-shuffles with frozen scale...")

t0 = time.time()
shuffle_rows = []

# Pre-extract arrays for speed
fold_data = []
for r in folds:
    seed, fold = r["seed"], r["fold"]
    tr_sids, te_sids = r["train_ids"], r["test_ids"]
    y_tr = np.array([labels_dict[sid] for sid in tr_sids])
    y_te = np.array([labels_dict[sid] for sid in te_sids])
    mu_l1, sc_l1, sg_l1 = intact_scalers_l1[(seed, fold)]
    mu_j, sc_j, sg_j = intact_scalers_jump[(seed, fold)]
    fold_data.append({
        "tr_sids": tr_sids,
        "te_sids": te_sids,
        "y_tr": y_tr,
        "y_te": y_te,
        "mu_l1": mu_l1, "sc_l1": sc_l1, "sg_l1": sg_l1,
        "mu_j": mu_j, "sc_j": sc_j, "sg_j": sg_j,
    })

for rep in range(N_PERM):
    shuffled_l1 = {}
    shuffled_jump = {}
    
    for sid in subject_ids:
        sdata = series_data[sid]
        g = sdata["g"]
        adjacent = sdata["adjacent"]
        mu = sdata["mu"]
        
        # Synchronous permute across blocks
        perm = rng.permutation(sdata["n_blocks"])
        g_perm = g[perm]
        
        lag1 = safe_corr(g_perm[:-1][adjacent], g_perm[1:][adjacent])
        jump = -float(np.abs(np.diff(g_perm)[adjacent]).mean() / (mu + 1e-12))
        
        c_glob, c_spat, cv, r_inter = canonical_dict[sid]
        shuffled_l1[sid] = np.array([c_glob, c_spat, cv, r_inter, lag1])
        shuffled_jump[sid] = np.array([c_glob, c_spat, cv, r_inter, jump])

    fold_aucs_c_l1 = []
    fold_aucs_d_l1 = []
    fold_aucs_c_j = []
    fold_aucs_d_j = []

    for fd in fold_data:
        tr_sids, te_sids = fd["tr_sids"], fd["te_sids"]
        y_tr, y_te = fd["y_tr"], fd["y_te"]

        X_tr_l1 = np.array([shuffled_l1[s] for s in tr_sids])
        X_te_l1 = np.array([shuffled_l1[s] for s in te_sids])
        X_tr_j = np.array([shuffled_jump[s] for s in tr_sids])
        X_te_j = np.array([shuffled_jump[s] for s in te_sids])

        # Frozen intact scale for Linear C (Lag1)
        z_te_l1 = (X_te_l1 - fd["mu_l1"]) / fd["sc_l1"] * fd["sg_l1"]
        pred_c_l1 = np.mean(z_te_l1, axis=1)
        fold_aucs_c_l1.append(roc_auc_score(y_te, pred_c_l1))

        # MDCO D (Lag1)
        m_d_l1 = TemporalMDCO(p=1.0).fit(X_tr_l1, y_tr)
        pred_d_l1 = m_d_l1.predict(X_te_l1)
        fold_aucs_d_l1.append(roc_auc_score(y_te, pred_d_l1))

        # Frozen intact scale for Linear C (Jump)
        z_te_j = (X_te_j - fd["mu_j"]) / fd["sc_j"] * fd["sg_j"]
        pred_c_j = np.mean(z_te_j, axis=1)
        fold_aucs_c_j.append(roc_auc_score(y_te, pred_c_j))

        # MDCO D (Jump)
        m_d_j = TemporalMDCO(p=1.0).fit(X_tr_j, y_tr)
        pred_d_j = m_d_j.predict(X_te_j)
        fold_aucs_d_j.append(roc_auc_score(y_te, pred_d_j))

    row = {
        "repeat": rep,
        "plus_lag1__C": float(np.mean(fold_aucs_c_l1)),
        "plus_lag1__D": float(np.mean(fold_aucs_d_l1)),
        "plus_increment__C": float(np.mean(fold_aucs_c_j)),
        "plus_increment__D": float(np.mean(fold_aucs_d_j)),
    }
    shuffle_rows.append(row)

    if (rep + 1) % 2000 == 0:
        elapsed = time.time() - t0
        print(f"Iteration {rep+1}/{N_PERM} ({elapsed:.1f}s):")
        print(f"  Linear C mean: {np.mean([r['plus_lag1__C'] for r in shuffle_rows]):.4f}, max: {np.max([r['plus_lag1__C'] for r in shuffle_rows]):.4f}")
        print(f"  MDCO D mean:   {np.mean([r['plus_lag1__D'] for r in shuffle_rows]):.4f}, max: {np.max([r['plus_lag1__D'] for r in shuffle_rows]):.4f}")

df_10000 = pd.DataFrame(shuffle_rows)
out_dir = (REPO_ROOT / "v3/results/temporal_dynamics_v1") if (REPO_ROOT / "v3/results/temporal_dynamics_v1").exists() else (REPO_ROOT / "results/temporal_dynamics_v1")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "shuffle_summary.csv"
out_csv_10000 = out_dir / "shuffle_summary_10000.csv"
df_10000.to_csv(out_csv, index=False)
df_10000.to_csv(out_csv_10000, index=False)
print(f"[4/4] Saved 10,000 permutation results to {out_csv} and {out_csv_10000}!")

c_shuff = df_10000["plus_lag1__C"].values
d_shuff = df_10000["plus_lag1__D"].values
j_shuff = df_10000["plus_increment__C"].values

p_c = (np.sum(c_shuff >= c_real) + 1) / (N_PERM + 1)
p_d = (np.sum(d_shuff >= d_real) + 1) / (N_PERM + 1)
p_j = (np.sum(j_shuff >= j_real) + 1) / (N_PERM + 1)

print("\n=== FINAL 10,000 PERMUTATION SUMMARY STATISTICS ===")
print(f"Plus Lag-1 Linear C (Real={c_real:.4f}): Shuffled {np.mean(c_shuff):.4f} +/- {np.std(c_shuff):.4f}, Range: [{np.min(c_shuff):.4f}, {np.max(c_shuff):.4f}], p = {p_c:.6f} ({np.sum(c_shuff >= c_real)}/{N_PERM})")
print(f"Plus Lag-1 MDCO D   (Real={d_real:.4f}): Shuffled {np.mean(d_shuff):.4f} +/- {np.std(d_shuff):.4f}, Range: [{np.min(d_shuff):.4f}, {np.max(d_shuff):.4f}], p = {p_d:.6f} ({np.sum(d_shuff >= d_real)}/{N_PERM})")
print(f"Plus Jump Linear C  (Real={j_real:.4f}): Shuffled {np.mean(j_shuff):.4f} +/- {np.std(j_shuff):.4f}, Range: [{np.min(j_shuff):.4f}, {np.max(j_shuff):.4f}], p = {p_j:.6f} ({np.sum(j_shuff >= j_real)}/{N_PERM})")
