import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

CURRENT = Path(__file__).resolve()
V3_ROOT = CURRENT.parents[1]
REPO_ROOT = CURRENT.parents[2]
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.temporal_dynamics.operators import compute_temporal_order_features, extract_clean_theta_series
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

# Load raw series
print("[2/4] Loading raw EEG series across 41 patients...")
series_by_subject = {}
intact_features = {}
for sid in subject_ids:
    npz_path = REPO_ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
    theta, indices = extract_clean_theta_series(npz_path)
    series_by_subject[sid] = (theta, indices)
    feats = compute_temporal_order_features(theta, indices)
    c_glob, c_spat, cv, r_inter = canonical_dict[sid]
    lag1 = feats["lag1"]
    jump = feats["neg_increment"]
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

print(f"Intact plus_lag1 C: {np.mean(intact_aucs['plus_lag1_C']):.4f}")
print(f"Intact plus_lag1 D: {np.mean(intact_aucs['plus_lag1_D']):.4f}")
print(f"Intact plus_jump C: {np.mean(intact_aucs['plus_jump_C']):.4f}")
print(f"Intact plus_jump D: {np.mean(intact_aucs['plus_jump_D']):.4f}")

# Run 1000 synchronous permutations with frozen intact scale
rng = np.random.default_rng(20261014)
N_PERM = 1000
print(f"\n[3/4] Running {N_PERM} synchronous joint time-shuffles with frozen scale...")

shuffle_rows = []

for rep in range(N_PERM):
    shuffled_l1 = {}
    shuffled_jump = {}
    for sid in subject_ids:
        theta, indices = series_by_subject[sid]
        n_blocks = len(theta)
        perm = rng.permutation(n_blocks)
        theta_perm = theta[perm]
        feats = compute_temporal_order_features(theta_perm, indices)
        c_glob, c_spat, cv, r_inter = canonical_dict[sid]
        lag1 = feats["lag1"]
        jump = feats["neg_increment"]
        shuffled_l1[sid] = np.array([c_glob, c_spat, cv, r_inter, lag1])
        shuffled_jump[sid] = np.array([c_glob, c_spat, cv, r_inter, jump])

    fold_aucs_c_l1 = []
    fold_aucs_d_l1 = []
    fold_aucs_c_j = []
    fold_aucs_d_j = []

    for r in folds:
        seed, fold = r["seed"], r["fold"]
        tr_sids, te_sids = r["train_ids"], r["test_ids"]
        y_tr = np.array([labels_dict[sid] for sid in tr_sids])
        y_te = np.array([labels_dict[sid] for sid in te_sids])

        X_tr_l1 = np.array([shuffled_l1[s] for s in tr_sids])
        X_te_l1 = np.array([shuffled_l1[s] for s in te_sids])

        X_tr_j = np.array([shuffled_jump[s] for s in tr_sids])
        X_te_j = np.array([shuffled_jump[s] for s in te_sids])

        # Frozen intact scale for Linear C (Lag1)
        mu_intact, scale_intact, signs_intact = intact_scalers_l1[(seed, fold)]
        z_te_l1 = (X_te_l1 - mu_intact) / scale_intact * signs_intact
        pred_c_l1 = np.mean(z_te_l1, axis=1)
        fold_aucs_c_l1.append(roc_auc_score(y_te, pred_c_l1))

        # MDCO D (Lag1)
        m_d_l1 = TemporalMDCO(p=1.0).fit(X_tr_l1, y_tr)
        pred_d_l1 = m_d_l1.predict(X_te_l1)
        fold_aucs_d_l1.append(roc_auc_score(y_te, pred_d_l1))

        # Frozen intact scale for Linear C (Jump)
        mu_j, scale_j, signs_j = intact_scalers_jump[(seed, fold)]
        z_te_j = (X_te_j - mu_j) / scale_j * signs_j
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

    if (rep + 1) % 250 == 0:
        print(f"Iteration {rep+1}/{N_PERM}:")
        print(f"  Linear C mean: {np.mean([r['plus_lag1__C'] for r in shuffle_rows]):.4f}, max: {np.max([r['plus_lag1__C'] for r in shuffle_rows]):.4f}")
        print(f"  MDCO D mean:   {np.mean([r['plus_lag1__D'] for r in shuffle_rows]):.4f}, max: {np.max([r['plus_lag1__D'] for r in shuffle_rows]):.4f}")

df_1000 = pd.DataFrame(shuffle_rows)
out_csv = REPO_ROOT / "v3/results/temporal_dynamics_v1/shuffle_summary.csv"
out_csv_1000 = REPO_ROOT / "v3/results/temporal_dynamics_v1/shuffle_summary_1000.csv"
df_1000.to_csv(out_csv, index=False)
df_1000.to_csv(out_csv_1000, index=False)
print(f"[4/4] Saved 1,000 permutation results to {out_csv} and {out_csv_1000}!")

c_real = np.mean(intact_aucs["plus_lag1_C"])
d_real = np.mean(intact_aucs["plus_lag1_D"])
j_real = np.mean(intact_aucs["plus_jump_C"])

c_shuff = df_1000["plus_lag1__C"].values
d_shuff = df_1000["plus_lag1__D"].values
j_shuff = df_1000["plus_increment__C"].values

print("\n=== SUMMARY STATISTICS ===")
print(f"Plus Lag-1 Linear C (Real={c_real:.4f}): Shuffled {np.mean(c_shuff):.4f} +/- {np.std(c_shuff):.4f}, Range: [{np.min(c_shuff):.4f}, {np.max(c_shuff):.4f}], p = {(np.sum(c_shuff >= c_real) + 1)/(N_PERM+1):.4f}")
print(f"Plus Lag-1 MDCO D   (Real={d_real:.4f}): Shuffled {np.mean(d_shuff):.4f} +/- {np.std(d_shuff):.4f}, Range: [{np.min(d_shuff):.4f}, {np.max(d_shuff):.4f}], p = {(np.sum(d_shuff >= d_real) + 1)/(N_PERM+1):.4f}")
print(f"Plus Jump Linear C  (Real={j_real:.4f}): Shuffled {np.mean(j_shuff):.4f} +/- {np.std(j_shuff):.4f}, Range: [{np.min(j_shuff):.4f}, {np.max(j_shuff):.4f}], p = {(np.sum(j_shuff >= j_real) + 1)/(N_PERM+1):.4f}")
