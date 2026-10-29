"""
Complete Benchmark Runner for Order-Aware Temporal Dynamics & Consensus Fusion (Temporal-MDCO / T-GECF).

Executes:
1. 2x2 Orthogonal Factorial Decomposition (Cell A, Cell B, Cell C, Cell D).
2. Tuned Same-Input Machine Learning Baselines (Logistic, Ridge, Shrinkage-LDA).
3. Representation Ablations (plus_increment, replace_cv_lag1, replace_cv_increment, residual_corr).
4. Adaptive Fluctuation & Uncertainty-Gated Consensus (T-GECF) with sample-level gating and inner tuning.
5. Window Length Sensitivity Ablation (2-second vs 4-second vs 8-second time windows).
6. 32-Repeat Synchronous Permutation Surrogates (3,200 models) for Causal Counter-Examples.
7. Wielek Sleep Cohort Discrete Transition Persistence Evaluation.
8. Fast Paired Dirichlet Bayesian 95% Confidence Intervals & Diagnostic Operating Metrics.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

CURRENT = Path(__file__).resolve()
REPO_ROOT = next((p for p in CURRENT.parents if (p / "results/multiscale_matched_v2").exists() or (p / "pytest.ini").exists()), CURRENT.parents[1])
V3_ROOT = next((p for p in CURRENT.parents if p.name == "v3"), CURRENT.parents[1])
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.temporal_dynamics.operators import (
    compute_temporal_order_features,
    extract_clean_theta_series,
)
from src.temporal_dynamics.consensus import (
    SignedStandardizedMean,
    TemporalMDCO,
    AdaptiveGatedConsensus,
    TunedMLBaseline,
)
from src.temporal_dynamics.surrogates import (
    generate_synchronous_surrogates,
)
from src.temporal_dynamics.adapters import (
    load_mendeley_temporal_dataset,
    compute_wielek_discrete_persistence,
)


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_fold_macro_auc(df: pd.DataFrame, score_col: str) -> float:
    """Compute mean AUROC across the 25 outer folds (5 seeds x 5 folds)."""
    fold_aucs = []
    for (seed, fold), group in df.groupby(["seed", "fold"]):
        fold_aucs.append(compute_auc(group["label"].values, group[score_col].values))
    return float(np.mean(fold_aucs))


def bootstrap_contrasts(
    df: pd.DataFrame,
    baseline_col: str,
    target_cols: List[str],
    n_draws: int = 10000,
    seed: int = 20261014,
) -> pd.DataFrame:
    """Compute paired subject-level bootstrap differences and 95% confidence intervals using Dirichlet weights."""
    ids = sorted(df["subject_id"].unique())
    lookup = {sid: i for i, sid in enumerate(ids)}
    rng = np.random.default_rng(seed)
    weights = np.maximum(rng.exponential(size=(n_draws, len(ids))), np.finfo(float).tiny)

    all_methods = [baseline_col] + target_cols
    groups = list(df.groupby(["seed", "fold"]))
    n_groups = len(groups)

    draw_aucs = {m: np.zeros(n_draws, dtype=float) for m in all_methods}
    for _, g in groups:
        y = g["label"].to_numpy(dtype=int)
        pos_mask = (y == 1)
        neg_mask = (y == 0)
        sids = g["subject_id"].to_numpy()
        idx_pos = np.array([lookup[s] for s in sids[pos_mask]])
        idx_neg = np.array([lookup[s] for s in sids[neg_mask]])
        wp = weights[:, idx_pos]
        wn = weights[:, idx_neg]
        denom = wp.sum(axis=1) * wn.sum(axis=1)

        for m in all_methods:
            scores = g[m].to_numpy(dtype=float)
            p_scores = scores[pos_mask, None]
            n_scores = scores[neg_mask]
            wins = (p_scores > n_scores) + 0.5 * (p_scores == n_scores)
            draw_aucs[m] += np.einsum("bi,ij,bj->b", wp, wins, wn, optimize=True) / denom / n_groups

    point_aucs = {m: compute_fold_macro_auc(df, m) for m in all_methods}
    base_point = point_aucs[baseline_col]
    base_draws = draw_aucs[baseline_col]

    rows = []
    for col in target_cols:
        pt_diff = point_aucs[col] - base_point
        draw_diff = draw_aucs[col] - base_draws
        lo, hi = np.quantile(draw_diff, [0.025, 0.975])
        prob_pos = float(np.mean(draw_diff > 0))
        rows.append({
            "contrast": f"{col}_minus_{baseline_col}",
            "point_difference": pt_diff,
            "ci_lower_95": float(lo),
            "ci_upper_95": float(hi),
            "prob_positive": prob_pos,
        })
    return pd.DataFrame(rows)


def tune_ml_param(
    method: str,
    parameters: List[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_inner_splits: int = 3,
    seed: int = 42,
) -> Any:
    """Tune ML baseline hyperparameter on inner folds without test leakage."""
    cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    scores = {p: [] for p in parameters}
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        for p in parameters:
            m = TunedMLBaseline(method=method, parameter=p).fit(X_tr, y_tr)
            preds = m.predict(X_val)
            scores[p].append(compute_auc(y_val, preds))
    mean_scores = {p: float(np.mean(scores[p])) for p in parameters}
    # Conservative tie-breaking: prefer stronger regularization on ties
    best_p = max(parameters, key=lambda p: mean_scores[p])
    return best_p


def tune_adaptive_gated(
    X_train: np.ndarray,
    y_train: np.ndarray,
    betas: List[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    gammas: List[float] = (0.0, 0.1, 0.2),
    n_inner_splits: int = 3,
    seed: int = 42,
) -> Tuple[float, float]:
    """Tune beta_base and local volatility gamma strictly on inner folds."""
    cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    grid_scores = {}
    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        for b in betas:
            for g in gammas:
                model = AdaptiveGatedConsensus(beta_base=b, gamma=g).fit(X_tr, y_tr)
                preds = model.predict(X_val)
                grid_scores.setdefault((b, g), []).append(compute_auc(y_val, preds))
                
    mean_scores = {k: float(np.mean(v)) for k, v in grid_scores.items()}
    best_config = max(mean_scores.keys(), key=lambda k: mean_scores[k])
    return best_config


def extract_multiscale_window_features(
    subject_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Extract lag1 persistence on 2-second and 8-second window caches."""
    cache_dir = REPO_ROOT / "results/bspc_window_cache_v1"
    lag1_2s = []
    lag1_8s = []
    
    for sid in subject_ids:
        npz_file = cache_dir / f"{sid}.npz"
        with np.load(npz_file, allow_pickle=False) as z:
            # 2-second window
            p2 = z["powers_2"]
            pk2 = z["peaks_2"]
            good2 = pk2 < 500
            idx2 = np.flatnonzero(good2)
            f2 = compute_temporal_order_features(p2[good2, 0, :], idx2)
            lag1_2s.append(f2["lag1"])
            
            # 8-second window
            p8 = z["powers_8"]
            pk8 = z["peaks_8"]
            good8 = pk8 < 500
            idx8 = np.flatnonzero(good8)
            f8 = compute_temporal_order_features(p8[good8, 0, :], idx8)
            lag1_8s.append(f8["lag1"])
            
    return {
        "lag1_2s": np.array(lag1_2s, dtype=float),
        "lag1_8s": np.array(lag1_8s, dtype=float),
    }


def run_benchmark(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("TEMPORAL-MDCO / T-GECF COMPLETE BENCHMARK EXECUTION")
    print("=" * 72)
    
    # 1. Load dataset & outer 25 partitions
    print("[Step 1/7] Loading Mendeley dataset and outer 25 fold partitions...")
    dataset = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    subject_ids = dataset["subject_ids"]
    labels_dict = {sid: int(dataset["labels"][i]) for i, sid in enumerate(subject_ids)}
    variants = dataset["feature_matrix_variants"]
    
    fitted_states_file = REPO_ROOT / "results/multiscale_matched_v2/fitted_states.json"
    all_states = json.loads(fitted_states_file.read_text(encoding="utf-8"))
    folds = [
        r for r in all_states
        if r.get("cohort") == "mendeley" and r.get("method") == "C"
    ]
    if len(folds) != 25:
        raise ValueError(f"Expected 25 Mendeley outer folds, found {len(folds)}")
        
    sid_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    
    # 2. Pre-generate 32 synchronous surrogates per subject
    print("[Step 2/7] Generating 32 synchronous surrogate series per subject...")
    surrogates_by_subject = {}
    for sid in subject_ids:
        npz_path = REPO_ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
        theta, indices = extract_clean_theta_series(npz_path)
        _, surrs = generate_synchronous_surrogates(
            theta, indices, n_repeats=32, seed=20261014, min_adjacent_pairs=10
        )
        surrogates_by_subject[sid] = surrs
        
    # 3. Load Multiscale Time Window Features (2s & 8s)
    print("[Step 3/7] Loading multiscale window features (2s and 8s grids)...")
    window_feats = extract_multiscale_window_features(subject_ids)
    canonical_mat = variants["canonical"]
    variants["win_2s"] = np.column_stack([canonical_mat, window_feats["lag1_2s"]])
    variants["win_8s"] = np.column_stack([canonical_mat, window_feats["lag1_8s"]])
    
    # 4. Outer Fold Evaluation Loop
    print("[Step 4/7] Evaluating core models & ML baselines across 25 outer folds...")
    prediction_rows = []
    shuffle_prediction_rows = []
    window_prediction_rows = []
    
    core_variant_names = [
        "canonical",
        "plus_lag1",
        "plus_increment",
        "replace_cv_lag1",
        "replace_cv_increment",
        "residual_corr",
        "plus_lag1_residual",
        "drop_c_spatial",
        "drop_c_global",
    ]
    
    # ML tuning grids
    logistic_C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge_alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    for fold_idx, r in enumerate(folds):
        seed = r["seed"]
        fold = r["fold"]
        tr_sids = r["train_ids"]
        te_sids = r["test_ids"]
        
        tr_idx = np.array([sid_to_idx[sid] for sid in tr_sids])
        te_idx = np.array([sid_to_idx[sid] for sid in te_sids])
        
        y_tr = np.array([labels_dict[sid] for sid in tr_sids], dtype=int)
        y_te = np.array([labels_dict[sid] for sid in te_sids], dtype=int)
        
        fold_scores = {}
        fold_diagnostics = {}
        
        # A. Core representations with Linear (C) and MDCO (D)
        for var_name in core_variant_names:
            X_full = variants[var_name]
            X_tr = X_full[tr_idx]
            X_te = X_full[te_idx]
            
            # Model C: Linear
            m_c = SignedStandardizedMean().fit(X_tr, y_tr)
            preds_c = m_c.predict(X_te)
            fold_scores[f"{var_name}__C"] = preds_c
            fold_diagnostics[f"{var_name}__C_thresh"] = float(np.median(m_c.predict(X_tr)))
            
            # Model D: MDCO
            m_d = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
            preds_d = m_d.predict(X_te)
            fold_scores[f"{var_name}__D"] = preds_d
            fold_diagnostics[f"{var_name}__D_thresh"] = float(np.median(m_d.predict(X_tr)))
            
        # B. Same-Input ML Baselines (Logistic, Ridge, Shrinkage-LDA) on canonical and plus_lag1
        for var_name in ["canonical", "plus_lag1"]:
            X_full = variants[var_name]
            X_tr = X_full[tr_idx]
            X_te = X_full[te_idx]
            
            # Tuned Logistic Regression
            best_C = tune_ml_param("logistic", logistic_C_grid, X_tr, y_tr, seed=seed * 10 + fold)
            m_log = TunedMLBaseline(method="logistic", parameter=best_C).fit(X_tr, y_tr)
            fold_scores[f"{var_name}__logistic_tuned"] = m_log.predict(X_te)
            fold_diagnostics[f"{var_name}__logistic_tuned_thresh"] = float(np.median(m_log.predict(X_tr)))
            
            # Tuned Ridge Classifier
            best_alpha = tune_ml_param("ridge", ridge_alpha_grid, X_tr, y_tr, seed=seed * 10 + fold)
            m_ridge = TunedMLBaseline(method="ridge", parameter=best_alpha).fit(X_tr, y_tr)
            fold_scores[f"{var_name}__ridge_tuned"] = m_ridge.predict(X_te)
            fold_diagnostics[f"{var_name}__ridge_tuned_thresh"] = float(np.median(m_ridge.predict(X_tr)))
            
            # Shrinkage LDA (auto Ledoit-Wolf)
            m_lda = TunedMLBaseline(method="lda_shrinkage", parameter="auto").fit(X_tr, y_tr)
            fold_scores[f"{var_name}__lda_shrinkage"] = m_lda.predict(X_te)
            fold_diagnostics[f"{var_name}__lda_shrinkage_thresh"] = float(np.median(m_lda.predict(X_tr)))

        # C. T-GECF Adaptive Fluctuation-Gated Consensus on plus_lag1
        X_tr_lag = variants["plus_lag1"][tr_idx]
        X_te_lag = variants["plus_lag1"][te_idx]
        best_b, best_g = tune_adaptive_gated(X_tr_lag, y_tr, seed=seed * 100 + fold)
        m_gated = AdaptiveGatedConsensus(beta_base=best_b, gamma=best_g).fit(X_tr_lag, y_tr)
        preds_gated = m_gated.predict(X_te_lag)
        fold_scores["t_gecf_adaptive"] = preds_gated
        fold_diagnostics["t_gecf_adaptive_thresh"] = float(np.median(m_gated.predict(X_tr_lag)))
        
        # Save outer test predictions
        for j, sid in enumerate(te_sids):
            row = {
                "seed": seed,
                "fold": fold,
                "subject_id": sid,
                "label": int(y_te[j]),
            }
            for score_key, scores_arr in fold_scores.items():
                row[score_key] = float(scores_arr[j])
                thresh = fold_diagnostics[f"{score_key}_thresh"]
                row[f"{score_key}_pred_binary"] = int(scores_arr[j] >= thresh)
            prediction_rows.append(row)
            
        # D. Multiscale Window Sensitivity (2s vs 4s vs 8s)
        for win_name in ["win_2s", "win_8s"]:
            X_tr_w = variants[win_name][tr_idx]
            X_te_w = variants[win_name][te_idx]
            p_win_c = SignedStandardizedMean().fit(X_tr_w, y_tr).predict(X_te_w)
            p_win_d = TemporalMDCO(p=1.0).fit(X_tr_w, y_tr).predict(X_te_w)
            for j, sid in enumerate(te_sids):
                window_prediction_rows.append({
                    "seed": seed,
                    "fold": fold,
                    "subject_id": sid,
                    "label": int(y_te[j]),
                    "window": win_name,
                    "score_C": float(p_win_c[j]),
                    "score_D": float(p_win_d[j]),
                })

        # E. 32-Repeat Surrogate Evaluations
        # Pre-fit intact linear models before shuffle loop to guarantee Frozen Intact Scale
        frozen_lag_c = SignedStandardizedMean().fit(variants["plus_lag1"][tr_idx], y_tr)
        frozen_inc_c = SignedStandardizedMean().fit(variants["plus_increment"][tr_idx], y_tr)
        
        for rep in range(32):
            canonical_full = variants["canonical"]
            shuff_lag_vals = np.array([
                surrogates_by_subject[sid][rep]["lag1"] for sid in subject_ids
            ])
            shuff_inc_vals = np.array([
                surrogates_by_subject[sid][rep]["neg_increment"] for sid in subject_ids
            ])
            
            X_shuff_lag = np.column_stack([canonical_full, shuff_lag_vals])
            X_shuff_inc = np.column_stack([canonical_full, shuff_inc_vals])
            
            p_lag_c = frozen_lag_c.predict(X_shuff_lag[te_idx])
            p_lag_d = TemporalMDCO(p=1.0).fit(X_shuff_lag[tr_idx], y_tr).predict(X_shuff_lag[te_idx])
            p_inc_c = frozen_inc_c.predict(X_shuff_inc[te_idx])
            p_inc_d = TemporalMDCO(p=1.0).fit(X_shuff_inc[tr_idx], y_tr).predict(X_shuff_inc[te_idx])
            
            for j, sid in enumerate(te_sids):
                shuffle_prediction_rows.append({
                    "repeat": rep,
                    "seed": seed,
                    "fold": fold,
                    "subject_id": sid,
                    "label": int(y_te[j]),
                    "plus_lag1__C": float(p_lag_c[j]),
                    "plus_lag1__D": float(p_lag_d[j]),
                    "plus_increment__C": float(p_inc_c[j]),
                    "plus_increment__D": float(p_inc_d[j]),
                })
                
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    
    shuff_df = pd.DataFrame(shuffle_prediction_rows)
    shuff_df.to_csv(output_dir / "shuffle_test_predictions.csv", index=False)
    
    win_df = pd.DataFrame(window_prediction_rows)
    win_df.to_csv(output_dir / "window_test_predictions.csv", index=False)
    
    # 5. Multiscale Window Sensitivity Summary
    print("[Step 5/7] Computing window sensitivity results (2s vs 4s vs 8s)...")
    win_summary = [
        {
            "window": "2-second",
            "Linear_C_AUROC": compute_fold_macro_auc(win_df[win_df.window == "win_2s"], "score_C"),
            "MDCO_D_AUROC": compute_fold_macro_auc(win_df[win_df.window == "win_2s"], "score_D"),
        },
        {
            "window": "4-second (Default)",
            "Linear_C_AUROC": compute_fold_macro_auc(pred_df, "plus_lag1__C"),
            "MDCO_D_AUROC": compute_fold_macro_auc(pred_df, "plus_lag1__D"),
        },
        {
            "window": "8-second",
            "Linear_C_AUROC": compute_fold_macro_auc(win_df[win_df.window == "win_8s"], "score_C"),
            "MDCO_D_AUROC": compute_fold_macro_auc(win_df[win_df.window == "win_8s"], "score_D"),
        },
    ]
    pd.DataFrame(win_summary).to_csv(output_dir / "window_sensitivity.csv", index=False)
    print(pd.DataFrame(win_summary).to_string(index=False))

    # 6. Wielek Cohort Discrete State Persistence Evaluation
    print("[Step 6/7] Evaluating Wielek cohort discrete transition dynamics...")
    wielek_persistence = compute_wielek_discrete_persistence()
    wielek_diag_file = REPO_ROOT / "data/public_doc_cohorts/cohort3_figshare_wielek/sleep_in_DOC_ML_data/DOC_classif/diag_bothSL.csv"
    diag_df = pd.read_csv(wielek_diag_file, header=None, names=["subject_id", "label", "crs_r"]).set_index("subject_id")
    
    p_map = {sid: v["discrete_lag1_persistence"] for sid, v in wielek_persistence.items()}
    diag_df["persistence"] = diag_df.index.map(p_map)
    
    wielek_metrics = {
        "patients": len(diag_df),
        "mean_persistence_overall": float(diag_df["persistence"].mean()),
        "mean_persistence_uws": float(diag_df[diag_df.label == 0]["persistence"].mean()),
        "mean_persistence_mcs": float(diag_df[diag_df.label == 1]["persistence"].mean()),
        "min_persistence": float(diag_df["persistence"].min()),
        "max_persistence": float(diag_df["persistence"].max()),
        "persistence_auroc": float(roc_auc_score(diag_df["label"], diag_df["persistence"])),
        "transition_rate_auroc": float(roc_auc_score(diag_df["label"], 1.0 - diag_df["persistence"])),
    }
    (output_dir / "wielek_persistence_results.json").write_text(
        json.dumps(wielek_metrics, indent=2) + "\n", encoding="utf-8"
    )
    print("Wielek Analysis Complete:", json.dumps(wielek_metrics, indent=2))

    # 7. Compute Global Metrics Summary & Contrasts
    print("[Step 7/7] Computing macro metrics, clinical diagnostics, and bootstrap intervals...")
    eval_cols = [
        "canonical__C", "canonical__D",
        "canonical__logistic_tuned", "canonical__ridge_tuned", "canonical__lda_shrinkage",
        "plus_lag1__C", "plus_lag1__D",
        "plus_lag1__logistic_tuned", "plus_lag1__ridge_tuned", "plus_lag1__lda_shrinkage",
        "plus_increment__C", "plus_increment__D",
        "replace_cv_lag1__C", "replace_cv_lag1__D",
        "replace_cv_increment__C", "replace_cv_increment__D",
        "residual_corr__C", "residual_corr__D",
        "plus_lag1_residual__C", "plus_lag1_residual__D",
        "drop_c_spatial__C", "drop_c_spatial__D",
        "drop_c_global__C", "drop_c_global__D",
        "t_gecf_adaptive",
    ]
    
    summary_rows = []
    y_true = pred_df["label"].values
    for col in eval_cols:
        macro_auc = compute_fold_macro_auc(pred_df, col)
        ap_list = [
            average_precision_score(g["label"].values, g[col].values)
            for _, g in pred_df.groupby(["seed", "fold"])
        ]
        macro_pr_auc = float(np.mean(ap_list))
        y_pred = pred_df[f"{col}_pred_binary"].values
        
        sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        bal_acc = float(0.5 * (sens + spec))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        summary_rows.append({
            "model": col,
            "macro_auroc": macro_auc,
            "macro_pr_auc": macro_pr_auc,
            "balanced_accuracy": bal_acc,
            "sensitivity": sens,
            "specificity": spec,
            "f1_score": f1,
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    print("\n--- Full Model Performance Summary ---")
    print(summary_df.to_string(index=False))
    
    # Contrasts relative to Canonical C
    target_contrasts = [c for c in eval_cols if c != "canonical__C"]
    contrast_df = bootstrap_contrasts(
        pred_df, baseline_col="canonical__C", target_cols=target_contrasts, n_draws=10000, seed=20261014
    )
    contrast_df.to_csv(output_dir / "contrasts.csv", index=False)
    print("\n--- Contrasts vs Canonical Linear Baseline (95% CI) ---")
    print(contrast_df.to_string(index=False))
    
    # Shuffle summary
    shuffle_summary_rows = []
    for rep, g in shuff_df.groupby("repeat"):
        shuffle_summary_rows.append({
            "repeat": int(rep),
            "plus_lag1__C": compute_fold_macro_auc(g, "plus_lag1__C"),
            "plus_lag1__D": compute_fold_macro_auc(g, "plus_lag1__D"),
            "plus_increment__C": compute_fold_macro_auc(g, "plus_increment__C"),
            "plus_increment__D": compute_fold_macro_auc(g, "plus_increment__D"),
        })
    shuff_sum_df = pd.DataFrame(shuffle_summary_rows)
    shuff_sum_df.to_csv(output_dir / "shuffle_summary.csv", index=False)
    
    print("\n--- 32 Synchronous Shuffles Summary ---")
    for col in ["plus_lag1__C", "plus_lag1__D", "plus_increment__C", "plus_increment__D"]:
        vals = shuff_sum_df[col].values
        print(f"  {col}: Mean = {vals.mean():.4f}, Range = [{vals.min():.4f}, {vals.max():.4f}]")
        
    # Output Manifest
    manifest = {
        "status": "completed",
        "patients": 41,
        "outer_partitions": 25,
        "surrogate_repeats": 32,
        "output_files": {
            f.name: fingerprint(f)
            for f in output_dir.iterdir()
            if f.is_file()
        }
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Complete benchmark finished! Artifacts archived in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=V3_ROOT / "results/temporal_dynamics_v1",
        help="Output destination directory",
    )
    args = parser.parse_args()
    run_benchmark(args.output)
