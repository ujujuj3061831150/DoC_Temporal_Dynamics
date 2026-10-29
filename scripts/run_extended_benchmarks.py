"""
Extended Benchmarking Suite for Schemes 1, 2, and 4:
1. Scheme 1: 20 Seeds x 5 Folds (100 Partitions) for High-Power Variance Reduction.
2. Scheme 2: 10-Fold CV (100 Partitions) and Leave-One-Out (LOOCV, 41 Folds).
3. Scheme 4: Cost-Sensitive / Youden-Optimal Thresholding Fitted Strictly on Training Folds.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

CURRENT = Path(__file__).resolve()
V3_ROOT = CURRENT.parents[1]
REPO_ROOT = CURRENT.parents[2]
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.temporal_dynamics.adapters import load_mendeley_temporal_dataset
from src.temporal_dynamics.consensus import (
    SignedStandardizedMean,
    TemporalMDCO,
    AdaptiveGatedConsensus,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except Exception:
        pr_auc = 0.5
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float((sens + spec) / 2.0)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        "auroc": auc,
        "pr_auc": pr_auc,
        "balanced_acc": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def find_youden_threshold(tr_scores: np.ndarray, y_tr: np.ndarray) -> float:
    """Find threshold maximizing Youden's J = Sensitivity + Specificity - 1 strictly on training set."""
    candidates = np.unique(tr_scores)
    best_j = -1e9
    best_thresh = float(np.median(tr_scores))
    
    for t in candidates:
        pred = (tr_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_tr, pred, labels=[0, 1]).ravel()
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = se + sp - 1.0
        if j > best_j:
            best_j = j
            best_thresh = float(t)
    return best_thresh


def run_extended_benchmarks():
    print("=" * 72)
    print("STARTING EXTENDED BENCHMARKS: SCHEMES 1, 2, and 4")
    print("=" * 72)
    
    dataset = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    subject_ids = dataset["subject_ids"]
    labels = np.asarray(dataset["labels"], dtype=int)
    variants = dataset["feature_matrix_variants"]
    
    # Feature subsets to evaluate
    eval_variants = {
        "canonical": variants["canonical"],
        "plus_lag1": variants["plus_lag1"],
        "plus_increment": variants["plus_increment"],
        "replace_cv_lag1": variants["replace_cv_lag1"],
        "replace_cv_increment": variants["replace_cv_increment"],
        "residual_corr": variants["residual_corr"],
        "plus_lag1_residual": variants["plus_lag1_residual"],
        "drop_cspatial": variants["plus_lag1_residual"][:, [0, 2, 3, 4]],
        "drop_cglobal": variants["plus_lag1_residual"][:, [1, 2, 3, 4]],
    }
    
    # =========================================================================
    # PART 1: Scheme 1 (20 Seeds x 5 Folds = 100 Partitions) & Scheme 4 (Thresholds)
    # =========================================================================
    print("\n>>> [1/3] Running Scheme 1 (20 Seeds x 5 Folds = 100 Partitions) & Scheme 4...")
    seeds_20 = [42 + 100 * i for i in range(20)]
    
    results_records_100 = []
    
    for seed_idx, seed in enumerate(seeds_20):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(subject_ids, labels)):
            y_tr, y_te = labels[tr_idx], labels[te_idx]
            
            for var_name, X in eval_variants.items():
                X_tr, X_te = X[tr_idx], X[te_idx]
                
                # Model C: Linear
                m_c = SignedStandardizedMean().fit(X_tr, y_tr)
                tr_scores_c = m_c.predict(X_tr)
                te_scores_c = m_c.predict(X_te)
                
                # Default Median Threshold
                thresh_med_c = float(np.median(tr_scores_c))
                m_c_med = compute_metrics(y_te, te_scores_c, thresh_med_c)
                
                # Youden-Optimal Threshold (Scheme 4)
                thresh_youden_c = find_youden_threshold(tr_scores_c, y_tr)
                m_c_youden = compute_metrics(y_te, te_scores_c, thresh_youden_c)
                
                results_records_100.append({
                    "scheme": "20seeds_5fold",
                    "seed": seed,
                    "fold": fold,
                    "variant": var_name,
                    "model": "Linear_C",
                    "auroc": m_c_med["auroc"],
                    "pr_auc": m_c_med["pr_auc"],
                    "bal_acc_median": m_c_med["balanced_acc"],
                    "sens_median": m_c_med["sensitivity"],
                    "spec_median": m_c_med["specificity"],
                    "bal_acc_youden": m_c_youden["balanced_acc"],
                    "sens_youden": m_c_youden["sensitivity"],
                    "spec_youden": m_c_youden["specificity"],
                })
                
                # Model D: MDCO
                m_d = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
                tr_scores_d = m_d.predict(X_tr)
                te_scores_d = m_d.predict(X_te)
                
                thresh_med_d = float(np.median(tr_scores_d))
                m_d_med = compute_metrics(y_te, te_scores_d, thresh_med_d)
                thresh_youden_d = find_youden_threshold(tr_scores_d, y_tr)
                m_d_youden = compute_metrics(y_te, te_scores_d, thresh_youden_d)
                
                results_records_100.append({
                    "scheme": "20seeds_5fold",
                    "seed": seed,
                    "fold": fold,
                    "variant": var_name,
                    "model": "MDCO_D",
                    "auroc": m_d_med["auroc"],
                    "pr_auc": m_d_med["pr_auc"],
                    "bal_acc_median": m_d_med["balanced_acc"],
                    "sens_median": m_d_med["sensitivity"],
                    "spec_median": m_d_med["specificity"],
                    "bal_acc_youden": m_d_youden["balanced_acc"],
                    "sens_youden": m_d_youden["sensitivity"],
                    "spec_youden": m_d_youden["specificity"],
                })
                
                # T-GECF on plus_lag1 and plus_lag1_residual
                if var_name in ["plus_lag1", "plus_lag1_residual"]:
                    m_gecf = AdaptiveGatedConsensus(theta_0=1.645, kappa=2.0).fit(X_tr, y_tr)
                    tr_scores_g = m_gecf.predict(X_tr)
                    te_scores_g = m_gecf.predict(X_te)
                    
                    thresh_med_g = float(np.median(tr_scores_g))
                    m_g_med = compute_metrics(y_te, te_scores_g, thresh_med_g)
                    thresh_youden_g = find_youden_threshold(tr_scores_g, y_tr)
                    m_g_youden = compute_metrics(y_te, te_scores_g, thresh_youden_g)
                    
                    results_records_100.append({
                        "scheme": "20seeds_5fold",
                        "seed": seed,
                        "fold": fold,
                        "variant": var_name,
                        "model": "T_GECF",
                        "auroc": m_g_med["auroc"],
                        "pr_auc": m_g_med["pr_auc"],
                        "bal_acc_median": m_g_med["balanced_acc"],
                        "sens_median": m_g_med["sensitivity"],
                        "spec_median": m_g_med["specificity"],
                        "bal_acc_youden": m_g_youden["balanced_acc"],
                        "sens_youden": m_g_youden["sensitivity"],
                        "spec_youden": m_g_youden["specificity"],
                    })

    df_100 = pd.DataFrame(results_records_100)
    out_100_csv = REPO_ROOT / "v3/results/temporal_dynamics_v1/metrics_100_folds.csv"
    df_100.to_csv(out_100_csv, index=False)
    print(f"Saved 100-fold partitions results to {out_100_csv}!")

    # Aggregate by (variant, model)
    summary_100 = []
    for (var, model), g in df_100.groupby(["variant", "model"]):
        summary_100.append({
            "variant": var,
            "model": model,
            "auroc_mean": float(g["auroc"].mean()),
            "auroc_std": float(g["auroc"].std()),
            "pr_auc_mean": float(g["pr_auc"].mean()),
            "bal_acc_med_mean": float(g["bal_acc_median"].mean()),
            "sens_med_mean": float(g["sens_median"].mean()),
            "spec_med_mean": float(g["spec_median"].mean()),
            "bal_acc_youden_mean": float(g["bal_acc_youden"].mean()),
            "sens_youden_mean": float(g["sens_youden"].mean()),
            "spec_youden_mean": float(g["spec_youden"].mean()),
        })
    df_summary_100 = pd.DataFrame(summary_100)
    out_sum_csv = REPO_ROOT / "v3/results/temporal_dynamics_v1/metrics_100_folds_summary.csv"
    df_summary_100.to_csv(out_sum_csv, index=False)

    # =========================================================================
    # PART 2: Scheme 2 (10-Fold CV with 10 Seeds = 100 Partitions)
    # =========================================================================
    print("\n>>> [2/3] Running Scheme 2 (10-Fold CV with 10 Seeds = 100 Partitions)...")
    results_records_10fold = []
    
    for seed in seeds_20[:10]:
        skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for fold, (tr_idx, te_idx) in enumerate(skf10.split(subject_ids, labels)):
            y_tr, y_te = labels[tr_idx], labels[te_idx]
            for var_name in ["canonical", "plus_lag1", "plus_lag1_residual"]:
                X = eval_variants[var_name]
                X_tr, X_te = X[tr_idx], X[te_idx]
                
                m_c = SignedStandardizedMean().fit(X_tr, y_tr)
                auc_c = float(roc_auc_score(y_te, m_c.predict(X_te)))
                
                m_d = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
                auc_d = float(roc_auc_score(y_te, m_d.predict(X_te)))
                
                results_records_10fold.append({
                    "seed": seed, "fold": fold, "variant": var_name,
                    "Linear_C_auroc": auc_c, "MDCO_D_auroc": auc_d
                })
                
    df_10fold = pd.DataFrame(results_records_10fold)
    summary_10fold = {}
    for var, g in df_10fold.groupby("variant"):
        summary_10fold[var] = {
            "Linear_C_auroc": float(g["Linear_C_auroc"].mean()),
            "MDCO_D_auroc": float(g["MDCO_D_auroc"].mean()),
        }

    # =========================================================================
    # PART 3: Scheme 2b (Leave-One-Out LOOCV, 41 Folds)
    # =========================================================================
    print("\n>>> [3/3] Running Scheme 2b (Leave-One-Out LOOCV, 41 Folds)...")
    loo = LeaveOneOut()
    summary_loocv = {}
    
    for var_name in ["canonical", "plus_lag1", "plus_lag1_residual"]:
        X = eval_variants[var_name]
        loo_c_preds = np.zeros(len(subject_ids))
        loo_d_preds = np.zeros(len(subject_ids))
        
        for tr_idx, te_idx in loo.split(subject_ids, labels):
            y_tr, y_te = labels[tr_idx], labels[te_idx]
            X_tr, X_te = X[tr_idx], X[te_idx]
            
            m_c = SignedStandardizedMean().fit(X_tr, y_tr)
            loo_c_preds[te_idx[0]] = m_c.predict(X_te)[0]
            
            m_d = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
            loo_d_preds[te_idx[0]] = m_d.predict(X_te)[0]
            
        summary_loocv[var_name] = {
            "Linear_C_auroc": float(roc_auc_score(labels, loo_c_preds)),
            "MDCO_D_auroc": float(roc_auc_score(labels, loo_d_preds)),
        }

    # Final combined report
    final_output = {
        "scheme_1_100_folds_summary": summary_100,
        "scheme_2_10_fold_summary": summary_10fold,
        "scheme_2_loocv_summary": summary_loocv,
    }
    
    out_json = REPO_ROOT / "v3/results/temporal_dynamics_v1/extended_schemes_summary.json"
    out_json.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(f"\nALL EXTENDED BENCHMARKS COMPLETE! Results saved to {out_json}!")
    
    print("\n=== SUMMARY OF KEY RESULTS ===")
    print("100-Fold (20 Seeds x 5 Folds) Results:")
    for row in summary_100:
        if row["variant"] in ["canonical", "plus_lag1", "plus_lag1_residual"] and row["model"] == "Linear_C":
            print(f"  {row['variant']:20s} Linear C: AUROC={row['auroc_mean']:.4f}+/-{row['auroc_std']:.4f} | Med Thresh: Sens={row['sens_med_mean']:.3f}, Spec={row['spec_med_mean']:.3f} | Youden Thresh: Sens={row['sens_youden_mean']:.3f}, Spec={row['spec_youden_mean']:.3f}")
            
    print("\n10-Fold CV Results (100 Partitions):")
    for var, vals in summary_10fold.items():
        print(f"  {var:20s} Linear C={vals['Linear_C_auroc']:.4f}, MDCO D={vals['MDCO_D_auroc']:.4f}")
        
    print("\nLOOCV (41 Folds) Results:")
    for var, vals in summary_loocv.items():
        print(f"  {var:20s} Linear C={vals['Linear_C_auroc']:.4f}, MDCO D={vals['MDCO_D_auroc']:.4f}")


if __name__ == "__main__":
    run_extended_benchmarks()
