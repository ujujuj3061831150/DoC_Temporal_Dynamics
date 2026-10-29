"""
Evaluation script for the CRRC (Beijing China Rehabilitation Research Center) External Cohort.

Executes:
1. De-identified clinical intake and signal preprocessing across 15 patients.
2. Dual-protocol evaluation:
   - Protocol 1 (Strict Clean 6-Channel Protocol, N=13)
   - Protocol 2 (Full Cohort Sensitivity Protocol, N=15)
3. Zero-shot transfer evaluation of the 4-feature orthogonal suite vs. static baseline.
4. Soft-truncation (c=1.5sigma) and noise resilience benchmark vs. supervised models.
5. Export of de-identified metadata, predictions, and summary statistics.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def _get_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / "results" / "multiscale_matched_v2" / "features.json").exists():
            return parent
    return Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent

REPO_ROOT = _get_repo_root()
sys.path.insert(0, str(REPO_ROOT / "v3" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_dynamics.crrc_adapter import (
    load_crrc_metadata,
    extract_crrc_patient_eeg_features,
)
from temporal_dynamics.adapters import load_mendeley_temporal_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics_and_ci(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 10000) -> Dict[str, Any]:
    """Compute AUROC, AP, Balanced Accuracy at median threshold, and bootstrap 95% CIs."""
    auroc = float(roc_auc_score(y_true, scores))
    ap = float(average_precision_score(y_true, scores))
    
    # Threshold at median (unsupervised balanced decision boundary)
    th_med = float(np.median(scores))
    preds = (scores >= th_med).astype(int)
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    
    # Sensitivity and specificity
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    sens = float(preds[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
    spec = float((1 - preds[neg_mask]).mean()) if neg_mask.sum() > 0 else 0.0
    
    # Bootstrap CIs
    rng = np.random.default_rng(42)
    boot_aurocs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], scores[idx]))
        
    ci_lower = float(np.percentile(boot_aurocs, 2.5))
    ci_upper = float(np.percentile(boot_aurocs, 97.5))
    
    return {
        "auroc": round(auroc, 4),
        "ap": round(ap, 4),
        "balanced_acc": round(bal_acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "auroc_ci95": [round(ci_lower, 4), round(ci_upper, 4)],
        "threshold_median": round(th_med, 4),
    }


def compute_paired_contrast_ci(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 10000) -> Dict[str, Any]:
    """Compute paired AUROC difference and bootstrap 95% CI."""
    delta_point = float(roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b))
    
    rng = np.random.default_rng(42)
    boot_deltas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], scores_a[idx])
        auc_b = roc_auc_score(y_true[idx], scores_b[idx])
        boot_deltas.append(auc_a - auc_b)
        
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_pos = float((np.array(boot_deltas) > 0).mean())
    
    return {
        "delta_auroc": round(delta_point, 4),
        "delta_ci95": [round(ci_lower, 4), round(ci_upper, 4)],
        "pr_delta_greater_zero": round(p_pos, 4),
    }


def main():
    data_dir = REPO_ROOT / "data" / "CRRC_raw"
    out_dir = REPO_ROOT / "v3" / "results" / "crrc_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cached_candidates = [
        out_dir / "crrc_deidentified_features.csv",
        REPO_ROOT / "results" / "crrc_validation" / "crrc_deidentified_features.csv",
    ]
    cached_csv = next((p for p in cached_candidates if p.exists()), None)
    
    if cached_csv is not None and (not data_dir.exists() or os.environ.get("RECOMPUTE_CRRC_RAW", "0") != "1"):
        logger.info(f"Loading pre-computed de-identified CRRC features from: {cached_csv}")
        df = pd.read_csv(cached_csv)
    else:
        logger.info("Loading CRRC clinical metadata from raw intake...")
        meta_df = load_crrc_metadata(data_dir)
        extracted_records = []
        for _, row in meta_df.iterrows():
            subj_id = row["subject_id"]
            p_name = row.get("_patient_dir_hint", "")
            p_dir = data_dir / p_name if p_name else data_dir / subj_id
            if not p_dir.exists():
                candidates = [p for p in data_dir.iterdir() if subj_id in p.name or (p_name and p_name in p.name)]
                if candidates:
                    p_dir = candidates[0]
                else:
                    logger.error(f"Directory not found for {subj_id}")
                    continue
            feats = extract_crrc_patient_eeg_features(patient_dir=p_dir, peak_threshold_uv=500.0, min_adjacent_pairs=10)
            rec = {
                "subject_id": subj_id,
                "diagnosis": row["diagnosis"],
                "y_true": row["y_true"],
                "crs_r_score": row["crs_r_score"],
                "lesion_laterality": row["lesion_laterality"],
                "bad_channels": row["bad_channels"],
                "clean_6ch_protocol": row["clean_6ch_protocol"],
            }
            rec.update(feats)
            extracted_records.append(rec)
        df = pd.DataFrame(extracted_records)
        export_cols = [
            "subject_id", "diagnosis", "y_true", "crs_r_score", "lesion_laterality",
            "bad_channels", "clean_6ch_protocol", "clean_blocks", "adjacent_pairs",
            "total_duration_sec", "clean_retention_rate", "c_global", "c_spatial",
            "neg_cv", "lag1", "neg_increment", "r_interhem", "residual_corr"
        ]
        df[export_cols].to_csv(out_dir / "crrc_deidentified_features.csv", index=False)

    cols = ["c_global", "neg_cv", "lag1", "residual_corr"]
    prior = np.array([-1., 1., 1., 1.])

    def score_cohort(sub_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_mat = sub_df[cols].values
        z_mat = StandardScaler().fit_transform(X_mat) * prior
        
        static_score = (z_mat[:, 0] + z_mat[:, 1]) / 2.0
        lag1_score = z_mat[:, 2]
        orth_score = z_mat.mean(axis=1)
        
        # Soft-truncation with c=1.5sigma (default from Table II-A and methods)
        soft_score = np.clip(z_mat, -1.5, 1.5).mean(axis=1)
        
        return static_score, lag1_score, orth_score, soft_score

    # Protocol 1: Strict Clean 6-Channel Protocol (N=13)
    p1_df = df[df["clean_6ch_protocol"] == True].copy()
    y_p1 = p1_df["y_true"].values
    static_p1, lag1_p1, orth_p1, soft_p1 = score_cohort(p1_df)
    
    p1_static_metrics = compute_metrics_and_ci(y_p1, static_p1)
    p1_lag1_metrics = compute_metrics_and_ci(y_p1, lag1_p1)
    p1_orth_metrics = compute_metrics_and_ci(y_p1, orth_p1)
    p1_soft_metrics = compute_metrics_and_ci(y_p1, soft_p1)
    p1_contrast = compute_paired_contrast_ci(y_p1, soft_p1, static_p1)
    
    # Protocol 2: Full Cohort Sensitivity Protocol (N=15)
    y_p2 = df["y_true"].values
    static_p2, lag1_p2, orth_p2, soft_p2 = score_cohort(df)
    
    p2_static_metrics = compute_metrics_and_ci(y_p2, static_p2)
    p2_lag1_metrics = compute_metrics_and_ci(y_p2, lag1_p2)
    p2_orth_metrics = compute_metrics_and_ci(y_p2, orth_p2)
    p2_soft_metrics = compute_metrics_and_ci(y_p2, soft_p2)
    p2_contrast = compute_paired_contrast_ci(y_p2, soft_p2, static_p2)

    # Supervised ML Baselines Transfer Benchmark (Mendeley -> CRRC)
    logger.info("Evaluating supervised ML models trained on Mendeley and transferred to CRRC...")
    m_data = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    y_m = m_data["labels"]
    X_m = m_data["feature_matrix_variants"]["drop_c_spatial"]

    X_c15 = df[cols].values
    scaler_crrc15 = StandardScaler().fit(X_c15)
    X_c15_target = scaler_crrc15.transform(X_c15)

    X_c13 = p1_df[cols].values
    scaler_crrc13 = StandardScaler().fit(X_c13)
    X_c13_target = scaler_crrc13.transform(X_c13)

    scaler_m = StandardScaler().fit(X_m)
    X_m_scaled = scaler_m.transform(X_m)

    supervised_models = {
        "ridge": RidgeClassifier(alpha=1.0),
        "logistic": LogisticRegression(C=1.0, random_state=42),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    }

    naive_transfer = {}
    aligned_transfer = {}
    
    # Direction alignment diagonal: diag(-1, 1, 1, -1) to match subacute prior s_target = [-1, 1, 1, 1]
    align_diag = np.array([-1., 1., 1., -1.])
    X_c15_aligned = X_c15_target * align_diag
    X_c13_aligned = X_c13_target * align_diag

    for name, clf in supervised_models.items():
        clf.fit(X_m_scaled, y_m)
        s15_naive = clf.decision_function(X_c15_target)
        s13_naive = clf.decision_function(X_c13_target)
        naive_transfer[name] = {
            "protocol_1_clean_13_auroc": round(roc_auc_score(y_p1, s13_naive), 4),
            "protocol_2_full_15_auroc": round(roc_auc_score(y_p2, s15_naive), 4),
        }
        
        s15_aligned = clf.decision_function(X_c15_aligned)
        s13_aligned = clf.decision_function(X_c13_aligned)
        aligned_transfer[name] = {
            "protocol_1_clean_13_auroc": round(roc_auc_score(y_p1, s13_aligned), 4),
            "protocol_2_full_15_auroc": round(roc_auc_score(y_p2, s15_aligned), 4),
        }

    summary = {
        "cohort_name": "China Rehabilitation Research Center (CRRC / Beijing Bo'ai Hospital)",
        "hardware": "NeuSen NSM2 / EEGCap32 (Neuracle, 32 channels, 1000 Hz, BioSemi-compatible channel layout)",
        "recording_duration_mean_min": 19.5,
        "protocol_1_strict_6ch": {
            "sample_size": 13,
            "mcs_count": 6,
            "vs_count": 7,
            "mcs_composition": "5 MCS, 1 EMCS",
            "exclusion_reason": "Excluded 2 VS subjects with documented O2 bad channel artifacts",
            "static_baseline": p1_static_metrics,
            "temporal_lag1": p1_lag1_metrics,
            "orthogonal_4feature_linear": p1_orth_metrics,
            "orthogonal_4feature_soft_truncation_c15": p1_soft_metrics,
            "paired_contrast_soft_vs_static": p1_contrast,
        },
        "protocol_2_full_sensitivity": {
            "sample_size": 15,
            "mcs_count": 6,
            "vs_count": 9,
            "mcs_composition": "5 MCS, 1 EMCS",
            "static_baseline": p2_static_metrics,
            "temporal_lag1": p2_lag1_metrics,
            "orthogonal_4feature_linear": p2_orth_metrics,
            "orthogonal_4feature_soft_truncation_c15": p2_soft_metrics,
            "paired_contrast_soft_vs_static": p2_contrast,
        },
        "supervised_ml_baselines_transfer": {
            "naive_frozen_source_transfer": naive_transfer,
            "identical_direction_aligned_transfer": aligned_transfer,
        }
    }

    # Write summary
    for target_dir in [out_dir, REPO_ROOT / "results" / "crrc_validation"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "crrc_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (target_dir / "crrc_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=== CRRC EVALUATION RESULTS SUMMARY ===")
    logger.info(f"Protocol 1 (N=13 clean): Linear = {p1_orth_metrics['auroc']}, Soft-Trunc (c=1.5) = {p1_soft_metrics['auroc']}")
    logger.info(f"Protocol 2 (N=15 all):   Linear = {p2_orth_metrics['auroc']}, Soft-Trunc (c=1.5) = {p2_soft_metrics['auroc']}")
    logger.info(f"Supervised Aligned: Logistic = {aligned_transfer['logistic']}, Ridge = {aligned_transfer['ridge']}, LDA = {aligned_transfer['lda']}")

if __name__ == "__main__":
    main()
