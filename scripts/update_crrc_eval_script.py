# -*- coding: utf-8 -*-
import sys
from pathlib import Path

path = Path("scripts/evaluate_crrc_cohort.py")
with open(path, "r", encoding="utf-8") as f:
    text = f.read().replace("\r\n", "\n")

# Check imports
if "from sklearn.linear_model import RidgeClassifier, LogisticRegression" not in text:
    old_imp = "from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score"
    new_imp = (
        "from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score\n"
        "from sklearn.linear_model import RidgeClassifier, LogisticRegression\n"
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from temporal_dynamics.adapters import load_mendeley_temporal_dataset"
    )
    assert old_imp in text
    text = text.replace(old_imp, new_imp)

# In main(), add supervised ML benchmark evaluation after p2_contrast
target_code = """    p2_contrast = compute_paired_contrast_ci(y_p2, orth_p2, static_p2)"""

supervised_code = """    p2_contrast = compute_paired_contrast_ci(y_p2, orth_p2, static_p2)

    # Supervised ML Baselines Transfer Benchmark (Mendeley -> CRRC)
    logger.info("Evaluating supervised ML models trained on Mendeley and zero-shot transferred to CRRC...")
    m_data = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    y_m = m_data["labels"]
    X_m = m_data["feature_matrix_variants"]["drop_c_spatial"]

    # Target-site standardized features for CRRC
    X_c15 = df[["c_global", "neg_cv", "lag1", "residual_corr"]].values
    scaler_crrc15 = StandardScaler().fit(X_c15)
    X_c15_target = scaler_crrc15.transform(X_c15)

    X_c13 = p1_df[["c_global", "neg_cv", "lag1", "residual_corr"]].values
    scaler_crrc13 = StandardScaler().fit(X_c13)
    X_c13_target = scaler_crrc13.transform(X_c13)

    scaler_m = StandardScaler().fit(X_m)
    X_m_scaled = scaler_m.transform(X_m)

    supervised_models = {
        "ridge": RidgeClassifier(alpha=1.0),
        "logistic": LogisticRegression(C=1.0, random_state=42),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    }

    supervised_results = {}
    for name, clf in supervised_models.items():
        clf.fit(X_m_scaled, y_m)
        if hasattr(clf, "decision_function"):
            s15 = clf.decision_function(X_c15_target)
            s13 = clf.decision_function(X_c13_target)
        else:
            s15 = clf.predict_proba(X_c15_target)[:, 1]
            s13 = clf.predict_proba(X_c13_target)[:, 1]
        
        auc13 = float(roc_auc_score(y_p1, s13))
        auc15 = float(roc_auc_score(y_p2, s15))
        supervised_results[name] = {
            "protocol_1_clean_13_auroc": round(auc13, 4),
            "protocol_2_full_15_auroc": round(auc15, 4),
        }
    logger.info(f"Supervised ML transfer collapse results: {supervised_results}")"""

assert target_code in text
text = text.replace(target_code, supervised_code)

# Add supervised_results to summary_report
old_summary = """        "protocol_2_full_sensitivity": {
            "sample_size": len(df),
            "mcs_count": int(df["y_true"].sum()),
            "vs_count": int((1 - df["y_true"]).sum()),
            "note": "Includes all 15 clinical records regardless of bad channels",
            "static_baseline": p2_static_metrics,
            "temporal_lag1": p2_lag1_metrics,
            "orthogonal_4feature_suite": p2_orth_metrics,
            "paired_contrast_orthogonal_vs_static": p2_contrast,
        }
    }"""

new_summary = """        "protocol_2_full_sensitivity": {
            "sample_size": len(df),
            "mcs_count": int(df["y_true"].sum()),
            "vs_count": int((1 - df["y_true"]).sum()),
            "note": "Includes all 15 clinical records regardless of bad channels",
            "static_baseline": p2_static_metrics,
            "temporal_lag1": p2_lag1_metrics,
            "orthogonal_4feature_suite": p2_orth_metrics,
            "paired_contrast_orthogonal_vs_static": p2_contrast,
        },
        "supervised_ml_baselines_transfer_collapse": supervised_results,
    }"""

assert old_summary in text
text = text.replace(old_summary, new_summary)

# Also write crrc_benchmark_summary.json as alias for compatibility
old_write = """    with open(out_dir / "crrc_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)"""

new_write = """    with open(out_dir / "crrc_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    with open(out_dir / "crrc_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)"""

assert old_write in text
text = text.replace(old_write, new_write)

with open(path, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(text)

print("Successfully updated scripts/evaluate_crrc_cohort.py")
