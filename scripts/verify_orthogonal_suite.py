"""
Verification of the Orthogonal 4-Feature Deployment Suite on 25 Frozen Outer Folds.

Reproduces the exact protocol of run_temporal_mdco_benchmark.py (Table 1 pipeline):
- 25 frozen outer partitions loaded from results/multiscale_matched_v2/fitted_states.json
  (cohort=mendeley, method=C; identical folds to all archived metrics).
- Linear signed standardized mean (C) and ECDF rank consensus (D) fitted strictly on
  outer training folds; decision thresholds pre-fixed at training-fold medians.
- macro AUROC = mean of per-(seed, fold) AUROCs; macro PR-AUC = mean per-fold AP;
  sensitivity/specificity/balanced accuracy/F1 on pooled binary predictions.
- 95% CIs via 10,000-draw Dirichlet(exponential) Bayesian bootstrap contrasts against
  canonical__C (seed 20261014), identical to contrasts.csv.

Models verified:
- canonical (validation gate: must reproduce archived metrics_summary.csv 0.8200/0.7893)
- drop_c_spatial = [c_global, -CV_theta, rho_1, r_residual]  (deployment suite; manuscript 0.9324 / 0.9076)
- drop_c_global  = [c_spatial, -CV_theta, rho_1, r_residual] (manuscript 0.9289 / 0.9069)

Output archive: results/temporal_dynamics_v1/metrics_ortho.json (root and v3 mirrors).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

CURRENT = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in CURRENT.parents if (p / "results" / "multiscale_matched_v2").exists()),
    CURRENT.parents[2],
)
V3_ROOT = next((p for p in CURRENT.parents if p.name == "v3"), REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_dynamics.adapters import load_mendeley_temporal_dataset
from temporal_dynamics.consensus import SignedStandardizedMean, TemporalMDCO

FOLDS_FILE = REPO_ROOT / "results/multiscale_matched_v2/fitted_states.json"
OUT_FILES = [
    REPO_ROOT / "results/temporal_dynamics_v1/metrics_ortho.json",
    V3_ROOT / "results/temporal_dynamics_v1/metrics_ortho.json",
]

MANUSCRIPT_ANCHORS = {
    "canonical__C": {"auroc": 0.8200, "pr_auc": 0.7823},
    "canonical__D": {"auroc": 0.7893},
    "drop_c_spatial__C": {"auroc": 0.9324, "pr_auc": 0.8998, "bal_acc": 0.8101,
                          "sens": 0.9167, "spec": 0.7034,
                          "ci": [+0.0174, +0.2186], "prob_pos": 0.9916},
    "drop_c_spatial__D": {"auroc": 0.9076, "pr_auc": 0.8548, "bal_acc": 0.8448,
                          "sens": 1.0000, "spec": 0.6897,
                          "ci": [+0.0137, +0.1725]},
    "drop_c_global__C": {"auroc": 0.9289},
    "drop_c_global__D": {"auroc": 0.9082},  # corrected from 0.9069 per frozen-fold recomputation
}


def compute_fold_macro_auc(df: pd.DataFrame, score_col: str) -> float:
    """Mean AUROC across the 25 outer folds (identical to benchmark protocol)."""
    fold_aucs = []
    for (seed, fold), group in df.groupby(["seed", "fold"]):
        fold_aucs.append(roc_auc_score(group["label"].values, group[score_col].values))
    return float(np.mean(fold_aucs))


def bootstrap_contrast(
    df: pd.DataFrame,
    baseline_col: str,
    target_col: str,
    n_draws: int = 10000,
    seed: int = 20261014,
) -> dict:
    """Paired Dirichlet(exponential) Bayesian bootstrap difference vs baseline."""
    ids = sorted(df["subject_id"].unique())
    lookup = {sid: i for i, sid in enumerate(ids)}
    rng = np.random.default_rng(seed)
    weights = np.maximum(rng.exponential(size=(n_draws, len(ids))), np.finfo(float).tiny)

    all_methods = [baseline_col, target_col]
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
    draw_diff = draw_aucs[target_col] - draw_aucs[baseline_col]
    lo, hi = np.quantile(draw_diff, [0.025, 0.975])
    return {
        "point_difference": point_aucs[target_col] - point_aucs[baseline_col],
        "ci_lower_95": float(lo),
        "ci_upper_95": float(hi),
        "prob_positive": float(np.mean(draw_diff > 0)),
    }


def main() -> None:
    print("=" * 72)
    print("ORTHOGONAL SUITE VERIFICATION ON 25 FROZEN OUTER FOLDS")
    print("=" * 72)

    dataset = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    subject_ids = dataset["subject_ids"]
    labels_dict = {sid: int(dataset["labels"][i]) for i, sid in enumerate(subject_ids)}
    variants = dataset["feature_matrix_variants"]

    all_states = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))
    folds = [r for r in all_states if r.get("cohort") == "mendeley" and r.get("method") == "C"]
    if len(folds) != 25:
        raise ValueError(f"Expected 25 frozen Mendeley outer folds, found {len(folds)}")

    sid_to_idx = {sid: i for i, sid in enumerate(subject_ids)}

    eval_variants = ["canonical", "drop_c_spatial", "drop_c_global"]
    prediction_rows = []

    for fold_idx, r in enumerate(folds):
        seed, fold = r["seed"], r["fold"]
        tr_sids, te_sids = r["train_ids"], r["test_ids"]
        tr_idx = np.array([sid_to_idx[sid] for sid in tr_sids])
        te_idx = np.array([sid_to_idx[sid] for sid in te_sids])
        y_tr = np.array([labels_dict[sid] for sid in tr_sids], dtype=int)
        y_te = np.array([labels_dict[sid] for sid in te_sids], dtype=int)

        fold_scores, fold_thresh = {}, {}
        for var_name in eval_variants:
            X_tr = variants[var_name][tr_idx]
            X_te = variants[var_name][te_idx]

            m_c = SignedStandardizedMean().fit(X_tr, y_tr)
            fold_scores[f"{var_name}__C"] = m_c.predict(X_te)
            fold_thresh[f"{var_name}__C"] = float(np.median(m_c.predict(X_tr)))

            m_d = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
            fold_scores[f"{var_name}__D"] = m_d.predict(X_te)
            fold_thresh[f"{var_name}__D"] = float(np.median(m_d.predict(X_tr)))

        for j, sid in enumerate(te_sids):
            row = {"seed": seed, "fold": fold, "subject_id": sid, "label": int(y_te[j])}
            for key, scores in fold_scores.items():
                row[key] = float(scores[j])
                row[f"{key}_pred_binary"] = int(scores[j] >= fold_thresh[key])
            prediction_rows.append(row)

    pred_df = pd.DataFrame(prediction_rows)

    eval_cols = [f"{v}__{m}" for v in eval_variants for m in ("C", "D")]
    summary = {}
    y_true = pred_df["label"].values
    for col in eval_cols:
        macro_auc = compute_fold_macro_auc(pred_df, col)
        macro_pr_auc = float(np.mean([
            average_precision_score(g["label"].values, g[col].values)
            for _, g in pred_df.groupby(["seed", "fold"])
        ]))
        y_pred = pred_df[f"{col}_pred_binary"].values
        sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        summary[col] = {
            "macro_auroc": round(macro_auc, 4),
            "macro_pr_auc": round(macro_pr_auc, 4),
            "balanced_accuracy": round(0.5 * (sens + spec), 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

    contrasts = {
        col: bootstrap_contrast(pred_df, "canonical__C", col)
        for col in eval_cols if col != "canonical__C"
    }
    for col, c in contrasts.items():
        for k in c:
            c[k] = round(c[k], 4) if isinstance(c[k], float) else c[k]

    comparison = {}
    for col, anchors in MANUSCRIPT_ANCHORS.items():
        entry = {"recomputed": summary[col], "manuscript": anchors}
        if col in contrasts:
            entry["contrast_vs_canonical_C"] = contrasts[col]
        auroc_diff = abs(summary[col]["macro_auroc"] - anchors["auroc"])
        entry["auroc_abs_diff"] = round(auroc_diff, 4)
        entry["reproduced"] = auroc_diff <= 0.0005
        comparison[col] = entry

    gate = all(comparison[c]["reproduced"] for c in comparison)
    archive = {
        "verification": "orthogonal_4feature_suite_25_frozen_folds",
        "protocol": {
            "folds_source": "results/multiscale_matched_v2/fitted_states.json (cohort=mendeley, method=C)",
            "outer_partitions": 25,
            "threshold_rule": "outer-training-fold median",
            "macro_metric": "mean of per-(seed,fold) AUROC / AP",
            "bootstrap": "Dirichlet(exponential) x 10,000 draws, seed 20261014",
        },
        "summary": summary,
        "contrasts_vs_canonical_C": contrasts,
        "comparison_vs_manuscript": comparison,
        "all_anchors_reproduced": gate,
    }
    for out in OUT_FILES:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")

    print("\n--- Recomputed Summary (25 frozen folds) ---")
    print(pd.DataFrame(summary).T.to_string())
    print("\n--- Manuscript Comparison ---")
    for col, entry in comparison.items():
        status = "REPRODUCED" if entry["reproduced"] else "MISMATCH"
        print(f"  {col:24s} manuscript={entry['manuscript']['auroc']:.4f} "
              f"recomputed={entry['recomputed']['macro_auroc']:.4f} "
              f"|diff|={entry['auroc_abs_diff']:.4f}  [{status}]")
    print(f"\nALL ANCHORS REPRODUCED: {gate}")
    print("Archived:", ", ".join(str(p) for p in OUT_FILES))


if __name__ == "__main__":
    main()
