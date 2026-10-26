#!/usr/bin/env python3
"""
Hyperparameter Stability Grid Runner for Soft Truncation and T-GECF Gating.

Evaluates:
1. Winsorized soft truncation threshold c in [1.0, 1.5, 2.0, 2.5, 3.0, inf].
2. Continuous Sigmoid local volatility gating parameters (theta_0, kappa) across 4x4 grid.

Saves verified outputs to results/temporal_dynamics_v1/hyperparam_grid_results.json.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
)


def find_repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "results" / "multiscale_matched_v2").exists() or (parent / ".git").exists():
            return parent
    return cur.parents[2]


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if (REPO_ROOT / "v3/src").exists() and str(REPO_ROOT / "v3/src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "v3/src"))
elif (REPO_ROOT / "src").exists() and str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_dynamics.consensus import AdaptiveGatedConsensus, SignedStandardizedMean
from temporal_dynamics.adapters import load_mendeley_temporal_dataset


def run_hyperparam_grid(output_path: Path | None = None) -> dict:
    if output_path is None:
        target_dir = (REPO_ROOT / "v3/results/temporal_dynamics_v1") if (REPO_ROOT / "v3/results/temporal_dynamics_v1").exists() else (REPO_ROOT / "results/temporal_dynamics_v1")
        output_path = target_dir / "hyperparam_grid_results.json"

    dataset = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    subject_ids = dataset["subject_ids"]
    labels = dataset["labels"]
    sid_to_idx = {sid: i for i, sid in enumerate(subject_ids)}

    fitted_states_file = REPO_ROOT / "results/multiscale_matched_v2/fitted_states.json"
    all_states = json.loads(fitted_states_file.read_text(encoding="utf-8"))
    folds = [r for r in all_states if r.get("cohort") == "mendeley" and r.get("method") == "C"]

    X_all = dataset["feature_matrix_variants"]["plus_lag1_residual"]

    # 1. Soft truncation sweep computed dynamically across folds
    c_values = [1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    soft_c_results = {}
    for c_val in c_values:
        c_key = "inf" if np.isinf(c_val) else f"{c_val:.1f}"
        param_c = None if np.isinf(c_val) else c_val
        aucs, aps = [], []
        y_trues_all, y_preds_binary = [], []
        for f in folds:
            tr_idx = [sid_to_idx[s] for s in f["train_ids"]]
            te_idx = [sid_to_idx[s] for s in f["test_ids"]]
            X_tr, y_tr = X_all[tr_idx], labels[tr_idx]
            X_te, y_te = X_all[te_idx], labels[te_idx]

            m = SignedStandardizedMean(c=param_c)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            aucs.append(roc_auc_score(y_te, preds))
            aps.append(average_precision_score(y_te, preds))

            thresh = float(np.median(m.predict(X_tr)))
            y_trues_all.extend(y_te)
            y_preds_binary.extend((preds >= thresh).astype(int))

        soft_c_results[c_key] = {
            "auroc": round(float(np.mean(aucs)), 4),
            "pr_auc": round(float(np.mean(aps)), 4),
            "bal_acc": round(float(balanced_accuracy_score(y_trues_all, y_preds_binary)), 4),
            "sens": round(float(recall_score(y_trues_all, y_preds_binary)), 4),
            "spec": round(float(recall_score(y_trues_all, y_preds_binary, pos_label=0)), 4),
        }

    # 2. T-GECF 4x4 parameter grid
    thetas = [(1.282, "1.282_90th"), (1.645, "1.645_95th"), (1.960, "1.960_97.5th"), (2.326, "2.326_99th")]
    kappas = [0.5, 1.0, 2.0, 5.0]

    gecf_results = {}
    for th_val, th_name in thetas:
        row_dict = {}
        for kp in kappas:
            aucs = []
            for f in folds:
                tr_idx = [sid_to_idx[s] for s in f["train_ids"]]
                te_idx = [sid_to_idx[s] for s in f["test_ids"]]
                X_tr, y_tr = X_all[tr_idx], labels[tr_idx]
                X_te, y_te = X_all[te_idx], labels[te_idx]

                m = AdaptiveGatedConsensus(gating_type="sigmoid", theta_0=th_val, kappa=kp)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                aucs.append(roc_auc_score(y_te, preds))
            row_dict[f"kappa_{kp}"] = round(float(np.mean(aucs)), 4)
        gecf_results[f"theta_0_{th_name}"] = row_dict

    # 3. Fixed beta=0.5 baseline ablation computed dynamically
    beta_aucs = []
    for f in folds:
        tr_idx = [sid_to_idx[s] for s in f["train_ids"]]
        te_idx = [sid_to_idx[s] for s in f["test_ids"]]
        X_tr, y_tr = X_all[tr_idx], labels[tr_idx]
        X_te, y_te = X_all[te_idx], labels[te_idx]
        m = AdaptiveGatedConsensus(beta=0.5)
        m.fit(X_tr, y_tr)
        beta_aucs.append(roc_auc_score(y_te, m.predict(X_te)))

    out_data = {
        "soft_truncation_c": soft_c_results,
        "t_gecf_sigmoid_grid_4x4": gecf_results,
        "baseline_ablations": {
            "fixed_beta_0.5": round(float(np.mean(beta_aucs)), 4),
            "unconstrained_c_inf": soft_c_results["inf"]["auroc"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[*] Hyperparameter grid results saved to {output_path}")
    return out_data


if __name__ == "__main__":
    run_hyperparam_grid()
