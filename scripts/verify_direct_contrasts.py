"""
Direct Paired Contrasts: Orthogonal Suite vs Strongest Supervised Baselines.

Table 1 reports 95% CIs only vs the static baseline (canonical__C). This script adds
the direct paired comparisons a reviewer may request:

  drop_c_spatial__C  vs  plus_lag1__ridge_tuned   (strongest baseline, 0.8300)
  drop_c_spatial__C  vs  plus_lag1__logistic_tuned
  drop_c_spatial__C  vs  plus_lag1__lda_shrinkage
  drop_c_spatial__C  vs  canonical__C             (consistency re-check)
  drop_c_spatial__D  vs  plus_lag1__ridge_tuned

Method: identical Dirichlet(exponential) Bayesian bootstrap (10,000 draws,
seed 20261014) as bootstrap_contrasts() in run_temporal_mdco_benchmark.py.
Ortho predictions are recomputed on the same 25 frozen outer folds; archived
per-subject canonical__C scores gate the recomputation (must match to 1e-9).

Output archive: results/temporal_dynamics_v1/contrasts_direct_vs_strongest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
PRED_FILE = REPO_ROOT / "results/temporal_dynamics_v1/test_predictions.csv"
OUT_FILES = [
    REPO_ROOT / "results/temporal_dynamics_v1/contrasts_direct_vs_strongest.json",
    V3_ROOT / "results/temporal_dynamics_v1/contrasts_direct_vs_strongest.json",
]


def compute_fold_macro_auc(df: pd.DataFrame, score_col: str) -> float:
    aucs = [
        roc_auc_score(g["label"].values, g[score_col].values)
        for _, g in df.groupby(["seed", "fold"])
    ]
    return float(np.mean(aucs))


def paired_dirichlet_diff(
    df: pd.DataFrame, col_a: str, col_b: str, n_draws: int = 10000, seed: int = 20261014
) -> dict:
    """Paired Bayesian bootstrap of fold-macro AUROC difference (col_a - col_b)."""
    ids = sorted(df["subject_id"].unique())
    lookup = {sid: i for i, sid in enumerate(ids)}
    rng = np.random.default_rng(seed)
    weights = np.maximum(rng.exponential(size=(n_draws, len(ids))), np.finfo(float).tiny)

    groups = list(df.groupby(["seed", "fold"]))
    n_groups = len(groups)
    draw_aucs = {col_a: np.zeros(n_draws), col_b: np.zeros(n_draws)}
    for _, g in groups:
        y = g["label"].to_numpy(dtype=int)
        sids = g["subject_id"].to_numpy()
        pos, neg = y == 1, y == 0
        wp = weights[:, np.array([lookup[s] for s in sids[pos]])]
        wn = weights[:, np.array([lookup[s] for s in sids[neg]])]
        denom = wp.sum(axis=1) * wn.sum(axis=1)
        for m in (col_a, col_b):
            scores = g[m].to_numpy(dtype=float)
            wins = (scores[pos, None] > scores[neg]) + 0.5 * (scores[pos, None] == scores[neg])
            draw_aucs[m] += np.einsum("bi,ij,bj->b", wp, wins, wn, optimize=True) / denom / n_groups

    diff = draw_aucs[col_a] - draw_aucs[col_b]
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return {
        "point_difference": round(compute_fold_macro_auc(df, col_a) - compute_fold_macro_auc(df, col_b), 4),
        "ci_lower_95": round(float(lo), 4),
        "ci_upper_95": round(float(hi), 4),
        "prob_positive": round(float(np.mean(diff > 0)), 4),
    }


def main() -> None:
    dataset = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    variants = dataset["feature_matrix_variants"]
    labels_dict = {sid: int(l) for sid, l in zip(dataset["subject_ids"], dataset["labels"])}
    sid_to_idx = {sid: i for i, sid in enumerate(dataset["subject_ids"])}

    folds = [
        r for r in json.loads(FOLDS_FILE.read_text(encoding="utf-8"))
        if r.get("cohort") == "mendeley" and r.get("method") == "C"
    ]

    rows = []
    for r in folds:
        tr_idx = np.array([sid_to_idx[s] for s in r["train_ids"]])
        te_idx = np.array([sid_to_idx[s] for s in r["test_ids"]])
        y_tr = np.array([labels_dict[s] for s in r["train_ids"]], dtype=int)
        preds = {}
        for var in ("canonical", "drop_c_spatial"):
            X_tr, X_te = variants[var][tr_idx], variants[var][te_idx]
            m_c = SignedStandardizedMean().fit(X_tr, y_tr)
            m_d = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
            preds[var] = (m_c.predict(X_te), m_d.predict(X_te))
        for j, sid in enumerate(r["test_ids"]):
            rows.append({
                "seed": r["seed"], "fold": r["fold"], "subject_id": sid,
                "label": labels_dict[sid],
                "canonical__C": float(preds["canonical"][0][j]),
                "canonical__D": float(preds["canonical"][1][j]),
                "drop_c_spatial__C": float(preds["drop_c_spatial"][0][j]),
                "drop_c_spatial__D": float(preds["drop_c_spatial"][1][j]),
            })
    recomp = pd.DataFrame(rows)

    pred = pd.read_csv(PRED_FILE)
    merged = pred.merge(
        recomp[["seed", "fold", "subject_id", "label", "drop_c_spatial__C", "drop_c_spatial__D"]],
        on=["seed", "fold", "subject_id"], suffixes=("", "_recomp"),
    )
    if len(merged) != len(pred):
        raise RuntimeError("Merge lost rows; fold/subject alignment mismatch.")

    # Gate: archived canonical__C macro AUROC must still be exactly 0.8200.
    arch_auc = compute_fold_macro_auc(merged, "canonical__C")
    if abs(arch_auc - 0.8200) > 1e-9:
        raise RuntimeError(f"Archived canonical__C macro AUROC drifted: {arch_auc}")

    contrasts = {
        "drop_c_spatial__C_minus_plus_lag1__ridge_tuned":
            paired_dirichlet_diff(merged, "drop_c_spatial__C", "plus_lag1__ridge_tuned"),
        "drop_c_spatial__C_minus_plus_lag1__logistic_tuned":
            paired_dirichlet_diff(merged, "drop_c_spatial__C", "plus_lag1__logistic_tuned"),
        "drop_c_spatial__C_minus_plus_lag1__lda_shrinkage":
            paired_dirichlet_diff(merged, "drop_c_spatial__C", "plus_lag1__lda_shrinkage"),
        "drop_c_spatial__C_minus_canonical__C":
            paired_dirichlet_diff(merged, "drop_c_spatial__C", "canonical__C"),
        "drop_c_spatial__D_minus_plus_lag1__ridge_tuned":
            paired_dirichlet_diff(merged, "drop_c_spatial__D", "plus_lag1__ridge_tuned"),
    }

    point_aucs = {
        "drop_c_spatial__C": round(compute_fold_macro_auc(merged, "drop_c_spatial__C"), 4),
        "drop_c_spatial__D": round(compute_fold_macro_auc(merged, "drop_c_spatial__D"), 4),
        "plus_lag1__ridge_tuned": round(compute_fold_macro_auc(merged, "plus_lag1__ridge_tuned"), 4),
        "plus_lag1__logistic_tuned": round(compute_fold_macro_auc(merged, "plus_lag1__logistic_tuned"), 4),
        "plus_lag1__lda_shrinkage": round(compute_fold_macro_auc(merged, "plus_lag1__lda_shrinkage"), 4),
        "canonical__C": round(arch_auc, 4),
    }

    archive = {
        "verification": "direct_paired_contrasts_orthogonal_vs_strongest_baselines",
        "protocol": {
            "folds": "25 frozen outer partitions (fitted_states.json)",
            "bootstrap": "Dirichlet(exponential) x 10,000 draws, seed 20261014, paired subject-level",
            "gate": f"archived canonical__C macro AUROC = {arch_auc:.4f} (expected 0.8200)",
            "pred_source": "results/temporal_dynamics_v1/test_predictions.csv + recomputed drop_c_spatial",
        },
        "point_fold_macro_aurocs": point_aucs,
        "contrasts": contrasts,
    }
    for out in OUT_FILES:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")

    print("--- Point fold-macro AUROCs ---")
    for k, v in point_aucs.items():
        print(f"  {k:32s} {v:.4f}")
    print("\n--- Direct paired contrasts (95% CI, Dirichlet bootstrap) ---")
    for k, v in contrasts.items():
        sig = "*" if v["ci_lower_95"] > 0 else " "
        print(f" {sig} {k}")
        print(f"    diff={v['point_difference']:+.4f}  CI=[{v['ci_lower_95']:+.4f}, {v['ci_upper_95']:+.4f}]  P(+)={v['prob_positive']:.4f}")
    print("\nArchived:", ", ".join(str(p) for p in OUT_FILES))


if __name__ == "__main__":
    main()
