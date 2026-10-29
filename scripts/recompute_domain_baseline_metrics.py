"""Align existing LZC/entropy operating metrics with frozen training thresholds."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parents[1].name == "v3" else Path(__file__).resolve().parents[1]
OUT = ROOT / "results/domain_baseline_metrics_v1"
METHODS = ("canonical", "lzc", "spectral_entropy")


def run():
    features_path = OUT / "features.csv"
    folds_path = ROOT / "results/temporal_dynamics_v1/test_predictions.csv"
    features = pd.read_csv(features_path).set_index("subject_id")
    frozen = pd.read_csv(folds_path)
    labels = frozen.drop_duplicates("subject_id").set_index("subject_id").label
    ids = sorted(labels.index)
    assert len(ids) == 41 and features.index.is_unique
    assert set(ids) == set(features.index)
    assert np.isfinite(features[list(METHODS[1:])].to_numpy()).all()
    rows, states = [], []
    groups = list(frozen.groupby(["seed", "fold"]))
    assert len(groups) == 25
    for (seed, fold), test in groups:
        test_ids = test.subject_id.tolist()
        train_ids = sorted(set(ids) - set(test_ids))
        y_train = labels.loc[train_ids].to_numpy()
        scores = {"canonical": test.canonical__C.to_numpy()}
        thresholds = {}
        for method in METHODS[1:]:
            x = features.loc[train_ids, method].to_numpy()
            corr = np.corrcoef(x, y_train)[0, 1] if np.std(x) > 0 else 0.
            sign = -1. if np.isfinite(corr) and corr < 0 else 1.
            thresholds[method] = float(np.median(sign * x))
            scores[method] = sign * features.loc[test_ids, method].to_numpy()
            states.append({"seed": int(seed), "fold": int(fold), "method": method,
                           "train_ids": train_ids, "test_ids": test_ids, "sign": sign,
                           "training_score_median": thresholds[method]})
        for i, sid in enumerate(test_ids):
            row = {"seed": int(seed), "fold": int(fold), "subject_id": sid,
                   "label": int(test.label.iloc[i]), "canonical": float(scores["canonical"][i]),
                   "canonical_binary": int(test.canonical__C_pred_binary.iloc[i])}
            for method in METHODS[1:]:
                row.update({method: float(scores[method][i]), method + "_threshold": thresholds[method],
                            method + "_binary": int(scores[method][i] >= thresholds[method])})
            rows.append(row)
    frame = pd.DataFrame(rows)
    assert (frame.groupby(["seed", "subject_id"]).size() == 1).all()
    draws = 10000
    seed = 20261026
    weights = np.random.default_rng(seed).exponential(size=(draws, len(ids)))
    lookup = {sid: i for i, sid in enumerate(ids)}
    sampled = {m: np.zeros(draws) for m in METHODS}
    aucs, aps = {m: [] for m in METHODS}, {m: [] for m in METHODS}
    for _, group in frame.groupby(["seed", "fold"]):
        y = group.label.to_numpy()
        pos, neg = y == 1, y == 0
        idx = np.array([lookup[s] for s in group.subject_id])
        wp, wn = weights[:, idx[pos]], weights[:, idx[neg]]
        denominator = wp.sum(axis=1) * wn.sum(axis=1)
        for method in METHODS:
            scores = group[method].to_numpy()
            aucs[method].append(roc_auc_score(y, scores))
            aps[method].append(average_precision_score(y, scores))
            wins = (scores[pos, None] > scores[neg]) + .5 * (scores[pos, None] == scores[neg])
            sampled[method] += np.einsum("bi,ij,bj->b", wp, wins, wn, optimize=True) / denominator / len(groups)
    metrics = {}
    for method in METHODS:
        y, pred = frame.label.to_numpy(), frame[method + "_binary"].to_numpy()
        sens, spec = recall_score(y, pred), recall_score(y, pred, pos_label=0)
        metrics[method] = {"auroc": float(np.mean(aucs[method])), "average_precision": float(np.mean(aps[method])),
                           "balanced_accuracy": float((sens + spec) / 2), "sensitivity": float(sens),
                           "specificity": float(spec), "f1": float(f1_score(y, pred)),
                           "true_positive": int(np.sum((y == 1) & (pred == 1))),
                           "true_negative": int(np.sum((y == 0) & (pred == 0))),
                           "positive_decisions": int(np.sum(y == 1)), "negative_decisions": int(np.sum(y == 0))}
        if method != "canonical":
            metrics[method]["difference_vs_static"] = metrics[method]["auroc"] - float(np.mean(aucs["canonical"]))
            metrics[method]["difference_ci95"] = np.quantile(sampled[method] - sampled["canonical"], [.025, .975]).tolist()
    assert np.isclose(metrics["lzc"]["auroc"], 0.5135555555555555)
    assert np.isclose(metrics["spectral_entropy"]["auroc"], 0.6953333333333334)
    frame.to_csv(OUT / "subject_predictions.csv", index=False)
    (OUT / "fitted_states.json").write_text(json.dumps(states, indent=2), encoding="utf-8")
    summary = {"protocol": {"folds": 25, "source": "Existing single-feature cache; unchanged feature values and frozen subject folds",
                "orientation": "Feature-label correlation fitted on each outer training set only",
                "threshold": "Outer-training signed-score median, >= comparison",
                "aggregation": "Fold-macro AUROC/AP; pooled outer-test decisions across five repeats",
                "bootstrap": {"type": "Paired subject-level Dirichlet conditional on fixed scores", "draws": draws, "seed": seed}},
               "metrics": metrics}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [features_path, folds_path]},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
