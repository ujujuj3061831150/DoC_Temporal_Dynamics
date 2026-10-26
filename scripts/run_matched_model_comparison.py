"""Same-feature, same-preprocessing comparisons on discovery and external cohorts.

Run: python scripts/run_matched_model_comparison.py
Outputs include every subject score, source-training threshold, fitted transform,
inner-CV selection, paired subject-level Bayesian bootstrap, and input hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "results/multiscale_matched_v2/features.json").exists())
sys.path.insert(0, str(ROOT / "src"))
from temporal_dynamics.adapters import load_mendeley_temporal_dataset
from temporal_dynamics.consensus import SignedStandardizedMean
from temporal_dynamics.figshare_adapter import load_figshare_pdoc_dataset

LEARNED = ("ridge", "logistic", "lda_shrinkage")
METHODS = ("equal_mean", "equal_mean_c15", *LEARNED)
FEATURES = {
    "canonical": ["c_global", "c_spatial", "neg_cv", "r_interhem"],
    "orthogonal_lag1": ["c_global", "neg_cv", "lag1", "residual_corr"],
    "short_increment": ["c_global", "c_spatial", "neg_increment", "r_interhem"],
}
# Preserve the original benchmark grid order; an exact tie takes the first candidate.
GRIDS = {"ridge": [.001, .01, .1, 1., 10., 100., 1000.],
         "logistic": [.001, .01, .1, 1., 10., 100., 1000.]}


def classifier(method, parameter):
    if method == "ridge":
        return RidgeClassifier(alpha=parameter)
    if method == "logistic":
        return LogisticRegression(C=parameter, solver="lbfgs", max_iter=2000)
    if method == "lda_shrinkage":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(method)


def fit_transform(X, y):
    # All methods receive exactly these same oriented coordinates.
    reference = SignedStandardizedMean().fit(X, y)
    scaler, signs = reference.scaler, reference.signs_.copy()
    return scaler, signs, scaler.transform(X) * signs


def tune_parameters(X, y, seed):
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    scores = {m: {p: [] for p in grid} for m, grid in GRIDS.items()}
    for tr, va in splits:
        scaler, signs, Z_tr = fit_transform(X[tr], y[tr])
        Z_va = scaler.transform(X[va]) * signs
        for method, grid in GRIDS.items():
            for parameter in grid:
                model = classifier(method, parameter).fit(Z_tr, y[tr])
                scores[method][parameter].append(roc_auc_score(
                    y[va], model.decision_function(Z_va)))
    selected = {m: max(grid, key=lambda p: np.mean(scores[m][p]))
                for m, grid in GRIDS.items()}
    trace = {m: {str(p): float(np.mean(v)) for p, v in ps.items()}
             for m, ps in scores.items()}
    return selected, {"seed": seed, "mean_validation_auroc": trace,
                      "splits": [{"train": tr.tolist(), "validation": va.tolist()}
                                 for tr, va in splits]}


def fit_models(X, y, seed=42):
    X, y = np.asarray(X, float), np.asarray(y, int)
    parameters, selection = tune_parameters(X, y, seed)
    scaler, signs, Z = fit_transform(X, y)
    models = {m: classifier(m, parameters.get(m)).fit(Z, y) for m in LEARNED}
    scores = predict_coordinates(Z, models)
    thresholds = {m: float(np.median(s)) for m, s in scores.items()}
    return {"scaler": scaler, "signs": signs, "models": models,
            "thresholds": thresholds, "parameters": parameters, "selection": selection}


def predict_coordinates(Z, models):
    return {"equal_mean": np.mean(Z, axis=1),
            "equal_mean_c15": np.clip(Z, -1.5, 1.5).mean(axis=1),
            **{m: model.decision_function(Z) for m, model in models.items()}}


def target_coordinates(X, fitted, protocol, target_signs=None):
    X = np.asarray(X, float)
    if protocol == "source_frozen":
        scaler, signs = fitted["scaler"], fitted["signs"]
    elif protocol in {"target_standardized", "target_aligned"}:
        # Explicit transductive batch protocol. No target labels enter this function.
        scaler = StandardScaler().fit(X)
        signs = fitted["signs"] if protocol == "target_standardized" else np.asarray(target_signs, float)
        if signs.shape != (X.shape[1],) or not np.isin(signs, [-1., 1.]).all():
            raise ValueError("Explicit target signs must match the feature columns.")
    else:
        raise ValueError(protocol)
    return scaler.transform(X) * signs, {
        "protocol": protocol, "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "signs": signs.tolist()}


def source_state(fitted, ids, columns):
    return {"training_ids": list(ids), "features": columns,
            "mean": fitted["scaler"].mean_.tolist(),
            "scale": fitted["scaler"].scale_.tolist(),
            "signs": fitted["signs"].tolist(),
            "parameters": {**fitted["parameters"], "lda_shrinkage": "auto"},
            "thresholds": fitted["thresholds"], "inner_cv": fitted["selection"],
            "coefficients": {m: {"coef": model.coef_.tolist(),
                                    "intercept": model.intercept_.tolist()}
                             for m, model in fitted["models"].items()}}


def prediction_rows(ids, y, scores, thresholds, seed=-1, fold=-1):
    rows = []
    for i, sid in enumerate(ids):
        row = {"subject_id": sid, "label": int(y[i]), "seed": seed, "fold": fold}
        for method in METHODS:
            row[method] = float(scores[method][i])
            row[method + "_threshold"] = thresholds[method]
            row[method + "_binary"] = int(scores[method][i] >= thresholds[method])
        rows.append(row)
    return rows


def summarize(frame, n_draws, seed=20261026):
    """Fold-macro AUC/AP, pooled decisions, paired subject-level Dirichlet draws."""
    ids = sorted(frame.subject_id.unique())
    lookup = {sid: i for i, sid in enumerate(ids)}
    weights = np.random.default_rng(seed).exponential(size=(n_draws, len(ids)))
    groups = list(frame.groupby(["seed", "fold"]))
    draws = {m: np.zeros(n_draws) for m in METHODS}
    aucs, aps = {m: [] for m in METHODS}, {m: [] for m in METHODS}
    for _, group in groups:
        y = group.label.to_numpy(); pos, neg = y == 1, y == 0
        idx = np.array([lookup[s] for s in group.subject_id])
        wp, wn = weights[:, idx[pos]], weights[:, idx[neg]]
        denominator = wp.sum(axis=1) * wn.sum(axis=1)
        for method in METHODS:
            s = group[method].to_numpy()
            aucs[method].append(roc_auc_score(y, s))
            aps[method].append(average_precision_score(y, s))
            wins = (s[pos, None] > s[neg]) + .5 * (s[pos, None] == s[neg])
            draws[method] += np.einsum("bi,ij,bj->b", wp, wins, wn, optimize=True) / denominator / len(groups)
    metrics, contrasts = {}, {}
    for method in METHODS:
        y, pred = frame.label.to_numpy(), frame[method + "_binary"].to_numpy()
        sensitivity, specificity = recall_score(y, pred, zero_division=0), recall_score(y, pred, pos_label=0, zero_division=0)
        metrics[method] = {"auroc": float(np.mean(aucs[method])), "average_precision": float(np.mean(aps[method])),
                           "balanced_accuracy": float((sensitivity + specificity) / 2),
                           "sensitivity": float(sensitivity), "specificity": float(specificity),
                           "f1": float(f1_score(y, pred, zero_division=0)),
                           "auroc_ci95": np.quantile(draws[method], [.025, .975]).tolist()}
    for reference in ("equal_mean", "equal_mean_c15"):
        for method in METHODS:
            if method == reference:
                continue
            diff = draws[reference] - draws[method]
            contrasts[reference + "_minus_" + method] = {
                "auroc_difference": metrics[reference]["auroc"] - metrics[method]["auroc"],
                "ci95": np.quantile(diff, [.025, .975]).tolist(),
                "prob_positive": float(np.mean(diff > 0))}
    return {"subjects": len(ids), "prediction_rows": len(frame),
            "metrics": metrics, "paired_contrasts": contrasts}


def run(output, n_draws=10000, include_figshare=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dataset = load_mendeley_temporal_dataset(repo_root=ROOT)
    ids, y = dataset["subject_ids"], dataset["labels"]
    source = {"canonical": dataset["feature_matrix_variants"]["canonical"],
              "orthogonal_lag1": dataset["feature_matrix_variants"]["drop_c_spatial"],
              "short_increment": dataset["feature_matrix_variants"]["replace_cv_increment"]}
    folds_path = ROOT / "results/multiscale_matched_v2/fitted_states.json"
    folds = [r for r in json.loads(folds_path.read_text(encoding="utf-8"))
             if r.get("cohort") == "mendeley" and r.get("method") == "C"]
    if len(folds) != 25 or len(set(ids)) != len(ids):
        raise ValueError("Expected 25 frozen folds and unique source subject IDs.")
    lookup = {s: i for i, s in enumerate(ids)}
    all_rows, summary, states = [], {}, {}

    def save_evaluation(key, rows, state):
        frame = pd.DataFrame(rows)
        result = summarize(frame, n_draws)
        if not key.startswith("mendeley/"):
            auxiliary = {}
            for method in METHODS:
                threshold = float(np.median(frame[method]))
                pred = frame[method].to_numpy() >= threshold
                sensitivity = float(recall_score(frame.label, pred, zero_division=0))
                specificity = float(recall_score(frame.label, pred, pos_label=0, zero_division=0))
                auxiliary[method] = {"threshold": threshold, "sensitivity": sensitivity,
                                     "specificity": specificity,
                                     "balanced_accuracy": (sensitivity + specificity) / 2}
            result["auxiliary_target_median_operating_points"] = auxiliary
        summary[key] = result; states[key] = state
        all_rows.extend({"evaluation": key, **r} for r in rows)
        print(key, {m: round(v["auroc"], 4) for m, v in result["metrics"].items()}, flush=True)

    for feature_set, X in source.items():
        rows, source_states = [], []
        for fold in folds:
            tr = np.array([lookup[s] for s in fold["train_ids"]]); te = np.array([lookup[s] for s in fold["test_ids"]])
            if set(tr) & set(te) or len(set(tr) | set(te)) != len(ids):
                raise ValueError("Invalid subject-level outer partition.")
            fitted = fit_models(X[tr], y[tr], int(fold["seed"]) * 10 + int(fold["fold"]))
            Z, _ = target_coordinates(X[te], fitted, "source_frozen")
            scores = predict_coordinates(Z, fitted["models"])
            rows.extend(prediction_rows(fold["test_ids"], y[te], scores, fitted["thresholds"], fold["seed"], fold["fold"]))
            source_states.append({"seed": fold["seed"], "fold": fold["fold"],
                                  **source_state(fitted, fold["train_ids"], FEATURES[feature_set])})
        save_evaluation("mendeley/" + feature_set, rows, source_states)

    crrc_path = ROOT / "results/crrc_validation/crrc_deidentified_features.csv"
    crrc = pd.read_csv(crrc_path)
    externals = []
    for name, frame in [("crrc_full15", crrc), ("crrc_clean13", crrc[crrc.clean_6ch_protocol])]:
        externals.append((name, "orthogonal_lag1", frame.subject_id.tolist(), frame.y_true.to_numpy(),
                          frame[FEATURES["orthogonal_lag1"]].to_numpy(), [-1., 1., 1., 1.]))
    if include_figshare:
        figshare = load_figshare_pdoc_dataset()
        for feature_set, columns in FEATURES.items():
            X_target = np.array([[figshare["feature_dicts"][s][c] for c in columns] for s in figshare["subject_ids"]])
            externals.append(("figshare50", feature_set, figshare["subject_ids"], figshare["labels"], X_target, None))
    fits = {name: fit_models(X, y) for name, X in source.items()}
    for cohort, feature_set, target_ids, target_y, target_X, prior in externals:
        fitted = fits[feature_set]
        for protocol in (["source_frozen", "target_standardized", "target_aligned"] if prior is not None
                         else ["source_frozen", "target_standardized"]):
            Z, calibration = target_coordinates(target_X, fitted, protocol, prior)
            scores = predict_coordinates(Z, fitted["models"])
            rows = prediction_rows(target_ids, target_y, scores, fitted["thresholds"])
            state = {"source": source_state(fitted, ids, FEATURES[feature_set]),
                     "target": {**calibration, "calibration_ids": target_ids if protocol != "source_frozen" else []}}
            save_evaluation("/".join([cohort, feature_set, protocol]), rows, state)

    pd.DataFrame(all_rows).to_csv(output / "subject_predictions.csv", index=False)
    # Cross-representation references for the existing Table I static-baseline column.
    all_predictions = pd.DataFrame(all_rows)
    key_columns = ["seed", "fold", "subject_id"]
    static = all_predictions[all_predictions.evaluation == "mendeley/canonical"].set_index(key_columns)
    orthogonal = all_predictions[all_predictions.evaluation == "mendeley/orthogonal_lag1"].set_index(key_columns).copy()
    static = static.reindex(orthogonal.index)
    if static.label.isna().any() or not np.array_equal(static.label, orthogonal.label):
        raise ValueError("Static and orthogonal predictions do not align by subject and fold.")
    for column in ["equal_mean", "equal_mean_threshold", "equal_mean_binary"]:
        orthogonal[column] = static[column]
    reference_result = summarize(orthogonal.reset_index(), n_draws)
    vs_static = {}
    for method in LEARNED:
        contrast = reference_result["paired_contrasts"]["equal_mean_minus_" + method]
        vs_static[method] = {"auroc_difference": -contrast["auroc_difference"],
                             "ci95": [-contrast["ci95"][1], -contrast["ci95"][0]]}
    (output / "contrasts_vs_static.json").write_text(json.dumps(vs_static, indent=2), encoding="utf-8")
    flat = [{"evaluation": k, "model": m, **v} for k, result in summary.items() for m, v in result["metrics"].items()]
    pd.DataFrame(flat).to_csv(output / "metrics.csv", index=False)
    protocol = {"features": FEATURES, "methods": list(METHODS), "inner_cv": "3-fold stratified; source-only",
                "parameter_grids": GRIDS, "threshold": "source-training score median for every model and protocol",
                "soft_truncation": "equal_mean_c15 clips shared oriented coordinates at fixed c=1.5 within its scoring rule; learned models receive the same unmodified coordinates",
                "internal_aggregation": "fold-macro AUROC/AP; pooled decisions across 5 repeats",
                "external_calibration": "source_frozen is inductive; target_standardized/target_aligned are unlabeled batch adaptation",
                "crrc_target_prior": [-1., 1., 1., 1.],
                "auxiliary_threshold": "Separately labeled unlabeled target-batch median; never used for source-only tuning or primary predictions",
                "bootstrap": {"type": "paired subject-level Dirichlet, conditional on fitted scores", "draws": n_draws, "seed": 20261026},
                "source_parameters": "Re-selected inside each outer training fold; full-source selection reused across external protocols"}
    (output / "summary.json").write_text(json.dumps({"protocol": protocol, "evaluations": summary}, indent=2), encoding="utf-8")
    (output / "fitted_states.json").write_text(json.dumps(states, indent=2), encoding="utf-8")
    lines = ["# Matched model comparison", "", "All models use the same feature columns, source-only selection, common transforms and source-training median thresholds.",
             "Target-batch adaptation is reported separately from frozen-source transfer. External labels are used only for evaluation.", "",
             "The clipped equal-mean variant uses fixed c=1.5 in its scoring rule. It does not change the input coordinates or model-selection conditions for other methods.", "",
             "| Evaluation | Equal mean | Equal mean c=1.5 | Ridge tuned | Logistic tuned | Shrinkage-LDA |", "|---|---:|---:|---:|---:|---:|"]
    for key, result in summary.items():
        lines.append("| " + key + " | " + " | ".join(f"{result['metrics'][m]['auroc']:.4f}" for m in METHODS) + " |")
    lines += ["", "## Direct paired contrasts", "", "| Evaluation | Reference | Comparator | Delta AUROC | 95% interval |", "|---|---|---|---:|---|"]
    for key, result in summary.items():
        for comparator, contrast in result["paired_contrasts"].items():
            lo, hi = contrast["ci95"]
            reference, method = comparator.split("_minus_")
            lines.append(f"| {key} | {reference} | {method} | {contrast['auroc_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    input_paths = [folds_path, crrc_path, ROOT / "results/multiscale_matched_v2/features.json"]
    input_paths += [ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz" for sid in ids]
    if include_figshare:
        input_paths += [ROOT / f"figshare_6ch_cache/{sid}.npy" for sid in figshare["subject_ids"]]
        input_paths.append(ROOT / "figshare_23552964_meta.json")
        for candidate in [ROOT / "figshare_clinical_diagnoses.csv", ROOT / "data/figshare_clinical_diagnoses.csv"]:
            if candidate.exists():
                input_paths.append(candidate)
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    (output / "manifest.json").write_text(json.dumps({"inputs": inputs, "outputs": files,
        "environment": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn"]},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/matched_model_comparison_v1")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--skip-figshare", action="store_true")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    run(args.output, args.draws, not args.skip_figshare)
