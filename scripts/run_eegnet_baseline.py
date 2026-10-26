"""
Empirical EEGNet-Style Deep Baseline on the 25 Frozen Outer Folds (Mendeley N=41).

Purpose: the manuscript argues conceptually (VC capacity overflow, >2,600 weights
vs N_train ~= 32) that deep networks overfit sparse DoC cohorts. This script turns
that argument into an empirical benchmark under the exact Table 1 protocol:

- Input: per-subject clean 4-s block theta powers (peaks < 500 uV), first T blocks,
  per-channel z-scored within subject (unsupervised; identical information stream
  the physical operators consume). Shape (1, 6 sensors, T blocks).
- Model: EEGNet-style compact CNN (Lawhern et al. 2018 layout: temporal conv ->
  depthwise channel conv -> separable conv -> global average pool -> dense),
  parameter count reported exactly (target > 2,600 to match the manuscript claim).
- Protocol: identical 25 frozen subject-level outer folds; within each training
  fold, subjects split stratified 75/25 for early stopping (val AUROC, patience 40,
  max 400 epochs, Adam 1e-3, weight decay 1e-4); test on held-out subjects.
- Diagnostics: in-sample (train) AUROC vs out-of-fold test AUROC per fold to
  exhibit the overfitting gap directly.

Output archive: results/temporal_dynamics_v1/eegnet_baseline_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn

CURRENT = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in CURRENT.parents if (p / "results" / "multiscale_matched_v2").exists()),
    CURRENT.parents[2],
)
V3_ROOT = next((p for p in CURRENT.parents if p.name == "v3"), REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal_dynamics.adapters import get_repo_root, load_mendeley_temporal_dataset
from temporal_dynamics.operators import extract_clean_theta_series

FOLDS_FILE = REPO_ROOT / "results/multiscale_matched_v2/fitted_states.json"
OUT_FILES = [
    REPO_ROOT / "results/temporal_dynamics_v1/eegnet_baseline_results.json",
    V3_ROOT / "results/temporal_dynamics_v1/eegnet_baseline_results.json",
]

T_MAX = 1600          # first N clean blocks (uniform truncation, >= all subjects' minimum)
SEED_BASE = 20261024
MAX_EPOCHS = 400
PATIENCE = 40


class EEGNetSmall(nn.Module):
    """EEGNet-style compact CNN: temporal -> depthwise -> separable -> GAP -> dense."""

    def __init__(self, n_sensors: int = 6, n_classes: int = 2, t_kernel: int = 128,
                 dw_kernel: int = 32, sep_kernel: int = 64, f1: int = 8, f2: int = 16,
                 dropout: float = 0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, t_kernel), padding=(0, t_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * 2, (1, dw_kernel), groups=f1, padding=(0, dw_kernel // 2)),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * 2, f2, (1, sep_kernel), groups=f1 * 2, padding=(0, sep_kernel // 2)),
            nn.Conv2d(f2, f2, (1, 1)),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(f2 * n_sensors, n_classes)

    def forward(self, x):                    # x: (B, 1, 6, T)
        h = self.block2(self.block1(x))
        h = h.mean(dim=3)                    # global average pool over time
        return self.head(h.flatten(1))


def load_subject_sequences() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per-subject z-scored clean theta block series, truncated to T_MAX blocks."""
    root = get_repo_root()
    dataset = load_mendeley_temporal_dataset(repo_root=root)
    seqs, labels = {}, {sid: int(l) for sid, l in zip(dataset["subject_ids"], dataset["labels"])}
    for sid in dataset["subject_ids"]:
        theta, _ = extract_clean_theta_series(root / f"results/target_profiles_v11/raw_channels/{sid}.npz")
        x = theta[:T_MAX].T                                  # (6, T)
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        seqs[sid] = x.astype(np.float32)
    n_min = min(v.shape[1] for v in seqs.values())
    if n_min < 512:
        raise RuntimeError(f"Shortest subject sequence {n_min} blocks unexpectedly small.")
    return seqs, labels


def run_fold(model_factory, X: dict, y: dict, train_sids, test_sids, device, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_labels = np.array([y[s] for s in train_sids])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_idx, va_idx = next(sss.split(np.zeros(len(train_sids)), tr_labels))
    fit_sids = [train_sids[i] for i in tr_idx]
    val_sids = [train_sids[i] for i in va_idx]

    def pack(sids):
        arr = np.stack([X[s] for s in sids])[:, None, :, :]     # (B, 1, 6, T)
        return (torch.from_numpy(arr).to(device),
                torch.from_numpy(np.array([y[s] for s in sids])).to(device))

    Xf, yf = pack(fit_sids)
    Xv, yv = pack(val_sids)
    Xt, yt = pack(test_sids)

    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    best_val, best_state, patience = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xf), yf)
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xv)
                va = roc_auc_score(yv.cpu().numpy(), val_logits[:, 1].cpu().numpy()) \
                    if len(set(yv.tolist())) == 2 else 0.5
            if va > best_val + 1e-4:
                best_val, patience = va, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE // 5:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tr_auc = roc_auc_score(yf.cpu().numpy(), model(Xf)[:, 1].cpu().numpy())
        te_logits = model(Xt)[:, 1].cpu().numpy()
    te_auc = roc_auc_score(yt.cpu().numpy(), te_logits) if len(set(yt.tolist())) == 2 else float("nan")
    return te_auc, tr_auc, best_val, te_logits, [y[s] for s in test_sids]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Device: {device}")

    X, y = load_subject_sequences()
    n_params = sum(p.numel() for p in EEGNetSmall().parameters())
    print(f"Subjects: {len(X)} | shortest sequence: {min(v.shape[1] for v in X.values())} blocks"
          f" | EEGNet params: {n_params}")

    folds = [
        r for r in json.loads(FOLDS_FILE.read_text(encoding="utf-8"))
        if r.get("cohort") == "mendeley" and r.get("method") == "C"
    ]

    t0 = time.time()
    records, oof_scores, oof_labels = [], [], []
    for k, r in enumerate(folds):
        seed = SEED_BASE + r["seed"] * 100 + r["fold"]
        te_auc, tr_auc, va_auc, logits, labels = run_fold(
            EEGNetSmall, X, y, r["train_ids"], r["test_ids"], device, seed
        )
        records.append({
            "seed": r["seed"], "fold": r["fold"], "test_auroc": None if np.isnan(te_auc) else round(float(te_auc), 4),
            "train_auroc": round(float(tr_auc), 4), "best_val_auroc": round(float(va_auc), 4),
        })
        oof_scores.extend(logits.tolist())
        oof_labels.extend(labels)
        print(f"  [{k+1:2d}/25] test={te_auc:.4f}  train={tr_auc:.4f}  val={va_auc:.4f}")

    fold_aucs = [q["test_auroc"] for q in records if q["test_auroc"] is not None]
    macro_auc = float(np.mean(fold_aucs))
    pooled_auc = float(roc_auc_score(oof_labels, oof_scores))
    train_gap = float(np.mean([q["train_auroc"] for q in records])) - macro_auc

    archive = {
        "benchmark": "eegnet_empirical_baseline_25_frozen_folds",
        "model": {
            "architecture": "EEGNet-style (temporal conv k128 -> depthwise k32 -> separable k64 -> GAP -> dense)",
            "n_parameters": n_params,
            "input": f"clean theta block powers, first {T_MAX} blocks, per-channel z-scored, shape (1, 6, T)",
        },
        "protocol": {
            "folds": "25 frozen outer subject-level partitions (fitted_states.json)",
            "early_stopping": f"inner stratified 75/25 subject split, patience {PATIENCE}, max {MAX_EPOCHS} epochs",
            "optimizer": "Adam lr=1e-3, weight_decay=1e-4, full-batch",
        },
        "results": {
            "fold_macro_auroc": round(macro_auc, 4),
            "pooled_oof_auroc": round(pooled_auc, 4),
            "mean_train_auroc": round(float(np.mean([q['train_auroc'] for q in records])), 4),
            "overfit_gap_train_minus_test": round(train_gap, 4),
            "reference_orthogonal_suite": 0.9324,
            "reference_ridge_plus_lag1": 0.8300,
            "runtime_sec": round(time.time() - t0, 1),
        },
        "per_fold": records,
    }
    for out in OUT_FILES:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")

    print("\n--- EEGNet Empirical Baseline Summary ---")
    print(f"  fold-macro AUROC : {macro_auc:.4f}   (orthogonal suite 0.9324, ridge+rho1 0.8300)")
    print(f"  pooled OOF AUROC : {pooled_auc:.4f}")
    print(f"  mean TRAIN AUROC : {np.mean([q['train_auroc'] for q in records]):.4f}  "
          f"(overfit gap {train_gap:+.4f})")
    print("Archived:", ", ".join(str(p) for p in OUT_FILES))


if __name__ == "__main__":
    main()
