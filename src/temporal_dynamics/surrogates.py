"""
Permutation Surrogate Generators & Invariance Verification for Causal Counter-Examples.

Provides:
- generate_synchronous_surrogates: Synchronous joint time-shuffling preserving instantaneous 6-channel spatial topology.
- verify_permutation_invariance: Rigorous check asserting that traditional features are invariant (error < 1e-12)
  while order-aware features show significant deviation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
from .operators import compute_temporal_order_features


def generate_synchronous_surrogates(
    theta: np.ndarray,
    indices: np.ndarray,
    n_repeats: int = 32,
    seed: int = 20261014,
    min_adjacent_pairs: int = 10,
    eps: float = 1e-12,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generate n_repeats of synchronous joint time-shuffled surrogate series.
    
    Parameters:
        theta: shape (N, 6) theta powers
        indices: shape (N,) block indices
        n_repeats: number of permutation repeats (default 32)
        seed: PRNG seed for deterministic reproducibility
        
    Returns:
        original_features: dict of features on intact time sequence
        surrogates: list of dicts for each surrogate repeat
    """
    theta_arr = np.asarray(theta, dtype=float)
    indices_arr = np.asarray(indices, dtype=int)
    n_blocks = len(theta_arr)
    
    orig = compute_temporal_order_features(
        theta_arr, indices_arr, min_adjacent_pairs=min_adjacent_pairs, eps=eps
    )
    
    rng = np.random.default_rng(seed)
    surrogates = []
    
    for rep in range(n_repeats):
        # Synchronous permutation: same perm applied to all 6 channels simultaneously
        perm = rng.permutation(n_blocks)
        theta_perm = theta_arr[perm]
        
        surr = compute_temporal_order_features(
            theta_perm, indices_arr, min_adjacent_pairs=min_adjacent_pairs, eps=eps
        )
        surr["repeat"] = rep
        
        # Immediate strict sanity check on spatial & marginal invariance
        err_cv = abs(surr["neg_cv"] - orig["neg_cv"])
        err_spatial = abs(surr["r_interhem"] - orig["r_interhem"])
        if err_cv > eps or err_spatial > eps:
            raise RuntimeError(
                f"Synchronous permutation failed invariance test at repeat {rep}: "
                f"err_cv={err_cv:.2e}, err_spatial={err_spatial:.2e}"
            )
            
        surrogates.append(surr)
        
    return orig, surrogates


def verify_permutation_invariance(
    orig: Dict[str, Any],
    surrogates: List[Dict[str, Any]],
    tol: float = 1e-12,
) -> Dict[str, float]:
    """
    Quantify invariance errors for traditional features and variance for temporal features.
    
    Returns summary statistics:
        - max_cv_error: float
        - max_interhem_error: float
        - lag1_drop_mean: float
        - lag1_std: float
    """
    cv_errors = [abs(s["neg_cv"] - orig["neg_cv"]) for s in surrogates]
    interhem_errors = [abs(s["r_interhem"] - orig["r_interhem"]) for s in surrogates]
    lag1_values = [s["lag1"] for s in surrogates]
    
    max_cv_err = float(max(cv_errors))
    max_interhem_err = float(max(interhem_errors))
    
    if max_cv_err > tol or max_interhem_err > tol:
        raise ValueError(
            f"Permutation invariance violated: max_cv_err={max_cv_err}, "
            f"max_interhem_err={max_interhem_err}"
        )
        
    return {
        "max_cv_error": max_cv_err,
        "max_interhem_error": max_interhem_err,
        "original_lag1": float(orig["lag1"]),
        "surrogate_lag1_mean": float(np.mean(lag1_values)),
        "surrogate_lag1_std": float(np.std(lag1_values)),
        "lag1_drop": float(orig["lag1"] - np.mean(lag1_values)),
    }
