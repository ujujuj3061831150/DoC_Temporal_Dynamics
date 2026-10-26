"""
Order-Aware Temporal Operators and Residual Resonance Extraction for Bedside EEG.

Implements:
1. Lag-1 autoregressive memory persistence (rho_1) on physically adjacent blocks.
2. Normalized negative adjacent jump rate (D_1).
3. Residual inter-hemispheric bilateral resonance (r_residual) via common-mode regression.
4. Traditional permutation-invariant baseline features (-CV_theta, r_interhem).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np


def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Pearson correlation coefficient with degenerate variance safeguard."""
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return 0.0
    std_a = float(np.std(a_arr))
    std_b = float(np.std(b_arr))
    if std_a < eps or std_b < eps:
        return 0.0
    c = float(np.corrcoef(a_arr, b_arr)[0, 1])
    return 0.0 if not np.isfinite(c) else float(np.clip(c, -1.0, 1.0))


def extract_clean_theta_series(
    npz_path: Union[str, Path],
    peak_threshold: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 6-channel EEG theta powers and block indices meeting artifact quality criteria.
    
    npz contents:
        'powers': shape (N_total, 2, 6) where band 0 is theta (4-8 Hz), band 1 is alpha.
        'peaks': shape (N_total,) peak amplitude in microvolts.
    
    Returns:
        theta: (N_clean, 6) float array
        indices: (N_clean,) int array of original recording block indices
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG cache file not found: {path}")
    
    with np.load(path, allow_pickle=False) as data:
        peaks = data["peaks"]
        powers = data["powers"]
        
    good = peaks < peak_threshold
    clean_indices = np.flatnonzero(good)
    clean_theta = powers[good, 0, :].astype(float)
    return clean_theta, clean_indices


def residual_bilateral_covariance(
    powers: np.ndarray,
    left_right_pairs: Sequence[Tuple[int, int]] = ((0, 1), (2, 3), (4, 5)),
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Compute raw bilateral correlation, residual correlation, and common-mode R^2.
    
    For each (left, right) symmetric pair:
        1. Regress out the mean of remaining 4 channels (global nuisance covariate).
        2. Correlate OLS projection residuals.
    
    Returns:
        mean_raw_interhem: float
        mean_residual_corr: float
        mean_r2: float
    """
    powers = np.asarray(powers, dtype=float)
    if powers.ndim != 2 or powers.shape[1] != 6:
        raise ValueError("6-channel power matrix required.")
    if len(powers) < 2:
        return 0.0, 0.0, 0.0
    
    n_blocks = len(powers)
    raw_corrs = []
    res_corrs = []
    r2_all = []
    
    for left, right in left_right_pairs:
        raw_corrs.append(safe_corr(powers[:, left], powers[:, right], eps=eps))
        
        other_indices = [k for k in range(6) if k not in (left, right)]
        other_mean = powers[:, other_indices].mean(axis=1)
        
        # Center and scale nuisance regressor for numerical stability
        z_std = float(np.std(other_mean))
        z = (other_mean - float(np.mean(other_mean))) / (z_std if z_std > eps else 1.0)
        
        design = np.column_stack([np.ones(n_blocks, dtype=float), z])
        values = powers[:, [left, right]]
        
        # Least squares projection
        coefs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        residuals = values - design @ coefs
        
        # Relative variance safeguard against floating-point roundoff artifacts
        var_orig = np.var(values, axis=0)
        var_res = np.var(residuals, axis=0)
        if (var_res[0] <= np.maximum(eps, 1e-10 * var_orig[0])) or (var_res[1] <= np.maximum(eps, 1e-10 * var_orig[1])):
            res_corrs.append(0.0)
        else:
            res_corrs.append(safe_corr(residuals[:, 0], residuals[:, 1], eps=eps))
        
        denom = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
        ss_res = np.sum(residuals ** 2, axis=0)
        zero_var = denom <= eps
        r2 = np.where(zero_var, 0.0, 1.0 - ss_res / np.maximum(denom, eps))
        r2_all.extend(np.clip(r2, 0.0, 1.0).tolist())
        
    return float(np.mean(raw_corrs)), float(np.mean(res_corrs)), float(np.mean(r2_all))


def compute_temporal_order_features(
    theta: np.ndarray,
    indices: np.ndarray,
    min_adjacent_pairs: int = 10,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Extract order-aware temporal dynamics and spatial controls from 6-channel theta power series.
    
    Parameters:
        theta: shape (N, 6) theta powers for N clean blocks.
        indices: shape (N,) original recording temporal block indices.
        min_adjacent_pairs: minimum number of physically consecutive blocks required.
        eps: numerical stability constant.
        
    Returns:
        Dict containing:
            - 'neg_cv': traditional negative theta coefficient of variation (-std/mean)
            - 'r_interhem': traditional average bilateral Pearson correlation
            - 'lag1': order-aware lag-1 persistence rho_1 on physical adjacent pairs
            - 'neg_increment': order-aware normalized negative adjacent jump rate D_1
            - 'residual_corr': common-mode residual bilateral correlation r_residual
            - 'commonmode_mean_r2': average variance explained by global common mode (around 47%)
            - 'clean_blocks': total clean blocks analyzed
            - 'adjacent_pairs': number of verified physically adjacent pairs
    """
    theta = np.asarray(theta, dtype=float)
    indices = np.asarray(indices, dtype=int)
    
    if theta.ndim != 2 or theta.shape[1] != 6 or len(theta) != len(indices):
        raise ValueError("theta must be shape (N, 6) matching indices length (N,).")
    
    adjacent = (np.diff(indices) == 1)
    n_adj = int(adjacent.sum())
    if n_adj < min_adjacent_pairs:
        raise ValueError(
            f"Insufficient physical adjacent pairs: {n_adj} found, {min_adjacent_pairs} required."
        )
    
    g = theta.mean(axis=1)
    mu = float(g.mean())
    if not np.isfinite(theta).all() or mu <= 0:
        raise ValueError("Theta powers contain non-finite values or zero mean.")
    
    # 1. Permutation-invariant CV
    cv = -float(g.std() / (mu + eps))
    
    # 2. Order-aware Lag-1 Autocorrelation (rho_1)
    g_curr = g[:-1][adjacent]
    g_next = g[1:][adjacent]
    lag1 = safe_corr(g_curr, g_next, eps=eps)
    
    # 3. Order-aware Normalized Negative Jump Rate (D_1)
    diff_abs_mean = float(np.abs(np.diff(g)[adjacent]).mean())
    neg_increment = -float(diff_abs_mean / (mu + eps))
    
    # 4. Bilateral and residual covariance
    r_interhem, residual_corr, commonmode_r2 = residual_bilateral_covariance(theta, eps=eps)
    
    return {
        "neg_cv": cv,
        "r_interhem": r_interhem,
        "lag1": lag1,
        "neg_increment": neg_increment,
        "residual_corr": residual_corr,
        "commonmode_mean_r2": commonmode_r2,
        "clean_blocks": len(theta),
        "adjacent_pairs": n_adj,
    }
