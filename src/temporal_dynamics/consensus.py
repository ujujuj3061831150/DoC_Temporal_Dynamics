"""
Consensus and Linear Fusion Estimators with Strict Subject-Level Training State Isolation.

Provides:
- SignedStandardizedMean: Linear signed equal-weight mean (Cell A / Cell C, Method C).
- TemporalMDCO: Non-parametric empirical CDF rank power-mean consensus (Cell B / Cell D, Method D).
- AdaptiveGatedConsensus: T-GECF hybrid operator with sample-wise local fluctuation gating.
- TunedMLBaseline: Scikit-learn Logistic, Ridge, and Shrinkage-LDA baselines with nested tuning.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import percentileofscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler


class SignedStandardizedMean:
    """
    Cell A & Cell C (Method C):
    Training-centered, signed equal-weight mean of standardized features.
    Supports optional Winsorized soft truncation with threshold c (e.g. c=2.0).
    """

    def __init__(self, c: Optional[float] = None, eps: float = 1e-12):
        self.c = float(c) if c is not None else None
        self.eps = eps
        self.scaler = StandardScaler()
        self.signs_: Optional[np.ndarray] = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SignedStandardizedMean":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if X_arr.ndim != 2 or len(X_arr) != len(y_arr) or not np.isfinite(X_arr).all():
            raise ValueError("Invalid training input matrix X or label vector y.")
        if set(y_arr) != {0, 1}:
            raise ValueError("Binary labels {0, 1} required.")

        self.scaler.fit(X_arr)
        signs = []
        for v in X_arr.T:
            if np.std(v) <= self.eps or np.corrcoef(v, y_arr)[0, 1] >= 0:
                signs.append(1.0)
            else:
                signs.append(-1.0)
        self.signs_ = np.array(signs, dtype=float)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.signs_ is None:
            raise RuntimeError("Estimator must be fitted before predict.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or not np.isfinite(X_arr).all():
            raise ValueError("Invalid prediction matrix X.")
        scaled = self.scaler.transform(X_arr)
        if self.c is not None:
            scaled = np.clip(scaled, -self.c, self.c)
        return (scaled * self.signs_).mean(axis=1)

    def state(self) -> Dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Estimator is not fitted.")
        return {
            "type": "signed_standardized_mean",
            "c": self.c,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "signs": self.signs_.tolist(),
        }


class TemporalMDCO:
    """
    Cell B & Cell D (Method D):
    Multiscale Dynamic Consensus Operator using non-parametric empirical CDF rank mapping.
    Matches standard MDCO (p=1.0, scaling='rank', equal weighting) evaluated on continuous DoC features.
    """

    def __init__(self, p: float = 1.0, eps: float = 1e-12):
        if float(p) <= 0:
            raise ValueError(f"Curvature parameter p must be strictly positive, got {p}")
        self.p = float(p)
        self.eps = eps
        self.signs_: Optional[np.ndarray] = None
        self.rank_training_values_: Optional[List[List[float]]] = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalMDCO":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if X_arr.ndim != 2 or len(X_arr) != len(y_arr) or not np.isfinite(X_arr).all():
            raise ValueError("Invalid training data.")
        if set(y_arr) != {0, 1}:
            raise ValueError("Binary labels {0, 1} required.")

        signs = []
        for v in X_arr.T:
            if np.std(v) <= self.eps or np.corrcoef(v, y_arr)[0, 1] >= 0:
                signs.append(1.0)
            else:
                signs.append(-1.0)
        self.signs_ = np.array(signs, dtype=float)
        
        signed_X = X_arr * self.signs_
        self.rank_training_values_ = [
            signed_X[:, k].copy().tolist() for k in range(X_arr.shape[1])
        ]
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.signs_ is None or self.rank_training_values_ is None:
            raise RuntimeError("Estimator must be fitted before predict.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or not np.isfinite(X_arr).all():
            raise ValueError("Invalid prediction matrix X.")
        if X_arr.shape[1] != len(self.signs_):
            raise ValueError(f"Feature count mismatch: expected {len(self.signs_)}, got {X_arr.shape[1]}.")
        signed_X = X_arr * self.signs_
        K = signed_X.shape[1]
        scores = np.zeros(len(X_arr), dtype=float)

        for k in range(K):
            train_dist = np.asarray(self.rank_training_values_[k], dtype=float)
            test_vals = signed_X[:, k]
            ranks = np.array([
                percentileofscore(train_dist, v, kind="mean") / 100.0
                for v in test_vals
            ])
            scores += (1.0 / K) * (ranks ** self.p)

        return scores ** (1.0 / self.p)

    def state(self) -> Dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Estimator is not fitted.")
        return {
            "type": "frozen_mdco",
            "p": self.p,
            "scaling": "rank",
            "K": len(self.signs_),
            "signs": self.signs_.tolist(),
            "rank_training_values": self.rank_training_values_,
        }


class AdaptiveGatedConsensus:
    """
    T-GECF: Fluctuation-Gated Adaptive Consensus Operator.
    
    Dynamically modulates the mixing ratio between non-parametric rank consensus (bounded in [0, 1])
    and standardized linear average using sample-level local volatility / feature dispersion:
        beta(x_i) = clip( beta_base + gamma * (v_i - mu_v) / sigma_v, 0.0, 1.0 )
        Score(x_i) = beta(x_i) * MDCO_norm(x_i) + (1 - beta(x_i)) * Linear_norm(x_i)
        
    When a subject exhibits high feature volatility / outlier dispersion v_i, beta increases
    towards 1.0, adaptively shifting weight to the bounded, non-parametric rank consensus operator (MDCO)
    to provide minimax anti-collapse protection against extreme electrode noise or sensor pops.
    When features are calm and mutually consistent, linear metric separability is prioritized.
    """

    def __init__(
        self,
        beta_base: float = 0.5,
        beta: Optional[float] = None,
        gamma: float = 0.0,
        theta_0: float = 1.645,
        kappa: float = 2.0,
        gating_type: str = "clip",
        c: Optional[float] = None,
        p: float = 1.0,
        eps: float = 1e-12,
    ):
        if beta is not None:
            self.beta_base = float(beta)
        else:
            self.beta_base = float(beta_base)
        self.beta = self.beta_base
        self.gamma = float(gamma)
        self.theta_0 = float(theta_0)
        self.kappa = float(kappa)
        self.gating_type = str(gating_type).lower()
        if self.gating_type not in {"clip", "sigmoid"}:
            raise ValueError(f"Unsupported gating_type: {gating_type}. Expected clip or sigmoid.")
        self.c = c
        self.p = float(p)
        self.eps = eps
        self.linear_model = SignedStandardizedMean(c=c, eps=eps)
        self.mdco_model = TemporalMDCO(p=p, eps=eps)
        self.train_score_stats_: Optional[Dict[str, float]] = None
        self.train_volatility_stats_: Optional[Dict[str, float]] = None
        self.is_fitted_ = False

    def _compute_local_volatility(self, X: np.ndarray) -> np.ndarray:
        """Compute sample-level stream dispersion / volatility across signed features."""
        signs = self.linear_model.signs_
        scaled = self.linear_model.scaler.transform(X) * signs
        return np.std(scaled, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveGatedConsensus":
        self.linear_model.fit(X, y)
        self.mdco_model.fit(X, y)
        
        lin_tr = self.linear_model.predict(X)
        mdco_tr = self.mdco_model.predict(X)
        self.train_score_stats_ = {
            "lin_mean": float(np.mean(lin_tr)),
            "lin_std": max(float(np.std(lin_tr)), self.eps),
            "mdco_mean": float(np.mean(mdco_tr)),
            "mdco_std": max(float(np.std(mdco_tr)), self.eps),
        }
        
        # Training volatility moments
        vol_tr = self._compute_local_volatility(X)
        self.train_volatility_stats_ = {
            "vol_mean": float(np.mean(vol_tr)),
            "vol_std": max(float(np.std(vol_tr)), self.eps),
        }
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.train_score_stats_ is None:
            raise RuntimeError("Estimator is not fitted.")
        
        lin_raw = self.linear_model.predict(X)
        mdco_raw = self.mdco_model.predict(X)
        
        stats = self.train_score_stats_
        lin_norm = (lin_raw - stats["lin_mean"]) / stats["lin_std"]
        mdco_norm = (mdco_raw - stats["mdco_mean"]) / stats["mdco_std"]
        
        # Sample-specific local fluctuation gating: high volatility -> shift to bounded MDCO
        vols = self._compute_local_volatility(X)
        v_stats = self.train_volatility_stats_
        vol_z = (vols - v_stats["vol_mean"]) / v_stats["vol_std"]
        
        if self.gating_type == "sigmoid":
            exp_arg = np.clip(-self.kappa * (vol_z - self.theta_0), -50.0, 50.0)
            betas = 1.0 / (1.0 + np.exp(exp_arg))
        else:
            betas = np.clip(self.beta_base + self.gamma * vol_z, 0.0, 1.0)
        return betas * mdco_norm + (1.0 - betas) * lin_norm

    def state(self) -> Dict[str, Any]:
        return {
            "type": "adaptive_gated_consensus",
            "gating_type": self.gating_type,
            "theta_0": self.theta_0,
            "kappa": self.kappa,
            "c": self.c,
            "beta_base": self.beta_base,
            "gamma": self.gamma,
            "p": self.p,
            "train_score_stats": self.train_score_stats_,
            "train_volatility_stats": self.train_volatility_stats_,
            "linear_state": self.linear_model.state(),
            "mdco_state": self.mdco_model.state(),
        }


class TunedMLBaseline:
    """
    Scikit-learn ML baselines (Logistic, Ridge, LDA shrinkage) fitted on standardized inputs.
    """

    def __init__(self, method: str = "logistic", parameter: Any = 1.0):
        self.method = method
        self.parameter = parameter
        self.scaler = StandardScaler()
        self.model_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TunedMLBaseline":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self.scaler.fit(X_arr)
        scaled_X = self.scaler.transform(X_arr)
        
        if self.method == "logistic":
            self.model_ = LogisticRegression(
                C=float(self.parameter), solver="lbfgs", max_iter=2000
            )
        elif self.method == "ridge":
            self.model_ = RidgeClassifier(alpha=float(self.parameter))
        elif self.method == "lda_shrinkage":
            self.model_ = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage=self.parameter
            )
        else:
            raise ValueError(f"Unknown baseline method: {self.method}")
            
        self.model_.fit(scaled_X, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Estimator must be fitted before predict.")
        scaled_X = self.scaler.transform(X)
        return self.model_.decision_function(scaled_X)
