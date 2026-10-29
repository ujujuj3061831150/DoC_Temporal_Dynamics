"""
Production Engine for Longitudinal Follow-up Audit, Linkage Verification,
and Pre-specified Prognostic Statistical Analysis.

This module implements:
1. Master traceability ingestion across DoC cohorts (Figshare, Wielek, Mendeley, CRRC).
2. Fail-closed linkage gating (preventing unverified pairings or guessed labels from entering prognostic analysis).
3. Frozen model state cryptographic hash verification.
4. Pre-specified statistical testing (Mann-Whitney U, Logistic Regression, Likelihood Ratio Test for incremental value over CRS-R).
"""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix


class NoConfirmedFollowupPairsError(Exception):
    """Raised when an analysis requires verified longitudinal pairs, but none are confirmed."""
    pass


class FrozenModelHashMismatchError(Exception):
    """Raised when the serialized frozen model state has been tampered with or modified."""
    pass


def get_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pytest.ini").exists() or (parent / "results").exists():
            return parent
    return current.parents[1]


class FollowupAuditEngine:
    """Audits longitudinal follow-up data availability and linkage traceability."""

    def __init__(self, master_path: Optional[Union[str, Path]] = None):
        self.root = get_repo_root()
        self.master_path = Path(master_path) if master_path else self.root / "results/followup_analysis_v1/followup_traceability_master.csv"
        self.records: List[Dict[str, Any]] = []
        self._load_master()

    def _load_master(self) -> None:
        if not self.master_path.exists():
            raise FileNotFoundError(f"Missing master traceability file: {self.master_path}")
        with open(self.master_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)

    def summary_by_cohort(self) -> Dict[str, Dict[str, Any]]:
        """Compute breakdown of enrolled patients, follow-up availability, and linkage status."""
        summary: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            c = r["cohort_name"]
            if c not in summary:
                summary[c] = {
                    "total_enrolled": 0,
                    "confirmed_pairs": 0,
                    "ambiguous_candidates": 0,
                    "missing_information": 0,
                    "reported_followup_scale": r["outcome_scale"],
                    "source": r["source"]
                }
            summary[c]["total_enrolled"] += 1
            st = r["pairing_status"]
            if st == "CONFIRMED_PAIR":
                summary[c]["confirmed_pairs"] += 1
            elif st == "AMBIGUOUS_CANDIDATE":
                summary[c]["ambiguous_candidates"] += 1
            else:
                summary[c]["missing_information"] += 1
        return summary

    def get_actionable_pairs(self, allow_ambiguous: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve actionable records under strict ethical verification rules.
        Default: only CONFIRMED_PAIR records are actionable.
        """
        valid_statuses = {"CONFIRMED_PAIR"}
        if allow_ambiguous:
            valid_statuses.add("AMBIGUOUS_CANDIDATE")
        return [r for r in self.records if r["pairing_status"] in valid_statuses]


class PrognosticValidationSuite:
    """Executes pre-registered prognostic statistical analysis on verified cohorts."""

    EXPECTED_FROZEN_HASH = "e0d608e05f23bdabd7a1f45610adefad10785b03898f1b8daa126e9c7140fb6a"

    def __init__(self, fitted_states_path: Optional[Union[str, Path]] = None):
        self.root = get_repo_root()
        self.states_path = Path(fitted_states_path) if fitted_states_path else self.root / "results/matched_model_comparison_v1/fitted_states.json"
        self.verify_frozen_hash()
        self._load_frozen_params()

    def verify_frozen_hash(self) -> str:
        if not self.states_path.exists():
            raise FileNotFoundError(f"Missing fitted states archive: {self.states_path}")
        content = self.states_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != self.EXPECTED_FROZEN_HASH:
            raise FrozenModelHashMismatchError(
                f"Fitted states SHA256 mismatch! Expected {self.EXPECTED_FROZEN_HASH}, got {actual_hash}"
            )
        return actual_hash

    def _load_frozen_params(self) -> None:
        with open(self.states_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        fig_state = d["figshare50/orthogonal_lag1/source_frozen"]["source"]
        self.feature_names = fig_state["features"]
        self.mean = np.array(fig_state["mean"], dtype=float)
        self.scale = np.array(fig_state["scale"], dtype=float)
        self.signs = np.array(fig_state["signs"], dtype=float)
        self.threshold_neutral = float(fig_state["thresholds"]["equal_mean"])
        self.threshold_c15 = float(fig_state["thresholds"]["equal_mean_c15"])

    def compute_score(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        """Compute standard S(x) and soft-truncated S_1.5(x)."""
        z = self.signs * (feature_vector - self.mean) / self.scale
        score = float(np.mean(z))
        score_c15 = float(np.mean(np.clip(z, -1.5, 1.5)))
        return score, score_c15

    def run_prognostic_association(
        self,
        scores: np.ndarray,
        outcomes: np.ndarray,
        crs_r_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Execute pre-specified statistical hypotheses:
        1. Mann-Whitney U test (rank-biserial correlation).
        2. Univariate Logistic Regression (OR, 95% CI, p-value).
        3. Multivariable Logistic Regression adjusted for baseline CRS-R (if provided).
        4. Operating point metrics at fixed threshold.
        """
        n_samples = len(scores)
        if n_samples < 2:
            raise ValueError(f"Insufficient samples for prognostic association: N={n_samples}")

        y = np.array(outcomes, dtype=int)
        unique_y = np.unique(y)
        if len(unique_y) < 2:
            raise ValueError(f"Outcome vector contains only one class: {unique_y}")

        # 1. Mann-Whitney U test
        group1 = scores[y == 1]
        group0 = scores[y == 0]
        u_stat, p_mann = stats.mannwhitneyu(group1, group0, alternative="two-sided")
        # Rank biserial correlation r = 1 - 2U / (n1 * n0)
        n1 = len(group1)
        n0 = len(group0)
        rank_biserial = float(1.0 - (2.0 * u_stat) / (n1 * n0))

        # 2. Univariate Logistic Regression (unpenalized via large C)
        X_score = scores.reshape(-1, 1)
        clf = LogisticRegression(C=1e9, solver="lbfgs")
        clf.fit(X_score, y)
        coef = float(clf.coef_[0][0])
        intercept = float(clf.intercept_[0])
        odds_ratio = float(np.exp(coef))

        # Standard error & Wald CI via Hessian of log-likelihood
        p_pred = clf.predict_proba(X_score)[:, 1]
        w = p_pred * (1.0 - p_pred)
        w = np.maximum(w, 1e-6)
        x_design = np.column_stack([np.ones(n_samples), scores])
        hessian = x_design.T @ np.diag(w) @ x_design
        try:
            cov = np.linalg.inv(hessian)
            se_coef = float(np.sqrt(cov[1, 1]))
            z_crit = 1.95996
            ci_lower = float(np.exp(coef - z_crit * se_coef))
            ci_upper = float(np.exp(coef + z_crit * se_coef))
            p_wald = float(2.0 * (1.0 - stats.norm.cdf(abs(coef / se_coef))))
        except Exception:
            se_coef = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            p_wald = np.nan

        # AUC
        try:
            auc = float(roc_auc_score(y, scores))
        except Exception:
            auc = np.nan

        # 3. Fixed Operating Point (Threshold theta_neutral = 0.0451)
        y_pred = (scores >= self.threshold_neutral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        acc = float((tp + tn) / n_samples)

        results: Dict[str, Any] = {
            "n_total": n_samples,
            "n_favorable": int(n1),
            "n_unfavorable": int(n0),
            "mann_whitney": {
                "u_statistic": float(u_stat),
                "p_value": float(p_mann),
                "rank_biserial_correlation": rank_biserial,
            },
            "univariate_logistic": {
                "coef_per_sd": coef,
                "odds_ratio": odds_ratio,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "std_error": se_coef,
                "wald_p_value": p_wald,
                "auc": auc,
            },
            "fixed_operating_point": {
                "threshold": self.threshold_neutral,
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "npv": npv,
                "accuracy": acc,
                "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            },
        }

        # 4. Multivariable with CRS-R (if provided and sufficient sample size)
        if crs_r_scores is not None and len(crs_r_scores) == n_samples and not np.isnan(crs_r_scores).any():
            X_crs = crs_r_scores.reshape(-1, 1)
            clf_crs = LogisticRegression(C=1e9, solver="lbfgs").fit(X_crs, y)
            p_crs = clf_crs.predict_proba(X_crs)[:, 1]
            auc_crs = float(roc_auc_score(y, p_crs))
            ll_crs = float(np.sum(y * np.log(np.maximum(p_crs, 1e-12)) + (1 - y) * np.log(np.maximum(1 - p_crs, 1e-12))))

            # Full model: CRS-R + score
            X_full = np.column_stack([crs_r_scores, scores])
            clf_full = LogisticRegression(C=1e9, solver="lbfgs").fit(X_full, y)
            p_full = clf_full.predict_proba(X_full)[:, 1]
            auc_full = float(roc_auc_score(y, p_full))
            ll_full = float(np.sum(y * np.log(np.maximum(p_full, 1e-12)) + (1 - y) * np.log(np.maximum(1 - p_full, 1e-12))))

            # Likelihood Ratio Test (LRT)
            lr_stat = float(2.0 * (ll_full - ll_crs))
            p_lrt = float(stats.chi2.sf(max(lr_stat, 0.0), df=1))

            results["multivariable_incremental_value"] = {
                "crs_only_auc": auc_crs,
                "full_model_auc": auc_full,
                "delta_auc": float(auc_full - auc_crs),
                "log_likelihood_crs": ll_crs,
                "log_likelihood_full": ll_full,
                "likelihood_ratio_statistic": lr_stat,
                "lrt_p_value": p_lrt,
                "score_coef_adjusted": float(clf_full.coef_[0][1]),
                "score_or_adjusted": float(np.exp(clf_full.coef_[0][1])),
            }

        return results
