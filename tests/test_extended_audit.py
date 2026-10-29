"""Extended Data Integrity, Mathematical Rigor, and Boundary Stress Tests."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

CURRENT = Path(__file__).resolve()
V3_ROOT = next((p for p in CURRENT.parents if p.name == "v3"), CURRENT.parents[1])
REPO_ROOT = next((p for p in CURRENT.parents if (p / "results/multiscale_matched_v2").exists() or (p / "pytest.ini").exists()), CURRENT.parents[1])
sys.path.insert(0, str(V3_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.temporal_dynamics.operators import (
    compute_temporal_order_features,
    residual_bilateral_covariance,
    safe_corr,
)
from src.temporal_dynamics.consensus import (
    SignedStandardizedMean,
    TemporalMDCO,
    AdaptiveGatedConsensus,
    TunedMLBaseline,
)
from src.temporal_dynamics.adapters import (
    load_mendeley_temporal_dataset,
)
from scripts.run_temporal_mdco_benchmark import (
    bootstrap_contrasts,
    compute_fold_macro_auc,
)


def test_operators_extreme_boundaries():
    # 1. All-zero powers
    with pytest.raises(ValueError, match="Theta powers contain non-finite values or zero mean"):
        compute_temporal_order_features(np.zeros((50, 6)), np.arange(50))
    # 2. Constant positive signal (zero variance)
    f_const = compute_temporal_order_features(np.full((50, 6), 5.0), np.arange(50))
    assert np.isfinite(list(f_const.values())).all()
    assert f_const["lag1"] == 0.0
    # 3. Broken single channel (dead lead)
    theta_dead = np.random.default_rng(42).uniform(1.0, 10.0, size=(50, 6))
    theta_dead[:, 0] = 0.0
    f_dead = compute_temporal_order_features(theta_dead, np.arange(50))
    assert np.isfinite(list(f_dead.values())).all()
    # 4. Adjacent pairs boundary: 9 fails, 10 passes
    theta_adj = np.ones((20, 6)) * 2.0
    idx_9 = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27]
    with pytest.raises(ValueError, match="Insufficient physical adjacent pairs"):
        compute_temporal_order_features(theta_adj[:len(idx_9)], np.array(idx_9), min_adjacent_pairs=10)
    idx_10 = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28]
    f_10 = compute_temporal_order_features(theta_adj, np.array(idx_10), min_adjacent_pairs=10)
    assert f_10["adjacent_pairs"] == 10
    # 5. Reviewer M5: 6 identical channels with large scale (pure common mode)
    g = np.arange(1, 101, dtype=float)
    for scale in [1.0, 1e4, 1e8]:
        powers_common = np.column_stack([g * scale] * 6)
        _, res_corr, r2 = residual_bilateral_covariance(powers_common)
        assert np.isclose(r2, 1.0)
        assert res_corr == 0.0, f"Residual correlation should be 0.0 for pure common mode at scale {scale}, got {res_corr}"


def test_consensus_numerical_stress():
    X_tr = np.random.default_rng(1).normal(size=(20, 4))
    y_tr = np.array([0] * 10 + [1] * 10)
    # 1. Constant training feature in SignedStandardizedMean
    X_c = X_tr.copy()
    X_c[:, 0] = 5.0
    m_ssm = SignedStandardizedMean().fit(X_c, y_tr)
    p_ssm = m_ssm.predict(np.array([[5.0, 1.0, 2.0, 3.0]]))
    assert np.isfinite(p_ssm).all()
    # 2. Extreme out-of-distribution values in TemporalMDCO
    m_mdco = TemporalMDCO(p=1.0).fit(X_tr, y_tr)
    p_mdco_ood = m_mdco.predict(np.array([1e15 * m_mdco.signs_, -1e15 * m_mdco.signs_]))
    assert np.isfinite(p_mdco_ood).all()
    assert p_mdco_ood[0] == pytest.approx(1.0, abs=1e-5)
    assert p_mdco_ood[1] == pytest.approx(0.0, abs=1e-5)
    # 3. AdaptiveGatedConsensus with zero volatility variance
    X_flat_vol = np.tile([1.0, 2.0, 3.0, 4.0], (20, 1))
    X_flat_vol[:10, 0] += 0.01
    m_gated = AdaptiveGatedConsensus(beta_base=0.5, gamma=0.2).fit(X_flat_vol, y_tr)
    p_gated = m_gated.predict(X_flat_vol[:5])
    assert np.isfinite(p_gated).all()
    # 4. Collinear features in TunedMLBaseline
    X_coll = np.column_stack([X_tr, X_tr[:, 0]])
    for method, param in [("lda_shrinkage", "auto"), ("ridge", 1.0), ("logistic", 1.0)]:
        model = TunedMLBaseline(method=method, parameter=param).fit(X_coll, y_tr)
        preds = model.predict(X_coll[:3])
        assert np.isfinite(preds).all()


def test_dirichlet_bootstrap_mathematical_properties():
    pred_csv = V3_ROOT / "results/temporal_dynamics_v1/test_predictions.csv"
    pred_df = pd.read_csv(pred_csv)
    cdf_1 = bootstrap_contrasts(pred_df, "canonical__C", ["plus_lag1__C"], n_draws=3000, seed=42)
    cdf_2 = bootstrap_contrasts(pred_df, "canonical__C", ["plus_lag1__C"], n_draws=3000, seed=20261014)
    pt_diff1 = cdf_1.iloc[0]["point_difference"]
    pt_diff2 = cdf_2.iloc[0]["point_difference"]
    assert pt_diff1 == pytest.approx(pt_diff2, abs=1e-10)
    assert cdf_1.iloc[0]["prob_positive"] > 0.95
    assert cdf_2.iloc[0]["prob_positive"] > 0.95


def test_feature_variant_alignment():
    ds = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    assert len(ds["subject_ids"]) == 41
    vars = ds["feature_matrix_variants"]
    for k in [
        "canonical", "plus_lag1", "plus_increment",
        "replace_cv_lag1", "replace_cv_increment", "residual_corr",
        "plus_lag1_residual", "drop_c_spatial", "drop_c_global"
    ]:
        assert k in vars
        assert vars[k].shape[0] == 41
        assert np.isfinite(vars[k]).all()
    assert vars["drop_c_spatial"].shape[1] == 4
    assert vars["drop_c_global"].shape[1] == 4
    assert vars["plus_lag1_residual"].shape[1] == 5
