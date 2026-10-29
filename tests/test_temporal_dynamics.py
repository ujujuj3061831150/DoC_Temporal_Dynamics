"""
Unit and Integration Test Suite for Temporal Dynamics & Consensus Fusion (Temporal-MDCO / T-GECF).
"""

from pathlib import Path
import sys
import numpy as np
import pytest

CURRENT = Path(__file__).resolve()
REPO_ROOT = next((p for p in CURRENT.parents if (p / "results/multiscale_matched_v2").exists() or (p / "pytest.ini").exists()), CURRENT.parents[1])
V3_ROOT = next((p for p in CURRENT.parents if p.name == "v3"), CURRENT.parents[1])
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
from src.temporal_dynamics.surrogates import (
    generate_synchronous_surrogates,
    verify_permutation_invariance,
)
from src.temporal_dynamics.adapters import (
    load_mendeley_temporal_dataset,
    compute_wielek_discrete_persistence,
)


def test_physical_gap_preservation():
    """Ensure temporal dynamics never artificially bridge gaps created by artifact rejection."""
    # 4 time blocks: indices 0, 1, 10, 11 -> only 2 adjacent pairs: (0,1) and (10,11)
    powers = np.ones((4, 6), dtype=float)
    powers[0] = 10.0
    powers[1] = 20.0
    powers[2] = 100.0
    powers[3] = 110.0
    indices = np.array([0, 1, 10, 11])
    
    feats = compute_temporal_order_features(powers, indices, min_adjacent_pairs=2)
    assert feats["adjacent_pairs"] == 2
    # If gap (1 to 10) was bridged, jump between 20 and 100 would skew diff; verify it only computes (20-10) and (110-100)
    expected_diff_mean = (10.0 + 10.0) / 2.0
    expected_mu = float(powers.mean())
    assert feats["neg_increment"] == pytest.approx(-expected_diff_mean / expected_mu, abs=1e-10)
    
    with pytest.raises(ValueError, match="Insufficient physical adjacent pairs"):
        compute_temporal_order_features(powers, indices, min_adjacent_pairs=3)


def test_synchronous_permutation_invariance():
    """Verify that synchronous joint permutation strictly preserves CV and interhem correlation."""
    rng = np.random.default_rng(2026)
    n_blocks = 120
    # Simulate 6-channel series with temporal autocorrelation
    t = np.linspace(0, 10, n_blocks)
    base_signal = np.sin(t) + 3.0
    powers = np.zeros((n_blocks, 6))
    for c in range(6):
        powers[:, c] = base_signal + rng.normal(0, 0.1, n_blocks)
    indices = np.arange(n_blocks)
    
    orig, surrs = generate_synchronous_surrogates(
        powers, indices, n_repeats=10, seed=42, min_adjacent_pairs=10
    )
    
    stats = verify_permutation_invariance(orig, surrs, tol=1e-12)
    assert stats["max_cv_error"] < 1e-12
    assert stats["max_interhem_error"] < 1e-12
    # Time-shuffling must noticeably drop the smooth sine autocorrelation
    assert stats["lag1_drop"] > 0.4


def test_residual_regression_orthogonality():
    """Verify that OLS projection residuals are strictly orthogonal to the nuisance regressor."""
    rng = np.random.default_rng(888)
    n_blocks = 200
    # Create strong global common mode
    common_mode = rng.normal(0, 5, n_blocks)
    powers = np.zeros((n_blocks, 6))
    for c in range(6):
        powers[:, c] = 50.0 + common_mode + rng.normal(0, 1, n_blocks)
        
    raw_r, res_r, mean_r2 = residual_bilateral_covariance(powers)
    assert raw_r > 0.90  # Confounded by massive common mode
    assert res_r < 0.30  # Stripped down by orthogonalization
    assert mean_r2 > 0.70  # Common mode explains >70% variance in this test


def test_zero_leakage_estimator_state():
    """Verify that predicting on test data never mutates the fitted state of estimators."""
    rng = np.random.default_rng(999)
    X_train = rng.normal(size=(30, 5))
    y_train = np.array([0] * 15 + [1] * 15)
    
    # 1. SignedStandardizedMean
    lin = SignedStandardizedMean().fit(X_train, y_train)
    state_before = lin.state()
    _ = lin.predict(np.zeros((5, 5)))
    _ = lin.predict(np.full((5, 5), 1e8))
    assert lin.state() == state_before
    
    # 2. TemporalMDCO
    mdco = TemporalMDCO(p=1.0).fit(X_train, y_train)
    state_mdco_before = mdco.state()
    _ = mdco.predict(np.zeros((5, 5)))
    _ = mdco.predict(np.full((5, 5), 1e8))
    assert mdco.state() == state_mdco_before
    
    # 3. AdaptiveGatedConsensus
    gated = AdaptiveGatedConsensus(beta=0.4).fit(X_train, y_train)
    state_gated_before = gated.state()
    _ = gated.predict(np.zeros((5, 5)))
    _ = gated.predict(np.full((5, 5), 1e8))
    assert gated.state() == state_gated_before


def test_adaptive_gated_consensus_interpolation():
    """Verify that AdaptiveGatedConsensus smoothly interpolates between linear and MDCO models."""
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(20, 4))
    y_train = np.array([0] * 10 + [1] * 10)
    X_test = rng.normal(size=(6, 4))
    
    m_lin = SignedStandardizedMean().fit(X_train, y_train)
    m_mdco = TemporalMDCO(p=1.0).fit(X_train, y_train)
    
    gated_mdco = AdaptiveGatedConsensus(beta=1.0).fit(X_train, y_train)
    gated_lin = AdaptiveGatedConsensus(beta=0.0).fit(X_train, y_train)
    
    # Standardized scores must match perfectly
    s_lin = m_lin.predict(X_test)
    s_lin_norm = (s_lin - np.mean(m_lin.predict(X_train))) / np.std(m_lin.predict(X_train))
    np.testing.assert_allclose(gated_lin.predict(X_test), s_lin_norm, atol=1e-10)
    
    s_mdco = m_mdco.predict(X_test)
    s_mdco_norm = (s_mdco - np.mean(m_mdco.predict(X_train))) / np.std(m_mdco.predict(X_train))
    np.testing.assert_allclose(gated_mdco.predict(X_test), s_mdco_norm, atol=1e-10)


def test_mendeley_cohort_adapter_integration():
    """Verify that the Mendeley adapter accurately loads all 41 patients and computes dynamics."""
    data = load_mendeley_temporal_dataset(repo_root=REPO_ROOT)
    assert len(data["subject_ids"]) == 41
    assert data["labels"].sum() == 12  # 12 MCS, 29 VS/UWS
    assert "canonical" in data["feature_matrix_variants"]
    assert "plus_lag1" in data["feature_matrix_variants"]
    assert data["feature_matrix_variants"]["canonical"].shape == (41, 4)
    assert data["feature_matrix_variants"]["plus_lag1"].shape == (41, 5)


def test_wielek_discrete_persistence_adapter():
    """Verify that Wielek adapter computes discrete sleep stage persistence."""
    wielek_results = compute_wielek_discrete_persistence()
    assert len(wielek_results) == 23
    for sid, info in wielek_results.items():
        assert 0.0 <= info["discrete_lag1_persistence"] <= 1.0
        assert info["valid_adjacent_pairs"] > 500


def test_tuned_ml_baselines():
    """Verify that TunedMLBaseline correctly fits Logistic, Ridge, and Shrinkage-LDA models."""
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(20, 4))
    y_tr = np.array([0] * 10 + [1] * 10)
    X_te = rng.normal(size=(5, 4))

    for method, param in [
        ("logistic", 1.0),
        ("ridge", 0.5),
        ("lda_shrinkage", "auto"),
    ]:
        model = TunedMLBaseline(method=method, parameter=param).fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == 5
        assert np.isfinite(preds).all()


def test_local_volatility_gating_behavior():
    """Verify that sample-level volatility modulation dynamically adjusts blending weight."""
    rng = np.random.default_rng(77)
    X_tr = rng.normal(size=(25, 4))
    y_tr = np.array([0] * 12 + [1] * 13)

    # Sample A: low volatility across features; Sample B: high volatility / extreme dispersion
    X_te = np.array([
        [1.0, 1.0, 1.0, 1.0],      # uniform / low volatility
        [-5.0, 5.0, -5.0, 5.0],    # extreme conflict / high volatility
    ])

    model_base = AdaptiveGatedConsensus(beta_base=0.5, gamma=0.0).fit(X_tr, y_tr)
    model_gated = AdaptiveGatedConsensus(beta_base=0.5, gamma=0.3).fit(X_tr, y_tr)

    p_base = model_base.predict(X_te)
    p_gated = model_gated.predict(X_te)

    # For high volatility sample, gated score should shift closer to pure bounded MDCO
    mdco_preds = model_gated.mdco_model.predict(X_te)
    stats = model_gated.train_score_stats_
    mdco_norm = (mdco_preds - stats["mdco_mean"]) / stats["mdco_std"]

    # Difference from MDCO should be smaller for the high-volatility sample in the gated model
    diff_base = abs(p_base[1] - mdco_norm[1])
    diff_gated = abs(p_gated[1] - mdco_norm[1])
    assert diff_gated <= diff_base + 1e-10


def test_artifact_threshold_boundary(tmp_path):
    """Explicitly verify the strict 500.0 uV rejection boundary in extract_clean_theta_series."""
    from src.temporal_dynamics.operators import extract_clean_theta_series
    
    # 4 blocks: 499.9 uV (pass), 500.0 uV (reject), 500.1 uV (reject), 100.0 uV (pass)
    powers = np.ones((4, 2, 6), dtype=float) * 10.0
    peaks = np.array([499.9, 500.0, 500.1, 100.0], dtype=float)
    
    test_npz = tmp_path / "test_boundary.npz"
    np.savez(test_npz, powers=powers, peaks=peaks)
    
    clean_theta, clean_indices = extract_clean_theta_series(test_npz)
    assert len(clean_indices) == 2
    np.testing.assert_array_equal(clean_indices, [0, 3])
    np.testing.assert_allclose(clean_theta, np.ones((2, 6)) * 10.0)


def test_benchmark_script_symbols_and_smoke():
    """Verify that run_temporal_mdco_benchmark has no missing or undefined global symbols."""
    import importlib.util
    bench_path = V3_ROOT / "scripts/run_temporal_mdco_benchmark.py"
    spec = importlib.util.spec_from_file_location("bench_module", bench_path)
    bench_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench_mod)
    
    # Check that core helper functions and paths are cleanly resolvable
    assert hasattr(bench_mod, "run_benchmark")
    assert hasattr(bench_mod, "REPO_ROOT")
    assert hasattr(bench_mod, "V3_ROOT")
    assert (bench_mod.REPO_ROOT / "data").exists()
    assert (bench_mod.V3_ROOT / "scripts").exists()

