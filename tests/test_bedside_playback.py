"""
Unit and regression tests for Bedside Streaming and Playback Engine.
"""

from __future__ import annotations
import sys
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest

CURRENT = Path(__file__).resolve()
ROOT = next((p for p in CURRENT.parents if (p / "results/multiscale_matched_v2").exists() or (p / "data/Mendeley_Data").exists()), CURRENT.parents[1])
CODE_ROOT = CURRENT.parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from temporal_dynamics.streaming import BedsideStreamingEngine
from temporal_dynamics.adapters import load_mendeley_temporal_dataset
from temporal_dynamics.operators import compute_temporal_order_features
from scipy.signal import welch


def test_streaming_engine_initialization():
    engine = BedsideStreamingEngine(fs=256, peak_threshold=500.0)
    assert engine.fs == 256.0
    assert engine.n_samples == 1024
    assert len(engine.scaler_mean) == 4
    assert len(engine.signs) == 4
    assert engine.signs.tolist() == [1.0, 1.0, 1.0, -1.0]


def test_streaming_consistency_real_patients():
    """Check recorded powers and independently archived full-record features within tolerance."""
    ds = load_mendeley_temporal_dataset(ROOT)
    sids = ds["subject_ids"]
    ortho = ds["feature_matrix_variants"]["drop_c_spatial"]

    # Test both an MCS patient (P36) and a VS patient (P25)
    test_cases = ["Nekrasova_P36", "Nekrasova_P25"]

    for sid in test_cases:
        idx = sids.index(sid)
        expected = {
            "c_global": float(ortho[idx, 0]),
            "neg_cv": float(ortho[idx, 1]),
            "lag1": float(ortho[idx, 2]),
            "residual_corr": float(ortho[idx, 3]),
        }

        npz_path = ROOT / f"results/target_profiles_v11/raw_channels/{sid}.npz"
        assert npz_path.exists(), f"Missing cache file: {npz_path}"

        with np.load(npz_path) as data:
            peaks = data["peaks"]
            powers = data["powers"]

        engine = BedsideStreamingEngine(fs=256, peak_threshold=500.0)

        for i in range(len(peaks)):
            # Ingest powers; compute_features=False during fast playback ingestion
            engine.process_power_epoch(
                theta_power=powers[i, 0, :],
                alpha_power=powers[i, 1, :],
                peak_amp=peaks[i],
                epoch_idx=i,
                compute_features=False,
            )

        verif = engine.verify_consistency_with_offline(expected, atol=1e-7)
        assert verif["consistent"], f"Consistency failure for {sid}: {verif['differences']}"
        for k, v in verif["differences"].items():
            assert v["absolute_diff"] < 1e-7, f"{sid} feature {k} diff {v['absolute_diff']} exceeds tolerance"

        # Verify Assessment Score matches
        final_score = engine.get_assessment_score()
        x = ortho[idx]
        z = engine.signs * (x-engine.scaler_mean)/engine.scaler_scale
        assert final_score["score"] == pytest.approx(z.mean(), abs=1e-7, rel=0)
        assert final_score["score_c15"] == pytest.approx(np.clip(z,-1.5,1.5).mean(), abs=1e-7, rel=0)

    assert engine.operating_threshold == pytest.approx(np.median(((ortho-engine.scaler_mean)/engine.scaler_scale*engine.signs).mean(axis=1)), abs=1e-14)


def test_artifact_rejection_and_adjacency_queues():
    """Verify artifact gating and physical adjacency resets across corruptions."""
    engine = BedsideStreamingEngine(fs=256, peak_threshold=500.0)

    # 1. Clean epoch 0
    rng = np.random.default_rng(42)
    raw0 = rng.normal(0, 20, (6, 1024))
    r0 = engine.process_raw_epoch(raw0, epoch_idx=0)
    assert r0["is_clean"] is True
    assert r0["clinical_decision"] != "ARTIFACT_REJECTED"
    assert engine.clean_epochs_count == 1
    assert engine.adjacent_pairs_count == 0
    assert r0["score"] is None
    assert r0["clinical_decision"] == "INSUFFICIENT_DATA"

    # 2. Corrupt epoch 1 (amplitude 800 uV)
    raw1 = rng.normal(0, 20, (6, 1024))
    raw1[0, 500] = 850.0  # artifact spike
    r1 = engine.process_raw_epoch(raw1, epoch_idx=1)
    assert r1["is_clean"] is False
    assert r1["clinical_decision"] == "ARTIFACT_REJECTED"
    assert engine.clean_epochs_count == 1
    assert engine.rejected_epochs_count == 1
    assert engine.adjacent_pairs_count == 0

    # 3. Clean epoch 2 (non-adjacent to clean epoch 0 because epoch 1 was rejected)
    raw2 = rng.normal(0, 20, (6, 1024))
    r2 = engine.process_raw_epoch(raw2, epoch_idx=2)
    assert r2["is_clean"] is True
    assert engine.clean_epochs_count == 2
    # Since epoch 1 was missing, epoch 2 - epoch 0 = 2 != 1, so adjacency should NOT increment!
    assert engine.adjacent_pairs_count == 0

    # 4. Clean epoch 3 (adjacent to clean epoch 2)
    raw3 = rng.normal(0, 20, (6, 1024))
    r3 = engine.process_raw_epoch(raw3, epoch_idx=3)
    assert r3["is_clean"] is True
    assert engine.clean_epochs_count == 3
    # Now epoch 3 - epoch 2 = 1, adjacency increments to 1!
    assert engine.adjacent_pairs_count == 1


def test_latency_counts_only_executed_stages():
    """Skipped work must not enter event latency statistics as zero-duration work."""
    engine = BedsideStreamingEngine(fs=256, peak_threshold=500.0)
    rng = np.random.default_rng(7)
    for i in range(30):
        raw = rng.normal(0,25,(6,1024))
        engine.process_raw_epoch(raw, epoch_idx=i)
    profile = engine.get_latency_profile()
    assert profile["total_epochs_profiled"] == 30
    assert profile["preprocess"]["count"] == 30
    assert profile["features"]["count"] == profile["scoring"]["count"] == 20
    assert profile["raw_epoch_processing"]["count"] == 30
    assert profile["cached_ingest_only"]["count"] == 0
    assert profile["cached_ingest_only"]["mean_ms"] is None

    engine.reset()
    for i in range(12):
        result = engine.process_power_epoch(np.arange(1,7)+i,np.ones(6),1,i,compute_features=(i==11))
    assert result["score"] is not None
    profile = engine.get_latency_profile()
    assert profile["preprocess"]["count"] == 0
    assert profile["preprocess"]["p95_ms"] is None
    assert profile["features"]["count"] == 1
    assert profile["cached_ingest_only"]["count"] == 11


def test_frozen_state_and_threshold_counterexample():
    engine = BedsideStreamingEngine()
    archive = ROOT / "results/matched_model_comparison_v1/fitted_states.json"
    state = engine.model_metadata
    expected = json.loads(archive.read_text(encoding="utf-8"))["crrc_full15/orthogonal_lag1/target_aligned"]["source"]
    assert state["provenance"]["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    np.testing.assert_array_equal(engine.scaler_mean, expected["mean"])
    np.testing.assert_array_equal(engine.scaler_scale, expected["scale"])
    np.testing.assert_array_equal(engine.signs, expected["signs"])
    assert engine.operating_threshold == expected["thresholds"]["equal_mean"]
    assert engine.threshold_c15 == expected["thresholds"]["equal_mean_c15"]
    x = engine.scaler_mean + .02*engine.signs*engine.scaler_scale
    names = ("c_global","neg_cv","lag1","residual_corr")
    score = engine._evaluate_assessment_score(dict(zip(names,x)))["score"]
    assert score == pytest.approx(.02)
    assert engine.prioritization_state(score) == "LOWER_PRIORITY"
    assert engine.prioritization_state(engine.operating_threshold) == "PRIORITIZE_FOR_MULTIMODAL"


def test_insufficient_pairs_skipped_updates_and_frozen_scale():
    engine = BedsideStreamingEngine()
    original = (engine.scaler_mean.copy(),engine.scaler_scale.copy(),engine.signs.copy(),engine.operating_threshold)
    first = engine.process_power_epoch(np.arange(1,7),np.ones(6),1,0)
    assert first["features"] is None and first["score"] is None
    assert first["prioritization_state"] == "INSUFFICIENT_DATA"
    for i in range(1,11):
        engine.process_power_epoch(np.arange(1,7)+i,np.ones(6),1,i)
    skipped = engine.process_power_epoch(np.arange(1,7)+11,np.ones(6),1,11,compute_features=False)
    assert skipped["ready"] is True
    assert skipped["score"] is None and skipped["prioritization_state"] == "NOT_EVALUATED"
    assert skipped["step_latencies_ms"]["features"] is None
    for a,b in zip((engine.scaler_mean,engine.scaler_scale,engine.signs),original[:3]):
        np.testing.assert_array_equal(a,b)
    assert engine.operating_threshold == original[3]

    # Hundreds of clean blocks separated by gaps still have zero adjacent pairs.
    gaps = BedsideStreamingEngine()
    for i in range(30):
        result = gaps.process_power_epoch(np.arange(1,7)+i,np.ones(6),1,2*i)
    assert gaps.adjacent_pairs_count == 0 and result["score"] is None
    assert result["prioritization_state"] == "INSUFFICIENT_DATA"


def test_raw_entrypoint_against_independent_batch_periodogram():
    """Synthetic API validation only; does not assert real-patient raw replay."""
    rng = np.random.default_rng(19)
    raw = rng.normal(0,20,(20,6,1024))
    raw[6,:,100] = [500,-500,0,0,0,0]  # reference mean=0; exact threshold rejected
    engine = BedsideStreamingEngine()
    outputs = [engine.process_raw_epoch(block,i) for i,block in enumerate(raw)]
    assert outputs[6]["prioritization_state"] == "ARTIFACT_REJECTED"
    referenced = raw-raw.mean(axis=1,keepdims=True)
    good = np.max(np.abs(referenced),axis=(1,2)) < 500
    f, psd = welch(referenced[good],fs=256,nperseg=1024,noverlap=0,scaling='density',axis=-1)
    theta = psd[:,:,(f>=4)&(f<8)].mean(axis=-1)
    alpha = psd[:,:,(f>=8)&(f<13)].mean(axis=-1)
    np.testing.assert_allclose(np.array(engine.clean_theta_powers),theta,atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.array(engine.clean_alpha_powers),alpha,atol=1e-12,rtol=0)
    temporal = compute_temporal_order_features(theta,np.flatnonzero(good))
    t,a = theta.mean(),alpha.mean()
    expected = {"c_global":(t-a)/(t+a+1e-12), **{k:temporal[k] for k in ("neg_cv","lag1","residual_corr")}}
    assert engine.verify_consistency_with_offline(expected,atol=1e-12)["consistent"]
    assert engine.adjacent_pairs_count == np.count_nonzero(np.diff(np.flatnonzero(good))==1)


def test_invalid_inputs_do_not_advance_replay_state():
    engine = BedsideStreamingEngine()
    with pytest.raises(ValueError):
        engine.process_power_epoch(np.ones(5),np.ones(6),1,0)
    with pytest.raises(ValueError):
        engine.process_raw_epoch(np.full((6,1024),np.nan),0)
    assert engine.total_epochs_received == 0
    engine.process_power_epoch(np.ones(6),np.ones(6),1,2)
    with pytest.raises(ValueError):
        engine.process_power_epoch(np.ones(6),np.ones(6),1,2)
    assert engine.total_epochs_received == 1
