"""Replay cached patient power blocks with frozen source scoring.

    python scripts/run_bedside_playback.py --subject Nekrasova_P36 --speed 0

This engineering replay includes source-training subjects and is not a held-out
clinical evaluation. Cache reading and report writing are outside engine timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = next(p for p in (CODE_ROOT, *CODE_ROOT.parents) if (p / "results/multiscale_matched_v2/features.json").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
from temporal_dynamics.streaming import BedsideStreamingEngine, FEATURE_NAMES
from temporal_dynamics.operators import compute_temporal_order_features


def offline_prefix_features(powers, peaks, threshold=500.0, min_adjacent_pairs=10):
    """Independent batch application of the published operators to this prefix."""
    indices = np.flatnonzero(peaks < threshold)
    if np.count_nonzero(np.diff(indices) == 1) < min_adjacent_pairs:
        return None
    theta, alpha = powers[indices, 0, :], powers[indices, 1, :]
    temporal = compute_temporal_order_features(theta, indices, min_adjacent_pairs=min_adjacent_pairs)
    t, a = float(theta.mean()), float(alpha.mean())
    return {"c_global": (t - a) / (t + a + 1e-12), **{k: temporal[k] for k in FEATURE_NAMES[1:]}}


def run_playback(subject_id="Nekrasova_P36", max_epochs=-1, display_step=200,
                 peak_threshold=500.0, output_dir=CODE_ROOT / "results/bedside_playback_demo",
                 speed=0.0, model_path=None):
    if display_step < 1 or speed < 0 or not np.isfinite(speed):
        raise ValueError("display_step must be positive and speed finite/nonnegative (0 = unpaced).")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "results/multiscale_matched_v2/features.json").read_text(encoding="utf-8"))
    metadata = next((r for r in records if r["cohort"] == "mendeley" and r["subject_id"] == subject_id), None)
    if metadata is None:
        raise ValueError(f"Unknown Mendeley subject: {subject_id}")
    path = ROOT / f"results/target_profiles_v11/raw_channels/{subject_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        peaks, powers = data["peaks"], data["powers"]
        fs, channels = float(data["fs"]), data["channels"].tolist()
    if powers.shape != (len(peaks), 2, 6) or channels != ["F3", "F4", "C3", "C4", "O1", "O2"]:
        raise ValueError("Cache must contain the source model's six-channel layout and two bands.")
    n = len(peaks) if max_epochs <= 0 else min(max_epochs, len(peaks))
    engine = BedsideStreamingEngine(fs=fs, peak_threshold=peak_threshold, model_path=model_path)
    source_member = subject_id in engine.model_metadata.get("training_ids", [])
    provenance = {
        "subject_id": subject_id, "input_mode": "cached_power",
        "sampling_rate_hz": fs, "channels": channels,
        "power_definition": "Mean Welch spectral density within each band; six-channel average reference, periodic Hann, one 4-second segment.",
        "cache_path": path.relative_to(ROOT).as_posix(),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recording_epoch_seconds": engine.epoch_seconds, "qc_peak_threshold_uv": peak_threshold,
        "display_step": display_step, "feature_update_policy": "Every display_step blocks once ready; final clean block also evaluated.",
        "min_adjacent_pairs": engine.min_adjacent_pairs,
        "source_training_member": source_member,
        "interpretation": "Engineering demonstration with frozen source state; no held-out diagnostic-performance claim.",
    }
    steps = []
    clean_indices = np.flatnonzero(peaks[:n] < peak_threshold)
    final_clean = int(clean_indices[-1]) if len(clean_indices) else -1
    start = time.perf_counter()
    print(f"Cached-power replay: {subject_id}, {n} blocks, source threshold={engine.operating_threshold:.12g}")
    for i in range(n):
        compute = (i + 1) % display_step == 0 or i == final_clean
        res = engine.process_power_epoch(powers[i, 0], powers[i, 1], peaks[i], i, compute_features=compute)
        step = {k: res[k] for k in (
            "epoch_idx", "recording_seconds", "is_clean", "peak_amplitude", "total_clean_blocks",
            "rejected_blocks", "adjacent_pairs", "retention_rate", "ready", "features", "score",
            "score_c15", "prioritization_state", "step_latencies_ms")}
        step.update(theta_power=powers[i, 0].tolist(), alpha_power=powers[i, 1].tolist())
        steps.append(step)
        if res["score"] is not None:
            print(f"  block={i+1:5d} clean={engine.clean_epochs_count:5d} adjacent={engine.adjacent_pairs_count:5d} "
                  f"score={res['score']:+.6f} {res['prioritization_state']}")
        if speed > 0:
            remaining = start + (i + 1) * engine.epoch_seconds / speed - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    elapsed = time.perf_counter() - start

    expected = offline_prefix_features(powers[:n], peaks[:n], peak_threshold, engine.min_adjacent_pairs)
    verification = engine.verify_consistency_with_offline(expected or {}, atol=1e-7)
    scores = engine.get_assessment_score()
    if expected is not None:
        x = np.array([expected[k] for k in FEATURE_NAMES])
        z = engine.signs * (x - engine.scaler_mean) / engine.scaler_scale
        expected_scores = {"score": float(z.mean()), "score_c15": float(np.clip(z, -engine.clip_c, engine.clip_c).mean())}
        score_diffs = {k: abs(scores[k] - expected_scores[k]) for k in expected_scores}
        verification["score_absolute_differences"] = score_diffs
        verification["scores_within_tolerance"] = all(v <= 1e-7 for v in score_diffs.values())
    else:
        verification["scores_within_tolerance"] = None

    archived = None
    if n == len(peaks) and peak_threshold == 500 and expected is not None:
        archived_expected = dict(expected, c_global=float(metadata["new"][0]))
        archived = engine.verify_consistency_with_offline(archived_expected, atol=1e-7)

    report = {
        "schema_version": 2, **provenance,
        "clinical_diagnosis_context_only": "MCS" if metadata["label"] == 1 else "VS/UWS",
        "total_epochs": n, "clean_blocks": engine.clean_epochs_count,
        "rejected_blocks": engine.rejected_epochs_count, "adjacent_pairs": engine.adjacent_pairs_count,
        "retention_rate": engine.clean_epochs_count / max(1, n),
        "threshold_source_train": engine.operating_threshold, "threshold_source_train_c15": engine.threshold_c15,
        "frozen_model": engine.model_metadata,
        "consistency_verified": verification["consistent"], "consistency": verification,
        "full_record_archive_comparison": archived,
        "final_features": engine.get_cumulative_features(),
        "final_assessment_score": scores["score"], "final_score_c15": scores["score_c15"],
        "prioritization_state": engine.prioritization_state(scores["score"]),
        "latency_profile_ms": engine.get_latency_profile(), "replay_wall_time_seconds": elapsed,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "numpy": np.__version__,
                        "thread_policy": "No CPU-affinity or BLAS-thread restriction was imposed."},
        "trajectory_file": f"playback_trajectory_{subject_id}.json",
    }
    trajectory = {"schema_version": 2, **provenance, "threshold_source_train": engine.operating_threshold,
                  "threshold_source_train_c15": engine.threshold_c15,
                  "model_state_sha256": engine.model_metadata.get("state_sha256"), "steps": steps}
    (output_dir / report["trajectory_file"]).write_text(json.dumps(trajectory, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / f"playback_report_{subject_id}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Final prefix consistency: {verification['consistent']}; score={scores['score']}; exported to {output_dir}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential cached-power bedside engineering replay")
    parser.add_argument("--subject", default="Nekrasova_P36")
    parser.add_argument("--max-epochs", type=int, default=-1)
    parser.add_argument("--display-step", type=int, default=200)
    parser.add_argument("--speed", type=float, default=0, help="0: unpaced; 1: recorded time; >1: accelerated")
    parser.add_argument("--output-dir", type=Path, default=CODE_ROOT / "results/bedside_playback_demo")
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()
    run_playback(args.subject, args.max_epochs, args.display_step, output_dir=args.output_dir,
                 speed=args.speed, model_path=args.model_path)
