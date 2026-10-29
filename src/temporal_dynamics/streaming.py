"""Sequential power-block replay with frozen source scoring and event-level timing.

Features are recomputed over the accumulated clean history, using the same
physical-adjacency and OLS operators as the offline analysis. The optional raw
epoch entry point measures its own periodogram preprocessing; cached powers do
not provide evidence of raw-EEG preprocessing latency.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import periodogram, windows

from .operators import compute_temporal_order_features


FEATURE_NAMES = ("c_global", "neg_cv", "lag1", "residual_corr")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "results/bedside_playback_demo/frozen_source_model.json"


class BedsideStreamingEngine:
    def __init__(
        self, fs=256.0, epoch_seconds=4.0, peak_threshold=500.0,
        theta_band=(4.0, 8.0), alpha_band=(8.0, 13.0),
        scaler_mean=None, scaler_scale=None, signs=None,
        operating_threshold=None, clip_c=1.5, eps=1e-12,
        min_adjacent_pairs=10, model_path=None,
    ):
        self.fs = float(fs)
        self.epoch_seconds = float(epoch_seconds)
        self.n_samples = int(round(self.fs * self.epoch_seconds))
        self.peak_threshold = float(peak_threshold)
        self.theta_band, self.alpha_band = theta_band, alpha_band
        self.clip_c, self.eps = float(clip_c), float(eps)
        self.min_adjacent_pairs = int(min_adjacent_pairs)
        if self.min_adjacent_pairs < 2:
            raise ValueError("At least two physical adjacent pairs are required.")
        # Match the source cache: six-channel average reference, periodic Hann,
        # one full-block Welch segment, and mean spectral density within each band.
        self.window = windows.hann(self.n_samples, sym=False)
        self.freq_res = self.fs / self.n_samples

        custom = (scaler_mean, scaler_scale, signs, operating_threshold)
        if any(value is not None for value in custom):
            if not all(value is not None for value in custom):
                raise ValueError("Custom scoring requires mean, scale, signs and training threshold together.")
            self.model_metadata = {"model": "explicit_custom_frozen_state", "training_ids": []}
            self.threshold_c15 = None
        else:
            path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            raw = path.read_bytes()
            state = json.loads(raw)
            if tuple(state["features"]) != FEATURE_NAMES:
                raise ValueError("Frozen model feature order does not match the replay engine.")
            if float(state["clip_c"]) != self.clip_c:
                raise ValueError("Clipping must match the frozen model state.")
            scaler_mean, scaler_scale, signs = state["mean"], state["scale"], state["signs"]
            operating_threshold = state["thresholds"]["equal_mean"]
            self.threshold_c15 = float(state["thresholds"]["equal_mean_c15"])
            self.model_metadata = dict(state, state_sha256=hashlib.sha256(raw).hexdigest())

        self.scaler_mean = np.asarray(scaler_mean, dtype=float)
        self.scaler_scale = np.asarray(scaler_scale, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        if any(a.shape != (4,) or not np.isfinite(a).all()
               for a in (self.scaler_mean, self.scaler_scale, self.signs)):
            raise ValueError("Frozen mean, scale and signs must be finite four-element vectors.")
        if np.any(self.scaler_scale <= 0) or not np.isin(self.signs, [-1, 1]).all():
            raise ValueError("Scales must be positive and signs must be +/-1.")
        self.operating_threshold = float(operating_threshold)
        if not np.isfinite(self.operating_threshold):
            raise ValueError("Training threshold must be finite.")
        self.reset()

    def reset(self):
        self.total_epochs_received = self.clean_epochs_count = self.rejected_epochs_count = 0
        self.clean_indices, self.clean_theta_powers, self.clean_alpha_powers = [], [], []
        self.last_received_idx = self.last_clean_idx = None
        self.adjacent_pairs_count = 0
        self.latencies = {k: [] for k in (
            "preprocess", "features", "scoring", "cached_ingest_only",
            "evaluated_updates", "raw_epoch_processing", "all_updates")}

    @property
    def ready(self):
        return self.clean_epochs_count >= 2 and self.adjacent_pairs_count >= self.min_adjacent_pairs

    def _epoch_index(self, epoch_idx):
        idx = (0 if self.last_received_idx is None else self.last_received_idx + 1) if epoch_idx is None else epoch_idx
        if int(idx) != idx or idx < 0:
            raise ValueError("Epoch index must be a nonnegative integer.")
        idx = int(idx)
        if self.last_received_idx is not None and idx <= self.last_received_idx:
            raise ValueError("Epoch indices must increase strictly, including rejected blocks.")
        return idx

    def process_raw_epoch(self, raw_data, epoch_idx=None, compute_features=True):
        start = time.perf_counter_ns()
        idx = self._epoch_index(epoch_idx)
        raw = np.asarray(raw_data, dtype=float)
        if raw.shape != (6, self.n_samples) or not np.isfinite(raw).all():
            raise ValueError(f"Expected finite raw shape (6, {self.n_samples}).")
        raw = raw - raw.mean(axis=0, keepdims=True)
        peak = float(np.max(np.abs(raw)))
        clean = peak < self.peak_threshold
        theta = alpha = np.zeros(6)
        if clean:
            f, psd = periodogram(raw, fs=self.fs, window=self.window, axis=1)
            theta = psd[:, (f >= self.theta_band[0]) & (f < self.theta_band[1])].mean(axis=1)
            alpha = psd[:, (f >= self.alpha_band[0]) & (f < self.alpha_band[1])].mean(axis=1)
        pre = (time.perf_counter_ns() - start) / 1e6
        return self._ingest(theta, alpha, peak, idx, clean, compute_features, start, pre, "raw_epoch")

    def process_power_epoch(self, theta_power, alpha_power, peak_amp, epoch_idx=None, compute_features=True):
        start = time.perf_counter_ns()
        idx = self._epoch_index(epoch_idx)
        theta, alpha = np.asarray(theta_power, dtype=float), np.asarray(alpha_power, dtype=float)
        peak = float(peak_amp)
        if not np.isfinite(peak) or peak < 0:
            raise ValueError("Peak amplitude must be finite and nonnegative.")
        if any(a.shape != (6,) or not np.isfinite(a).all() or np.any(a < 0) for a in (theta, alpha)):
            raise ValueError("Cached powers must be finite nonnegative six-element vectors.")
        return self._ingest(theta, alpha, peak, idx, peak < self.peak_threshold,
                            compute_features, start, None, "cached_power")

    def _ingest(self, theta, alpha, peak, idx, clean, compute_features, start, pre, input_mode):
        self.total_epochs_received += 1
        self.last_received_idx = idx
        if clean:
            self.clean_epochs_count += 1
            self.clean_indices.append(idx)
            self.clean_theta_powers.append(theta.copy())
            self.clean_alpha_powers.append(alpha.copy())
            if self.last_clean_idx is not None and idx - self.last_clean_idx == 1:
                self.adjacent_pairs_count += 1
            self.last_clean_idx = idx
        else:
            self.rejected_epochs_count += 1

        features = None
        scores = {"score": None, "score_c15": None, "z_coordinates": None}
        feat_ms = score_ms = None
        if not clean:
            status = "ARTIFACT_REJECTED"
        elif not self.ready:
            status = "INSUFFICIENT_DATA"
        elif not compute_features:
            status = "NOT_EVALUATED"
        else:
            tick = time.perf_counter_ns()
            features = self._compute_cumulative_features()
            feat_ms = (time.perf_counter_ns() - tick) / 1e6
            tick = time.perf_counter_ns()
            scores = self._evaluate_assessment_score(features)
            score_ms = (time.perf_counter_ns() - tick) / 1e6
            status = self.prioritization_state(scores["score"])

        total = (time.perf_counter_ns() - start) / 1e6
        timings = {"preprocess": pre, "features": feat_ms, "scoring": score_ms, "total": total}
        for name in ("preprocess", "features", "scoring"):
            if timings[name] is not None:
                self.latencies[name].append(timings[name])
        self.latencies["all_updates"].append(total)
        if scores["score"] is not None:
            self.latencies["evaluated_updates"].append(total)
        if input_mode == "raw_epoch":
            self.latencies["raw_epoch_processing"].append(total)
        elif feat_ms is None:
            self.latencies["cached_ingest_only"].append(total)

        return {
            "epoch_idx": idx, "recording_seconds": (idx + 1) * self.epoch_seconds,
            "input_mode": input_mode, "is_clean": bool(clean), "peak_amplitude": peak,
            "total_clean_blocks": self.clean_epochs_count,
            "rejected_blocks": self.rejected_epochs_count,
            "adjacent_pairs": self.adjacent_pairs_count, "ready": self.ready,
            "retention_rate": self.clean_epochs_count / self.total_epochs_received,
            "features": features, "score": scores["score"], "score_c15": scores["score_c15"],
            "standardized_coordinates": scores["z_coordinates"],
            "threshold": self.operating_threshold, "threshold_c15": self.threshold_c15,
            "prioritization_state": status, "clinical_decision": status,
            "step_latencies_ms": timings,
        }

    def prioritization_state(self, score):
        if score is None:
            return "INSUFFICIENT_DATA"
        return "PRIORITIZE_FOR_MULTIMODAL" if score >= self.operating_threshold else "LOWER_PRIORITY"

    def get_cumulative_features(self):
        return self._compute_cumulative_features()

    def get_assessment_score(self):
        features = self._compute_cumulative_features()
        if features is None:
            return {"score": None, "score_c15": None, "z_coordinates": None}
        return self._evaluate_assessment_score(features)

    def _compute_cumulative_features(self):
        if not self.ready:
            return None
        theta = np.asarray(self.clean_theta_powers)
        alpha = np.asarray(self.clean_alpha_powers)
        temporal = compute_temporal_order_features(
            theta, np.asarray(self.clean_indices), min_adjacent_pairs=self.min_adjacent_pairs, eps=self.eps)
        mean_theta, mean_alpha = float(theta.mean()), float(alpha.mean())
        return {
            "c_global": (mean_theta - mean_alpha) / (mean_theta + mean_alpha + self.eps),
            **{k: temporal[k] for k in FEATURE_NAMES[1:]},
        }

    def _evaluate_assessment_score(self, features):
        x = np.array([features[k] for k in FEATURE_NAMES])
        z = self.signs * (x - self.scaler_mean) / self.scaler_scale
        return {"score": float(z.mean()), "score_c15": float(np.clip(z, -self.clip_c, self.clip_c).mean()),
                "z_coordinates": z.tolist()}

    def get_latency_profile(self):
        def stats(values):
            if not values:
                return {"count": 0, **{k: None for k in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms")}}
            x = np.asarray(values)
            return {"count": len(x), "mean_ms": float(x.mean()), "median_ms": float(np.median(x)),
                    "p95_ms": float(np.percentile(x, 95)), "p99_ms": float(np.percentile(x, 99)), "max_ms": float(x.max())}
        return {
            "total_epochs_profiled": self.total_epochs_received,
            "clock": "time.perf_counter_ns", "units": "milliseconds",
            "scope": "In-memory engine calls; excludes file I/O, report export, rendering and acquisition wait. Unexecuted stages are excluded, not zero-filled.",
            "feature_algorithm": "Cumulative history recomputation using offline operators; cost depends on history length.",
            **{name: stats(values) for name, values in self.latencies.items()},
        }

    def verify_consistency_with_offline(self, expected_features, atol=1e-7):
        features = self._compute_cumulative_features()
        if features is None:
            return {"consistent": None, "status": "INSUFFICIENT_DATA", "tolerance_atol": atol, "differences": {}}
        if not all(k in expected_features and np.isfinite(expected_features[k]) for k in FEATURE_NAMES):
            raise ValueError("Offline comparison requires all four finite features.")
        differences = {}
        for name in FEATURE_NAMES:
            diff = abs(features[name] - expected_features[name])
            differences[name] = {"stream_value": features[name], "offline_expected": expected_features[name],
                                 "absolute_diff": diff, "within_tolerance": bool(diff <= atol)}
        return {"consistent": all(v["within_tolerance"] for v in differences.values()),
                "tolerance_atol": atol, "tolerance_rtol": 0.0, "differences": differences}
