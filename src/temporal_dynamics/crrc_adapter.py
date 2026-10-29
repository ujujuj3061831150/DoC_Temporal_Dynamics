"""
CRRC (China Rehabilitation Research Center / Beijing Bo'ai Hospital) Cohort Adapter.

Provides standardized, de-identified intake, signal preprocessing,
and order-aware temporal dynamics feature extraction for the CRRC clinical cohort.
"""

from __future__ import annotations

import os
import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import mne

from temporal_dynamics.operators import compute_temporal_order_features

logger = logging.getLogger(__name__)

# Core 6 homologous channels
TARGET_6_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]

# Standardized anonymous subject identifiers CRRC_001 .. CRRC_015
ANONYMOUS_SUBJECT_IDS = [f"CRRC_{idx:03d}" for idx in range(1, 16)]


def get_deidentified_subject_id(patient_name: str) -> str:
    """Return standardized anonymous subject identifier CRRC_001 .. CRRC_015.

    Guarantees privacy by design: no raw personal health identifiers
    are retained or hardcoded in public distribution archives.
    """
    clean_name = patient_name.strip()
    if clean_name.startswith("CRRC_") and clean_name[5:].isdigit():
        return clean_name

    mapping_file = Path(__file__).resolve().parent / "_local_patient_mapping.json"
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if clean_name in mapping:
                return mapping[clean_name]
        except Exception:
            pass

    import hashlib
    h = hashlib.sha256(clean_name.encode("utf-8")).hexdigest()[:4]
    return f"CRRC_ANON_{h}"


def load_crrc_metadata(
    data_dir: Union[str, Path]
) -> pd.DataFrame:
    """
    Load and de-identify clinical metadata from CRRC Excel sheets.
    """
    data_path = Path(data_dir)
    score_file = data_path / "病人因数分析.xlsx"
    bad_file = data_path / "zk_bad_channels_info.xlsx"
    
    if not score_file.exists():
        raise FileNotFoundError(f"Score file not found: {score_file}")
        
    score_df = pd.read_excel(score_file)
    bad_dict = {}
    if bad_file.exists():
        bad_df = pd.read_excel(bad_file)
        for _, row in bad_df.iterrows():
            bad_dict[str(row["subject_name"]).strip()] = str(row["bad_channels"]).strip()
            
    records = []
    for _, row in score_df.iterrows():
        p_name = str(row["病人"]).strip()
        subj_id = get_deidentified_subject_id(p_name)
        diag = str(row["量表评分"]).strip()
        score = float(row["量表分数"])
        lesion_cn = str(row["缺损情况"]).strip()
        
        # Translate lesion description
        lesion_en = {
            "完好": "bilateral_intact",
            "左脑": "left_hemisphere",
            "右脑": "right_hemisphere"
        }.get(lesion_cn, "unspecified")
        
        raw_bads = bad_dict.get(p_name, "0")
        bad_ch_list = []
        if raw_bads != "0":
            bad_ch_list = [c.strip().upper() for c in raw_bads.replace("，", ",").split(",") if c.strip()]
            
        # Core 6-channel clean check
        has_core_bad = any(c in TARGET_6_CHANNELS for c in bad_ch_list)
        
        records.append({
            "subject_id": subj_id,
            "diagnosis": diag,
            "y_true": 1 if diag == "MCS" else 0,
            "crs_r_score": score,
            "lesion_laterality": lesion_en,
            "bad_channels": ",".join(bad_ch_list) if bad_ch_list else "none",
            "clean_6ch_protocol": not has_core_bad,
        })
        
    return pd.DataFrame(records)


def extract_crrc_patient_eeg_features(
    patient_dir: Union[str, Path],
    sampling_target: float = 250.0,
    peak_threshold_uv: float = 500.0,
    min_adjacent_pairs: int = 10,
) -> Dict[str, Any]:
    """
    Extract 6-channel order-aware temporal dynamics features from raw BDF files.
    """
    p_path = Path(patient_dir)
    resting_dir = p_path / "静息态"
    if not resting_dir.exists():
        raise FileNotFoundError(f"No resting EEG directory found at: {resting_dir}")
        
    bdf_files = sorted([
        f for f in glob.glob(str(resting_dir / "*.bdf"))
        if "evt" not in os.path.basename(f).lower()
    ])
    if not bdf_files:
        raise FileNotFoundError(f"No valid BDF data files found in: {resting_dir}")
        
    # Read raw recordings
    raws = []
    for f in bdf_files:
        try:
            r = mne.io.read_raw_bdf(f, preload=True, verbose=False)
            raws.append(r)
        except Exception as e:
            logger.warning(f"Failed to read BDF {f}: {e}")
            
    if not raws:
        raise RuntimeError(f"All BDF files failed to load in: {resting_dir}")
        
    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
    
    # Check channel availability
    ch_upper_map = {c.upper().strip(): c for c in raw.ch_names}
    for req_ch in TARGET_6_CHANNELS:
        if req_ch not in ch_upper_map:
            raise KeyError(f"Required channel {req_ch} not found in recording.")
            
    picks = [ch_upper_map[c] for c in TARGET_6_CHANNELS]
    raw.pick(picks)
    raw.reorder_channels(picks)
    
    # Common Average Reference across available cephalic sensors
    raw.set_eeg_reference("average", projection=False, verbose=False)
    
    # Bandpass filter (0.5 - 45 Hz) & 50 Hz notch
    raw.filter(l_freq=0.5, h_freq=45.0, phase="zero-double", verbose=False)
    raw.notch_filter(freqs=[50.0], phase="zero-double", verbose=False)
    
    sfreq = float(raw.info["sfreq"])
    # Microvolts conversion (MNE raw.get_data() returns Volts)
    data_uv = raw.get_data() * 1e6
    
    total_duration_sec = data_uv.shape[1] / sfreq
    block_len = int(4.0 * sfreq)
    n_blocks = data_uv.shape[1] // block_len
    
    clean_blocks = []
    clean_indices = []
    for b in range(n_blocks):
        blk = data_uv[:, b * block_len : (b + 1) * block_len]
        ptp = float(np.ptp(blk, axis=1).max())
        if ptp < peak_threshold_uv:
            clean_blocks.append(blk)
            clean_indices.append(b)
            
    if len(clean_blocks) < min_adjacent_pairs:
        raise ValueError(f"Insufficient clean blocks: {len(clean_blocks)} < {min_adjacent_pairs}")
        
    clean_blocks_arr = np.array(clean_blocks) # (K, 6, block_len)
    clean_indices_arr = np.array(clean_indices)
    
    # Hann FFT for Theta (4-8 Hz) and Alpha (8-13 Hz)
    freqs = np.fft.rfftfreq(block_len, d=1.0 / sfreq)
    theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    
    window = np.hanning(block_len)
    fft_vals = np.fft.rfft(clean_blocks_arr * window, axis=2)
    psd = (np.abs(fft_vals) ** 2) / (np.sum(window ** 2) * sfreq)
    
    theta_pow = psd[:, :, theta_mask].mean(axis=2) # (K, 6) in uV^2 / Hz
    alpha_pow = psd[:, :, alpha_mask].mean(axis=2) # (K, 6) in uV^2 / Hz
    
    # Compute temporal order dynamics
    temporal_feats = compute_temporal_order_features(
        theta=theta_pow,
        indices=clean_indices_arr,
        min_adjacent_pairs=min_adjacent_pairs
    )
    
    # Spectral contrast metrics
    mean_theta = float(theta_pow.mean())
    mean_alpha = float(alpha_pow.mean())
    c_global = float((mean_theta - mean_alpha) / (mean_theta + mean_alpha + 1e-12))
    
    channel_contrasts = [
        float((theta_pow[:, ch].mean() - alpha_pow[:, ch].mean()) / (theta_pow[:, ch].mean() + alpha_pow[:, ch].mean() + 1e-12))
        for ch in range(6)
    ]
    c_spatial = float(np.mean(channel_contrasts))
    
    results = dict(temporal_feats)
    results.update({
        "c_global": c_global,
        "c_spatial": c_spatial,
        "total_duration_sec": total_duration_sec,
        "clean_retention_rate": len(clean_blocks) / max(1, n_blocks),
        "total_blocks": n_blocks,
    })
    return results
