"""
Figshare Cohort Adapter: High-Density 62-Channel Continuous EEG for pDoC Validation.

Provides data extraction, clinical group partitioning, and feature construction for:
Ling et al. / Xu et al. pDoC cohort (Figshare DOI: 10.6084/m9.figshare.23552964):
- 19 Minimally Conscious State (MCS)
- 31 Vegetative State / Unresponsive Wakefulness Syndrome (VS/UWS)
- 13 Exit-MCS (EMCS)
- 32 Healthy Controls (HC)
Total N = 95 subjects with continuous 62-channel 250 Hz resting EEG.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import welch

from .operators import compute_temporal_order_features


def get_repo_root() -> Path:
    """Resolve repository root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "figshare_6ch_cache").exists() or (parent / "pytest.ini").exists():
            return parent
    return current.parents[2]


def get_figshare_group_partition(
    meta_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    diagnosis_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """
    Partition Figshare subjects into clinical diagnostic tiers based on independent clinical diagnoses
    or official repository metadata.
    
    Parameters:
        meta_path: path to Figshare metadata JSON.
        cache_dir: path to directory containing downloaded numpy arrays.
        diagnosis_path: path to independent clinical diagnosis CSV (columns: subject_id, diagnosis).
        
    Returns:
        Dict with keys 'mcs' (19), 'vs' (31), 'emcs' (13), 'hc' (32).
    """
    import csv
    import warnings
    
    root = get_repo_root()
    path = (
        Path(diagnosis_path) if diagnosis_path is not None
        else root / "figshare_clinical_diagnoses.csv"
    )
    
    # 1. Independent clinical diagnosis table if available
    if path.exists():
        groups = {k: [] for k in ("mcs", "vs", "emcs", "hc")}
        seen = set()
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sid = row["subject_id"].strip()
                group = row["diagnosis"].strip().lower()
                if not sid or sid in seen or group not in groups:
                    raise ValueError(f"Invalid clinical diagnosis row: {row}")
                seen.add(sid)
                groups[group].append(sid)

        expected = {"mcs": 19, "vs": 31, "emcs": 13, "hc": 32}
        if {k: len(v) for k, v in groups.items()} != expected:
            raise ValueError(f"Clinical cohort counts { {k: len(v) for k, v in groups.items()} } do not match protocol {expected}.")
        return groups
        
    if diagnosis_path is not None:
        raise FileNotFoundError(f"Independent clinical diagnosis CSV not found at: {diagnosis_path}")
        
    # 2. Metadata-based clinical partition
    m_path = Path(meta_path) if meta_path else root / "figshare_23552964_meta.json"
    c_dir = Path(cache_dir) if cache_dir else root / "figshare_6ch_cache"
    
    if not m_path.exists():
        raise FileNotFoundError(f"Missing Figshare metadata JSON: {m_path}")
        
    with open(m_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
        
    dats = sorted([x for x in meta["files"] if x["name"].endswith(".dat")], key=lambda x: x["id"])
    b1_files = [x["name"][:-4] for x in dats if x["id"] < 41350000]
    b2_dats = [x for x in dats if x["id"] > 41350000]
    b2_p1_files = [x["name"][:-4] for x in b2_dats[:26]]
    
    # Clinical pDoC patients in Batch 2 (strictly defined by repository upload batch IDs)
    mcs_files = [x["name"][:-4] for x in b2_dats[26:45]]  # 19 MCS
    vs_files = [x["name"][:-4] for x in b2_dats[45:]]     # 31 VS
    
    # In the absence of an independent clinical diagnosis table, HC (32) and EMCS (13) cannot
    # be derived from EEG signals to prevent circular label dependency (Axiom 2: independent clinical truth).
    return {
        "mcs": mcs_files,
        "vs": vs_files,
        "emcs": [],
        "hc": [],
    }


def load_figshare_pdoc_dataset(
    cache_dir: Optional[Union[str, Path]] = None,
    block_sec: float = 2.0,
    fs: int = 250,
    diagnosis_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Extract canonical and order-aware dynamic features for Figshare pDoC cohort (19 MCS vs 31 VS).
    
    Parameters:
        cache_dir: path to directory containing downloaded (6, 30000) numpy arrays.
        block_sec: temporal epoch duration (default: 2.0 seconds).
        fs: sampling rate (default: 250 Hz).
        diagnosis_path: path to independent clinical diagnosis CSV.
        
    Returns:
        Dict containing:
            - 'subject_ids': list of 50 patient IDs
            - 'labels': (50,) int array (1 for MCS, 0 for VS)
            - 'feature_matrix_variants': dict mapping variant names to (50, K) matrices
            - 'feature_dicts': per-subject feature dictionary
    """
    root = get_repo_root()
    c_dir = Path(cache_dir) if cache_dir else root / "figshare_6ch_cache"
    if diagnosis_path is None:
        default_csv = root / "data" / "figshare_clinical_diagnoses.csv"
        if default_csv.exists():
            diagnosis_path = default_csv
    groups = get_figshare_group_partition(cache_dir=c_dir, diagnosis_path=diagnosis_path)
    
    mcs_files = groups["mcs"]
    vs_files = groups["vs"]
    pdoc_ids = mcs_files + vs_files
    labels = np.array([1] * len(mcs_files) + [0] * len(vs_files), dtype=int)
    
    pts_per_block = int(block_sec * fs)
    n_blocks = 30000 // pts_per_block
    
    feature_dicts = {}
    canonical_list = []
    
    for sid in pdoc_ids:
        npy_path = c_dir / f"{sid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing cached 6-channel EEG file: {npy_path}")
        mat = np.load(npy_path)
        
        # Spectral contrasts (c_global, c_spatial)
        f, psd = welch(mat, fs=fs, nperseg=500, axis=1)
        th = np.mean(psd[:, (f >= 4) & (f < 8)], axis=1)
        al = np.mean(psd[:, (f >= 8) & (f <= 13)], axis=1)
        c_g = float((np.mean(th) - np.mean(al)) / (np.mean(th) + np.mean(al) + 1e-12))
        c_s = float(np.mean((th - al) / (th + al + 1e-12)))
        
        # Block-level powers
        theta_blocks = np.zeros((n_blocks, 6))
        for b in range(n_blocks):
            seg = mat[:, b * pts_per_block : (b + 1) * pts_per_block]
            f_b, psd_b = welch(seg, fs=fs, nperseg=min(fs, pts_per_block), axis=1)
            theta_blocks[b] = np.mean(psd_b[:, (f_b >= 4) & (f_b < 8)], axis=1)
            
        indices = np.arange(n_blocks)
        res = compute_temporal_order_features(theta_blocks, indices, min_adjacent_pairs=5)
        
        feature_dicts[sid] = {
            "c_global": c_g,
            "c_spatial": c_s,
            "neg_cv": res["neg_cv"],
            "r_interhem": res["r_interhem"],
            "lag1": res["lag1"],
            "neg_increment": res["neg_increment"],
            "residual_corr": res["residual_corr"],
        }
        canonical_list.append([c_g, c_s, res["neg_cv"], res["r_interhem"]])
        
    canonical_mat = np.array(canonical_list, dtype=float)
    
    variants = {
        "canonical": canonical_mat,
        "plus_lag1": np.column_stack([canonical_mat, [feature_dicts[s]["lag1"] for s in pdoc_ids]]),
        "plus_increment": np.column_stack([canonical_mat, [feature_dicts[s]["neg_increment"] for s in pdoc_ids]]),
        "replace_cv_lag1": np.column_stack([
            canonical_mat[:, :2],
            [feature_dicts[s]["lag1"] for s in pdoc_ids],
            canonical_mat[:, 3],
        ]),
        "replace_cv_increment": np.column_stack([
            canonical_mat[:, :2],
            [feature_dicts[s]["neg_increment"] for s in pdoc_ids],
            canonical_mat[:, 3],
        ]),
        "residual_corr": np.column_stack([
            canonical_mat[:, :3],
            [feature_dicts[s]["residual_corr"] for s in pdoc_ids],
        ]),
        "plus_lag1_residual": np.column_stack([
            canonical_mat[:, :3],
            [feature_dicts[s]["residual_corr"] for s in pdoc_ids],
            [feature_dicts[s]["lag1"] for s in pdoc_ids],
        ]),
    }
    
    return {
        "subject_ids": pdoc_ids,
        "labels": labels,
        "feature_matrix_variants": variants,
        "feature_dicts": feature_dicts,
        "groups": groups,
    }
