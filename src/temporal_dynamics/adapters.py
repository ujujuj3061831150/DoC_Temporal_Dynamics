"""
Cohort Data Adapters for Mendeley (Continuous EEG Powers) and Wielek (Discrete Sleep Dynamics).

Ensures explicit semantic boundaries:
- Mendeley: 41 subjects with continuous 6-channel EEG theta spectral dynamics (4s blocks).
- Wielek: 23 subjects with discrete sleep stage transitions (30s epochs), respecting time_record boundaries.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .operators import compute_temporal_order_features, extract_clean_theta_series


def get_repo_root() -> Path:
    """Resolve repository root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "results/multiscale_matched_v2").exists() or (parent / "pytest.ini").exists():
            return parent
    return current.parents[2]


def load_mendeley_temporal_dataset(
    repo_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load Mendeley 41-patient cohort with canonical 4 features and newly computed temporal dynamics.
    
    Returns dict:
        - 'subject_ids': list of 41 subject IDs
        - 'labels': (41,) int array (1 for MCS, 0 for VS/UWS)
        - 'canonical_4': (41, 4) float array [c_global, c_spatial, neg_theta_cv, r_interhem]
        - 'temporal_features': dict of {subject_id: temporal_order_features_dict}
        - 'feature_matrix_variants': dict mapping variant names to (41, K) feature matrices:
            * 'canonical': 4 features
            * 'plus_lag1': 5 features (canonical + lag1)
            * 'plus_increment': 5 features (canonical + D1)
            * 'replace_cv_lag1': 4 features [c_global, c_spatial, lag1, r_interhem]
            * 'replace_cv_increment': 4 features [c_global, c_spatial, D1, r_interhem]
            * 'residual_corr': 4 features [c_global, c_spatial, neg_cv, r_residual]
    """
    root = Path(repo_root) if repo_root else get_repo_root()
    features_json = root / "results/multiscale_matched_v2/features.json"
    if not features_json.exists():
        raise FileNotFoundError(f"Missing {features_json}")
        
    records = [
        r for r in json.loads(features_json.read_text(encoding="utf-8"))
        if r["cohort"] == "mendeley"
    ]
    if len(records) != 41:
        raise ValueError(f"Expected 41 Mendeley subjects, found {len(records)}")
        
    subject_ids = [r["subject_id"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=int)
    canonical_dict = {r["subject_id"]: np.array(r["new"], dtype=float) for r in records}
    
    temporal_dict = {}
    for sid in subject_ids:
        npz_path = root / f"results/target_profiles_v11/raw_channels/{sid}.npz"
        theta, indices = extract_clean_theta_series(npz_path)
        feat = compute_temporal_order_features(theta, indices)
        
        # Verify consistency with original stored features
        orig_4 = canonical_dict[sid]
        np.testing.assert_allclose(
            [feat["neg_cv"], feat["r_interhem"]],
            orig_4[2:],
            atol=1e-10,
            rtol=0,
            err_msg=f"Discrepancy in static features for subject {sid}",
        )
        temporal_dict[sid] = feat
        
    # Construct variants
    canonical_mat = np.array([canonical_dict[sid] for sid in subject_ids])
    
    variants = {
        "canonical": canonical_mat,
        "plus_lag1": np.column_stack([
            canonical_mat,
            [temporal_dict[sid]["lag1"] for sid in subject_ids]
        ]),
        "plus_increment": np.column_stack([
            canonical_mat,
            [temporal_dict[sid]["neg_increment"] for sid in subject_ids]
        ]),
        "replace_cv_lag1": np.column_stack([
            canonical_mat[:, :2],
            [temporal_dict[sid]["lag1"] for sid in subject_ids],
            canonical_mat[:, 3]
        ]),
        "replace_cv_increment": np.column_stack([
            canonical_mat[:, :2],
            [temporal_dict[sid]["neg_increment"] for sid in subject_ids],
            canonical_mat[:, 3]
        ]),
        "residual_corr": np.column_stack([
            canonical_mat[:, :3],
            [temporal_dict[sid]["residual_corr"] for sid in subject_ids]
        ]),
        "plus_lag1_residual": np.column_stack([
            canonical_mat[:, :3],
            [temporal_dict[sid]["residual_corr"] for sid in subject_ids],
            [temporal_dict[sid]["lag1"] for sid in subject_ids]
        ]),
        "drop_c_spatial": np.column_stack([
            canonical_mat[:, [0, 2]],
            [temporal_dict[sid]["lag1"] for sid in subject_ids],
            [temporal_dict[sid]["residual_corr"] for sid in subject_ids]
        ]),
        "drop_c_global": np.column_stack([
            canonical_mat[:, [1, 2]],
            [temporal_dict[sid]["lag1"] for sid in subject_ids],
            [temporal_dict[sid]["residual_corr"] for sid in subject_ids]
        ]),
    }
    
    return {
        "subject_ids": subject_ids,
        "labels": labels,
        "canonical_4": canonical_mat,
        "temporal_features": temporal_dict,
        "feature_matrix_variants": variants,
    }


def compute_wielek_discrete_persistence(
    wielek_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Compute discrete sleep stage self-transition persistence P(S_{t+1} == S_t) for Wielek cohort.
    
    Respects physical time_record boundaries and strictly tests adjacency without bridging gaps.
    """
    root = Path(wielek_dir) if wielek_dir else (
        get_repo_root() / "data/public_doc_cohorts/cohort3_figshare_wielek/sleep_in_DOC_ML_data/DOC_classif"
    )
    if not root.exists():
        raise FileNotFoundError(f"Wielek directory not found: {root}")
        
    diag_csv = root / "diag_bothSL.csv"
    if not diag_csv.exists():
        raise FileNotFoundError(f"Missing {diag_csv}")
        
    subject_results = {}
    for csv_file in sorted(root.glob("VP*_OCpred.csv")):
        sid = csv_file.stem.split("_")[0]
        df = pd.read_csv(csv_file)
        
        # Must respect time_record session blocks (e.g. day1, night1)
        total_adjacent_pairs = 0
        same_stage_pairs = 0
        
        for _, group in df.groupby("time_record"):
            group = group.sort_values("time")
            times = group["time"].values
            stages = group["myPredict"].values
            
            adj = (np.diff(times) == 1)
            if adj.sum() > 0:
                s_curr = stages[:-1][adj]
                s_next = stages[1:][adj]
                same_stage_pairs += int(np.sum(s_curr == s_next))
                total_adjacent_pairs += int(adj.sum())
                
        p_stay = float(same_stage_pairs / total_adjacent_pairs) if total_adjacent_pairs > 0 else 0.0
        subject_results[sid] = {
            "subject_id": sid,
            "discrete_lag1_persistence": p_stay,
            "valid_adjacent_pairs": total_adjacent_pairs,
            "stay_count": same_stage_pairs,
        }
        
    return subject_results
