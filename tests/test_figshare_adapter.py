"""
Unit tests for Figshare Continuous 62-Channel EEG Adapter.
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

from src.temporal_dynamics.figshare_adapter import (
    get_figshare_group_partition,
    load_figshare_pdoc_dataset,
)


def test_figshare_group_partition():
    groups = get_figshare_group_partition()
    assert "mcs" in groups
    assert "vs" in groups
    assert "emcs" in groups
    assert "hc" in groups
    
    assert len(groups["mcs"]) == 19
    assert len(groups["vs"]) == 31
    # Without independent clinical diagnosis table, emcs and hc are strictly empty to prevent circular labeling
    assert len(groups["emcs"]) == 0
    assert len(groups["hc"]) == 0


def test_load_figshare_pdoc_dataset():
    ds = load_figshare_pdoc_dataset(block_sec=2.0)
    assert len(ds["subject_ids"]) == 50
    assert len(ds["labels"]) == 50
    assert (ds["labels"] == 1).sum() == 19
    assert (ds["labels"] == 0).sum() == 31
    
    variants = ds["feature_matrix_variants"]
    assert "canonical" in variants
    assert "replace_cv_increment" in variants
    assert variants["canonical"].shape == (50, 4)
    assert variants["replace_cv_increment"].shape == (50, 4)
    assert np.isfinite(variants["replace_cv_increment"]).all()


def test_figshare_group_partition_from_csv(tmp_path):
    csv_file = tmp_path / "test_diagnoses.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("subject_id,diagnosis\n")
        for i in range(19):
            f.write(f"mcs_{i},mcs\n")
        for i in range(31):
            f.write(f"vs_{i},vs\n")
        for i in range(13):
            f.write(f"emcs_{i},emcs\n")
        for i in range(32):
            f.write(f"hc_{i},hc\n")
            
    groups = get_figshare_group_partition(diagnosis_path=csv_file)
    assert len(groups["mcs"]) == 19
    assert len(groups["vs"]) == 31
    assert len(groups["emcs"]) == 13
    assert len(groups["hc"]) == 32
    assert groups["mcs"][0] == "mcs_0"


def test_figshare_group_partition_invalid_csv(tmp_path):
    csv_file = tmp_path / "invalid_diagnoses.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("subject_id,diagnosis\n")
        f.write("sub1,unknown_group\n")
    with pytest.raises(ValueError):
        get_figshare_group_partition(diagnosis_path=csv_file)


def test_committed_figshare_clinical_diagnoses_csv():
    repo_root = REPO_ROOT
    csv_path = repo_root / "data/figshare_clinical_diagnoses.csv"
    assert csv_path.exists(), f"Missing clinical diagnosis CSV at {csv_path}"
    groups = get_figshare_group_partition(diagnosis_path=csv_path)
    assert len(groups["mcs"]) == 19
    assert len(groups["vs"]) == 31
    assert len(groups["emcs"]) == 13
    assert len(groups["hc"]) == 32


