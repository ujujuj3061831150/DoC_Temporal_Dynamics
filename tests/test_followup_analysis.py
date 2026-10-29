"""
Unit and Integration Tests for Longitudinal Follow-up Audit,
Linkage Verification, and Prognostic Analysis Engine.
"""

from __future__ import annotations
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

# Robust ROOT locator
CURRENT = Path(__file__).resolve()
ROOT = next((p for p in CURRENT.parents if (p / "results").exists() or (p / "pytest.ini").exists()), CURRENT.parents[1])
sys.path.insert(0, str(ROOT / "src"))

from temporal_dynamics.followup_analysis import (
    FollowupAuditEngine,
    PrognosticValidationSuite,
    FrozenModelHashMismatchError,
    NoConfirmedFollowupPairsError,
)


def test_master_traceability_table_integrity():
    engine = FollowupAuditEngine()
    summary = engine.summary_by_cohort()

    assert "Figshare_Ling2024" in summary
    assert "Wielek_2018" in summary
    assert "Mendeley_Nekrasova2019" in summary
    assert "CRRC_2024" in summary

    # Figshare must have 50 pDoC patients
    assert summary["Figshare_Ling2024"]["total_enrolled"] == 50
    assert summary["Figshare_Ling2024"]["confirmed_pairs"] == 0
    assert summary["Figshare_Ling2024"]["missing_information"] == 50

    # Wielek must have 23 patients
    assert summary["Wielek_2018"]["total_enrolled"] == 23
    assert summary["Wielek_2018"]["confirmed_pairs"] == 0
    assert summary["Wielek_2018"]["ambiguous_candidates"] == 4
    assert summary["Wielek_2018"]["missing_information"] == 19

    # Mendeley must have 41 patients
    assert summary["Mendeley_Nekrasova2019"]["total_enrolled"] == 41
    assert summary["Mendeley_Nekrasova2019"]["confirmed_pairs"] == 0

    # Total actionable pairs under strict rule must be 0
    actionable = engine.get_actionable_pairs(allow_ambiguous=False)
    assert len(actionable) == 0


def test_frozen_model_hash_verification(tmp_path):
    suite = PrognosticValidationSuite()
    actual_hash = suite.verify_frozen_hash()
    assert actual_hash == PrognosticValidationSuite.EXPECTED_FROZEN_HASH
    assert suite.threshold_neutral == pytest.approx(0.045108, abs=1e-4)

    # Test tampering detection
    corrupted_file = tmp_path / "fitted_states_tampered.json"
    corrupted_file.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(FrozenModelHashMismatchError):
        PrognosticValidationSuite(fitted_states_path=corrupted_file)


def test_statistical_engine_correctness():
    suite = PrognosticValidationSuite()

    # Controlled synthetic cohort: 20 favorable, 20 unfavorable
    np.random.seed(12345)
    y = np.array([1] * 20 + [0] * 20, dtype=int)
    scores = np.concatenate([
        np.random.normal(0.5, 0.2, size=20),
        np.random.normal(-0.5, 0.2, size=20)
    ])
    crs = np.concatenate([
        np.random.normal(14.0, 2.0, size=20),
        np.random.normal(6.0, 2.0, size=20)
    ])

    results = suite.run_prognostic_association(scores, y, crs)

    assert results["n_total"] == 40
    assert results["n_favorable"] == 20
    assert results["n_unfavorable"] == 20
    assert results["mann_whitney"]["p_value"] < 1e-4
    assert results["univariate_logistic"]["odds_ratio"] > 1.0
    assert results["fixed_operating_point"]["threshold"] == pytest.approx(0.045108, abs=1e-4)
    assert "multivariable_incremental_value" in results
    assert "lrt_p_value" in results["multivariable_incremental_value"]


def test_prognostic_score_computation():
    suite = PrognosticValidationSuite()
    features = np.array([0.451091, -1.661900, 0.334660, 0.334171], dtype=float)
    s, s_c15 = suite.compute_score(features)
    # At exact population mean, score should be 0.0
    assert abs(s) < 1e-5
    assert abs(s_c15) < 1e-5
