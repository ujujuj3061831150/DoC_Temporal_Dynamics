"""Exercise the CLI-to-dashboard path, including incomplete recording prefixes."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = next(p for p in (CODE_ROOT, *CODE_ROOT.parents) if (p / "results/multiscale_matched_v2/features.json").exists())


@pytest.mark.parametrize("epochs", [1, 30])
def test_cli_export_dashboard_and_prefix_reference(tmp_path, epochs):
    run = subprocess.run([sys.executable, str(CODE_ROOT / "scripts/run_bedside_playback.py"),
                          "--subject", "Nekrasova_P36", "--max-epochs", str(epochs),
                          "--display-step", "5", "--speed", "0", "--output-dir", str(tmp_path)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads((tmp_path / "playback_report_Nekrasova_P36.json").read_text(encoding="utf-8"))
    trace = json.loads((tmp_path / report["trajectory_file"]).read_text(encoding="utf-8"))
    assert len(trace["steps"]) == epochs
    assert report["latency_profile_ms"]["preprocess"]["count"] == 0
    assert report["latency_profile_ms"]["preprocess"]["mean_ms"] is None
    assert report["full_record_archive_comparison"] is None
    evaluated = [s for s in trace["steps"] if s["score"] is not None]
    assert report["latency_profile_ms"]["features"]["count"] == len(evaluated)
    assert all(s["features"] is None and s["score"] is None for s in trace["steps"][:10])
    assert report["consistency_verified"] is (True if epochs == 30 else None)
    if epochs == 1:
        assert report["prioritization_state"] == "INSUFFICIENT_DATA"
    for step in evaluated:
        expected_state = "PRIORITIZE_FOR_MULTIMODAL" if step["score"] >= report["threshold_source_train"] else "LOWER_PRIORITY"
        assert step["prioritization_state"] == expected_state
    with np.load(DATA_ROOT / "results/target_profiles_v11/raw_channels/Nekrasova_P36.npz") as data:
        np.testing.assert_array_equal(np.array([s["theta_power"] for s in trace["steps"]]),data["powers"][:epochs,0])

    spec = importlib.util.spec_from_file_location("playback_demo_builder", CODE_ROOT / "scripts/build_bedside_playback_demo.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    html_path = tmp_path / "demo.html"
    builder.build_demo(tmp_path,html_path)
    html = html_path.read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script id="playback-data" type="application/json">(.*?)</script>',html,re.S)[1])
    assert payload["traces"]["Nekrasova_P36"] == trace
    assert "Math.random" not in html and "isMCS" not in html
    assert "clinical_diagnosis" not in payload["traces"]["Nekrasova_P36"]
    assert (tmp_path / "BEDSIDE_PLAYBACK_VERIFICATION_REPORT.md").exists()
