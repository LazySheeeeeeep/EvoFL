"""Offline checks for stratification and retained-history exclusion."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_evidence_reflector_comparison as experiment


def test_audit_sample_covers_three_outcomes():
    cases = []
    for index, (top1, top5) in enumerate([(False, False), (False, True), (True, True)]):
        for n in range(2):
            cases.append({"instance_id": f"{index}-{n}", "function_ground_truth": ["a.py::f"],
                          "original_no_skill_metrics": {"top1": top1, "top5": top5}})
    selected = experiment.audit_sample(cases)
    assert len(selected) == 6
    assert {c["audit_stratum"] for c in selected} == {"top5_miss", "ranking_gap", "top1_hit"}


def test_history_exclusion_keeps_original_inputs(tmp_path, monkeypatch):
    prior = tmp_path / "previous"
    prior.mkdir()
    manifest = prior / "selected_cases.json"
    manifest.write_text(json.dumps([{"instance_id": "seen"}]))
    current = tmp_path / "current"
    current.mkdir()
    (current / "selected_cases.json").write_text(json.dumps([{"instance_id": "current"}]))
    monkeypatch.setattr(experiment, "OUT", current)
    ids, files = experiment.exposed_ids(tmp_path)
    assert ids == {"seen"}
    assert files == [str(manifest)]
    assert json.loads(manifest.read_text()) == [{"instance_id": "seen"}]
