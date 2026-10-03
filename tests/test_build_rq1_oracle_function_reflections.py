import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "build_rq1_oracle_function_reflections",
    SCRIPTS / "build_rq1_oracle_function_reflections.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": json.dumps({"applicability": "value changes across boundary",
                                       "lesson": "Compare the producer output with consumer expectation."})}


def episode_and_case():
    episode = {"instance_id": "case-1", "repo": "owner/repo", "issue": "Value ignored",
               "investigation": [{"tool": "grep", "purpose": "search"}],
               "predicted_functions": ["wrong.py::g"], "skill_conditioned": True}
    case = {"instance_id": "case-1", "repo": "owner/repo", "base_commit": "abc",
            "patch": "secret patch"}
    return episode, case


def test_oracle_reflection_uses_truth_but_not_trajectory():
    client = FakeClient()
    episode, case = episode_and_case()
    evidence = {"repo": "owner/repo", "problem_statement": "Value ignored",
                "ground_truth": {"functions": ["right.py::f"], "patch": "secret patch"},
                "ground_truth_source": "evolution_evidence"}
    result = MODULE.reflect_once(client, episode, case, evidence)
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["oracle_repaired_functions"] == ["right.py::f"]
    assert payload["historical_repair_patch"] == "secret patch"
    assert "investigation" not in payload and "prediction" not in payload
    assert "right.py::f" not in json.dumps(result)
    assert "secret patch" not in json.dumps(result)
    assert result["source_skill_conditioned"] is False


def test_oracle_truth_must_match_issue_repo_and_patch():
    episode, case = episode_and_case()
    evidence = {"repo": "owner/repo", "problem_statement": "Value ignored",
                "ground_truth": {"functions": ["right.py::f"], "patch": "secret patch"}}
    MODULE.oracle_input(episode, case, evidence)
    for field, value, message in (("repo", "other/repo", "identity mismatch"),
                                  ("problem_statement", "different", "issue mismatch")):
        bad = {**evidence, field: value}
        with pytest.raises(ValueError, match=message):
            MODULE.oracle_input(episode, case, bad)
    with pytest.raises(ValueError, match="patch mismatch"):
        MODULE.oracle_input(episode, case, {**evidence, "ground_truth": {"functions": ["right.py::f"], "patch": "other"}})
    with pytest.raises(ValueError, match="Missing oracle functions"):
        MODULE.oracle_input(episode, case, {**evidence, "ground_truth": {"functions": [], "patch": "secret patch"}})


def test_missing_evolution_evidence_uses_audited_function_mapping(tmp_path):
    episode, case = episode_and_case()
    source = tmp_path / "historical"
    audit = tmp_path / "audit"
    audit.mkdir()
    cached = tmp_path / "source.py"
    cached.write_text("def f(): pass\n")
    normalized_patch = case["patch"].replace("\r\n", "\n").strip()
    (audit / "case-1.json").write_text(json.dumps({
        "instance_id": "case-1", "repo": "owner/repo", "base_commit": "abc",
        "patch_sha256": hashlib.sha256(normalized_patch.encode()).hexdigest(),
        "sources": [{"cache_path": str(cached), "sha256": hashlib.sha256(cached.read_bytes()).hexdigest()}],
        "existing_function_targets": ["right.py::f"],
    }))
    evidence = MODULE.load_oracle_evidence(episode, case, source, audit)
    assert evidence["ground_truth_source"] == "audited_patch_function_mapping"
    assert MODULE.oracle_input(episode, case, evidence)["oracle_repaired_functions"] == ["right.py::f"]
    cached.write_text("changed\n")
    with pytest.raises(ValueError, match="source cache changed"):
        MODULE.load_oracle_evidence(episode, case, source, audit)


def test_missing_evolution_evidence_uses_frozen_manifest_label(tmp_path):
    episode, case = episode_and_case()
    case["function_ground_truth"] = ["right.py::f"]
    evidence = MODULE.load_oracle_evidence(episode, case, tmp_path)
    assert evidence["ground_truth_source"] == "frozen_training_manifest"
    assert MODULE.oracle_input(episode, case, evidence)["oracle_repaired_functions"] == ["right.py::f"]


def test_malformed_oracle_reflection_is_rejected():
    episode, case = episode_and_case()
    evidence = {"repo": "owner/repo", "problem_statement": "Value ignored",
                "ground_truth": {"functions": ["right.py::f"], "patch": "secret patch"}}
    client = FakeClient()
    client.chat = lambda **kwargs: {"content": "{}"}
    with pytest.raises(ValueError, match="did not return"):
        MODULE.reflect_once(client, episode, case, evidence)
