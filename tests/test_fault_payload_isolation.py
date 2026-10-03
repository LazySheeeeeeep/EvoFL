import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
from run_fault_payload_diagnostic import PrefixReplayAgent
from evolutefl.explorer.v5_agent import V5ExplorerAgent, fault_skill_model_view
from evolutefl.llm.client import FakeLLMClient
from evolutefl.skills import make_skill_bank
from test_v5_workflow import config, fault_args, tool


CARD = {"skill_id": "fault_demo", "skill_type": "fault_skill",
        "fault_family": "state_assignment", "fault_subtype": "lost_input",
        "title": "Lost input", "trigger": "An input value is dropped",
        "knowledge": ["Compare input and stored value."], "secret_debug": "not for model"}
SEARCH = {"fault_family": "state_assignment", "matched_skill": CARD,
          "search_trace": {"selector_attempts": [{"reasoning_content": "PRIVATE_RETRY_TEXT"}],
                           "validator": {"reason": "PRIVATE_VALIDATOR_TEXT"}}}


def test_model_view_allowlist_and_identical_null_results():
    view = fault_skill_model_view(SEARCH)
    assert view["matched_skill"]["knowledge"] == CARD["knowledge"]
    assert "PRIVATE" not in json.dumps(view) and "secret_debug" not in json.dumps(view)
    nulls = [{"fault_family": "state_assignment", "matched_skill": None, "search_trace": trace}
             for trace in ({"disabled": True}, {"selector": "rejected"}, {"error": "failed"})]
    assert all(fault_skill_model_view(n) == fault_skill_model_view(nulls[0]) for n in nulls)


def original_run(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def adapt(x):\n    return 0\n")
    directory = tmp_path / "original"
    client = FakeLLMClient([
        tool("read_file", {"path": "a.py"}), tool("load_fault_skill", fault_args()),
        tool("read_observation", {"observation_id": "obs-00002"}),
        tool("finish_localization", {"ranked_functions": ["a.py::adapt"], "summary": "done"})])
    monkeypatch.setattr(V5ExplorerAgent, "load_fault", lambda *a: copy.deepcopy(SEARCH))
    agent = V5ExplorerAgent(llm_client=client, skill_bank=make_skill_bank(cfg), config=cfg, system_prompt="test")
    task = {"repo": "demo/repo", "instance_id": "demo", "bug_report": "Input is lost",
            "repo_path": str(root), "run_dir": str(directory)}
    assert agent.run(task)["status"] == "completed"
    return cfg, root, directory, task, client


def test_live_and_observation_replay_exclude_diagnostics_but_keep_disk_audit(tmp_path, monkeypatch):
    _, _, directory, _, _ = original_run(tmp_path, monkeypatch)
    assert "PRIVATE_RETRY_TEXT" in (directory / "fault_skill_search.json").read_text()
    observations = (directory / "observations.jsonl").read_text()
    assert "PRIVATE" not in observations and "secret_debug" not in observations
    rows = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
    delivered = [r["message"]["content"] for r in rows if r["event"] == "tool_result"]
    assert "PRIVATE" not in json.dumps(delivered)


@pytest.mark.parametrize("changed", [False, True])
def test_prefix_replay_checks_source_before_live_call(tmp_path, monkeypatch, changed):
    cfg, root, directory, task, _ = original_run(tmp_path, monkeypatch)
    if changed:
        (root / "a.py").write_text("def adapt(x):\n    return x\n")
    client = FakeLLMClient([tool("finish_localization", {"ranked_functions": ["a.py::adapt"], "summary": "done"})])
    agent = PrefixReplayAgent(prefix_dir=directory, inject_skill=False, llm_client=client,
                             skill_bank=make_skill_bank(cfg), config=cfg, system_prompt="test")
    result = agent.run({**task, "run_dir": str(tmp_path / "replay")})
    assert agent.verified is (not changed)
    assert len(client.calls) == (0 if changed else 1)
    assert result["status"] == ("llm_request_failed" if changed else "completed")
    observations = (tmp_path / "replay/observations.jsonl").read_text()
    assert "PRIVATE" not in observations
