import json
from pathlib import Path

from evolutefl.llm.client import FakeLLMClient
from evolutefl.skills import make_skill_bank
from run_rq1_issue_only_ablation import IssueOnlyExplorer, ablation_prompt


def tool(name, arguments):
    return {"content": None, "tool_calls": [{"id": name, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def test_issue_only_retrieval_precedes_repository_observations(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def faulty(): return 0\n")
    cfg = json.loads(Path("config/evolutefl.global.json").read_text())
    cfg["skill_bank"]["path"] = str(tmp_path / "empty_skills.jsonl")
    cfg["explorer"]["fault_skill_attempt_step"] = 1
    cfg["explorer"]["max_steps"] = 6
    client = FakeLLMClient([
        tool("load_fault_skill", {"fault_family": "state_assignment",
            "fault_signature": "value not retained", "project_context": "guessed project details",
            "suspected_path": "guessed function"}),
        tool("finish_localization", {"ranked_functions": ["a.py::faulty"],
            "summary": "Issue-only smoke"}),
    ])
    source = Path("prompt_records/explorer/explorer_system_v5.txt").read_text()
    agent = IssueOnlyExplorer(llm_client=client, skill_bank=make_skill_bank(cfg),
        config=cfg, system_prompt=ablation_prompt(source))
    directory = tmp_path / "run"
    result = agent.run({"repo": "demo/repo", "base_commit": "abc", "repo_path": str(root),
        "run_dir": str(directory), "bug_report": "A value is not retained."})

    assert result["status"] == "completed"
    first = json.loads(client.calls[0]["messages"][1]["content"])
    assert first["bug_report"] == "A value is not retained."
    assert "repo" not in first and "base_commit" not in first
    context = json.loads((directory / "issue_only_retrieval_context.json").read_text())
    assert context["prior_investigation"] == []
    assert context["request"]["project_context"] == "repository not yet inspected"
    assert context["request"]["suspected_path"] == "undetermined"
    assert json.loads((directory / "fault_skill_request.json").read_text())["step"] == 1
    assert any("demo/repo" in (message.get("content") or "") for message in client.calls[1]["messages"])
