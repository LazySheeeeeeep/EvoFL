from __future__ import annotations

import json
from pathlib import Path

from evolutefl.explorer import run_explorer
from evolutefl.llm.client import FakeLLMClient
from evolutefl.skills.bank import SkillBankV0


def test_explorer_runs_native_tool_call_smoke(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "grep", "arguments": json.dumps({"repo_path": str(repo), "pattern": "culprit"})},
    }
    finish_call = {
        "id": "call_finish",
        "type": "function",
        "function": {
            "name": "finish_localization",
            "arguments": json.dumps({"ranked_functions": ["bug.py::culprit"], "summary": "Found culprit."}),
        },
    }
    fake = FakeLLMClient(
        [
            {"tool_calls": [tool_call], "content": ""},
            {"tool_calls": [finish_call], "content": ""},
        ]
    )
    config = {
        "explorer": {"max_steps": 5, "response_retries": 1, "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt"},
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 5, "max_per_dimension": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_dimension": {}},
        "llm": {},
    }
    result = run_explorer(
        task={
            "instance_id": "demo",
            "repo_path": str(repo),
            "repo": "demo/repo",
            "base_commit": "unknown",
            "bug_report": "culprit returns bad",
            "run_dir": str(run_dir),
        },
        config=config,
        llm_client=fake,
        skill_bank=SkillBankV0(skill_path),
    )
    assert result["ranked_functions"] == ["bug.py::culprit"]
    assert (run_dir / "trajectory.jsonl").exists()
    assert fake.calls[0]["tools"]
    assert any(tool["function"]["name"] == "finish_localization" for tool in fake.calls[0]["tools"])


def test_forced_finish_uses_no_tools_and_json_response_format(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "grep", "arguments": json.dumps({"repo_path": str(repo), "pattern": "culprit"})},
    }
    fake = FakeLLMClient(
        [
            {"tool_calls": [tool_call], "content": ""},
            {
                "content": json.dumps(
                    {
                        "thought": "best available evidence",
                        "action": {"type": "finish", "ranked_functions": ["bug.py::culprit"], "summary": "Forced finish."},
                    }
                )
            },
        ]
    )
    config = {
        "explorer": {
            "max_steps": 2,
            "response_retries": 0,
            "checkpoint_interval": 0,
            "finalization_steps": 1,
            "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt",
        },
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 5, "max_per_dimension": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_dimension": {}},
        "llm": {},
    }
    result = run_explorer(
        task={
            "instance_id": "demo",
            "repo_path": str(repo),
            "repo": "demo/repo",
            "base_commit": "unknown",
            "bug_report": "culprit returns bad",
            "run_dir": str(run_dir),
        },
        config=config,
        llm_client=fake,
        skill_bank=SkillBankV0(skill_path),
    )
    assert result["ranked_functions"] == ["bug.py::culprit"]
    assert fake.calls[-1]["tools"][0]["function"]["name"] == "finish_localization"
    assert fake.calls[-1]["tool_choice"] == "auto"


def test_unbounded_explorer_uses_runtime_timeout_for_forced_finish(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    fake = FakeLLMClient(
        [
            {
                "content": json.dumps(
                    {
                        "thought": "best available evidence",
                        "action": {"type": "finish", "ranked_functions": ["bug.py::culprit"], "summary": "Timeout finish."},
                    }
                )
            }
        ]
    )
    config = {
        "explorer": {
            "max_steps": None,
            "max_runtime_seconds": 0,
            "response_retries": 0,
            "checkpoint_interval": 0,
            "finalization_steps": 0,
            "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt",
        },
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 5, "max_per_dimension": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_dimension": {}},
        "llm": {},
    }
    result = run_explorer(
        task={
            "instance_id": "demo",
            "repo_path": str(repo),
            "repo": "demo/repo",
            "base_commit": "unknown",
            "bug_report": "culprit returns bad",
            "run_dir": str(run_dir),
        },
        config=config,
        llm_client=fake,
        skill_bank=SkillBankV0(skill_path),
    )
    trajectory = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    assert result["ranked_functions"] == ["bug.py::culprit"]
    assert "runtime_timeout" in trajectory
    assert fake.calls[-1]["tools"][0]["function"]["name"] == "finish_localization"
    assert fake.calls[-1]["tool_choice"] == "auto"


def test_empty_final_response_exhaustion_forces_finish_instead_of_interrupting(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    fake = FakeLLMClient(
        [
            {"content": ""},
            {"content": ""},
            {"content": ""},
        ]
    )
    config = {
        "explorer": {
            "max_steps": None,
            "max_runtime_seconds": 30,
            "response_retries": 1,
            "checkpoint_interval": 0,
            "finalization_steps": 0,
            "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt",
        },
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 5, "max_per_dimension": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_dimension": {}},
        "llm": {},
    }
    result = run_explorer(
        task={
            "instance_id": "demo",
            "repo_path": str(repo),
            "repo": "demo/repo",
            "base_commit": "unknown",
            "bug_report": "culprit returns bad",
            "run_dir": str(run_dir),
        },
        config=config,
        llm_client=fake,
        skill_bank=SkillBankV0(skill_path),
    )
    trajectory = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    assert result["status"] == "completed"
    assert result["ranked_functions"] == []
    assert "response_repair_exhausted" in trajectory
    assert "forced_finish" in trajectory


def test_empty_forced_finish_uses_deterministic_evidence_fallback(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "bug.py"})},
    }
    fake = FakeLLMClient(
        [
            {"tool_calls": [tool_call], "content": ""},
            {"content": ""},
            {"content": ""},
        ]
    )
    config = {
        "explorer": {
            "max_steps": 2,
            "response_retries": 0,
            "checkpoint_interval": 0,
            "finalization_steps": 1,
            "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt",
        },
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 5, "max_per_dimension": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_dimension": {}},
        "llm": {},
    }
    result = run_explorer(
        task={
            "instance_id": "demo",
            "repo_path": str(repo),
            "repo": "demo/repo",
            "base_commit": "unknown",
            "bug_report": "culprit returns bad",
            "run_dir": str(run_dir),
        },
        config=config,
        llm_client=fake,
        skill_bank=SkillBankV0(skill_path),
    )
    trajectory = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    assert result["ranked_functions"] == ["bug.py::culprit"]
    assert "deterministic_finish" in trajectory
