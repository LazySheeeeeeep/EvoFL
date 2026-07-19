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
        "issue_abstraction": {"enabled": False},
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 2, "max_per_skill_type": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_skill_type": {}},
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
    initial_payload = json.loads((run_dir / "initial_payload.json").read_text(encoding="utf-8"))
    assert "skill_context" not in initial_payload
    assert "matched_skills" not in initial_payload
    assert "skill_search_trace" not in initial_payload
    assert "rendered_localization_context" not in initial_payload
    assert fake.calls[0]["tools"]
    assert any(tool["function"]["name"] == "finish_localization" for tool in fake.calls[0]["tools"])


def test_explorer_injects_issue_abstraction_when_enabled(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"
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
            {
                "content": json.dumps(
                    {
                        "abstract_problem_signature": "A function returns an incorrect value.",
                        "project_type_key": "small_python_library",
                        "project_type_description": "A compact Python library exposing behavior through direct function APIs.",
                        "project_skill_query": "small python library and implementation boundary",
                        "strategy_skill_query": "inspect reported behavior and rank its implementing function",
                        "key_symptoms": ["wrong value returned"],
                    }
                )
            },
            {"tool_calls": [finish_call], "content": ""},
        ]
    )
    config = {
        "explorer": {"max_steps": 3, "response_retries": 1, "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt"},
        "issue_abstraction": {"enabled": True, "prompt_path": "prompt_records/explorer/issue_abstraction_v0.txt"},
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 2, "max_per_skill_type": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_skill_type": {}},
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

    initial_payload = json.loads((run_dir / "initial_payload.json").read_text(encoding="utf-8"))
    assert result["ranked_functions"] == ["bug.py::culprit"]
    assert initial_payload["issue_abstraction"]["abstract_problem_signature"] == "A function returns an incorrect value."
    assert initial_payload["issue_abstraction"]["project_type_key"] == "small_python_library"
    assert initial_payload["issue_abstraction"]["project_type_description"].startswith("A compact Python library")
    assert "project_skill_query" not in initial_payload["issue_abstraction"]
    assert "strategy_skill_query" not in initial_payload["issue_abstraction"]
    assert (run_dir / "issue_abstraction.json").exists()
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert fake.calls[1]["tools"]


def test_explorer_injects_one_structured_skill_context_without_duplicates(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text(
        json.dumps(
            {
                "skill_id": "strategy_trace_v1",
                "status": "active",
                "version": 1,
                "skill_type": "strategy_skill",
                "scope": "contextual",
                "value": "producer_consumer_trace",
                "retrieval_text": "producer consumer trace ignored downstream",
                "skill": {
                    "title": "Producer-consumer tracing",
                    "trigger": "Use when a producer value is ignored downstream.",
                    "knowledge": "Trace the value to each consumer and rank the first divergence.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    finish_call = {
        "id": "call_finish",
        "type": "function",
        "function": {
            "name": "finish_localization",
            "arguments": json.dumps({"ranked_functions": ["bug.py::culprit"], "summary": "Found culprit."}),
        },
    }
    config = {
        "explorer": {"max_steps": 2, "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt"},
        "issue_abstraction": {"enabled": False},
        "skill_bank": {
            "path": str(skill_path),
            "max_matched_skills": 2,
            "max_per_skill_type": {"project_skill": 1, "strategy_skill": 1},
            "min_score": 3.0,
        },
        "assembler": {"max_knowledge_chars": 900, "max_per_skill_type": {}},
        "llm": {},
    }
    run_dir = tmp_path / "run"
    result = run_explorer(
        task={
            "instance_id": "demo",
            "repo_path": str(repo),
            "repo": "demo/repo",
            "base_commit": "unknown",
            "bug_report": "producer value ignored downstream consumer trace",
            "run_dir": str(run_dir),
        },
        config=config,
        llm_client=FakeLLMClient([{"tool_calls": [finish_call], "content": ""}]),
        skill_bank=SkillBankV0(skill_path, min_score=3.0),
    )
    initial_payload = json.loads((run_dir / "initial_payload.json").read_text(encoding="utf-8"))
    assert result["ranked_functions"] == ["bug.py::culprit"]
    assert initial_payload["skill_context"]["sections"]["strategy_skill"][0]["source_skill_id"] == "strategy_trace_v1"
    assert "matched_skills" not in initial_payload
    assert "assembled_context" not in initial_payload
    assert "rendered_localization_context" not in initial_payload


def test_finalization_mode_forces_finish_tool_choice(tmpdir) -> None:
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
        "issue_abstraction": {"enabled": False},
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 2, "max_per_skill_type": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_skill_type": {}},
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
    assert fake.calls[-1]["tool_choice"] == {"type": "function", "function": {"name": "finish_localization"}}


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
        "issue_abstraction": {"enabled": False},
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 2, "max_per_skill_type": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_skill_type": {}},
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
    assert fake.calls[-1]["tool_choice"] == {"type": "function", "function": {"name": "finish_localization"}}


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
        "issue_abstraction": {"enabled": False},
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 2, "max_per_skill_type": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_skill_type": {}},
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
        "issue_abstraction": {"enabled": False},
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 2, "max_per_skill_type": {}},
        "assembler": {"max_knowledge_chars": 900, "max_per_skill_type": {}},
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
