from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolutefl.explorer import run_explorer
from evolutefl.explorer.agent import ExplorerAgent
from evolutefl.explorer.v3_agent import _observed_evidence_snapshot
from evolutefl.llm.client import FakeLLMClient
from evolutefl.skills.bank import SkillBankV0


def _call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _config(skill_path: Path, max_steps: int | None = 8) -> dict:
    return {
        "explorer": {
            "max_steps": max_steps,
            "response_retries": 1,
            "finalization_steps": 3,
            "system_prompt_path": "prompt_records/explorer/explorer_system_v0.txt",
        },
        "skill_bank": {"path": str(skill_path)},
        "llm": {},
    }


def _task(repo: Path, run_dir: Path) -> dict:
    return {
        "instance_id": "demo",
        "repo_path": str(repo),
        "repo": "demo/repo",
        "base_commit": "base",
        "bug_report": "The public API returns the wrong value.",
        "run_dir": str(run_dir),
    }


def test_v3_rejects_empty_issue_before_llm_call(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    task = _task(repo, root / "run")
    task["bug_report"] = "   "
    config = _config(skills)
    config["explorer"]["workflow_version"] = "v3"
    fake = FakeLLMClient([])

    with pytest.raises(ValueError, match="non-empty bug_report/problem_statement"):
        run_explorer(
            task=task,
            config=config,
            llm_client=fake,
            skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1),
        )

    assert fake.calls == []
    assert not (root / "run").exists()


class _ConstantEmbeddingClient:
    def embed_texts(self, texts: list[str], *, task: str | None = None) -> list[list[float]]:
        del task
        return [[1.0, 0.0] for _ in texts]


def test_v3_enters_strategy_after_sufficient_fault_evidence(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "first.py").write_text("def first_candidate():\n    return 'bad'\n", encoding="utf-8")
    (repo / "second.py").write_text("def second_candidate():\n    return 'also bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "service", "architecture_signature": "two handlers",
        "component_roles": ["handler"], "responsibility_boundary": "request boundary",
    }, "project")
    issue = _call("load_fault_skill", {
        "fault_family": "state_assignment", "fault_signature": "public value is wrong",
        "project_context": "handler boundary", "suspected_path": "two handlers",
    }, "issue")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none",
        "candidate_functions": ["first.py::first_candidate", "second.py::second_candidate"],
        "candidate_relationship": "competing handlers",
        "unresolved_question": "which handler creates the wrong output",
        "observed_evidence": "both functions were inspected",
        "selection_reason": "empty catalog",
    }, "strategy")
    finish = _call("finish_localization", {
        "ranked_functions": ["first.py::first_candidate"], "summary": "first handler is responsible",
    }, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]},
        {"tool_calls": [issue]},
        {"tool_calls": [
            _call("read_file", {"path": "first.py"}, "first"),
            _call("read_file", {"path": "second.py"}, "second"),
        ]},
        {"tool_calls": [strategy]},
        {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"].update({"workflow_version": "v3", "max_steps": 8})
    run_dir = root / "run"
    result = run_explorer(
        task=_task(repo, run_dir), config=config, llm_client=fake,
        skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1),
    )

    assert result["ranked_functions"] == ["first.py::first_candidate"]
    progress = json.loads((run_dir / "fault_exploration_progress.json").read_text(encoding="utf-8"))
    assert progress["evidence_sufficient"] is True
    assert progress["transition_reason"] == "evidence_sufficient"
    assert fake.calls[3]["tool_choice"]["function"]["name"] == "load_strategy_skill"
    assert fake.calls[4]["tool_choice"]["function"]["name"] == "finish_localization"
    ranking_progress = json.loads((run_dir / "diagnostic_ranking_progress.json").read_text(encoding="utf-8"))
    assert ranking_progress["transition_reason"] == "no_strategy_skill_matched"


def test_v3_candidate_progress_requires_read_definition_evidence(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    run_dir = root / "run"
    run_dir.mkdir()
    grep_only = {
        "event": "tool_result",
        "name": "grep",
        "content": json.dumps({
            "ok": True,
            "result": {"matches": [{"path": "bug.py", "line": 1, "text": "def culprit():"}]},
        }),
    }
    (run_dir / "trajectory.jsonl").write_text(json.dumps(grep_only) + "\n", encoding="utf-8")

    candidates, paths = _observed_evidence_snapshot(run_dir, repo)
    assert candidates == set()
    assert paths == {"bug.py"}

    read_definition = {
        "event": "tool_result",
        "name": "read_file",
        "content": json.dumps({
            "ok": True,
            "result": {"path": "bug.py", "content": [{"line": 1, "text": "def culprit():"}]},
        }),
    }
    with (run_dir / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(read_definition) + "\n")

    candidates, _ = _observed_evidence_snapshot(run_dir, repo)
    assert candidates == {"bug.py::culprit"}


def test_v3_finishes_after_one_strategy_verification(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "first.py").write_text("def first_candidate():\n    return 'bad'\n", encoding="utf-8")
    (repo / "second.py").write_text("def second_candidate():\n    return 'also bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    strategy_id = "strategy_compare_v1"
    skills.write_text(json.dumps({
        "skill_id": strategy_id,
        "status": "active",
        "version": 1,
        "skill_type": "strategy_skill",
        "value": "producer_reporter_comparison",
        "skill": {
            "title": "Producer-reporter comparison",
            "trigger": "Use when two functions compete over where wrong state originates.",
            "knowledge": "Compare the first state producer against its downstream reporter.",
        },
    }) + "\n", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "service", "architecture_signature": "two handlers",
        "component_roles": ["handler"], "responsibility_boundary": "request boundary",
    }, "project")
    issue = _call("load_fault_skill", {
        "fault_family": "state_assignment", "fault_signature": "public value is wrong",
        "project_context": "handler boundary", "suspected_path": "two handlers",
    }, "issue")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": strategy_id,
        "candidate_functions": ["first.py::first_candidate", "second.py::second_candidate"],
        "candidate_relationship": "producer and reporter", "unresolved_question": "where wrong state originates",
        "observed_evidence": "both candidates were inspected", "selection_reason": "The trigger matches.",
    }, "strategy")
    finish = _call("finish_localization", {
        "ranked_functions": ["first.py::first_candidate"], "summary": "first producer is responsible",
    }, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]},
        {"tool_calls": [issue]},
        {"tool_calls": [
            _call("read_file", {"path": "first.py"}, "first"),
            _call("read_file", {"path": "second.py"}, "second"),
        ]},
        {"tool_calls": [strategy]},
        {"content": json.dumps({
            "approved": True,
            "condition_evidence": ["Two candidates compete over where state originates."],
            "reason": "The comparison condition is established.",
        })},
        {"tool_calls": [_call("read_file", {"path": "first.py"}, "verify")]},
        {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"].update({"workflow_version": "v3", "max_steps": 10})
    run_dir = root / "run"
    result = run_explorer(
        task=_task(repo, run_dir), config=config, llm_client=fake,
        skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1),
    )

    assert result["ranked_functions"] == ["first.py::first_candidate"]
    assert fake.calls[6]["tool_choice"]["function"]["name"] == "finish_localization"
    progress = json.loads((run_dir / "diagnostic_ranking_progress.json").read_text(encoding="utf-8"))
    assert progress["transition_reason"] == "strategy_skill_verified"
    assert progress["strategy_validation_observations"] == 1


def test_v3_enters_strategy_after_issue_search_stagnates(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def only_candidate():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "service", "architecture_signature": "single handler",
        "component_roles": ["handler"], "responsibility_boundary": "request boundary",
    }, "project")
    issue = _call("load_fault_skill", {
        "fault_family": "state_assignment", "fault_signature": "public value is wrong",
        "project_context": "handler boundary", "suspected_path": "handler",
    }, "issue")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none", "candidate_functions": ["bug.py::only_candidate"],
        "candidate_relationship": "single unresolved handler", "unresolved_question": "confirm responsibility",
        "observed_evidence": "the same handler was repeatedly inspected", "selection_reason": "empty catalog",
    }, "strategy")
    finish = _call("finish_localization", {
        "ranked_functions": ["bug.py::only_candidate"], "summary": "only handler is responsible",
    }, "finish")
    reads = [
        {"tool_calls": [_call("read_file", {"path": "bug.py"}, f"read-{index}")]}
        for index in range(4)
    ]
    fake = FakeLLMClient([
        {"tool_calls": [project]}, {"tool_calls": [issue]}, *reads,
        {"tool_calls": [strategy]}, {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"].update({
        "workflow_version": "v3",
        "max_steps": 10,
        "issue_min_observations_before_stagnation": 4,
        "stage_stagnation_turns": 2,
    })
    run_dir = root / "run"
    run_explorer(
        task=_task(repo, run_dir), config=config, llm_client=fake,
        skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1),
    )

    progress = json.loads((run_dir / "fault_exploration_progress.json").read_text(encoding="utf-8"))
    assert progress["evidence_sufficient"] is False
    assert progress["search_stagnated"] is True
    assert progress["transition_reason"] == "search_stagnated"
    assert fake.calls[6]["tool_choice"]["function"]["name"] == "load_strategy_skill"


def test_explorer_runs_three_native_stages(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    project = _call(
        "load_project_skill",
        {
            "project_type": "layered API service system",
            "architecture_signature": "State producers feed public response consumers through a stable boundary.",
            "component_roles": ["producer", "consumer"],
            "responsibility_boundary": "state production boundary",
        },
        "project",
    )
    strategy = _call(
        "load_strategy_skill",
        {
            "selected_skill_id": "none",
            "candidate_functions": ["bug.py::culprit"],
            "candidate_relationship": "producer to reporter",
            "unresolved_question": "where the wrong state is created",
            "observed_evidence": "The public result is incorrect.",
            "selection_reason": "The empty catalog contains no applicable Strategy Skill.",
        },
        "strategy",
    )
    finish = _call(
        "finish_localization",
        {"ranked_functions": ["bug.py::culprit"], "summary": "Observed incorrect return."},
        "finish",
    )
    fake = FakeLLMClient(
        [
            {"tool_calls": [_call("grep", {"pattern": "culprit", "repo_path": str(repo)}, "grep")]},
            {"tool_calls": [project]},
            {"tool_calls": [_call("read_file", {"path": "bug.py"}, "read")]},
            {"tool_calls": [strategy]},
            {"tool_calls": [finish]},
        ]
    )
    run_dir = root / "run"
    result = run_explorer(
        task=_task(repo, run_dir),
        config=_config(skills),
        llm_client=fake,
        skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1),
    )
    assert result["ranked_functions"] == ["bug.py::culprit"]
    payload = json.loads((run_dir / "initial_payload.json").read_text(encoding="utf-8"))
    assert set(payload) == {"instance_id", "repo", "base_commit", "bug_report"}
    tool_names = [
        {tool["function"]["name"] for tool in call["tools"]}
        for call in fake.calls
    ]
    assert "load_project_skill" in tool_names[0] and "finish_localization" not in tool_names[0]
    assert "load_strategy_skill" in tool_names[2] and "load_project_skill" not in tool_names[2]
    assert "finish_localization" in tool_names[4] and "load_strategy_skill" not in tool_names[4]
    trajectory = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    for event in ("project_skill_request", "project_skill_loaded", "strategy_skill_request", "strategy_skill_loaded"):
        assert event in trajectory
    project_search = json.loads((run_dir / "project_skill_search.json").read_text(encoding="utf-8"))
    assert "project_type: layered API service system" in project_search["retrieval_query"]
    assert "architecture_signature: State producers feed" in project_search["retrieval_query"]
    assert "component_roles: producer, consumer" in project_search["retrieval_query"]
    assert "responsibility_boundary: state production boundary" in project_search["retrieval_query"]
    assert json.loads((run_dir / "loaded_skills.json").read_text(encoding="utf-8")) == {
        "project_skill": None,
        "strategy_skill": None,
    }


def test_v3_reveals_issue_only_after_project_attempt(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "layered service", "architecture_signature": "adapter feeds service boundary",
        "component_roles": ["adapter", "service"], "responsibility_boundary": "input boundary",
    }, "project")
    issue = _call("load_fault_skill", {
        "fault_family": "interface_contract", "fault_signature": "public result differs from input contract",
        "project_context": "adapter-service boundary", "suspected_path": "input normalization path",
    }, "issue")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none", "candidate_functions": ["bug.py::culprit"],
        "candidate_relationship": "producer to consumer", "unresolved_question": "where value changes",
        "observed_evidence": "public result is wrong", "selection_reason": "empty catalog",
    }, "strategy")
    finish = _call("finish_localization", {"ranked_functions": ["bug.py::culprit"], "summary": "return is wrong"}, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]}, {"tool_calls": [issue]}, {"tool_calls": [strategy]},
        {"tool_calls": [_call("read_file", {"path": "bug.py"}, "evidence")]}, {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"]["workflow_version"] = "v3"
    result = run_explorer(task=_task(repo, root / "v3_run"), config=config, llm_client=fake,
                          skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1))
    assert result["ranked_functions"] == ["bug.py::culprit"]
    payload = json.loads((root / "v3_run" / "initial_payload.json").read_text(encoding="utf-8"))
    assert "bug_report" not in payload and "repository_manifest" in payload
    assert "instance_id" not in payload
    assert payload["stage"] == "repository_orientation"
    assert payload["objective"]
    issue_payload = json.loads((root / "v3_run" / "issue_payload.json").read_text(encoding="utf-8"))
    assert issue_payload["issue"] == _task(repo, root / "x")["bug_report"]
    assert issue_payload["stage"] == "fault_skill_selection"
    # The issue stage transition follows the Project tool reply; otherwise an
    # OpenAI-compatible provider rejects the message history as malformed.
    second_messages = fake.calls[1]["messages"]
    project_call_index = next(
        index for index, message in enumerate(second_messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert second_messages[project_call_index + 1]["role"] == "tool"
    assert second_messages[project_call_index + 2]["role"] == "user"
    assert '"issue"' in second_messages[project_call_index + 2]["content"]
    third_messages = fake.calls[2]["messages"]
    issue_call_index = next(
        index for index, message in enumerate(third_messages)
        if message.get("role") == "assistant"
        and any(
            (call.get("function") or {}).get("name") == "load_fault_skill"
            for call in message.get("tool_calls", [])
        )
    )
    assert third_messages[issue_call_index + 1]["role"] == "tool"
    assert third_messages[issue_call_index + 2]["role"] == "user"
    assert '"stage": "fault_guided_localization"' in third_messages[issue_call_index + 2]["content"]
    trajectory = [
        json.loads(line)
        for line in (root / "v3_run" / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    stage_instructions = [event for event in trajectory if event.get("event") == "stage_instruction"]
    assert stage_instructions
    assert any(event.get("stage") == "diagnostic_ranking" for event in stage_instructions)
    tool_sets = [{tool["function"]["name"] for tool in call["tools"]} for call in fake.calls]
    assert "load_project_skill" in tool_sets[0]
    assert tool_sets[1] == {"load_fault_skill"}
    assert "load_strategy_skill" in tool_sets[2]
    assert "finish_localization" in tool_sets[4]


def test_v3_fault_selector_and_validator_receive_issue_and_orientation_context(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text(json.dumps({
        "skill_id": "fault_config_boundary_v1", "status": "active", "version": 1,
        "skill_type": "fault_skill",
        "fault_family": "transformation_representation",
        "fault_subtype": "configuration_input_boundary_mismatch",
        "skill": {
            "title": "Configuration input boundary mismatch",
            "trigger": "Use when configuration input diverges while crossing a normalization boundary.",
            "knowledge": ["Trace configuration input to normalization.", "Inspect defaults and merges.", "Confirm divergence before consumers."],
        },
    }) + "\n", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "configuration driven service", "architecture_signature": "input adapter feeds normalization boundary",
        "component_roles": ["input adapter", "normalizer"], "responsibility_boundary": "configuration normalization",
    }, "project")
    issue = _call("load_fault_skill", {
        "fault_family": "transformation_representation", "fault_signature": "configuration value has no effect",
        "project_context": "input adapter and normalization boundary", "suspected_path": "normalization and default merge",
    }, "issue")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none", "candidate_functions": ["bug.py::culprit"],
        "candidate_relationship": "input boundary to consumer", "unresolved_question": "where the value changes",
        "observed_evidence": "configuration value is ignored", "selection_reason": "no strategy card",
    }, "strategy")
    finish = _call("finish_localization", {"ranked_functions": ["bug.py::culprit"], "summary": "best"}, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]},
        {"tool_calls": [issue]},
        {"content": json.dumps({"selected_skill_id": "fault_config_boundary_v1", "reason": "The trigger matches the symptom and boundary."})},
        {"content": json.dumps({"applicable": True, "reason": "The request and orientation establish the boundary."})},
        {"tool_calls": [strategy]},
        {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"]["workflow_version"] = "v3"
    run_dir = root / "v3_issue_context"
    run_explorer(task=_task(repo, run_dir), config=config, llm_client=fake,
                 skill_bank=SkillBankV0(skills, min_score=0, project_skill_min_score=0))
    selector_payload = json.loads(fake.calls[2]["messages"][1]["content"])
    validator_payload = json.loads(fake.calls[3]["messages"][1]["content"])
    for payload in (selector_payload, validator_payload):
        assert payload["issue_report"] == _task(repo, root / "unused")["bug_report"]
        assert payload["orientation"]["project_request"]["responsibility_boundary"] == "configuration normalization"
        assert payload["request"]["fault_family"] == "transformation_representation"
    loaded = json.loads((run_dir / "loaded_skills.json").read_text(encoding="utf-8"))
    assert loaded["fault_skill"]["skill_id"] == "fault_config_boundary_v1"


def test_v3_fault_selector_retries_empty_content_and_records_protocol_diagnostics(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text(json.dumps({
        "skill_id": "fault_config_boundary_v1", "status": "active", "version": 1,
        "skill_type": "fault_skill",
        "fault_family": "transformation_representation",
        "fault_subtype": "configuration_input_boundary_mismatch",
        "skill": {
            "title": "Configuration input boundary mismatch",
            "trigger": "Configuration input diverges while crossing a normalization boundary.",
            "knowledge": ["Trace configuration input to normalization."],
        },
    }) + "\n", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "configuration service", "architecture_signature": "adapter to normalizer",
        "component_roles": ["adapter", "normalizer"], "responsibility_boundary": "normalization",
    }, "project")
    fault = _call("load_fault_skill", {
        "fault_family": "transformation_representation", "fault_signature": "value changes form",
        "project_context": "adapter and normalizer", "suspected_path": "normalization boundary",
    }, "fault")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none", "candidate_functions": ["bug.py::culprit"],
        "candidate_relationship": "boundary to consumer", "unresolved_question": "where it changes",
        "observed_evidence": "value differs", "selection_reason": "no strategy card",
    }, "strategy")
    finish = _call("finish_localization", {
        "ranked_functions": ["bug.py::culprit"], "summary": "best",
    }, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]},
        {"tool_calls": [fault]},
        {"content": "", "raw": {"model": "fake-reasoner", "choices": [{
            "finish_reason": "length", "message": {"content": "", "reasoning_content": "thinking"},
        }], "usage": {"completion_tokens": 4096}}},
        {"content": json.dumps({"selected_skill_id": None, "reason": "No mechanism match."})},
        {"tool_calls": [strategy]},
        {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"].update({
        "workflow_version": "v3",
        "fault_skill_selector_attempts": 3,
        "fault_skill_selector_max_tokens": 4096,
        "fault_skill_selector_retry_max_tokens": 8192,
    })
    run_dir = root / "v3_selector_retry"
    run_explorer(
        task=_task(repo, run_dir),
        config=config,
        llm_client=fake,
        skill_bank=SkillBankV0(skills, min_score=0, project_skill_min_score=0),
    )
    search = json.loads((run_dir / "fault_skill_search.json").read_text(encoding="utf-8"))
    attempts = search["search_trace"]["selector_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["valid"] is False
    assert attempts[0]["finish_reason"] == "length"
    assert attempts[0]["reasoning_char_count"] == len("thinking")
    assert attempts[1]["valid"] is True
    assert attempts[0]["requested_max_tokens"] == 4096
    assert attempts[1]["requested_max_tokens"] == 8192
    assert fake.calls[2]["max_tokens"] == 4096
    assert fake.calls[3]["max_tokens"] == 8192
    assert search["search_trace"]["selector_protocol_failed"] is False
    assert search["search_trace"]["selector"]["selected_skill_id"] is None


def test_v3_invalid_fault_family_is_rejected_without_consuming_attempt(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "service", "architecture_signature": "request handler",
        "component_roles": ["handler"], "responsibility_boundary": "request boundary",
    }, "project")
    invalid = _call("load_fault_skill", {
        "fault_family": "not_a_family", "fault_signature": "wrong output",
        "project_context": "request handler", "suspected_path": "handler to response",
    }, "invalid")
    valid = _call("load_fault_skill", {
        "fault_family": "state_assignment", "fault_signature": "wrong output state",
        "project_context": "request handler", "suspected_path": "handler to response",
    }, "valid")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none", "candidate_functions": [],
        "candidate_relationship": "no candidates", "unresolved_question": "where state changes",
        "observed_evidence": "no matching card", "selection_reason": "empty catalog",
    }, "strategy")
    finish = _call("finish_localization", {
        "ranked_functions": ["handler.py::handle"], "summary": "handler boundary",
    }, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]}, {"tool_calls": [invalid]}, {"tool_calls": [valid]},
        {"tool_calls": [strategy]}, {"tool_calls": [finish]},
    ])
    config = _config(skills, max_steps=7)
    config["explorer"]["workflow_version"] = "v3"
    run_dir = root / "invalid_family"

    result = run_explorer(
        task=_task(repo, run_dir), config=config, llm_client=fake,
        skill_bank=SkillBankV0(skills),
    )

    assert result["status"] == "completed"
    trajectory = [
        json.loads(line)
        for line in (run_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    requests = [event for event in trajectory if event.get("event") == "fault_skill_request"]
    assert len(requests) == 1
    assert requests[0]["request"]["fault_family"] == "state_assignment"
    assert any(
        event.get("event") == "tool_result" and "Invalid fault_family" in str(event.get("content"))
        for event in trajectory
    )


def test_stage_skill_is_injected_through_tool_message(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 0\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    records = [
        {
            "skill_id": "project_pipeline_v1", "status": "active", "version": 1,
            "skill_type": "project_skill", "value": "producer_consumer_pipeline_system",
            "skill": {"title": "Producer-consumer pipeline system", "trigger": "Use when a repository moves state through producer and consumer components.", "knowledge": "Pipelines separate state production from downstream exposure."},
        },
        {
            "skill_id": "strategy_contrast_v1", "status": "active", "version": 1,
                "skill_type": "strategy_skill", "value": "producer_reporter_metadata_contrast",
                "skill": {"title": "Producer reporter metadata contrast", "trigger": "Use when a producer and reporter disagree over canonical metadata.", "knowledge": "Compare the first wrong metadata state with later symptom exposure."},
        },
    ]
    skills.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    fake = FakeLLMClient([
        {"tool_calls": [_call("load_project_skill", {"project_type": "producer consumer pipeline system", "architecture_signature": "State producers feed downstream consumers.", "component_roles": ["producer", "consumer"], "responsibility_boundary": "producer consumer"}, "p")]},
        {"tool_calls": [_call("load_strategy_skill", {"selected_skill_id": "strategy_contrast_v1", "candidate_functions": [], "candidate_relationship": "producer reporter metadata", "unresolved_question": "cause", "observed_evidence": "The canonical metadata is wrong at the reporter.", "selection_reason": "The trigger covers the observed metadata disagreement."}, "s")]},
        {"content": json.dumps({"approved": True, "condition_evidence": ["The producer and reporter disagree over canonical metadata.", "The unresolved question is where that metadata first becomes wrong."], "reason": "The trigger conditions are established."})},
        {"tool_calls": [_call("read_file", {"path": "bug.py"}, "read")]},
        {"tool_calls": [_call("finish_localization", {"ranked_functions": ["bug.py::culprit"], "summary": "best"}, "f")]},
    ])
    run_dir = root / "run"
    run_explorer(task=_task(repo, run_dir), config=_config(skills), llm_client=fake, skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1))
    loaded = json.loads((run_dir / "loaded_skills.json").read_text(encoding="utf-8"))
    assert loaded["project_skill"]["skill_id"] == "project_pipeline_v1"
    assert loaded["strategy_skill"]["skill_id"] == "strategy_contrast_v1"
    strategy_tools = fake.calls[1]["tools"]
    strategy_schema = next(
        tool["function"] for tool in strategy_tools
        if tool["function"]["name"] == "load_strategy_skill"
    )
    assert "strategy_contrast_v1" in strategy_schema["description"]
    assert "disagree over canonical metadata" in strategy_schema["description"]
    assert "Compare the first wrong state" not in strategy_schema["description"]
    assert strategy_schema["parameters"]["properties"]["selected_skill_id"]["enum"] == [
        "strategy_contrast_v1",
        "none",
    ]
    second_messages = fake.calls[1]["messages"]
    project_tool = next(message for message in second_messages if message.get("name") == "load_project_skill")
    assert json.loads(project_tool["content"])["matched_skill"]["skill_id"] == "project_pipeline_v1"
    validation = json.loads((run_dir / "strategy_skill_validation.json").read_text(encoding="utf-8"))
    assert validation["approved"] is True
    validator_call = fake.calls[2]
    assert "tools" not in validator_call
    assert validator_call["response_format"] == {"type": "json_object"}
    post_strategy_tool_names = {tool["function"]["name"] for tool in fake.calls[3]["tools"]}
    assert post_strategy_tool_names == {"grep", "read_file", "write"}
    trajectory = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    assert "strategy_skill_evidence_observation" in trajectory


def test_embedding_project_validation_retries_an_empty_structured_response(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 0\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text(
        json.dumps(
            {
                "skill_id": "project_pipeline_v1",
                "status": "active",
                "version": 1,
                "skill_type": "project_skill",
                "value": "producer_consumer_pipeline_system",
                "skill": {
                    "title": "Producer-consumer pipeline system",
                    "trigger": "Use when state passes from a producer into downstream consumers.",
                    "knowledge": "Inspect the boundary that first creates the shared state.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    project_call = _call(
        "load_project_skill",
        {
            "project_type": "producer consumer pipeline system",
            "architecture_signature": "A producer creates state consumed by downstream components.",
            "component_roles": ["producer", "consumer"],
            "responsibility_boundary": "state production boundary",
        },
        "p",
    )
    fake = FakeLLMClient(
        [
            {"tool_calls": [project_call]},
                {
                    "tool_calls": [
                        _call(
                            "select_project_skill",
                        {
                            "selected_skill_id": "project_pipeline_v1",
                            "reason": "The component roles and state boundary match.",
                        },
                        "project-selector",
                        )
                    ]
                },
                {},
                {
                    "content": json.dumps(
                        {
                            "approved": True,
                            "role_evidence": [
                                "The producer creates state consumed by the downstream consumer.",
                                "The observed boundary transfers the produced state to the consumer.",
                            ],
                            "reason": "The producer-consumer boundary is established.",
                        }
                    )
                },
                {
                    "tool_calls": [
                        _call(
                        "load_strategy_skill",
                        {
                            "selected_skill_id": "none",
                            "candidate_functions": ["bug.py::culprit"],
                            "candidate_relationship": "single producer",
                            "unresolved_question": "where state is created",
                            "observed_evidence": "The producer returns the wrong state.",
                            "selection_reason": "No Strategy Skill is available.",
                        },
                        "s",
                    )
                ]
            },
            {
                "tool_calls": [
                    _call(
                        "finish_localization",
                        {
                            "ranked_functions": ["bug.py::culprit"],
                            "summary": "Observed the faulty producer.",
                        },
                        "f",
                    )
                ]
            },
        ]
    )
    run_dir = root / "run"
    bank = SkillBankV0(
        skills,
        retrieval_mode="embedding",
        embedding_client=_ConstantEmbeddingClient(),
        embedding_cache_path=root / "embeddings.jsonl",
        project_skill_embedding_min_score=0.9,
    )

    run_explorer(
        task=_task(repo, run_dir),
        config=_config(skills),
        llm_client=fake,
        skill_bank=bank,
    )

    loaded = json.loads((run_dir / "loaded_skills.json").read_text(encoding="utf-8"))
    assert loaded["project_skill"]["skill_id"] == "project_pipeline_v1"
    selector_call = fake.calls[1]
    selector_payload = json.loads(selector_call["messages"][1]["content"])
    candidate = selector_payload["candidate_project_skills"][0]
    assert candidate["skill_id"] == "project_pipeline_v1"
    assert "knowledge" not in candidate
    assert selector_call["tool_choice"]["function"]["name"] == "select_project_skill"
    search = json.loads((run_dir / "project_skill_search.json").read_text(encoding="utf-8"))
    assert search["search_trace"]["selector"]["selected_skill_id"] == "project_pipeline_v1"
    validation = json.loads((run_dir / "project_skill_validation.json").read_text(encoding="utf-8"))
    assert validation["approved"] is True
    assert validation["validation_attempt_count"] == 2
    validator_retry = fake.calls[3]
    assert "tools" not in validator_retry
    assert validator_retry["response_format"] == {"type": "json_object"}


def test_near_max_steps_forces_project_strategy_finish(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 0\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    fake = FakeLLMClient([
        {"tool_calls": [_call("load_project_skill", {"project_type": "unknown project type", "architecture_signature": "Unknown architecture.", "component_roles": ["producer"], "responsibility_boundary": "boundary"}, "p")]},
        {"tool_calls": [_call("load_strategy_skill", {}, "s")]},
        {"tool_calls": [_call("finish_localization", {"ranked_functions": ["bug.py::culprit"], "summary": "forced"}, "f")]},
    ])
    result = run_explorer(task=_task(repo, root / "run"), config=_config(skills, 3), llm_client=fake, skill_bank=SkillBankV0(skills))
    assert result["ranked_functions"] == ["bug.py::culprit"]
    choices = [call["tool_choice"]["function"]["name"] for call in fake.calls]
    assert choices == ["load_project_skill", "load_strategy_skill", "finish_localization"]


def test_strategy_only_ablation_records_project_stage_as_disabled(tmpdir) -> None:
    """A one-type ablation must skip, rather than fail, the disabled stage."""

    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 0\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text(
        json.dumps(
            {
                "skill_id": "strategy_source_reporter_v1",
                "status": "active",
                "version": 1,
                "skill_type": "strategy_skill",
                "value": "source_reporter_contract_decision",
                "skill": {
                    "title": "Source versus reporter contract decision",
                    "trigger": "Use when a source and reporter disagree about a canonical value.",
                    "knowledge": "Compare where a value first becomes invalid with where it is reported.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake = FakeLLMClient(
        [
            {
                "tool_calls": [
                    _call(
                        "load_project_skill",
                        {
                            "project_type": "source reporter pipeline",
                            "architecture_signature": "A source creates canonical values that a reporter exposes.",
                            "component_roles": ["source", "reporter"],
                            "responsibility_boundary": "canonical value handoff",
                        },
                        "project",
                    )
                ]
            },
            {
                "tool_calls": [
                    _call(
                        "load_strategy_skill",
                        {
                            "selected_skill_id": "strategy_source_reporter_v1",
                            "candidate_functions": ["bug.py::culprit"],
                            "candidate_relationship": "source and reporter disagree",
                            "unresolved_question": "where the value first becomes invalid",
                            "observed_evidence": "The source emits an invalid value before reporting.",
                            "selection_reason": "The source-reporter trigger applies.",
                        },
                        "strategy",
                    )
                ]
            },
            {
                "content": json.dumps(
                    {
                        "approved": True,
                        "condition_evidence": [
                            "The source and reporter disagree about a canonical value.",
                            "The source emits the value before the reporter exposes it.",
                        ],
                        "reason": "The trigger conditions are present.",
                    }
                )
            },
            {
                "tool_calls": [
                    _call(
                        "finish_localization",
                        {"ranked_functions": ["bug.py::culprit"], "summary": "source evidence"},
                        "finish",
                    )
                ]
            },
        ]
    )

    run_dir = root / "run"
    run_explorer(
        task=_task(repo, run_dir),
        config=_config(skills),
        llm_client=fake,
        skill_bank=SkillBankV0(skills, enabled_skill_types=["strategy_skill"]),
    )

    project_request = json.loads((run_dir / "project_skill_request.json").read_text(encoding="utf-8"))
    project_search = json.loads((run_dir / "project_skill_search.json").read_text(encoding="utf-8"))
    loaded = json.loads((run_dir / "loaded_skills.json").read_text(encoding="utf-8"))
    assert project_request["disabled"] is True
    assert project_search["search_trace"]["retrieval_mode"] == "disabled_skill_type_ablation_v1"
    assert loaded["project_skill"] is None
    assert loaded["strategy_skill"]["skill_id"] == "strategy_source_reporter_v1"
    assert {tool["function"]["name"] for tool in fake.calls[0]["tools"]} == {
        "grep",
        "read_file",
        "write",
        "load_project_skill",
    }
    assert {tool["function"]["name"] for tool in fake.calls[1]["tools"]} == {
        "grep",
        "read_file",
        "write",
        "load_strategy_skill",
    }


def test_v3_rejects_an_unavailable_tool_in_the_issue_stage(tmpdir) -> None:
    """A provider hallucinating grep must not bypass the Fault Skill transition."""

    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "layered service", "architecture_signature": "adapter feeds service",
        "component_roles": ["adapter", "service"], "responsibility_boundary": "input boundary",
    }, "project")
    issue = _call("load_fault_skill", {
        "fault_family": "interface_contract", "fault_signature": "public result differs from contract",
        "project_context": "adapter-service boundary", "suspected_path": "input normalization",
    }, "issue")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none", "candidate_functions": ["bug.py::culprit"],
        "candidate_relationship": "producer to consumer", "unresolved_question": "where value changes",
        "observed_evidence": "public result is wrong", "selection_reason": "empty catalog",
    }, "strategy")
    finish = _call("finish_localization", {
        "ranked_functions": ["bug.py::culprit"], "summary": "return is wrong",
    }, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]},
        {"tool_calls": [_call("grep", {"pattern": "culprit", "repo_path": str(repo)}, "illegal-grep")]},
        {"tool_calls": [issue]},
        {"tool_calls": [strategy]},
        {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"]["workflow_version"] = "v3"
    run_dir = root / "v3_stage_gate"
    result = run_explorer(
        task=_task(repo, run_dir), config=config, llm_client=fake,
        skill_bank=SkillBankV0(skills, min_score=1, project_skill_min_score=1),
    )

    assert result["ranked_functions"] == ["bug.py::culprit"]
    tool_sets = [{tool["function"]["name"] for tool in call["tools"]} for call in fake.calls]
    assert tool_sets[1] == {"load_fault_skill"}
    trajectory = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    assert '"event":"stage_tool_rejected"' in trajectory
    assert '"tool_name":"grep"' in trajectory


def test_v3_requires_fault_skill_evidence_before_strategy(tmpdir) -> None:
    """A loaded Fault Skill receives repository evidence before strategy selection."""

    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text(json.dumps({
        "skill_id": "issue_return_boundary_v1", "status": "active", "version": 1,
        "skill_type": "fault_skill",
        "fault_family": "interface_contract",
        "fault_subtype": "return_value_boundary_mismatch",
        "skill": {
            "title": "Return boundary mismatch",
            "trigger": "Use when public output diverges at a normalization boundary.",
            "knowledge": ["Trace the value through the boundary."],
        },
    }) + "\n", encoding="utf-8")
    project = _call("load_project_skill", {
        "project_type": "layered service", "architecture_signature": "adapter feeds service",
        "component_roles": ["adapter", "service"], "responsibility_boundary": "input boundary",
    }, "project")
    issue = _call("load_fault_skill", {
        "fault_family": "interface_contract", "fault_signature": "public result differs from contract",
        "project_context": "adapter-service boundary", "suspected_path": "input normalization",
    }, "issue")
    strategy = _call("load_strategy_skill", {
        "selected_skill_id": "none", "candidate_functions": ["bug.py::culprit"],
        "candidate_relationship": "producer to consumer", "unresolved_question": "where value changes",
        "observed_evidence": "return is wrong", "selection_reason": "empty strategy catalog",
    }, "strategy")
    finish = _call("finish_localization", {
        "ranked_functions": ["bug.py::culprit"], "summary": "return is wrong",
    }, "finish")
    fake = FakeLLMClient([
        {"tool_calls": [project]},
        {"tool_calls": [issue]},
        {"content": json.dumps({"selected_skill_id": "issue_return_boundary_v1", "reason": "The trigger matches."})},
        {"content": json.dumps({"applicable": True, "reason": "The boundary is established."})},
        {"tool_calls": [strategy]},
        {"tool_calls": [_call("grep", {"pattern": "culprit", "repo_path": str(repo)}, "grep")]},
        {"tool_calls": [_call("read_file", {"path": "bug.py"}, "read")]},
        {"tool_calls": [strategy]},
        {"tool_calls": [finish]},
    ])
    config = _config(skills)
    config["explorer"].update({"workflow_version": "v3", "fault_skill_evidence_budget": 2})
    run_dir = root / "v3_issue_evidence"
    result = run_explorer(
        task=_task(repo, run_dir), config=config, llm_client=fake,
        skill_bank=SkillBankV0(skills, min_score=0, project_skill_min_score=0),
    )

    assert result["ranked_functions"] == ["bug.py::culprit"]
    assert result["stage_completion"]["fault_skill_evidence_observations"] == 2
    tool_sets = [
        {tool["function"]["name"] for tool in call["tools"]}
        for call in fake.calls
        if "tools" in call
    ]
    assert "load_strategy_skill" not in tool_sets[2]
    assert "load_strategy_skill" in tool_sets[5]
    trajectory = (run_dir / "trajectory.jsonl").read_text(encoding="utf-8")
    assert trajectory.count('"event":"fault_skill_evidence_observation"') == 2


def test_forced_finish_uses_only_compact_context(tmpdir) -> None:
    root = Path(str(tmpdir))
    repo = root / "repo"
    repo.mkdir()
    (repo / "bug.py").write_text("def culprit():\n    return 'bad'\n", encoding="utf-8")
    skills = root / "skills.jsonl"
    skills.write_text("", encoding="utf-8")
    fake = FakeLLMClient([{"tool_calls": [_call("finish_localization", {
        "ranked_functions": ["bug.py::culprit"], "summary": "compact evidence",
    }, "finish")] }])
    agent = ExplorerAgent(
        llm_client=fake,
        skill_bank=SkillBankV0(skills),
        config=_config(skills),
        system_prompt="test prompt",
    )
    trace: list[dict] = []
    result = agent._forced_finish(
        [{"role": "system", "content": "long old history"}, {"role": "user", "content": "old"}],
        root / "run",
        str(repo),
        trace,
        reason="test convergence",
        compact_context={"issue": "wrong return", "observed_function_candidates": ["bug.py::culprit"]},
    )

    assert result["action"]["ranked_functions"] == ["bug.py::culprit"]
    assert len(fake.calls[0]["messages"]) == 2
    context = json.loads((root / "run" / "forced_finish_context.json").read_text(encoding="utf-8"))
    assert context["observed_function_candidates"] == ["bug.py::culprit"]


def test_project_stage_loads_exact_repository_skill_without_selector(tmpdir) -> None:
    root = Path(str(tmpdir))
    skills = root / "skills.jsonl"
    skills.write_text(json.dumps({
        "skill_id": "project_skill_demo_repo",
        "status": "active",
        "version": 1,
        "skill_type": "project_skill",
        "repo_id": "demo/repo",
        "skill": {
            "title": "Demo repository architecture",
            "knowledge": ["Requests pass from the public facade into a shared dispatcher."],
        },
    }) + "\n", encoding="utf-8")
    fake = FakeLLMClient([])
    agent = ExplorerAgent(
        llm_client=fake,
        skill_bank=SkillBankV0(skills),
        config=_config(skills),
        system_prompt="test prompt",
    )
    run_dir = root / "run"
    run_dir.mkdir()

    _, loaded = agent._load_stage_skill(
        _call("load_project_skill", {
            "repository_summary": "A public facade delegates to a dispatcher.",
            "component_roles": ["facade", "dispatcher"],
            "responsibility_boundaries": ["facade-to-dispatcher"],
        }, "project"),
        "project_skill",
        run_dir,
        2,
        repo_id="Demo/Repo",
    )

    assert loaded is not None
    assert loaded["skill_id"] == "project_skill_demo_repo"
    search = json.loads((run_dir / "project_skill_search.json").read_text(encoding="utf-8"))
    assert search["search_trace"]["retrieval_mode"] == "exact_repo_id_v1"
    assert fake.calls == []
