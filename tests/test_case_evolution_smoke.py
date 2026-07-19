from __future__ import annotations

import json
from pathlib import Path

from evolutefl.llm.client import FakeLLMClient
from evolutefl.reflection import run_case_evolution


def _write_case(case_dir: Path, ranked_functions: list[str]) -> None:
    case_dir.mkdir()
    (case_dir / "result.json").write_text(
        json.dumps(
            {
                "instance_id": "case1",
                "status": "completed",
                "ranked_functions": ranked_functions,
                "final_summary": "Explorer summary.",
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "trajectory.jsonl").write_text(
        json.dumps({"event": "assistant_tool_calls", "step": 1, "tool_calls": []})
        + "\n"
        + json.dumps({"event": "finish", "step": 2})
        + "\n",
        encoding="utf-8",
    )


def _config(skill_path: Path) -> dict:
    return {
        "llm": {},
        "reflection": {"reflector_prompt_path": "prompt_records/reflection/reflector_skill_v0.txt"},
        "skill_bank": {
            "path": str(skill_path),
            "max_matched_skills": 2,
            "max_per_skill_type": {"project_skill": 1, "strategy_skill": 1},
        },
    }


def _no_update() -> dict:
    return {
        "decision": "no_update",
        "scope": None,
        "target_value": "unknown",
        "target_skill_id": None,
        "rationale": "No reusable lesson for this skill type.",
        "edit": None,
        "no_update_reason": "No reusable lesson for this skill type.",
    }


def test_case_evolution_failure_creates_strategy_skill(tmpdir) -> None:
    case_dir = Path(str(tmpdir)) / "case"
    _write_case(case_dir, ["wrong.py::symptom"])
    (case_dir / "issue_abstraction.json").write_text(
        json.dumps(
            {
                "abstract_problem_signature": "A symptom reporter outranks the state producer.",
                "project_skill_query": "small library with state producer and reporter components",
                "strategy_skill_query": "compare producer and reporter then rank first causal state change",
                "key_symptoms": ["symptom reporter is related but not causal"],
            }
        ),
        encoding="utf-8",
    )
    skill_path = Path(str(tmpdir)) / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    reflector = {
        "case_summary": "Create one strategy skill.",
        "outcome_type": "failure",
        "skill_updates": {
            "project_skill": _no_update(),
            "strategy_skill": {
                "decision": "create_new",
                "scope": "global",
                "target_value": "root_cause_before_reporter",
                "target_skill_id": None,
                "rationale": "The trajectory exposes a reusable ranking policy.",
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {
                        "text": "Rank the first evidence-backed state producer above functions that only propagate or report the symptom.",
                        "title": "Root cause before reporter",
                        "trigger": "Use when reporters are related to the symptom but do not create the invalid state.",
                        "retrieval_text": "state producer downstream reporter causal ranking",
                    },
                },
                "no_update_reason": None,
            },
        },
        "no_update_reason": None,
    }
    fake = FakeLLMClient([{"content": json.dumps(reflector)}])
    summary = run_case_evolution(
        case_run_dir=case_dir,
        repo="demo/repo",
        issue="Bug report",
        config=_config(skill_path),
        llm_client=fake,
        ground_truth_functions=["right.py::producer"],
        ground_truth_patch="diff --git a/right.py b/right.py",
    )
    assert summary["eligible"] is True
    assert summary["updated_skill_types"] == ["strategy_skill"]
    assert summary["updated_skill_ids"]
    reflector_payload = json.loads(fake.calls[0]["messages"][1]["content"])
    assert set(reflector_payload["skill_search_context"]["skill_type_slots"]) == {
        "project_skill",
        "strategy_skill",
    }


def test_case_evolution_success_can_preserve_strategy_skill(tmpdir) -> None:
    case_dir = Path(str(tmpdir)) / "case"
    _write_case(case_dir, ["right.py::producer"])
    skill_path = Path(str(tmpdir)) / "skills.jsonl"
    skill_path.write_text(
        json.dumps(
            {
                "skill_id": "strategy_root_cause_v1",
                "status": "active",
                "version": 1,
                "skill_type": "strategy_skill",
                "scope": "global",
                "value": "root_cause_before_reporter",
                "retrieval_text": "root cause ranking reporter producer symptom cause",
                "skill": {
                    "title": "Root cause ranking",
                    "trigger": "Use when symptom reporters compete with state producers.",
                    "knowledge": "Rank evidence-supported state producers above symptom reporters.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reflector = {
        "case_summary": "Successful trajectory supports the strategy.",
        "outcome_type": "success",
        "skill_updates": {
            "project_skill": _no_update(),
            "strategy_skill": {
                "decision": "preserve_existing",
                "scope": "global",
                "target_value": "root_cause_before_reporter",
                "target_skill_id": "strategy_root_cause_v1",
                "rationale": "The retrieved strategy explains the successful ranking.",
                "edit": None,
                "no_update_reason": None,
            },
        },
        "no_update_reason": None,
    }
    fake = FakeLLMClient([{"content": json.dumps(reflector)}])
    summary = run_case_evolution(
        case_run_dir=case_dir,
        repo="demo/repo",
        issue="root cause ranking reporter producer symptom cause",
        config=_config(skill_path),
        llm_client=fake,
        ground_truth_functions=["right.py::producer"],
    )
    assert summary["eligible"] is True
    assert summary["applied_edits"][0]["action"] == "preserve"


def test_case_evolution_can_refresh_issue_abstraction(tmpdir) -> None:
    case_dir = Path(str(tmpdir)) / "case"
    _write_case(case_dir, ["wrong.py::symptom"])
    (case_dir / "issue_abstraction.json").write_text(
        json.dumps(
            {
                "abstract_problem_signature": "Old abstraction.",
                "project_skill_query": "old project query",
                "strategy_skill_query": "old strategy query",
                "key_symptoms": [],
            }
        ),
        encoding="utf-8",
    )
    skill_path = Path(str(tmpdir)) / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    config = _config(skill_path)
    config["issue_abstraction"] = {
        "enabled": True,
        "prompt_path": "prompt_records/explorer/issue_abstraction_v0.txt",
    }
    abstraction = {
        "abstract_problem_signature": "Equivalent inputs diverge before backend execution.",
        "project_type_key": "configuration_adapter_system",
        "project_type_description": "A system that normalizes external configuration before backend construction.",
        "project_skill_query": "Configuration systems separate normalization from backend construction.",
        "strategy_skill_query": "Compare equivalent paths at their first shared boundary.",
        "key_symptoms": ["equivalent inputs diverge"],
    }
    reflector = {
        "case_summary": "No reusable update.",
        "outcome_type": "failure",
        "skill_updates": {
            "project_skill": _no_update(),
            "strategy_skill": _no_update(),
        },
        "no_update_reason": "The case adds no reusable lesson.",
    }
    fake = FakeLLMClient(
        [
            {"content": json.dumps(abstraction)},
            {"content": json.dumps(reflector)},
        ]
    )
    output_dir = Path(str(tmpdir)) / "rerun"

    summary = run_case_evolution(
        case_run_dir=case_dir,
        repo="demo/repo",
        issue="Bug report",
        config=config,
        llm_client=fake,
        force=True,
        output_dir=output_dir,
        refresh_issue_abstraction=True,
    )

    assert summary["issue_abstraction"] == abstraction
    assert json.loads((output_dir / "issue_abstraction.json").read_text(encoding="utf-8")) == abstraction
    assert len(fake.calls) == 2
