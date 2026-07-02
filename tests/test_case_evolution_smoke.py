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
        json.dumps({"event": "assistant_tool_calls", "step": 1, "tool_calls": []}) + "\n"
        + json.dumps({"event": "finish", "step": 2}) + "\n",
        encoding="utf-8",
    )


def _config(skill_path: Path) -> dict:
    return {
        "llm": {},
        "reflection": {
            "insight_prompt_path": "prompt_records/reflection/reflection_insight_v1.txt",
            "reflector_prompt_path": "prompt_records/reflection/reflector_skill_v0.txt",
        },
        "skill_bank": {"path": str(skill_path), "max_matched_skills": 5, "max_per_dimension": {}},
    }


def _no_update() -> dict:
    return {
        "decision": "no_update",
        "target_value": "unknown",
        "target_skill_id": None,
        "rationale": "No reusable lesson for this dimension.",
        "edit": None,
    }


def test_case_evolution_failure_uses_trajectory_direct_reflector(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    case_dir = tmp_path / "case"
    _write_case(case_dir, ["wrong.py::symptom"])
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text("", encoding="utf-8")
    reflector = {
        "case_summary": "One ranking update.",
        "outcome_type": "failure",
        "optimization_intent": "correct",
        "dimension_assessment": {
            "general": {
                "learnable": "weak",
                "candidate_value": "unknown",
                "what_can_be_learned": "A possible residual ranking lesson can be considered later.",
            },
            "project_type": {"learnable": "none", "candidate_value": "unknown", "what_can_be_learned": "None."},
            "fault_mode": {"learnable": "none", "candidate_value": "unknown", "what_can_be_learned": "None."},
            "strategy_type": {
                "learnable": "strong",
                "candidate_value": "root_cause_before_reporter_ranking",
                "what_can_be_learned": "Rank state producers above symptom reporters.",
            },
        },
        "dimension_updates": {
            "general": {
                "decision": "no_update",
                "target_value": "unknown",
                "target_skill_id": None,
                "rationale": "General updates are handled by batch-level residual reflection.",
                "edit": None,
            },
            "project_type": _no_update(),
            "fault_mode": _no_update(),
            "strategy_type": {
                "decision": "create_new",
                "target_value": "root_cause_before_reporter_ranking",
                "target_skill_id": None,
                "rationale": "The transferable lesson is a strategy-level root-cause ranking principle.",
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "evidence": ["Top-5 missed producer."],
                    "content": {
                        "text": (
                            "Core principle: A localization candidate is stronger when code evidence shows it creates the incorrect state rather than merely reporting it.\n"
                            "Evidence requirement: Require a read_file or grep-backed causal link from the state producer to the observed symptom.\n"
                            "Ranking implication: Rank the producer above symptom reporters when the reporter only propagates or formats the wrong state."
                        ),
                        "title": "Root cause ranking",
                        "trigger": "Use when symptom reporter functions outrank likely state producers.",
                        "retrieval_text": "root cause ranking reporter producer symptom cause",
                    },
                    "expected_effect": "Improves root-cause ranking.",
                    "risk": "Could over-prioritize producers without evidence.",
                },
            },
        },
        "residual_card": {
            "instance_id": "case1",
            "outcome_type": "failure",
            "dimension_coverage": {
                "project_type": {"verdict": "irrelevant", "value": "unknown", "evidence": "No project-type lesson."},
                "fault_mode": {"verdict": "irrelevant", "value": "unknown", "evidence": "No fault-mode lesson."},
                "strategy_type": {
                    "verdict": "covered",
                    "value": "root_cause_before_reporter_ranking",
                    "evidence": "Strategy update captures the ranking lesson.",
                },
            },
            "best_explaining_dimension": "strategy_type",
            "residual_lesson": "",
            "residual_reason": "Strategy dimension explains the lesson.",
            "supporting_evidence": ["Top-5 missed producer."],
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
        force=False,
        ground_truth_functions=["right.py::producer"],
        ground_truth_patch="diff --git a/right.py b/right.py",
    )
    assert summary["eligible"] is True
    assert summary["outcome"]["label"] == "failure"
    assert summary["updated_skill_ids"]
    assert (case_dir / "case_evolution" / "trajectory_evidence.json").exists()
    assert (case_dir / "case_evolution" / "residual_card.json").exists()
    assert not (case_dir / "case_evolution" / "insight.json").exists()
    assert len(fake.calls) == 1


def test_case_evolution_success_can_preserve_existing_skill(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    case_dir = tmp_path / "case"
    _write_case(case_dir, ["right.py::producer"])
    skill_path = tmp_path / "skills.jsonl"
    skill_path.write_text(
        json.dumps(
            {
                "skill_id": "strategy_root_cause_ranking_v1",
                "status": "active",
                "version": 1,
                "dimension": "strategy_type",
                "value": "root_cause_ranking",
                "retrieval_text": "root cause ranking reporter producer symptom cause",
                "skill": {
                    "title": "Root cause ranking",
                    "trigger": "Use when symptom reporter functions outrank likely state producers.",
                    "knowledge": "Rank evidence-supported root causes above symptom reporters.",
                    "anti_patterns": [],
                },
                "provenance": {"supported_by_cases": [], "supported_by_successes": [], "supported_by_failures": [], "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reflector = {
        "case_summary": "Successful trajectory reinforces root-cause ranking.",
        "outcome_type": "success",
        "optimization_intent": "preserve",
        "dimension_assessment": {
            "general": {
                "learnable": "none",
                "candidate_value": "unknown",
                "what_can_be_learned": "No residual general lesson.",
            },
            "project_type": {"learnable": "none", "candidate_value": "unknown", "what_can_be_learned": "None."},
            "fault_mode": {"learnable": "none", "candidate_value": "unknown", "what_can_be_learned": "None."},
            "strategy_type": {
                "learnable": "strong",
                "candidate_value": "root_cause_ranking",
                "what_can_be_learned": "Existing strategy skill explains the success.",
            },
        },
        "dimension_updates": {
            "general": {
                "decision": "no_update",
                "target_value": "unknown",
                "target_skill_id": None,
                "rationale": "No residual general lesson.",
                "edit": None,
            },
            "project_type": _no_update(),
            "fault_mode": _no_update(),
            "strategy_type": {
                "decision": "preserve_existing",
                "target_value": "root_cause_ranking",
                "target_skill_id": "strategy_root_cause_ranking_v1",
                "rationale": "The retrieved strategy ranking skill explains the successful trajectory.",
                "edit": {
                    "operation": "preserve",
                    "field": "skill.knowledge",
                    "evidence": ["Top-5 hit ground truth."],
                    "content": {},
                    "expected_effect": "Record support without expanding text.",
                    "risk": "None.",
                },
            },
        },
        "residual_card": {
            "instance_id": "case1",
            "outcome_type": "success",
            "dimension_coverage": {
                "project_type": {"verdict": "irrelevant", "value": "unknown", "evidence": "No project-type lesson."},
                "fault_mode": {"verdict": "irrelevant", "value": "unknown", "evidence": "No fault-mode lesson."},
                "strategy_type": {
                    "verdict": "covered",
                    "value": "root_cause_ranking",
                    "evidence": "Strategy skill explains the trajectory.",
                },
            },
            "best_explaining_dimension": "strategy_type",
            "residual_lesson": "",
            "residual_reason": "Strategy dimension explains the lesson.",
            "supporting_evidence": ["Top-5 hit ground truth."],
        },
        "no_update_reason": None,
    }
    fake = FakeLLMClient([{"content": json.dumps(reflector)}])
    summary = run_case_evolution(
        case_run_dir=case_dir,
        repo="demo/repo",
        issue="producer symptom cause reporter",
        config=_config(skill_path),
        llm_client=fake,
        force=False,
        ground_truth_functions=["right.py::producer"],
    )
    assert summary["eligible"] is True
    assert summary["outcome"]["label"] == "success"
    assert summary["applied_edits"][0]["action"] == "preserve"
