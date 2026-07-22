from __future__ import annotations

import json
from pathlib import Path

from evolutefl.skills.bank import SkillBankV0


def write_seed(path: Path) -> None:
    records = [
        {
            "skill_id": "project_config_adapter_v1",
            "status": "active",
            "version": 1,
            "skill_type": "project_skill",
            "scope": "architecture_family",
            "value": "configuration_adapter",
            "skill": {
                "title": "Configuration adapter boundary",
                "trigger": "Use when external configuration is accepted before normalization.",
                "knowledge": "Configuration adapters connect external options to normalized consumer state.",
            },
        },
        {
            "skill_id": "strategy_path_trace_v1",
            "status": "active",
            "version": 1,
            "skill_type": "strategy_skill",
            "scope": "contextual",
            "value": "producer_consumer_trace",
            "skill": {
                "title": "Producer-consumer tracing",
                "trigger": "Use when an accepted value is ignored downstream.",
                "knowledge": "Trace the value through producers and consumers and rank the first divergence.",
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_search_for_explorer_returns_two_skill_types(tmpdir) -> None:
    bank_path = Path(str(tmpdir)) / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path, project_skill_min_score=4.0)
    result = bank.search_for_explorer(
        "demo/repo",
        "configuration adapter option normalization consumer boundary ignored trace producer first divergence",
    )
    assert {skill["skill_type"] for skill in result["matched_skills"]} == {
        "project_skill",
        "strategy_skill",
    }
    assert result["skill_type_slots"]["project_skill"]["status"] == "matched"
    assert result["skill_type_slots"]["strategy_skill"]["status"] == "matched"


def test_search_uses_independent_abstraction_queries(tmpdir) -> None:
    bank_path = Path(str(tmpdir)) / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path, min_score=3.0, project_skill_min_score=3.0)
    result = bank.search_for_explorer(
        "demo/repo",
        "Concrete text without retrieval terms.",
        issue_abstraction={
            "abstract_problem_signature": "Configured value is ignored.",
            "project_skill_query": "configuration adapter option normalization consumer boundary",
            "strategy_skill_query": "trace option producer consumer first divergence ranking",
            "key_symptoms": ["configured option ignored"],
        },
    )
    assert len(result["matched_skills"]) == 2
    trace = result["skill_search_trace"]
    assert trace["retrieval_mode"] == "abstracted_skill_type_retrieval_v1"
    assert set(trace["skill_type_queries"]) == {"project_skill", "strategy_skill"}


def test_reflector_receives_strategy_top_k_below_explorer_threshold(tmpdir) -> None:
    bank_path = Path(str(tmpdir)) / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path)
    explorer_search = {
        "matched_skills": [],
        "skill_type_slots": {
            "project_skill": {"status": "missing", "matched_skills": [], "weak_candidates": []},
            "strategy_skill": {
                "status": "weak",
                "matched_skills": [],
                "weak_candidates": [
                    {
                        "skill_id": "strategy_path_trace_v1",
                        "skill_type": "strategy_skill",
                        "scope": "contextual",
                        "value": "producer_consumer_trace",
                        "title": "Producer-consumer tracing",
                        "trigger": "Use when an accepted value is ignored downstream.",
                        "knowledge": "Trace the value through producers and consumers and rank the first divergence.",
                    }
                ],
            },
        },
        "skill_search_trace": {
            "skill_type_queries": {
                "project_skill": "project query",
                "strategy_skill": "strategy query",
            },
            "skill_type_traces": {
                "strategy_skill": {
                    "top_scores": [
                        {
                            "skill_id": "strategy_path_trace_v1",
                            "skill_type": "strategy_skill",
                            "score": 0.31,
                        }
                    ]
                }
            },
        },
    }

    context = bank.search_for_reflector(
        "demo/repo",
        "issue",
        explorer_search=explorer_search,
        strategy_candidate_limit=3,
    )

    assert context["matched_case_skills"] == []
    assert [item["skill_id"] for item in context["strategy_dedup_candidates"]] == [
        "strategy_path_trace_v1"
    ]
    assert context["strategy_dedup_candidates"][0]["retrieval_score"] == 0.31
    assert context["strategy_dedup_candidates"][0]["candidate_roles"] == [
        "strategy_skill_weak_candidate",
        "strategy_semantic_dedup_candidate",
    ]
    assert context["candidate_target_skills"][0]["skill_id"] == "strategy_path_trace_v1"


def test_reflector_prioritizes_exact_project_type_container(tmpdir) -> None:
    bank_path = Path(str(tmpdir)) / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path, min_score=100.0, project_skill_min_score=100.0)
    abstraction = {
        "abstract_problem_signature": "A configured value is lost.",
        "project_type_key": "configuration_adapter",
        "project_type_description": "A system that adapts external configuration into normalized consumer state.",
        "project_skill_query": "unrelated words that do not pass the Explorer threshold",
        "strategy_skill_query": "unrelated strategy words",
        "key_symptoms": [],
    }
    explorer_search = bank.search_for_explorer(
        "demo/repo",
        "issue",
        issue_abstraction=abstraction,
    )
    assert explorer_search["matched_skills"] == []

    context = bank.search_for_reflector(
        "demo/repo",
        "issue",
        issue_abstraction=abstraction,
        explorer_search=explorer_search,
    )

    assert context["project_type"] == {
        "key": "configuration_adapter",
        "description": "A system that adapts external configuration into normalized consumer state.",
        "identity_query": "A system that adapts external configuration into normalized consumer state.",
    }
    assert context["project_identity_trace"]["query"] == (
        "A system that adapts external configuration into normalized consumer state."
    )
    assert [item["skill_id"] for item in context["project_type_candidates"]][:1] == [
        "project_config_adapter_v1"
    ]
    assert "project_type_exact_candidate" in context["project_type_candidates"][0]["candidate_roles"]


def test_apply_update_create_rewrite_and_preserve(tmpdir) -> None:
    bank_path = Path(str(tmpdir)) / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path)
    created = bank.apply_update(
        {
            "operation": "create",
            "skill_type": "strategy_skill",
            "target_skill_id": None,
            "skill": {
                "scope": "global",
                "value": "parallel_path_contrast",
                "title": "Parallel path contrast",
                "trigger": "Use when equivalent paths behave differently.",
                "knowledge": "Compare equivalent paths at their shared boundary and rank the first divergence.",
            },
        }
    )
    assert created["action"] == "create_new"

    rewritten = bank.apply_update(
        {
            "operation": "rewrite",
            "skill_type": "project_skill",
            "target_skill_id": "project_config_adapter_v1",
            "skill": {
                "scope": "architecture_family",
                "value": "configuration_adapter",
                "title": "Configuration adapter boundary",
                "trigger": "Use when external configuration is accepted before normalization.",
                "knowledge": "Configuration adapters connect external options to normalized state consumed by downstream components.",
            },
        }
    )
    assert rewritten["action"] == "rewrite"
    rewritten_skill = {skill.skill_id: skill for skill in bank.active_skills()}["project_config_adapter_v1"]
    assert rewritten_skill.version == 2
    assert "downstream components" in rewritten_skill.knowledge
    assert {skill.skill_type for skill in bank.active_skills()} == {"project_skill", "strategy_skill"}

    preserved = bank.apply_update(
        {
            "operation": "preserve",
            "skill_type": "strategy_skill",
            "target_skill_id": "strategy_path_trace_v1",
            "skill": None,
        }
    )
    assert preserved["action"] == "preserve"


def test_load_skips_obsolete_fault_mode_records(tmpdir) -> None:
    bank_path = Path(str(tmpdir)) / "skills.jsonl"
    records = [
        {
            "skill_id": "obsolete_fault",
            "status": "active",
            "version": 1,
            "dimension": "fault_mode",
            "value": "old",
            "skill": {"title": "Old", "trigger": "Old", "knowledge": "Old"},
        },
        {
            "skill_id": "legacy_strategy",
            "status": "active",
            "version": 1,
            "dimension": "strategy_type",
            "value": "trace",
            "skill": {"title": "Trace", "trigger": "Use for tracing.", "knowledge": "Trace evidence."},
        },
    ]
    bank_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    active = SkillBankV0(bank_path).active_skills()
    assert len(active) == 1
    assert active[0].skill_type == "strategy_skill"
