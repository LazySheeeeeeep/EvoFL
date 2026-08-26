from __future__ import annotations

from pathlib import Path

import pytest

from evolutefl.reflection.reflector import validate_reflector_output
from evolutefl.skills.bank import SkillBankV0


def _issue_card() -> dict:
    return {
        "value": "configuration_input_boundary_mismatch",
        "title": "Configuration input changes at a boundary",
        "trigger": "Use when configuration behavior differs from equivalent API input.",
        "knowledge": [{"text": "Inspect the boundary that translates external configuration before backend consumption."}],
    }


def test_issue_create_assigns_numeric_knowledge_id(tmp_path: Path) -> None:
    bank = SkillBankV0(tmp_path / "skills.jsonl")
    result = bank.apply_update(
        {"operation": "create", "skill_type": "issue_skill", "target_skill_id": None, "skill": _issue_card()}
    )
    skill = bank.get_active_skill(result["updated_skill_id"], skill_type="issue_skill")
    assert skill is not None
    assert skill["knowledge"] == [
        "Inspect the boundary that translates external configuration before backend consumption."
    ]
    stored = bank.active_skills()[0]
    assert stored.knowledge == [{"id": 1, "text": skill["knowledge"][0]}]


def test_issue_rewrite_edits_one_numeric_proposition(tmp_path: Path) -> None:
    bank = SkillBankV0(tmp_path / "skills.jsonl")
    created = bank.apply_update(
        {"operation": "create", "skill_type": "issue_skill", "target_skill_id": None, "skill": _issue_card()}
    )
    updated = bank.apply_update(
        {
            "operation": "rewrite",
            "skill_type": "issue_skill",
            "target_skill_id": created["updated_skill_id"],
            "knowledge_edit": {
                "operation": "add",
                "target_knowledge_id": None,
                "text": "Compare configuration and API values immediately before and after that boundary.",
            },
        }
    )
    active = bank.active_skills()[0]
    assert updated["version"] == 2
    assert active.knowledge == [
        {"id": 1, "text": "Inspect the boundary that translates external configuration before backend consumption."},
        {"id": 2, "text": "Compare configuration and API values immediately before and after that boundary."},
    ]


def test_reflector_issue_rewrite_requires_candidate_knowledge_id() -> None:
    payload = {
        "case_summary": "case",
        "skill_updates": {
            "issue_skill": {
                "decision": "rewrite_existing",
                "target_skill_id": "issue_config_v1",
                "rationale": "The case adds a distinct boundary observation.",
                "skill": None,
                "knowledge_edit": {
                    "operation": "replace",
                    "target_knowledge_id": 1,
                    "text": "Compare values on both sides of the adaptation boundary.",
                },
                "no_update_reason": None,
            },
            "strategy_skill": {
                "decision": "no_update",
                "target_skill_id": None,
                "rationale": "No ranking lesson.",
                "skill": None,
                "no_update_reason": "No ranking lesson.",
            },
        },
    }
    context = {
        "candidate_target_skills": [
            {
                "skill_id": "issue_config_v1",
                "skill_type": "issue_skill",
                "value": "configuration_input_boundary_mismatch",
                "title": "Configuration input changes at a boundary",
                "trigger": "Use when configuration behavior differs from equivalent API input.",
                "knowledge": [{"id": 1, "text": "Inspect the adaptation boundary."}],
            }
        ]
    }
    result = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}, "case": {"instance_id": "case"}},
        skill_search_context=context,
        skill_types=("issue_skill", "strategy_skill"),
        strict_final_cards=True,
    )
    edit = result["materialized_updates"][0]["knowledge_edit"]
    assert edit["target_knowledge_id"] == 1


def test_reflector_issue_create_requires_one_initial_proposition() -> None:
    issue_card = _issue_card()
    issue_card["knowledge"].append(
        {"text": "A second proposition belongs in a later one-item rewrite, not initial creation."}
    )
    payload = {
        "case_summary": "case",
        "skill_updates": {
            "issue_skill": {
                "decision": "create_new",
                "target_skill_id": None,
                "rationale": "A reusable issue pattern was found.",
                "skill": issue_card,
                "knowledge_edit": None,
                "no_update_reason": None,
            },
            "strategy_skill": {
                "decision": "no_update",
                "target_skill_id": None,
                "rationale": "No ranking lesson.",
                "skill": None,
                "no_update_reason": "No ranking lesson.",
            },
        },
    }

    with pytest.raises(ValueError, match="exactly one initial knowledge proposition"):
        validate_reflector_output(
            payload,
            trajectory_evidence={"outcome": {"label": "failure"}, "case": {"instance_id": "case"}},
            skill_search_context={},
            skill_types=("issue_skill", "strategy_skill"),
            strict_final_cards=True,
        )


def test_isolated_validation_keeps_valid_issue_when_strategy_is_too_long() -> None:
    payload = {
        "case_summary": "case",
        "skill_updates": {
            "issue_skill": {
                "decision": "create_new",
                "target_skill_id": None,
                "rationale": "A reusable issue pattern was found.",
                "skill": _issue_card(),
                "knowledge_edit": None,
                "no_update_reason": None,
            },
            "strategy_skill": {
                "decision": "create_new",
                "target_skill_id": None,
                "rationale": "A reusable ranking contrast was found.",
                "skill": {
                    "value": "origin_vs_later_boundary",
                    "title": "Origin versus later boundary",
                    "trigger": "Use when two responsibility boundaries can introduce the same bad state.",
                    "knowledge": [
                        "x" * 361,
                        "Compare the state immediately before and after the handoff.",
                        "Rank the first boundary whose observable state violates the contract.",
                    ],
                },
                "knowledge_edit": None,
                "no_update_reason": None,
            },
        },
    }

    result = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}, "case": {"instance_id": "case"}},
        skill_search_context={},
        skill_types=("issue_skill", "strategy_skill"),
        strict_final_cards=True,
        isolate_skill_errors=True,
    )

    assert [item["skill_type"] for item in result["materialized_updates"]] == ["issue_skill"]
    assert result["skill_updates"]["issue_skill"]["decision"] == "create_new"
    assert result["skill_updates"]["strategy_skill"]["decision"] == "no_update"
    assert result["protocol_errors"][0]["skill_type"] == "strategy_skill"
