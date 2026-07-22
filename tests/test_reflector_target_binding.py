from __future__ import annotations

import pytest

from evolutefl.reflection.reflector import validate_reflector_output


def _project_card() -> dict:
    return {
        "scope": "architecture_family",
        "value": "adapter_boundary",
        "title": "Adapter boundary",
        "trigger": "Use in layered adapters.",
        "knowledge": "Adapters connect external input to normalized internal state.",
    }


def _strategy_card() -> dict:
    return {
        "scope": "contextual",
        "value": "producer_consumer_trace",
        "title": "Producer-consumer trace",
        "trigger": "Use when accepted state is ignored downstream.",
        "knowledge": "Trace state from producer to consumer and rank the first divergence.",
    }


def _context() -> dict:
    project = {"skill_id": "project_adapter_v1", "skill_type": "project_skill", **_project_card()}
    strategy = {"skill_id": "strategy_trace_v1", "skill_type": "strategy_skill", **_strategy_card()}
    return {
        "candidate_target_skills": [project, strategy],
        "skill_type_slots": {
            "project_skill": {"status": "matched", "matched_skills": [project], "weak_candidates": []},
            "strategy_skill": {"status": "matched", "matched_skills": [strategy], "weak_candidates": []},
        },
    }


def _no_update(reason: str = "No reusable lesson for this skill type.") -> dict:
    return {
        "decision": "no_update",
        "target_skill_id": None,
        "rationale": reason,
        "skill": None,
        "no_update_reason": reason,
    }


def _payload(project_update: dict, strategy_update: dict) -> dict:
    return {
        "case_summary": "Two-type skill assessment.",
        "outcome_type": "failure",
        "skill_updates": {"project_skill": project_update, "strategy_skill": strategy_update},
        "no_update_reason": None,
    }


def test_two_skill_types_can_materialize_atomic_updates() -> None:
    payload = _payload(
        {
            "decision": "create_new",
            "target_skill_id": None,
            "rationale": "The repository has a distinct response assembly boundary.",
            "skill": {
                "scope": "repository",
                "value": "response_assembly_boundary",
                "title": "Response assembly boundary",
                "trigger": "Use for response-construction behavior in this repository.",
                "knowledge": "Response handlers assemble parsed state before serializer output.",
            },
            "no_update_reason": None,
        },
        {
            "decision": "rewrite_existing",
            "target_skill_id": "strategy_trace_v1",
            "rationale": "The trajectory clarifies the ranking consequence.",
            "skill": {
                **_strategy_card(),
                "knowledge": "Trace state from producer to consumer and rank the first evidence-backed divergence above downstream reporters.",
            },
            "no_update_reason": None,
        },
    )
    validated = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}},
        skill_search_context=_context(),
    )
    updates = validated["materialized_updates"]
    assert [update["skill_type"] for update in updates] == ["project_skill", "strategy_skill"]
    assert [update["operation"] for update in updates] == ["create", "rewrite"]


def test_existing_update_must_target_retrieved_same_skill_type() -> None:
    payload = _payload(
        _no_update(),
        {
            "decision": "rewrite_existing",
            "target_skill_id": "project_adapter_v1",
            "rationale": "Wrong type target.",
            "skill": _strategy_card(),
            "no_update_reason": None,
        },
    )
    with pytest.raises(ValueError, match="same type"):
        validate_reflector_output(payload, skill_search_context=_context())


def test_create_new_requires_valid_scope_and_complete_card() -> None:
    payload = _payload(
        {
            "decision": "create_new",
            "target_skill_id": None,
            "rationale": "New project relation.",
            "skill": {**_project_card(), "scope": "global"},
            "no_update_reason": None,
        },
        _no_update(),
    )
    with pytest.raises(ValueError, match="Invalid scope"):
        validate_reflector_output(payload, skill_search_context=_context())


def test_create_new_defaults_scope_when_omitted() -> None:
    card = _project_card()
    card.pop("scope")
    card["value"] = "adapter_normalization_contract"
    payload = _payload(
        {
            "decision": "create_new",
            "target_skill_id": None,
            "rationale": "The case reveals a reusable component boundary.",
            "skill": card,
            "no_update_reason": None,
        },
        _no_update(),
    )
    validated = validate_reflector_output(payload, skill_search_context=_context())
    assert validated["skill_updates"]["project_skill"]["skill"]["scope"] == "architecture_family"


def test_preserve_existing_materializes_preserve() -> None:
    payload = _payload(
        _no_update(),
        {
            "decision": "preserve_existing",
            "target_skill_id": "strategy_trace_v1",
            "rationale": "The retrieved strategy already covers the trajectory.",
            "skill": None,
            "no_update_reason": None,
        },
    )
    validated = validate_reflector_output(payload, skill_search_context=_context())
    assert validated["materialized_updates"][0]["operation"] == "preserve"


def test_strategy_dedup_candidate_is_valid_existing_target() -> None:
    context = _context()
    strategy = next(skill for skill in context["candidate_target_skills"] if skill["skill_type"] == "strategy_skill")
    context["candidate_target_skills"] = [
        skill for skill in context["candidate_target_skills"] if skill["skill_type"] == "project_skill"
    ]
    context["strategy_dedup_candidates"] = [
        {**strategy, "candidate_role": "strategy_semantic_dedup_candidate", "retrieval_score": 0.31}
    ]
    payload = _payload(
        _no_update(),
        {
            "decision": "preserve_existing",
            "target_skill_id": "strategy_trace_v1",
            "rationale": "The candidate has the same diagnostic identity.",
            "skill": None,
            "no_update_reason": None,
        },
    )
    validated = validate_reflector_output(payload, skill_search_context=context)
    assert validated["materialized_updates"][0]["target_skill_id"] == "strategy_trace_v1"


def test_project_create_reuses_existing_project_type_container() -> None:
    context = _context()
    context["project_type"] = {
        "key": "adapter_boundary",
        "description": "Systems organized around an external-input adapter boundary.",
    }
    payload = _payload(
        {
            "decision": "create_new",
            "target_skill_id": None,
            "rationale": "Attempt to duplicate the same project type.",
            "skill": _project_card(),
            "no_update_reason": None,
        },
        _no_update(),
    )
    with pytest.raises(ValueError, match="container already exists"):
        validate_reflector_output(payload, skill_search_context=context)


def test_rewrite_requires_complete_skill_card() -> None:
    payload = _payload(
        {
            "decision": "rewrite_existing",
            "target_skill_id": "project_adapter_v1",
            "rationale": "Refine relation.",
            "skill": {"knowledge": "Replacement only."},
            "no_update_reason": None,
        },
        _no_update(),
    )
    with pytest.raises(ValueError, match="missing required fields"):
        validate_reflector_output(payload, skill_search_context=_context())


def test_old_edit_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="skill_updates"):
        validate_reflector_output({"dimension_updates": {}})
