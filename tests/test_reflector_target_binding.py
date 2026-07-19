from __future__ import annotations

import pytest

from evolutefl.reflection.reflector import validate_reflector_output


def _context() -> dict:
    project = {
        "skill_id": "project_adapter_v1",
        "skill_type": "project_skill",
        "scope": "architecture_family",
        "value": "adapter_boundary",
        "title": "Adapter boundary",
        "trigger": "Use in layered adapters.",
        "knowledge": "Adapters connect external input to normalized internal state.",
        "retrieval_text": "adapter external input normalized state",
    }
    strategy = {
        "skill_id": "strategy_trace_v1",
        "skill_type": "strategy_skill",
        "scope": "contextual",
        "value": "producer_consumer_trace",
        "title": "Producer-consumer trace",
        "trigger": "Use when accepted state is ignored downstream.",
        "knowledge": "Trace state from producer to consumer and rank the first divergence.",
        "retrieval_text": "producer consumer trace first divergence",
    }
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
        "scope": None,
        "target_value": "unknown",
        "target_skill_id": None,
        "rationale": reason,
        "edit": None,
        "no_update_reason": reason,
    }


def _payload(project_update: dict, strategy_update: dict) -> dict:
    return {
        "case_summary": "Two-type skill assessment.",
        "outcome_type": "failure",
        "skill_updates": {
            "project_skill": project_update,
            "strategy_skill": strategy_update,
        },
        "no_update_reason": None,
    }


def test_two_skill_types_can_materialize_independent_edits() -> None:
    payload = _payload(
        {
            "decision": "create_new",
            "scope": "repository",
            "target_value": "response_assembly_boundary",
            "target_skill_id": None,
            "rationale": "The repository has a distinct response assembly boundary.",
            "edit": {
                "operation": "add",
                "field": "skill.knowledge",
                "content": {
                    "text": "Response handlers assemble parsed request state before serializer output.",
                    "title": "Response assembly boundary",
                    "trigger": "Use for response-construction behavior in this repository.",
                    "retrieval_text": "response handler parsed state serializer boundary",
                },
            },
            "no_update_reason": None,
        },
        {
            "decision": "update_existing",
            "scope": "contextual",
            "target_value": "producer_consumer_trace",
            "target_skill_id": "strategy_trace_v1",
            "rationale": "The trajectory clarifies the ranking consequence.",
            "edit": {
                "operation": "replace",
                "field": "skill.knowledge",
                "content": {
                    "old_text": "Trace state from producer to consumer and rank the first divergence.",
                    "new_text": "Trace state from producer to consumer and rank the first evidence-backed divergence above downstream reporters.",
                },
            },
            "no_update_reason": None,
        },
    )
    validated = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}},
        skill_search_context=_context(),
    )
    edits = validated["materialized_edits"]
    assert [edit["target"]["skill_type"] for edit in edits] == ["project_skill", "strategy_skill"]


def test_existing_edit_must_target_retrieved_same_skill_type() -> None:
    payload = _payload(
        _no_update(),
        {
            "decision": "update_existing",
            "scope": "contextual",
            "target_value": "trace",
            "target_skill_id": "project_adapter_v1",
            "rationale": "Wrong type target.",
            "edit": {"operation": "add", "field": "skill.knowledge", "content": {"text": "More."}},
            "no_update_reason": None,
        },
    )
    with pytest.raises(ValueError, match="same type"):
        validate_reflector_output(payload, skill_search_context=_context())


def test_create_new_requires_valid_scope_and_complete_card() -> None:
    payload = _payload(
        {
            "decision": "create_new",
            "scope": "global",
            "target_value": "adapter_boundary",
            "target_skill_id": None,
            "rationale": "New project relation.",
            "edit": {
                "operation": "add",
                "field": "skill.knowledge",
                "content": {"text": "A project relation."},
            },
            "no_update_reason": None,
        },
        _no_update(),
    )
    with pytest.raises(ValueError, match="Invalid scope"):
        validate_reflector_output(payload, skill_search_context=_context())


def test_create_new_defaults_internal_scope_when_prompt_omits_it() -> None:
    payload = _payload(
        {
            "decision": "create_new",
            "target_value": "adapter_normalization_contract",
            "target_skill_id": None,
            "rationale": "The case reveals a reusable component boundary.",
            "edit": {
                "operation": "add",
                "field": "skill.knowledge",
                "content": {
                    "title": "Adapter normalization contract",
                    "trigger": "Configuration-driven systems with adapter boundaries.",
                    "retrieval_text": "configuration normalization adapter backend contract",
                    "text": "Configuration-driven systems separate input normalization from backend construction.",
                },
            },
            "no_update_reason": None,
        },
        _no_update(),
    )

    validated = validate_reflector_output(payload, skill_search_context=_context())

    assert validated["skill_updates"]["project_skill"]["scope"] == "architecture_family"


def test_preserve_existing_uses_null_prompt_edit_but_materializes_preserve() -> None:
    payload = _payload(
        _no_update(),
        {
            "decision": "preserve_existing",
            "scope": "contextual",
            "target_value": "producer_consumer_trace",
            "target_skill_id": "strategy_trace_v1",
            "rationale": "The retrieved strategy already covers the trajectory.",
            "edit": None,
            "no_update_reason": None,
        },
    )
    validated = validate_reflector_output(payload, skill_search_context=_context())
    assert validated["materialized_edits"][0]["operation"] == "preserve"


def test_strategy_dedup_candidate_is_a_valid_existing_target() -> None:
    context = _context()
    strategy = next(
        skill for skill in context["candidate_target_skills"] if skill["skill_type"] == "strategy_skill"
    )
    context["candidate_target_skills"] = [
        skill for skill in context["candidate_target_skills"] if skill["skill_type"] == "project_skill"
    ]
    context["skill_type_slots"]["strategy_skill"] = {
        "status": "weak",
        "matched_skills": [],
        "weak_candidates": [],
    }
    context["strategy_dedup_candidates"] = [
        {**strategy, "candidate_role": "strategy_semantic_dedup_candidate", "retrieval_score": 0.31}
    ]
    payload = _payload(
        _no_update(),
        {
            "decision": "preserve_existing",
            "scope": "contextual",
            "target_value": "producer_consumer_trace",
            "target_skill_id": "strategy_trace_v1",
            "rationale": "The lower-threshold candidate has the same evidence-action-ranking identity.",
            "edit": None,
            "no_update_reason": None,
        },
    )

    validated = validate_reflector_output(payload, skill_search_context=context)

    assert validated["materialized_edits"][0]["target"]["skill_id"] == "strategy_trace_v1"


def test_project_create_reuses_existing_project_type_container() -> None:
    context = _context()
    project = next(
        skill for skill in context["candidate_target_skills"] if skill["skill_type"] == "project_skill"
    )
    context["project_type"] = {
        "key": "adapter_boundary",
        "description": "Systems organized around an external-input adapter boundary.",
    }
    context["project_type_candidates"] = [project]
    payload = _payload(
        {
            "decision": "create_new",
            "scope": "architecture_family",
            "target_value": "adapter_boundary",
            "target_skill_id": None,
            "rationale": "Attempt to duplicate the same project type.",
            "edit": {
                "operation": "add",
                "field": "skill.knowledge",
                "content": {
                    "text": "Adapter systems normalize external input before internal consumption.",
                    "title": "Adapter systems",
                    "trigger": "Use for adapter-based systems.",
                    "retrieval_text": "adapter system normalization boundary",
                },
            },
            "no_update_reason": None,
        },
        _no_update(),
    )

    with pytest.raises(ValueError, match="container already exists"):
        validate_reflector_output(payload, skill_search_context=context)


def test_replace_requires_exact_old_text() -> None:
    payload = _payload(
        {
            "decision": "update_existing",
            "scope": "architecture_family",
            "target_value": "adapter_boundary",
            "target_skill_id": "project_adapter_v1",
            "rationale": "Refine relation.",
            "edit": {
                "operation": "replace",
                "field": "skill.knowledge",
                "content": {"old_text": "missing text", "new_text": "replacement"},
            },
            "no_update_reason": None,
        },
        _no_update(),
    )
    with pytest.raises(ValueError, match="old_text was not found"):
        validate_reflector_output(payload, skill_search_context=_context())


def test_old_dimension_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="skill_updates"):
        validate_reflector_output({"dimension_updates": {}})
