from __future__ import annotations

import pytest

from evolutefl.reflection.reflector import validate_reflector_output


def _strategy_knowledge_card() -> str:
    return (
        "Prefer evidence that connects caller assumptions to callee return behavior at a shared boundary. "
        "Rank the function that first violates the caller-callee contract above callers that only expose the mismatch."
    )


def _project_knowledge_card() -> str:
    return (
        "Config pipeline repositories often route external values through schema normalization before downstream consumers see them. "
        "Architecture evidence should distinguish normalization boundaries from option consumers."
    )


def _context() -> dict:
    return {
        "dimension_slots": {
            "fault_mode": {
                "status": "matched",
                "matched_skills": [
                    {
                        "skill_id": "fault_config_schema_v1",
                        "dimension": "fault_mode",
                        "value": "config_schema_mismatch",
                        "title": "Config schema mismatch",
                        "knowledge": "Check schema normalization boundaries.",
                    }
                ],
                "weak_candidates": [],
            },
            "strategy_type": {
                "status": "weak",
                "matched_skills": [],
                "weak_candidates": [
                    {
                        "skill_id": "strategy_caller_callee_v1",
                        "dimension": "strategy_type",
                        "value": "caller_callee_contrast",
                        "title": "Caller callee contrast",
                        "knowledge": "Compare caller assumptions with callee behavior.",
                    }
                ],
            },
        },
        "candidate_target_skills": [
            {
                "skill_id": "fault_config_schema_v1",
                "dimension": "fault_mode",
                "value": "config_schema_mismatch",
                "title": "Config schema mismatch",
                "knowledge": "Check schema normalization boundaries.",
            }
        ],
    }


def _no_update(reason: str = "No bounded update for this dimension.") -> dict:
    return {
        "decision": "no_update",
        "target_value": "unknown",
        "target_skill_id": None,
        "rationale": reason,
        "edit": None,
    }


def _base_payload(dimension_updates: dict) -> dict:
    return {
        "case_summary": "Case teaches dimension-specific localization knowledge.",
        "outcome_type": "failure",
        "optimization_intent": "correct",
        "dimension_assessment": {
            "general": {"learnable": "none", "candidate_value": "unknown", "what_can_be_learned": "None."},
            "project_type": {
                "learnable": "strong",
                "candidate_value": "config_pipeline",
                "what_can_be_learned": "Config values pass through normalization before downstream consumers.",
            },
            "fault_mode": {
                "learnable": "strong",
                "candidate_value": "config_schema_mismatch",
                "what_can_be_learned": "Config schema mismatches arise at normalization boundaries.",
            },
            "strategy_type": {
                "learnable": "weak",
                "candidate_value": "caller_callee_contrast",
                "what_can_be_learned": "Caller/callee contrast can help ranking.",
            },
        },
        "dimension_updates": dimension_updates,
        "no_update_reason": None,
    }


def test_dimension_updates_can_apply_multiple_dimension_edits() -> None:
    payload = _base_payload(
        {
            "general": _no_update(),
            "project_type": {
                "decision": "create_new",
                "target_value": "config_pipeline",
                "target_skill_id": None,
                "rationale": "The architecture lesson belongs in project_type.",
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {
                        "text": _project_knowledge_card(),
                        "title": "Config pipeline normalization",
                        "trigger": "Use when issues mention config values ignored before consumer logic.",
                        "retrieval_text": "config pipeline normalization downstream consumer schema",
                    },
                    "evidence": ["Config value is transformed before consumer code."],
                },
            },
            "fault_mode": {
                "decision": "update_existing",
                "target_value": "config_schema_mismatch",
                "target_skill_id": "fault_config_schema_v1",
                "rationale": "The mechanism refines the matched fault-mode skill.",
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {"text": "Also inspect default merges that overwrite explicit config values."},
                    "evidence": ["Patch changed default merge behavior."],
                },
            },
            "strategy_type": _no_update("No strategy update."),
        }
    )

    validated = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}},
        skill_search_context=_context(),
    )

    edits = validated["materialized_edits"]
    assert [edit["target"]["dimension"] for edit in edits] == ["project_type", "fault_mode"]
    assert edits[0]["target"]["skill_id"] is None
    assert edits[1]["target"]["skill_id"] == "fault_config_schema_v1"
    assert "dimension_decision" not in validated
    assert "proposed_edits" not in validated


def test_missing_dimension_updates_object_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension_updates"):
        validate_reflector_output(
            {"case_summary": "old protocol", "dimension_decision": {}, "proposed_edits": []},
            trajectory_evidence={"outcome": {"label": "failure"}},
            skill_search_context=_context(),
        )


def test_existing_skill_edit_must_target_retrieved_same_dimension_skill() -> None:
    payload = _base_payload(
        {
            "general": _no_update(),
            "project_type": _no_update(),
            "fault_mode": {
                "decision": "update_existing",
                "target_value": "config_schema_mismatch",
                "target_skill_id": "unretrieved_skill_v1",
                "rationale": "This target is not available.",
                "edit": {
                    "operation": "replace",
                    "field": "skill.knowledge",
                    "content": {"old_text": "old", "new_text": "new"},
                },
            },
            "strategy_type": _no_update(),
        }
    )

    with pytest.raises(ValueError, match="not among this case's retrieved"):
        validate_reflector_output(payload, trajectory_evidence={"outcome": {"label": "failure"}}, skill_search_context=_context())


def test_new_skill_add_can_use_null_target_skill_id() -> None:
    payload = _base_payload(
        {
            "general": _no_update(),
            "project_type": _no_update(),
            "fault_mode": _no_update(),
            "strategy_type": {
                "decision": "create_new",
                "target_value": "caller_callee_contrast",
                "target_skill_id": None,
                "rationale": "The weak strategy candidate is too broad; this case needs a narrower contrast skill.",
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {
                        "text": _strategy_knowledge_card(),
                        "title": "Caller callee contrast",
                        "trigger": "Use when caller expectations and callee behavior diverge.",
                        "retrieval_text": "caller callee contrast expectations return behavior",
                    },
                },
            },
        }
    )

    validated = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}},
        skill_search_context=_context(),
    )

    edit = validated["materialized_edits"][0]
    assert edit["target"]["skill_id"] is None
    assert edit["target"]["dimension"] == "strategy_type"
    assert edit["target"]["slot_status"] == "weak"


def test_create_new_in_existing_dimension_requires_rationale() -> None:
    payload = _base_payload(
        {
            "general": _no_update(),
            "project_type": _no_update(),
            "fault_mode": {
                "decision": "create_new",
                "target_value": "config_schema_mismatch",
                "target_skill_id": None,
                "rationale": "",
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {
                        "text": "Inspect config normalization before downstream option consumers.",
                        "title": "Config normalization boundary",
                        "trigger": "Use when config file values are ignored or transformed.",
                        "retrieval_text": "config normalization option consumer ignored transformed",
                    },
                },
            },
            "strategy_type": _no_update(),
        }
    )

    with pytest.raises(ValueError, match="requires rationale"):
        validate_reflector_output(payload, trajectory_evidence={"outcome": {"label": "failure"}}, skill_search_context=_context())


def test_weak_dimension_candidate_can_be_existing_update_target() -> None:
    payload = _base_payload(
        {
            "general": _no_update(),
            "project_type": _no_update(),
            "fault_mode": _no_update(),
            "strategy_type": {
                "decision": "update_existing",
                "target_value": "caller_callee_contrast",
                "target_skill_id": "strategy_caller_callee_v1",
                "rationale": "The reusable lesson is a comparison strategy.",
                "edit": {
                    "operation": "add",
                    "field": "skill.knowledge",
                    "content": {"text": "Prefer shared boundary evidence over isolated caller names."},
                },
            },
        }
    )

    validated = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}},
        skill_search_context=_context(),
    )
    target = validated["materialized_edits"][0]["target"]

    assert target["skill_id"] == "strategy_caller_callee_v1"
    assert target["dimension"] == "strategy_type"
    assert target["value"] == "caller_callee_contrast"


def test_no_update_requires_rationale_or_reason() -> None:
    payload = _base_payload(
        {
            "general": {"decision": "no_update", "target_value": "unknown", "target_skill_id": None, "edit": None},
            "project_type": _no_update(),
            "fault_mode": _no_update(),
            "strategy_type": _no_update(),
        }
    )

    with pytest.raises(ValueError, match="requires rationale"):
        validate_reflector_output(payload, trajectory_evidence={"outcome": {"label": "failure"}}, skill_search_context=_context())


def test_replace_edit_requires_exact_old_text_in_target_field() -> None:
    payload = _base_payload(
        {
            "general": _no_update(),
            "project_type": _no_update(),
            "fault_mode": {
                "decision": "update_existing",
                "target_value": "config_schema_mismatch",
                "target_skill_id": "fault_config_schema_v1",
                "rationale": "The reusable lesson is a config schema fault mode.",
                "edit": {
                    "operation": "replace",
                    "field": "skill.knowledge",
                    "content": {
                        "old_text": "This text is not in the current skill.",
                        "new_text": "A replacement should not be accepted without an exact source span.",
                    },
                },
            },
            "strategy_type": _no_update(),
        }
    )

    with pytest.raises(ValueError, match="old_text was not found"):
        validate_reflector_output(payload, trajectory_evidence={"outcome": {"label": "failure"}}, skill_search_context=_context())


def test_replace_edit_accepts_exact_old_text_in_target_field() -> None:
    payload = _base_payload(
        {
            "general": _no_update(),
            "project_type": _no_update(),
            "fault_mode": {
                "decision": "update_existing",
                "target_value": "config_schema_mismatch",
                "target_skill_id": "fault_config_schema_v1",
                "rationale": "The reusable lesson is a config schema fault mode.",
                "edit": {
                    "operation": "replace",
                    "field": "skill.knowledge",
                    "content": {
                        "old_text": "Check schema normalization boundaries.",
                        "new_text": "Check schema normalization and default-merge boundaries before downstream consumers.",
                    },
                },
            },
            "strategy_type": _no_update(),
        }
    )

    validated = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}},
        skill_search_context=_context(),
    )

    assert validated["materialized_edits"][0]["content"]["old_text"] == "Check schema normalization boundaries."
