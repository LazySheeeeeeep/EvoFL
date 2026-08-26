from __future__ import annotations

import pytest

from evolutefl.reflection.reflector import validate_reflector_output


EVIDENCE = {"outcome": {"label": "failure"}, "case": {"instance_id": "case1"}}
CONTEXT = {
    "candidate_target_skills": [
        {"skill_id": "project_adapter_v1", "skill_type": "project_skill", "value": "adapter", "title": "Adapter", "trigger": "adapter", "knowledge": "boundary"},
        {"skill_id": "strategy_trace_v1", "skill_type": "strategy_skill", "value": "trace", "title": "Trace", "trigger": "trace", "knowledge": "compare"},
    ]
}


def _none() -> dict:
    return {"decision": "no_update", "target_skill_id": None, "rationale": "none", "skill": None, "no_update_reason": "none"}


def _payload(project: dict, strategy: dict) -> dict:
    return {"case_summary": "case", "skill_updates": {"project_skill": project, "strategy_skill": strategy}, "no_update_reason": None}


def test_rewrite_must_target_retrieved_same_type() -> None:
    strategy = {"decision": "rewrite_existing", "target_skill_id": "project_adapter_v1", "rationale": "wrong", "skill": {"value": "trace", "title": "Trace", "trigger": "Use", "knowledge": ["Compare candidate states.", "Rank the first divergence."]}, "no_update_reason": None}
    with pytest.raises(ValueError, match="same type"):
        validate_reflector_output(_payload(_none(), strategy), trajectory_evidence=EVIDENCE, skill_search_context=CONTEXT)


def test_create_and_preserve_structural_contracts() -> None:
    project = {"decision": "create_new", "target_skill_id": None, "rationale": "new", "skill": {"value": "layered_service_system", "title": "Layered service system", "trigger": "Use when a service separates request handling, state creation, and response rendering.", "knowledge": ["Services separate state creation from exposure.", "Responsibility boundaries identify where invalid state first appears."]}, "no_update_reason": None}
    strategy = {"decision": "preserve_existing", "target_skill_id": "strategy_trace_v1", "rationale": "covered", "skill": None, "no_update_reason": None}
    result = validate_reflector_output(_payload(project, strategy), trajectory_evidence=EVIDENCE, skill_search_context=CONTEXT)
    assert [item["operation"] for item in result["materialized_updates"]] == ["create", "preserve"]


def test_rewrite_can_target_skill_loaded_during_this_case() -> None:
    context = {"candidate_target_skills": [], "runtime_loaded_skills": [CONTEXT["candidate_target_skills"][1]]}
    strategy = {
        "decision": "rewrite_existing",
        "target_skill_id": "strategy_trace_v1",
        "rationale": "The runtime-loaded card shares the same decision center.",
        "skill": {
            "value": "producer_consumer_trace",
            "title": "Producer consumer trace",
            "trigger": "Use when values are ignored downstream.",
            "knowledge": ["Inspect both roles.", "Compare their contracts.", "Rank the first divergence."],
        },
        "no_update_reason": None,
    }
    result = validate_reflector_output(_payload(_none(), strategy), trajectory_evidence=EVIDENCE, skill_search_context=context)
    assert result["materialized_updates"][0]["target_skill_id"] == "strategy_trace_v1"


def test_old_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="skill_updates"):
        validate_reflector_output({"dimension_updates": {}}, trajectory_evidence=EVIDENCE, skill_search_context=CONTEXT)


def test_new_skill_requires_compact_atomic_knowledge() -> None:
    project = {
        "decision": "create_new",
        "target_skill_id": None,
        "rationale": "new",
        "skill": {
            "value": "service_boundary",
            "title": "Service boundary",
            "trigger": "Use for service boundaries.",
            "knowledge": ["Only one statement."],
        },
        "no_update_reason": None,
    }
    with pytest.raises(ValueError, match="2-4 atomic statements"):
        validate_reflector_output(
            _payload(project, _none()),
            trajectory_evidence=EVIDENCE,
            skill_search_context=CONTEXT,
        )


def test_failure_with_patch_and_ranked_candidates_can_skip_strategy_update() -> None:
    evidence = {
        **EVIDENCE,
        "prediction": {"ranked_functions": ["wrong.py::initializer"]},
        "ground_truth": {"patch_context": "diff --git a/right.py b/right.py"},
    }
    result = validate_reflector_output(
        _payload(_none(), _none()),
        trajectory_evidence=evidence,
        skill_search_context=CONTEXT,
    )
    assert result["skill_updates"]["strategy_skill"]["decision"] == "no_update"
    assert result["materialized_updates"] == []
