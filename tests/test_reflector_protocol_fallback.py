from __future__ import annotations

import pytest

from evolutefl.llm.client import FakeLLMClient
from evolutefl.reflection.reflector import (
    _remaining_source_terms,
    _validate_semantic_center,
    finalize_skill_cards,
    generate_evolution_queries,
    run_reflector,
)
from evolutefl.skills.schema import validate_portable_skill_card


def test_reflector_protocol_failure_returns_no_update(tmpdir) -> None:
    output_path = tmpdir.join("reflector_debug.json")
    fake = FakeLLMClient([{"content": ""}, {"content": ""}])
    output = run_reflector(
        trajectory_evidence={"outcome": {"label": "failure"}, "case": {"instance_id": "case1"}},
        evolution_queries={
            "project_skill_query": "project",
            "strategy_diagnostic_context": "strategy context",
            "selected_strategy_skill_id": None,
            "strategy_selection_reason": "none applies",
        },
        skill_search_context={"candidate_target_skills": []},
        llm_client=fake,
        prompt="Return JSON.",
        attempts=2,
        output_path=str(output_path),
    )
    assert output["materialized_updates"] == []
    assert all(item["decision"] == "no_update" for item in output["skill_updates"].values())


def test_reflector_replays_truncated_deepseek_json_without_protocol_repair(tmpdir) -> None:
    output_path = tmpdir.join("reflector_debug.json")
    fake = FakeLLMClient([
        {
            "content": "",
            "raw": {"choices": [{"finish_reason": "length", "message": {
                "reasoning_content": "unfinished"
            }}], "usage": {"completion_tokens": 4096}},
        },
        {
            "content": (
                '{"case_summary":"No portable lesson.","outcome_type":"success",'
                '"skill_updates":{"fault_skill":{"decision":"no_update",'
                '"target_skill_id":null,"rationale":"Existing guidance is sufficient.",'
                '"skill":null,"no_update_reason":"Existing guidance is sufficient."}}}'
            ),
            "raw": {"choices": [{"finish_reason": "stop", "message": {}}]},
        },
    ])
    fake.base_url = "https://api.deepseek.com"
    fake.model = "deepseek-v4-flash"
    fake.max_tokens = 4096

    output = run_reflector(
        trajectory_evidence={"outcome": {"label": "success"}, "case": {"instance_id": "case1"}},
        evolution_queries={"fault_family": "validation_control", "fault_subtype_query": "branch outcome"},
        skill_search_context={"candidate_target_skills": []},
        llm_client=fake,
        prompt="Return JSON.",
        attempts=1,
        output_path=str(output_path),
        skill_types=("fault_skill",),
        truncation_retries=1,
        truncation_retry_max_tokens=8192,
    )

    assert output["skill_updates"]["fault_skill"]["decision"] == "no_update"
    assert len(fake.calls) == 2
    assert fake.calls[0]["messages"] == fake.calls[1]["messages"]
    assert fake.calls[1]["max_tokens"] == 8192
    assert fake.calls[1]["extra_body"]["thinking"] == {"type": "disabled"}


def test_evolution_query_retries_when_query_is_too_long() -> None:
    fake = FakeLLMClient(
        [
            {
                "content": (
                    '{"project_skill_query":"' + ("x" * 801)
                    + '","strategy_diagnostic_context":"short",'
                    '"selected_strategy_skill_id":null,'
                    '"strategy_selection_reason":"none"}'
                )
            },
            {
                "content": (
                    '{"project_skill_query":"compact project boundary",'
                    '"strategy_diagnostic_context":"compact diagnostic context",'
                    '"selected_strategy_skill_id":"strategy_trace_v1",'
                    '"strategy_selection_reason":"trigger applies"}'
                )
            },
        ]
    )
    result = generate_evolution_queries(
        trajectory_evidence={},
        strategy_catalog=[
            {
                "skill_id": "strategy_trace_v1",
                "title": "Trace",
                "trigger": "Use when producers compete with reporters.",
            }
        ],
        llm_client=fake,
        prompt="Return JSON.",
        attempts=2,
    )
    assert result["project_skill_query"] == "compact project boundary"
    assert result["selected_strategy_skill_id"] == "strategy_trace_v1"
    assert len(fake.calls) == 2


def test_evolution_query_degrades_unknown_strategy_catalog_id_to_no_match() -> None:
    fake = FakeLLMClient(
        [
            {
                "content": (
                    '{"project_skill_query":"component boundary",'
                    '"strategy_diagnostic_context":"producer and consumer disagree",'
                    '"selected_strategy_skill_id":"invented_card_v1",'
                    '"strategy_selection_reason":"The relationship seems relevant."}'
                )
            }
        ]
    )

    result = generate_evolution_queries(
        trajectory_evidence={},
        strategy_catalog=[],
        llm_client=fake,
        prompt="Return JSON.",
    )

    assert result["selected_strategy_skill_id"] is None
    assert result["unavailable_strategy_skill_id"] == "invented_card_v1"
    assert "No supplied Strategy Skill matched" in result["strategy_selection_reason"]


def test_card_finalizer_replaces_materialized_complete_card() -> None:
    output = {
        "skill_updates": {
            "project_skill": {"skill": {"value": "draft", "title": "Draft", "trigger": "Use when a boundary converts values.", "knowledge": ["A boundary receives values.", "Consumers read converted values."]}},
            "strategy_skill": {"skill": None},
        },
        "materialized_updates": [
            {
                "update_id": "update_001_project_skill",
                "operation": "create",
                "skill_type": "project_skill",
                "skill": {"value": "draft", "title": "Draft", "trigger": "Use when a boundary converts values.", "knowledge": ["A boundary receives values.", "Consumers read converted values."]},
            }
        ],
    }
    fake = FakeLLMClient(
        [
            {
                "content": (
                    '{"skill":{"value":"normalization_boundary",'
                    '"title":"Input normalization boundary",'
                    '"trigger":"Use when an adapter transforms external values into a canonical representation.",'
                    '"knowledge":["The adapter owns normalization at the input boundary.",'
                    '"The boundary preserves value semantics during the handoff.",'
                    '"Downstream consumers rely on the canonical representation."]}}'
                )
            }
        ]
    )
    finalized = finalize_skill_cards(
        reflector_output=output,
        llm_client=fake,
        prompt="Return JSON.",
    )
    assert finalized["materialized_updates"][0]["skill"]["value"] == "normalization_boundary"
    assert finalized["skill_updates"]["project_skill"]["skill"]["title"] == "Input normalization boundary"


def test_card_finalizer_excludes_remaining_source_symbol() -> None:
    output = {
        "skill_updates": {
            "project_skill": {"skill": {"value": "draft", "title": "NamedApi boundary", "trigger": "Use when NamedApi converts input.", "knowledge": ["NamedApi normalizes input.", "Consumers use the result."]}},
            "strategy_skill": {"skill": None},
        },
        "materialized_updates": [
            {"update_id": "update_001_project_skill", "operation": "create", "skill_type": "project_skill", "skill": {"value": "draft", "title": "NamedApi boundary", "trigger": "Use when NamedApi converts input.", "knowledge": ["NamedApi normalizes input.", "Consumers use the result."]}},
        ],
    }
    fake = FakeLLMClient(
        [
            {"content": '{"skill":{"value":"draft","title":"NamedApi boundary","trigger":"Use when NamedApi converts input.","knowledge":["NamedApi normalizes input.","The handoff preserves value semantics.","Consumers use the result."]}}'},
            {"content": '{"skill":{"value":"input_normalization_boundary","title":"Input normalization boundary","trigger":"Use when an adapter converts external input into a canonical representation.","knowledge":["The adapter owns input normalization.","The boundary provides a canonical representation.","Consumers rely on the normalized contract."]}}'},
        ]
    )
    finalized = finalize_skill_cards(reflector_output=output, llm_client=fake, prompt="Return JSON.", attempts=2)
    assert finalized["materialized_updates"][0]["skill"]["value"] == "input_normalization_boundary"
    assert len(fake.calls) == 2


def test_card_finalizer_drops_card_when_every_portability_attempt_is_rejected() -> None:
    output = {
        "skill_updates": {
            "project_skill": {
                "decision": "create_new",
                "target_skill_id": None,
                "skill": {
                    "value": "draft",
                    "title": "NamedApi boundary",
                    "trigger": "Use when NamedApi converts input.",
                    "knowledge": ["NamedApi normalizes input.", "The handoff preserves value semantics.", "Consumers use the result."],
                },
            },
            "strategy_skill": {"skill": None},
        },
        "materialized_updates": [
            {
                "update_id": "update_001_project_skill",
                "operation": "create",
                "skill_type": "project_skill",
                "skill": {
                    "value": "draft",
                    "title": "NamedApi boundary",
                    "trigger": "Use when NamedApi converts input.",
                    "knowledge": ["NamedApi normalizes input.", "The handoff preserves value semantics.", "Consumers use the result."],
                },
            }
        ],
    }
    fake = FakeLLMClient(
        [
            {"content": '{"skill":{"value":"draft","title":"NamedApi boundary","trigger":"Use when NamedApi converts input.","knowledge":["NamedApi normalizes input.","The handoff preserves value semantics.","Consumers use the result."]}}'},
            {"content": '{"skill":{"value":"draft","title":"NamedApi boundary","trigger":"Use when NamedApi converts input.","knowledge":["NamedApi normalizes input.","The handoff preserves value semantics.","Consumers use the result."]}}'},
        ]
    )

    finalized = finalize_skill_cards(reflector_output=output, llm_client=fake, prompt="Return JSON.", attempts=2)

    assert finalized["materialized_updates"] == []
    assert finalized["skill_updates"]["project_skill"]["decision"] == "no_update"


def test_card_finalizer_rejects_new_snake_case_identifier_not_in_draft() -> None:
    output = {
        "skill_updates": {
            "project_skill": {
                "skill": {
                    "value": "draft",
                    "title": "Boundary",
                    "trigger": "Use when a boundary converts input.",
                    "knowledge": ["A boundary transforms input.", "The handoff preserves the contract.", "Consumers use the result."],
                }
            },
            "strategy_skill": {"skill": None},
        },
        "materialized_updates": [
            {
                "update_id": "update_001_project_skill",
                "operation": "create",
                "skill_type": "project_skill",
                "skill": {
                    "value": "draft",
                    "title": "Boundary",
                    "trigger": "Use when a boundary converts input.",
                    "knowledge": ["A boundary transforms input.", "The handoff preserves the contract.", "Consumers use the result."],
                },
            }
        ],
    }
    fake = FakeLLMClient(
        [
            {
                "content": (
                    '{"skill":{"value":"normalization_boundary",'
                    '"title":"Normalization boundary",'
                    '"trigger":"Use when an adapter converts input.",'
                    '"knowledge":["The adapter sets derived_start_position.",'
                    '"The handoff preserves value semantics.",'
                    '"Consumers use the canonical representation."]}}'
                )
            },
            {
                "content": (
                    '{"skill":{"value":"normalization_boundary",'
                    '"title":"Normalization boundary",'
                    '"trigger":"Use when an adapter converts input.",'
                    '"knowledge":["The adapter establishes the derived position.",'
                    '"The handoff preserves value semantics.",'
                    '"Consumers use the canonical representation."]}}'
                )
            },
        ]
    )

    finalized = finalize_skill_cards(reflector_output=output, llm_client=fake, prompt="Return JSON.", attempts=2)

    assert finalized["materialized_updates"][0]["skill"]["knowledge"][0] == "The adapter establishes the derived position."
    assert len(fake.calls) == 2


def test_source_symbol_alias_with_punctuation_is_rejected() -> None:
    skill = {
        "value": "representation_adapter_boundary",
        "title": "Hook-manager representation boundary",
        "trigger": "Use when a wrapper presents a public representation.",
        "knowledge": [
            "The hook-manager adapts raw values for consumers.",
            "Consumers rely on the wrapper contract.",
        ],
    }

    assert _remaining_source_terms(skill, ["hook_manager.py"]) == ["hook_manager.py"]


def test_portable_card_requires_three_short_statements() -> None:
    card = {
        "value": "representation_adapter_contract",
        "title": "Representation adapter contract",
        "trigger": "Use when a wrapper translates a public representation before delegation.",
        "knowledge": [
            "The wrapper owns translation between public and delegated representations.",
            "The handoff preserves value semantics before delegation.",
            "Consumers rely on consistent public behavior across the boundary.",
        ],
    }
    assert validate_portable_skill_card(card, skill_type="project_skill") == card
    card["knowledge"] = card["knowledge"][:2]
    with pytest.raises(ValueError, match="exactly 3"):
        validate_portable_skill_card(card, skill_type="project_skill")


def test_strategy_semantic_center_keeps_one_causal_contrast() -> None:
    center = _validate_semantic_center(
        {
            "semantic_center": (
                "Decide whether invalid state first appears in the upstream representation producer "
                "or is introduced when a downstream interpreter applies that representation."
            ),
            "roles": ["upstream representation producer", "downstream interpreter"],
            "contract": (
                "Use evidence from both sides of the handoff to determine whether the representation "
                "was already invalid before interpretation."
            ),
        },
        skill_type="strategy_skill",
    )
    assert center == {
        "semantic_center": "Determine where the first contract violation occurs across a responsibility handoff.",
        "roles": ["upstream responsibility boundary", "later responsibility boundary"],
        "contract": (
            "Compare observable state before and after the handoff; the first contradictory "
            "state determines the higher-ranked boundary."
        ),
    }


def test_card_finalizer_writes_from_distilled_semantic_center() -> None:
    draft = {
        "value": "source_wrapper",
        "title": "Source wrapper",
        "trigger": "Use when local buffering delegates output.",
        "knowledge": ["The wrapper buffers output.", "The delegate emits output."],
    }
    output = {
        "skill_updates": {"project_skill": {"skill": draft}, "strategy_skill": {"skill": None}},
        "materialized_updates": [
            {
                "update_id": "update_001_project_skill",
                "operation": "create",
                "skill_type": "project_skill",
                "skill": draft,
            }
        ],
    }
    fake = FakeLLMClient(
        [
            {
                "content": (
                    '{"semantic_center":"A public representation adapter hands complete values to a delegated emitter.",'
                    '"roles":["public representation adapter","delegated emitter"],'
                    '"contract":"The handoff preserves complete value semantics."}'
                )
            },
            {
                "content": (
                    '{"skill":{"value":"public_representation_delegate_contract",'
                    '"title":"Public representation and delegated operation contract",'
                    '"trigger":"Use when a wrapper translates a public representation before delegated operations consume it.",'
                    '"knowledge":["The wrapper owns translation between public and delegated representations.",'
                    '"The handoff preserves complete value semantics before delegation.",'
                    '"Consumers rely on consistent public behavior across the boundary."]}}'
                )
            },
        ]
    )

    finalized = finalize_skill_cards(
        reflector_output=output,
        llm_client=fake,
        prompt="Write card.",
        semantic_center_prompt="Distill center.",
    )

    assert finalized["materialized_updates"][0]["skill"]["value"] == "public_representation_delegate_contract"
    writer_payload = fake.calls[1]["messages"][1]["content"]
    assert "semantic_center" in writer_payload
    assert "A public representation adapter hands complete values to a delegated emitter." in writer_payload
    assert "draft_skill" not in writer_payload
    assert fake.calls[0]["max_tokens"] == 2048


def test_card_finalizer_uses_draft_when_semantic_center_request_is_empty() -> None:
    """A provider-empty distillation request must not discard a valid update."""
    draft = {
        "value": "source_wrapper",
        "title": "Source wrapper",
        "trigger": "Use when local buffering delegates output.",
        "knowledge": ["The wrapper buffers output.", "The delegate emits output."],
    }
    output = {
        "skill_updates": {"project_skill": {"skill": draft}, "strategy_skill": {"skill": None}},
        "materialized_updates": [
            {
                "update_id": "update_001_project_skill",
                "operation": "create",
                "skill_type": "project_skill",
                "skill": draft,
            }
        ],
    }
    fake = FakeLLMClient(
        [
            {"content": ""},
            {
                "content": (
                    '{"skill":{"value":"public_representation_delegate_contract",'
                    '"title":"Public representation and delegated operation contract",'
                    '"trigger":"Use when a wrapper translates a public representation before delegated operations consume it.",'
                    '"knowledge":["The wrapper owns translation between public and delegated representations.",'
                    '"The handoff preserves complete value semantics before delegation.",'
                    '"Consumers rely on consistent public behavior across the boundary."]}}'
                )
            },
        ]
    )

    finalized = finalize_skill_cards(
        reflector_output=output,
        llm_client=fake,
        prompt="Write card.",
        semantic_center_prompt="Distill center.",
        semantic_center_attempts=1,
    )

    assert finalized["materialized_updates"][0]["skill"]["value"] == "public_representation_delegate_contract"
    writer_payload = fake.calls[1]["messages"][1]["content"]
    assert "draft_skill" in writer_payload
