from __future__ import annotations

from pathlib import Path

import pytest

from evolutefl.reflection.reflector import validate_reflector_output
from evolutefl.skills.bank import SkillBankV0


def _fault_card() -> dict:
    return {
        "fault_family": "transformation_representation",
        "fault_subtype": "configuration_normalization_boundary",
        "title": "Configuration normalization boundary mismatch",
        "trigger": "Configuration input reaches consumers in a representation different from equivalent API input.",
        "knowledge": [
            "Inspect the boundary that normalizes external configuration before backend consumption."
        ],
    }


def test_fault_create_uses_explicit_family_and_subtype(tmpdir) -> None:
    bank = SkillBankV0(Path(str(tmpdir)) / "skills.jsonl")
    result = bank.apply_update({
        "operation": "create",
        "skill_type": "fault_skill",
        "target_skill_id": None,
        "skill": _fault_card(),
    })

    skill = bank.get_active_skill(result["updated_skill_id"], skill_type="fault_skill")
    assert skill is not None
    assert skill["fault_family"] == "transformation_representation"
    assert skill["fault_subtype"] == "configuration_normalization_boundary"
    assert skill["knowledge"] == _fault_card()["knowledge"]
    stored = bank.active_skills()[0].to_dict()
    assert "value" not in stored


def test_fault_rewrite_replaces_complete_card_and_preserves_identity(tmpdir) -> None:
    bank = SkillBankV0(Path(str(tmpdir)) / "skills.jsonl")
    created = bank.apply_update({
        "operation": "create", "skill_type": "fault_skill", "skill": _fault_card()
    })
    replacement = _fault_card()
    replacement["knowledge"] = [
        "Trace configuration through normalization and default merging before inspecting consumers.",
        "Compare the representation immediately before and after the normalization boundary.",
    ]
    updated = bank.apply_update({
        "operation": "rewrite",
        "skill_type": "fault_skill",
        "target_skill_id": created["updated_skill_id"],
        "skill": replacement,
    })

    active = bank.active_skills()[0]
    assert updated["version"] == 2
    assert active.knowledge == replacement["knowledge"]
    assert active.fault_family == "transformation_representation"
    assert active.fault_subtype == "configuration_normalization_boundary"


def test_fault_rewrite_rejects_identity_change(tmpdir) -> None:
    bank = SkillBankV0(Path(str(tmpdir)) / "skills.jsonl")
    created = bank.apply_update({
        "operation": "create", "skill_type": "fault_skill", "skill": _fault_card()
    })
    replacement = _fault_card()
    replacement["fault_subtype"] = "different_subtype"

    with pytest.raises(ValueError, match="semantic identity"):
        bank.apply_update({
            "operation": "rewrite",
            "skill_type": "fault_skill",
            "target_skill_id": created["updated_skill_id"],
            "skill": replacement,
        })


def test_reflector_fault_rewrite_materializes_complete_card() -> None:
    payload = {
        "case_summary": "case",
        "skill_updates": {
            "fault_skill": {
                "decision": "rewrite_existing",
                "target_skill_id": "fault_config_v1",
                "rationale": "The same subtype gains a clearer investigation direction.",
                "skill": _fault_card(),
                "no_update_reason": None,
            }
        },
    }
    context = {
        "candidate_target_skills": [{
            "skill_id": "fault_config_v1",
            "skill_type": "fault_skill",
            **_fault_card(),
        }]
    }

    result = validate_reflector_output(
        payload,
        trajectory_evidence={"outcome": {"label": "failure"}, "case": {"instance_id": "case"}},
        skill_search_context=context,
        skill_types=("fault_skill",),
        strict_final_cards=True,
    )

    update = result["materialized_updates"][0]
    assert update["operation"] == "rewrite"
    assert update["skill"]["fault_subtype"] == "configuration_normalization_boundary"
    assert "knowledge_edit" not in update


def test_invalid_fault_family_is_rejected() -> None:
    card = _fault_card()
    card["fault_family"] = "unknown"
    payload = {
        "skill_updates": {
            "fault_skill": {
                "decision": "create_new",
                "target_skill_id": None,
                "rationale": "test",
                "skill": card,
                "no_update_reason": None,
            }
        }
    }
    with pytest.raises(ValueError, match="Invalid fault_family"):
        validate_reflector_output(
            payload,
            trajectory_evidence={"outcome": {"label": "failure"}},
            skill_search_context={},
            skill_types=("fault_skill",),
            strict_final_cards=True,
        )
