from __future__ import annotations

import copy
import json

import pytest

from evolutefl.reflection.reflector import validate_reflector_output
from evolutefl.skills.bank import SkillBankV0
from evolutefl.skills.schema import DimensionSkill


PRIMARY = "validation_control"
ALIAS = "transformation_representation"


def card():
    return {"fault_family": PRIMARY, "fault_subtype": "state_loop_termination",
            "title": "Premature stream termination", "trigger": "Output disappears during token processing",
            "knowledge": ["Compare the state handler return value with the loop's output-drain condition."]}


def create(bank, value):
    return bank.apply_update({"operation": "create", "skill_type": "fault_skill", "skill": value})["updated_skill_id"]


def rewrite(bank, target, value):
    return bank.apply_update({"operation": "rewrite", "skill_type": "fault_skill", "target_skill_id": target, "skill": value})


def test_legacy_bank_load_adds_only_primary_in_memory_without_writing(tmp_path):
    path = tmp_path / "skills.jsonl"
    value = card()
    row = {"skill_id": "legacy", "skill_type": "fault_skill", "status": "active", "version": 1,
           "fault_family": value.pop("fault_family"), "fault_subtype": value.pop("fault_subtype"), "skill": value}
    path.write_text(json.dumps(row) + "\n")
    before = path.read_bytes()
    bank = SkillBankV0(path)
    assert bank.active_skills()[0].retrieval_families == [PRIMARY]
    assert bank.active_skills()[0].to_dict()["retrieval_families"] == [PRIMARY]
    assert not bank.fault_catalog(ALIAS)["candidate_skills"]
    assert path.read_bytes() == before


def test_shared_entry_rewrite_is_one_identity_and_removal_clears_alias(tmp_path):
    bank = SkillBankV0(tmp_path / "skills.jsonl")
    value = {**card(), "retrieval_families": [ALIAS, PRIMARY, ALIAS]}
    target = create(bank, value)
    for family in [PRIMARY, ALIAS]:
        entry = bank.fault_catalog(family)["candidate_skills"][0]
        assert entry["skill_id"] == target and "knowledge" not in entry
        assert entry["retrieval_families"] == [PRIMARY, ALIAS]
    union = bank.fault_evolution_catalog([PRIMARY, ALIAS, PRIMARY])
    assert len(union["candidate_skills"]) == 1
    assert union["candidate_skills"][0]["matched_retrieval_families"] == [PRIMARY, ALIAS]
    assert union["skill_search_trace"]["catalog_counts_by_family"] == {PRIMARY: 1, ALIAS: 1}
    replacement = {**value, "knowledge": ["Trace whether pending output is drained before the stop sentinel ends iteration."]}
    result = rewrite(bank, target, replacement)
    assert result["version"] == 2 and result["updated_skill_id"] == target
    assert len(bank.active_skills()) == 1
    assert bank.get_active_skill(target)["knowledge"] == replacement["knowledge"]
    stored = [json.loads(line) for line in bank.path.read_text().splitlines()]
    assert [(r["version"], r["status"]) for r in stored] == [(1, "active"), (1, "superseded"), (2, "active")]
    replacement["retrieval_families"] = [PRIMARY]
    rewrite(bank, target, replacement)
    assert not bank.fault_catalog(ALIAS)["candidate_skills"]
    assert bank.active_skills()[0].version == 3


def test_legacy_rewrite_preserves_aliases_and_preserve_does_not_write(tmp_path):
    bank = SkillBankV0(tmp_path / "skills.jsonl")
    target = create(bank, {**card(), "retrieval_families": [PRIMARY, ALIAS]})
    rewrite(bank, target, card())
    assert bank.get_active_skill(target)["retrieval_families"] == [PRIMARY, ALIAS]
    before = bank.path.read_bytes()
    bank.apply_update({"operation": "preserve", "skill_type": "fault_skill", "target_skill_id": target})
    assert bank.path.read_bytes() == before


@pytest.mark.parametrize("entries", [[], "validation_control", ["not_a_family"], [False]])
def test_invalid_entries_are_rejected_before_write(tmp_path, entries):
    bank = SkillBankV0(tmp_path / "skills.jsonl")
    with pytest.raises(ValueError):
        create(bank, {**card(), "retrieval_families": entries})
    assert not bank.active_skills()


def test_aliases_do_not_change_primary_semantic_identity(tmp_path):
    bank = SkillBankV0(tmp_path / "skills.jsonl")
    target = create(bank, {**card(), "retrieval_families": [PRIMARY, ALIAS]})
    before = bank.path.read_bytes()
    with pytest.raises(ValueError, match="semantic identity"):
        rewrite(bank, target, {**card(), "fault_family": ALIAS})
    assert bank.path.read_bytes() == before


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("entries", [None, [PRIMARY], [PRIMARY, ALIAS]])
def test_reflector_protocol_retains_or_explicitly_replaces_entries(tmp_path, strict, entries):
    bank = SkillBankV0(tmp_path / "skills.jsonl")
    target = create(bank, {**card(), "retrieval_families": [PRIMARY, ALIAS]})
    value = card()
    if entries is not None:
        value["retrieval_families"] = entries
    payload = {"skill_updates": {"fault_skill": {"decision": "rewrite_existing", "target_skill_id": target,
        "rationale": "One applicable investigation pattern", "skill": value}}}
    before = copy.deepcopy(payload)
    output = validate_reflector_output(payload, trajectory_evidence={"outcome": {"label": "failure"}},
        skill_search_context={"candidate_target_skills": [bank.get_active_skill(target)]},
        skill_types=("fault_skill",), strict_final_cards=strict)
    updated = output["materialized_updates"][0]
    assert updated["skill"]["retrieval_families"] == (entries if entries is not None else [PRIMARY, ALIAS])
    assert payload == before
    bank.apply_update(updated)
    assert bank.active_skills()[0].version == 2


def test_project_and_strategy_serialization_do_not_gain_fault_entries():
    for kind in ("project_skill", "strategy_skill"):
        skill = DimensionSkill("demo", "active", 1, kind, "demo", "Demo", "Some trigger", ["Some knowledge"])
        assert "retrieval_families" not in skill.to_dict()
        assert "retrieval_families" not in skill.compact_dict()
