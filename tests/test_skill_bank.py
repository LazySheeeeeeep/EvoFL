from __future__ import annotations

import json
from pathlib import Path

from evolutefl.skills.bank import SkillBankV0


def write_seed(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "skill_id": "config_schema_v1",
                "status": "active",
                "version": 1,
                "dimension": "fault_mode",
                "value": "config_schema_mismatch",
                "retrieval_text": "config schema option normalization default merge",
                "skill": {
                    "title": "Config schema mismatch",
                    "trigger": "Use for config schema mismatches.",
                    "knowledge": "Config mismatches often originate at schema normalization, option merge, or default handling boundaries.",
                    "anti_patterns": ["Do not rank downstream consumers before checking config normalization."],
                },
                "provenance": {"supported_by_cases": ["seed"], "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"},
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def test_search_for_explorer_returns_dimension_skills(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path)
    result = bank.search_for_explorer("demo/repo", "config option default merge is ignored")
    skill = result["matched_skills"][0]
    assert skill["skill_id"] == "config_schema_v1"
    assert "knowledge" in skill
    assert "items" not in skill
    assert result["dimension_slots"]["fault_mode"]["status"] == "matched"
    assert result["dimension_slots"]["project_type"]["status"] == "missing"
    assert result["skill_search_trace"]["dimension_slot_summary"]["fault_mode"] == "matched"


def test_search_threshold_can_reject_weak_match(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path, min_score=4.0, project_type_min_score=6.0)
    result = bank.search_for_explorer("demo/repo", "schema")
    assert result["matched_skills"] == []
    assert result["skill_search_trace"]["filtered_below_threshold"]
    assert result["dimension_slots"]["fault_mode"]["status"] == "weak"
    assert result["dimension_slots"]["fault_mode"]["weak_candidates"][0]["skill_id"] == "config_schema_v1"


def test_search_for_explorer_can_use_issue_abstraction_dimension_query(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path, min_score=4.0)
    result = bank.search_for_explorer(
        "demo/repo",
        "The concrete issue text does not share useful words.",
        issue_abstraction={
            "abstract_problem_signature": "configured option ignored before downstream use",
            "project_type_query": "configuration pipeline repository",
            "fault_mode_query": "config schema option normalization default merge",
            "strategy_type_query": "trace option propagation to downstream consumer",
            "key_symptoms": ["configured option ignored"],
        },
    )

    assert result["matched_skills"][0]["skill_id"] == "config_schema_v1"
    assert result["skill_search_trace"]["retrieval_mode"] == "abstracted_dimension_skill_retrieval_v1"
    assert result["skill_search_trace"]["dimension_queries"]["fault_mode"] == "config schema option normalization default merge"


def test_apply_update_create_replace_delete_and_preserve(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    write_seed(bank_path)
    bank = SkillBankV0(bank_path)
    created = bank.apply_update(
        {
            "operation": "add",
            "target": {"skill_id": None, "dimension": "strategy_type", "value": "path_contrast", "field": "skill.knowledge"},
            "content": {
                "text": "Path contrast bugs require comparing equivalent working and failing flows at their shared boundary.",
                "title": "Path contrast",
                "trigger": "Use for working/failing path mismatch.",
                "retrieval_text": "path compare mismatch working failing boundary",
            },
            "source_cases": ["case1"],
            "outcome_type": "failure",
        }
    )
    assert created["action"] == "create_new"

    replaced = bank.apply_update(
        {
            "operation": "replace",
            "target": {"skill_id": "config_schema_v1", "dimension": "fault_mode", "value": "config_schema_mismatch", "field": "skill.knowledge"},
            "content": {
                "old_text": "Config mismatches often originate at schema normalization, option merge, or default handling boundaries.",
                "new_text": "Config mismatches often originate at schema normalization, option merge, type conversion, or default handling boundaries.",
            },
            "source_cases": ["case2"],
            "outcome_type": "failure",
        }
    )
    assert replaced["action"] == "replace"
    active = {skill.skill_id: skill for skill in bank.active_skills()}
    assert active["config_schema_v1"].version == 2
    assert "type conversion" in active["config_schema_v1"].knowledge

    deleted = bank.apply_update(
        {
            "operation": "delete",
            "target": {"skill_id": "config_schema_v1", "dimension": "fault_mode", "value": "config_schema_mismatch", "field": "skill.knowledge"},
            "content": {"text": "type conversion, "},
            "source_cases": ["case3"],
            "outcome_type": "failure",
        }
    )
    assert deleted["action"] == "delete"
    preserved = bank.apply_update(
        {
            "operation": "preserve",
            "target": {"skill_id": "config_schema_v1", "dimension": "fault_mode", "value": "config_schema_mismatch", "field": "skill.knowledge"},
            "source_cases": ["case4"],
            "outcome_type": "success",
        }
    )
    assert preserved["action"] == "preserve"


def test_active_skills_keep_only_latest_record_per_skill_id(tmpdir) -> None:
    tmp_path = Path(str(tmpdir))
    bank_path = tmp_path / "skills.jsonl"
    records = [
        {
            "skill_id": "duplicate_v1",
            "status": "active",
            "version": 1,
            "dimension": "fault_mode",
            "value": "old_value",
            "retrieval_text": "old",
            "skill": {
                "title": "Old",
                "trigger": "old trigger",
                "knowledge": "old knowledge",
                "anti_patterns": [],
            },
            "provenance": {"supported_by_cases": []},
        },
        {
            "skill_id": "duplicate_v1",
            "status": "superseded",
            "version": 1,
            "dimension": "fault_mode",
            "value": "old_value",
            "retrieval_text": "old",
            "skill": {
                "title": "Old",
                "trigger": "old trigger",
                "knowledge": "old knowledge",
                "anti_patterns": [],
            },
            "provenance": {"supported_by_cases": []},
        },
        {
            "skill_id": "duplicate_v1",
            "status": "active",
            "version": 2,
            "dimension": "strategy_type",
            "value": "new_value",
            "retrieval_text": "new",
            "skill": {
                "title": "New",
                "trigger": "new trigger",
                "knowledge": "new knowledge",
                "anti_patterns": [],
            },
            "provenance": {"supported_by_cases": []},
        },
    ]
    bank_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    active = SkillBankV0(bank_path).active_skills()

    assert len(active) == 1
    assert active[0].version == 2
    assert active[0].dimension == "strategy_type"
