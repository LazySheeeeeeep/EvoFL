from __future__ import annotations

import pytest

from evolutefl.skills.schema import DimensionSkill


def test_new_schema_parse_to_dict() -> None:
    record = {
        "skill_id": "fault_missing_data_v1",
        "status": "active",
        "version": 1,
        "dimension": "fault_mode",
        "value": "missing_data",
        "retrieval_text": "missing data none empty value",
        "skill": {
            "title": "Missing data localization",
            "trigger": "Use when data is dropped.",
            "knowledge": "Missing-data bugs often originate where values are filtered, defaulted, or converted.",
            "anti_patterns": ["Do not rank only the final reporter."],
        },
        "provenance": {"supported_by_cases": ["case1"], "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"},
    }
    skill = DimensionSkill.from_dict(record)
    assert skill.dimension == "fault_mode"
    assert "filtered" in skill.knowledge
    saved = skill.to_dict()
    assert "items" not in saved["skill"]
    assert saved["skill"]["anti_patterns"] == ["Do not rank only the final reporter."]


def test_legacy_migration_merges_guidance_items_into_knowledge() -> None:
    legacy = {
        "skill_id": "legacy_skill",
        "status": "active",
        "version": 1,
        "taxonomy": {"project_type": "cli_tool", "fault_mode": "unknown", "strategy_type": "unknown"},
        "skill": {
            "title": "Legacy",
            "trigger": "Legacy trigger",
            "guidance": "Inspect legacy guidance.",
            "items": [{"stage": "rank", "text": "Rank producer above reporter.", "key": "legacy"}],
            "anti_patterns": ["Avoid keyword-only ranking."],
        },
    }
    skill = DimensionSkill.from_dict(legacy)
    assert skill.dimension == "project_type"
    assert "Inspect legacy guidance." in skill.knowledge
    assert "Rank producer above reporter." in skill.knowledge
    assert skill.anti_patterns == ["Avoid keyword-only ranking."]


def test_invalid_dimension_rejected() -> None:
    with pytest.raises(ValueError):
        DimensionSkill.from_dict(
            {
                "skill_id": "bad",
                "status": "active",
                "version": 1,
                "dimension": "stage",
                "value": "bad",
                "skill": {"title": "Bad", "trigger": "Bad", "knowledge": "Bad"},
            }
        )
