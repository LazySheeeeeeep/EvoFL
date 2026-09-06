from __future__ import annotations

import pytest

from evolutefl.skills.schema import (
    DimensionSkill,
    UnsupportedLegacySkillError,
    normalize_repo_id,
)


def test_new_project_skill_schema_round_trip() -> None:
    record = {
        "skill_id": "project_config_adapter_v1",
        "status": "active",
        "version": 1,
        "skill_type": "project_skill",
        "scope": "architecture_family",
        "value": "configuration_adapter",
        "retrieval_text": "configuration adapter external option normalized state",
        "skill": {
            "title": "Configuration adapter boundary",
            "trigger": "Use in layered configuration systems.",
            "knowledge": "Configuration adapters connect external options to normalized internal state.",
        },
    }
    skill = DimensionSkill.from_dict(record)
    assert skill.skill_type == "project_skill"
    saved = skill.to_dict()
    assert saved["skill_type"] == "project_skill"
    assert "scope" not in saved
    assert "dimension" not in saved
    assert "retrieval_text" not in saved
    assert saved["skill"]["knowledge"] == [
        "Configuration adapters connect external options to normalized internal state."
    ]
    assert "scope" not in skill.compact_dict()


def test_repo_project_skill_does_not_require_trigger() -> None:
    skill = DimensionSkill.from_dict({
        "skill_id": "project_skill_pallets_click",
        "status": "active",
        "version": 1,
        "skill_type": "project_skill",
        "repo_id": "Pallets/Click",
        "skill": {
            "title": "Click repository architecture",
            "knowledge": ["Commands are resolved before callbacks are invoked."],
        },
    })

    assert skill.repo_id == "pallets/click"
    assert skill.to_dict()["repo_id"] == "pallets/click"
    assert "trigger" not in skill.to_dict()["skill"]


def test_swesmith_snapshot_normalizes_to_upstream_repository() -> None:
    assert normalize_repo_id(
        "swesmith/pandas-dev__pandas.95280573"
    ) == "pandas-dev/pandas"


def test_legacy_project_and_strategy_types_are_mapped() -> None:
    legacy = {
        "skill_id": "legacy_strategy",
        "status": "active",
        "version": 1,
        "dimension": "strategy_type",
        "value": "path_contrast",
        "skill": {
            "title": "Path contrast",
            "trigger": "Use for divergent paths.",
            "guidance": "Compare equivalent paths at their shared boundary.",
        },
    }
    skill = DimensionSkill.from_dict(legacy)
    assert skill.skill_type == "strategy_skill"


def test_legacy_fault_mode_is_not_loaded_into_two_type_bank() -> None:
    with pytest.raises(UnsupportedLegacySkillError):
        DimensionSkill.from_dict(
            {
                "skill_id": "old_fault",
                "status": "active",
                "version": 1,
                "dimension": "fault_mode",
                "value": "missing_data",
                "skill": {"title": "Old", "trigger": "Old", "knowledge": "Old"},
            }
        )
