from __future__ import annotations

from evolutefl.skills.assembler import assemble_skill_context, render_skill_context


def test_assembler_groups_by_dimension_and_renders_without_stage_names() -> None:
    skills = [
        {
            "skill_id": "s1",
            "dimension": "fault_mode",
            "value": "missing_data",
            "title": "Missing data",
            "trigger": "Use for missing data.",
            "knowledge": "Missing data often originates at producer or normalization boundaries.",
            "score": 8,
        },
        {
            "skill_id": "s2",
            "dimension": "general",
            "value": "root_cause",
            "title": "Root cause",
            "trigger": "Use generally.",
            "knowledge": "Rank root cause above symptom reporter.",
            "score": 4,
        },
    ]
    assembled = assemble_skill_context(skills)
    assert assembled["template"] == "dimension_skill_context_v1"
    assert assembled["sections"]["fault_mode"][0]["source_skill_id"] == "s1"
    assert "anti_patterns" not in assembled["sections"]["fault_mode"][0]
    assert "provenance" not in assembled["sections"]["fault_mode"][0]
    assert "score" not in assembled["sections"]["fault_mode"][0]
    rendered = render_skill_context(assembled)
    assert "Fault-mode knowledge:" in rendered
    assert "General localization principles:" in rendered
    assert "not a mandatory workflow" in rendered
    assert "Anti-patterns:" not in rendered
    assert "Orient:" not in rendered
    assert "Rank:" not in rendered
