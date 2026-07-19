from __future__ import annotations

from evolutefl.skills.assembler import assemble_skill_context, render_skill_context


def test_assembler_groups_project_and_strategy_skills() -> None:
    skills = [
        {
            "skill_id": "p1",
            "skill_type": "project_skill",
            "scope": "architecture_family",
            "value": "configuration_adapter",
            "title": "Configuration adapter boundary",
            "trigger": "Use in layered configuration systems.",
            "knowledge": "Configuration adapters connect external options to normalized internal state.",
        },
        {
            "skill_id": "s1",
            "skill_type": "strategy_skill",
            "scope": "global",
            "value": "producer_consumer_trace",
            "title": "Producer-consumer tracing",
            "trigger": "Use when a value is accepted but ignored downstream.",
            "knowledge": "Trace the value from its producer to each consumer and rank the first divergence.",
        },
    ]

    assembled = assemble_skill_context(skills)
    assert assembled["template"] == "project_strategy_skill_context_v1"
    assert assembled["sections"]["project_skill"][0]["source_skill_id"] == "p1"
    assert assembled["sections"]["strategy_skill"][0]["source_skill_id"] == "s1"
    rendered = render_skill_context(assembled)
    assert "Project knowledge:" in rendered
    assert "Localization strategy knowledge:" in rendered
