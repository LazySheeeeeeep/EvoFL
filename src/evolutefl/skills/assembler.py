from __future__ import annotations

from typing import Any


SECTION_ORDER = ["project_skill", "strategy_skill"]
SECTION_TITLES = {
    "project_skill": "Project knowledge",
    "strategy_skill": "Localization strategy knowledge",
}
DEFAULT_SKILL_TYPE_LIMITS = {
    "project_skill": 1,
    "strategy_skill": 1,
}


def assemble_skill_context(
    matched_skills: list[dict[str, Any]],
    max_per_skill_type: dict[str, int] | None = None,
    max_knowledge_chars: int = 900,
) -> dict[str, Any]:
    limits = {**DEFAULT_SKILL_TYPE_LIMITS, **(max_per_skill_type or {})}
    sections: dict[str, list[dict[str, Any]]] = {skill_type: [] for skill_type in SECTION_ORDER}
    source_skill_ids: list[str] = []

    for skill in sorted(matched_skills, key=lambda item: (-float(item.get("score") or 0.0), item.get("skill_id", ""))):
        skill_type = skill.get("skill_type")
        if skill_type not in sections or len(sections[skill_type]) >= limits.get(skill_type, 1):
            continue
        knowledge = str(skill.get("knowledge") or "")
        if len(knowledge) > max_knowledge_chars:
            knowledge = knowledge[:max_knowledge_chars].rstrip() + "..."
        skill_id = str(skill.get("skill_id") or "")
        if skill_id:
            source_skill_ids.append(skill_id)
        sections[skill_type].append(
            {
                "title": skill.get("title", ""),
                "trigger": skill.get("trigger", ""),
                "knowledge": knowledge,
                "source_skill_id": skill_id,
                "skill_type": skill_type,
                "scope": skill.get("scope", ""),
                "value": skill.get("value", "unknown"),
            }
        )

    return {
        "template": "project_strategy_skill_context_v1",
        "source_skill_ids": list(dict.fromkeys(source_skill_ids)),
        "sections": sections,
    }


def render_skill_context(assembled_context: dict[str, Any]) -> str:
    lines = ["Retrieved localization knowledge:"]
    sections = assembled_context.get("sections", {})
    for skill_type in SECTION_ORDER:
        lines.append("")
        lines.append(f"{SECTION_TITLES[skill_type]}:")
        skills = sections.get(skill_type, []) or []
        if not skills:
            lines.append("- (none)")
            continue
        for skill in skills:
            title = skill.get("title") or skill.get("source_skill_id") or "Untitled skill"
            lines.append(f"- [{title}] {skill.get('knowledge', '')}")
    lines.append("")
    lines.append("Important:")
    lines.append("- These skills are reusable localization knowledge, not a mandatory workflow.")
    lines.append("- They are not ground truth.")
    lines.append("- You may ignore any skill if repository evidence contradicts it.")
    lines.append("- Final ranked_functions must be based on code evidence from grep/read_file.")
    return "\n".join(lines)
