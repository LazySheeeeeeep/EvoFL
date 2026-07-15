from __future__ import annotations

from typing import Any


SECTION_ORDER = ["project_type", "fault_mode", "strategy_type", "general"]
SECTION_TITLES = {
    "project_type": "Project-type knowledge",
    "fault_mode": "Fault-mode knowledge",
    "strategy_type": "Evidence-strategy knowledge",
    "general": "General localization principles",
}
DEFAULT_DIMENSION_LIMITS = {
    "project_type": 1,
    "fault_mode": 2,
    "strategy_type": 1,
    "general": 1,
}


def assemble_skill_context(
    matched_skills: list[dict[str, Any]],
    max_per_dimension: dict[str, int] | None = None,
    max_knowledge_chars: int = 900,
) -> dict[str, Any]:
    limits = {**DEFAULT_DIMENSION_LIMITS, **(max_per_dimension or {})}
    sections: dict[str, list[dict[str, Any]]] = {dimension: [] for dimension in SECTION_ORDER}
    source_skill_ids: list[str] = []

    for skill in sorted(matched_skills, key=lambda item: (-float(item.get("score") or 0.0), item.get("skill_id", ""))):
        dimension = skill.get("dimension", "general")
        if dimension not in sections:
            dimension = "general"
        if len(sections[dimension]) >= limits.get(dimension, 1):
            continue
        knowledge = str(skill.get("knowledge") or "")
        if len(knowledge) > max_knowledge_chars:
            knowledge = knowledge[:max_knowledge_chars].rstrip() + "..."
        skill_id = str(skill.get("skill_id") or "")
        if skill_id:
            source_skill_ids.append(skill_id)
        sections[dimension].append(
            {
                "title": skill.get("title", ""),
                "trigger": skill.get("trigger", ""),
                "knowledge": knowledge,
                "source_skill_id": skill_id,
                "dimension": dimension,
                "value": skill.get("value", "unknown"),
                "retrieval_text": skill.get("retrieval_text", ""),
            }
        )

    return {
        "template": "dimension_skill_context_v1",
        "source_skill_ids": list(dict.fromkeys(source_skill_ids)),
        "sections": sections,
    }


def render_skill_context(assembled_context: dict[str, Any]) -> str:
    lines = ["Retrieved localization knowledge:"]
    sections = assembled_context.get("sections", {})
    for dimension in SECTION_ORDER:
        lines.append("")
        lines.append(f"{SECTION_TITLES[dimension]}:")
        skills = sections.get(dimension, []) or []
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
