"""Compatibility wrapper for project/strategy skill context assembly."""

from __future__ import annotations

from typing import Any

from .assembler import assemble_skill_context, render_skill_context


def compose_skills(
    matched_skills: list[dict[str, Any]],
    max_per_skill_type: dict[str, int] | None = None,
    max_total_items: int = 10,
) -> dict[str, Any]:
    del max_total_items
    return assemble_skill_context(matched_skills, max_per_skill_type=max_per_skill_type)


def render_guidance_text(composed_guidance: dict[str, Any]) -> str:
    return render_skill_context(composed_guidance)
