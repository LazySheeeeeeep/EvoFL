from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


SKILL_TYPES = ("project_skill", "strategy_skill")
SKILL_STATUSES = ("active", "superseded", "rejected")
SKILL_SCOPES = {
    "project_skill": ("repository", "architecture_family"),
    "strategy_skill": ("global", "contextual"),
}
LEGACY_SKILL_TYPE_MAP = {
    "project_type": "project_skill",
    "strategy_type": "strategy_skill",
}


class UnsupportedLegacySkillError(ValueError):
    """Raised when an obsolete skill type has no safe two-type mapping."""


def slugify(value: str, fallback: str = "skill") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or fallback


@dataclass
class DimensionSkill:
    """A compact project-knowledge or localization-strategy skill card.

    The class name is retained for import compatibility. New serialized records
    use ``skill_type`` and contain only project_skill or strategy_skill.
    """

    skill_id: str
    status: str
    version: int
    skill_type: str
    scope: str
    value: str
    title: str
    trigger: str
    knowledge: str

    @property
    def dimension(self) -> str:
        """Compatibility accessor for internal callers during the v0.4 migration."""

        return self.skill_type

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DimensionSkill":
        normalized = migrate_legacy_skill(data)
        skill_id = str(normalized.get("skill_id") or "").strip()
        if not skill_id:
            raise ValueError("Skill missing skill_id.")
        status = str(normalized.get("status") or "active")
        if status not in SKILL_STATUSES:
            raise ValueError(f"Invalid skill status: {status!r}")
        version = int(normalized.get("version") or 1)
        skill_type = str(normalized.get("skill_type") or "")
        if skill_type not in SKILL_TYPES:
            raise ValueError(f"Invalid skill_type: {skill_type!r}")
        scope = str(normalized.get("scope") or _default_scope(skill_type))
        if scope not in SKILL_SCOPES[skill_type]:
            raise ValueError(f"Invalid scope {scope!r} for {skill_type}.")
        value = str(normalized.get("value") or "unknown")
        skill_block = normalized.get("skill") or {}
        title = str(skill_block.get("title") or "").strip()
        trigger = str(skill_block.get("trigger") or "").strip()
        knowledge = str(skill_block.get("knowledge") or "").strip()
        if not title or not trigger or not knowledge:
            raise ValueError(f"Skill {skill_id} missing title/trigger/knowledge.")
        return cls(
            skill_id=skill_id,
            status=status,
            version=version,
            skill_type=skill_type,
            scope=scope,
            value=value,
            title=title,
            trigger=trigger,
            knowledge=knowledge,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "status": self.status,
            "version": self.version,
            "skill_type": self.skill_type,
            "scope": self.scope,
            "value": self.value,
            "skill": {
                "title": self.title,
                "trigger": self.trigger,
                "knowledge": self.knowledge,
            },
        }

    def compact_dict(
        self,
        *,
        score: float | None = None,
        match_reasons: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the runtime view used by Explorer and Reflector."""

        del score, match_reasons
        return {
            "skill_id": self.skill_id,
            "skill_type": self.skill_type,
            "scope": self.scope,
            "value": self.value,
            "title": self.title,
            "trigger": self.trigger,
            "knowledge": self.knowledge,
        }

def migrate_legacy_skill(record: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(record)
    skill_block = normalized.setdefault("skill", {})

    knowledge_parts: list[str] = []
    if skill_block.get("knowledge"):
        knowledge_parts.append(str(skill_block.get("knowledge")))
    if skill_block.get("guidance"):
        knowledge_parts.append(str(skill_block.get("guidance")))
    for item in skill_block.get("items", []) or []:
        if isinstance(item, dict) and item.get("text"):
            knowledge_parts.append(str(item["text"]))
    if knowledge_parts:
        skill_block["knowledge"] = "\n".join(
            dict.fromkeys(part.strip() for part in knowledge_parts if part.strip())
        )

    skill_block.pop("anti_patterns", None)
    skill_block.pop("items", None)
    skill_block.pop("guidance", None)

    raw_type = str(normalized.get("skill_type") or normalized.get("dimension") or "").strip()
    if not raw_type:
        raw_type = _legacy_type_from_taxonomy(normalized)
    skill_type = LEGACY_SKILL_TYPE_MAP.get(raw_type, raw_type)
    if skill_type not in SKILL_TYPES:
        raise UnsupportedLegacySkillError(
            f"Legacy skill type {raw_type!r} is not part of the project/strategy SkillBank."
        )
    normalized["skill_type"] = skill_type
    normalized["scope"] = normalized.get("scope") or _default_scope(skill_type)
    normalized.pop("dimension", None)

    normalized.setdefault("value", "unknown")
    # Old records may contain a separately maintained retrieval_text. It is
    # intentionally discarded: title, trigger, and knowledge are now the
    # canonical retrieval document and are rewritten atomically.
    normalized.pop("retrieval_text", None)
    normalized.setdefault("status", "active")
    normalized.setdefault("version", 1)
    normalized.pop("taxonomy", None)
    normalized.pop("provenance", None)
    normalized.pop("parent_skill_id", None)
    normalized.pop("supersedes", None)
    return normalized


def _legacy_type_from_taxonomy(record: dict[str, Any]) -> str:
    taxonomy = record.get("taxonomy") if isinstance(record.get("taxonomy"), dict) else {}
    if str(taxonomy.get("project_type") or "unknown") != "unknown":
        return "project_type"
    if str(taxonomy.get("strategy_type") or "unknown") != "unknown":
        return "strategy_type"
    if str(taxonomy.get("fault_mode") or "unknown") != "unknown":
        return "fault_mode"
    return "general"


def _default_scope(skill_type: str) -> str:
    return "architecture_family" if skill_type == "project_skill" else "contextual"


# Preferred public name for the two-type framework. DimensionSkill remains an
# import-compatible alias for historical callers.
SkillCard = DimensionSkill
