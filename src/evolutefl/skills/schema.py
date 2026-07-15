from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


DIMENSIONS = ("general", "project_type", "fault_mode", "strategy_type")
SKILL_STATUSES = ("active", "superseded", "rejected")


def slugify(value: str, fallback: str = "skill") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or fallback


def infer_legacy_dimension_value(record: dict[str, Any]) -> tuple[str, str]:
    taxonomy = record.get("taxonomy") if isinstance(record.get("taxonomy"), dict) else {}
    for dimension in ("fault_mode", "strategy_type", "project_type"):
        value = str(taxonomy.get(dimension) or "unknown")
        if value and value != "unknown":
            return dimension, value
    return "general", "unknown"


@dataclass
class DimensionSkill:
    """Dimension-level natural-language localization knowledge card.

    A skill is not an executable workflow. It is reusable knowledge associated
    with one dimension: project_type, fault_mode, strategy_type, or general.
    """

    skill_id: str
    status: str
    version: int
    dimension: str
    value: str
    retrieval_text: str
    title: str
    trigger: str
    knowledge: str

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
        dimension = str(normalized.get("dimension") or "")
        if dimension not in DIMENSIONS:
            raise ValueError(f"Invalid skill dimension: {dimension!r}")
        value = str(normalized.get("value") or "unknown")
        skill_block = normalized.get("skill") or {}
        title = str(skill_block.get("title") or "").strip()
        trigger = str(skill_block.get("trigger") or "").strip()
        knowledge = str(skill_block.get("knowledge") or "").strip()
        if not title or not trigger or not knowledge:
            raise ValueError(f"Skill {skill_id} missing title/trigger/knowledge.")
        retrieval_text = str(normalized.get("retrieval_text") or build_retrieval_text(title, trigger, dimension, value))
        return cls(
            skill_id=skill_id,
            status=status,
            version=version,
            dimension=dimension,
            value=value,
            retrieval_text=retrieval_text,
            title=title,
            trigger=trigger,
            knowledge=knowledge,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "skill_id": self.skill_id,
            "status": self.status,
            "version": self.version,
            "dimension": self.dimension,
            "value": self.value,
            "retrieval_text": self.retrieval_text,
            "skill": {
                "title": self.title,
                "trigger": self.trigger,
                "knowledge": self.knowledge,
            },
        }
        return data

    def compact_dict(self, *, score: float | None = None, match_reasons: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the runtime skill view used by Explorer and Reflector.

        Runtime prompts should only see the reusable localization card itself,
        so case-specific audit fields do not leak into retrieval context or
        agent guidance.
        """
        del score, match_reasons
        data: dict[str, Any] = {
            "skill_id": self.skill_id,
            "dimension": self.dimension,
            "value": self.value,
            "retrieval_text": self.retrieval_text,
            "title": self.title,
            "trigger": self.trigger,
            "knowledge": self.knowledge,
        }
        return data


def build_retrieval_text(
    title: str,
    trigger: str,
    dimension: str,
    value: str,
) -> str:
    parts = [title, trigger, dimension, value]
    return " ".join(part for part in parts if part)


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
        skill_block["knowledge"] = "\n".join(dict.fromkeys(part.strip() for part in knowledge_parts if part.strip()))

    skill_block.pop("anti_patterns", None)
    skill_block.pop("items", None)
    skill_block.pop("guidance", None)

    if not normalized.get("dimension") or not normalized.get("value"):
        dimension, value = infer_legacy_dimension_value(normalized)
        normalized["dimension"] = normalized.get("dimension") or dimension
        normalized["value"] = normalized.get("value") or value

    if not normalized.get("retrieval_text"):
        normalized["retrieval_text"] = build_retrieval_text(
            str(skill_block.get("title") or ""),
            str(skill_block.get("trigger") or ""),
            str(normalized.get("dimension") or "general"),
            str(normalized.get("value") or "unknown"),
        )
    normalized.setdefault("status", "active")
    normalized.setdefault("version", 1)
    normalized.pop("taxonomy", None)
    normalized.pop("provenance", None)
    normalized.pop("parent_skill_id", None)
    normalized.pop("supersedes", None)
    return normalized
