from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


DIMENSIONS = ("general", "project_type", "fault_mode", "strategy_type")
SKILL_STATUSES = ("active", "superseded", "rejected")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str, fallback: str = "skill") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or fallback


def taxonomy_from(record: dict[str, Any] | None) -> dict[str, str]:
    source = record or {}
    return {
        "project_type": str(source.get("project_type") or "unknown"),
        "fault_mode": str(source.get("fault_mode") or "unknown"),
        "strategy_type": str(source.get("strategy_type") or "unknown"),
    }


def infer_dimension_value(taxonomy: dict[str, str]) -> tuple[str, str]:
    for dimension in ("fault_mode", "strategy_type", "project_type"):
        value = taxonomy.get(dimension, "unknown")
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
    anti_patterns: list[str] = field(default_factory=list)
    taxonomy: dict[str, str] = field(default_factory=dict)
    supported_by_cases: list[str] = field(default_factory=list)
    supported_by_successes: list[str] = field(default_factory=list)
    supported_by_failures: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    edit_history: list[dict[str, Any]] = field(default_factory=list)
    parent_skill_id: str | None = None
    supersedes: str | None = None

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
        taxonomy = taxonomy_from(normalized.get("taxonomy"))
        skill_block = normalized.get("skill") or {}
        title = str(skill_block.get("title") or "").strip()
        trigger = str(skill_block.get("trigger") or "").strip()
        knowledge = str(skill_block.get("knowledge") or "").strip()
        if not title or not trigger or not knowledge:
            raise ValueError(f"Skill {skill_id} missing title/trigger/knowledge.")
        anti_patterns = [str(item).strip() for item in skill_block.get("anti_patterns", []) if str(item).strip()]
        retrieval_text = str(
            normalized.get("retrieval_text")
            or build_retrieval_text(title, trigger, knowledge, anti_patterns, dimension, value)
        )
        provenance = normalized.get("provenance") or {}
        legacy_insights = [str(x) for x in provenance.get("supported_by_insights", []) if str(x)]
        supported_by_cases = [str(x) for x in provenance.get("supported_by_cases", []) if str(x)]
        supported_by_cases = list(dict.fromkeys(supported_by_cases + legacy_insights))
        return cls(
            skill_id=skill_id,
            status=status,
            version=version,
            taxonomy=taxonomy,
            dimension=dimension,
            value=value,
            retrieval_text=retrieval_text,
            title=title,
            trigger=trigger,
            knowledge=knowledge,
            anti_patterns=anti_patterns,
            supported_by_cases=supported_by_cases,
            supported_by_successes=[str(x) for x in provenance.get("supported_by_successes", []) if str(x)],
            supported_by_failures=[str(x) for x in provenance.get("supported_by_failures", []) if str(x)],
            created_at=str(provenance.get("created_at") or utc_now()),
            updated_at=str(provenance.get("updated_at") or utc_now()),
            edit_history=list(provenance.get("edit_history", []) or []),
            parent_skill_id=normalized.get("parent_skill_id"),
            supersedes=normalized.get("supersedes"),
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
                "anti_patterns": self.anti_patterns,
            },
            "provenance": {
                "supported_by_cases": self.supported_by_cases,
                "supported_by_successes": self.supported_by_successes,
                "supported_by_failures": self.supported_by_failures,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
        }
        if self.taxonomy:
            data["taxonomy"] = taxonomy_from(self.taxonomy)
        if self.edit_history:
            data["provenance"]["edit_history"] = self.edit_history
        if self.parent_skill_id:
            data["parent_skill_id"] = self.parent_skill_id
        if self.supersedes:
            data["supersedes"] = self.supersedes
        return data

    def compact_dict(self, *, score: float | None = None, match_reasons: dict[str, Any] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "skill_id": self.skill_id,
            "dimension": self.dimension,
            "value": self.value,
            "title": self.title,
            "trigger": self.trigger,
            "knowledge": self.knowledge,
            "anti_patterns": self.anti_patterns,
            "provenance": {
                "supported_by_cases": self.supported_by_cases,
                "supported_by_successes": self.supported_by_successes,
                "supported_by_failures": self.supported_by_failures,
            },
        }
        if score is not None:
            data["score"] = score
        if match_reasons is not None:
            data["match_reasons"] = match_reasons
        return data


def build_retrieval_text(
    title: str,
    trigger: str,
    knowledge: str,
    anti_patterns: list[str],
    dimension: str,
    value: str,
) -> str:
    del knowledge, anti_patterns
    parts = [title, trigger, dimension, value]
    return " ".join(part for part in parts if part)


def migrate_legacy_skill(record: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(record)
    normalized["taxonomy"] = taxonomy_from(normalized.get("taxonomy"))
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

    skill_block["anti_patterns"] = [
        str(item).strip() for item in skill_block.get("anti_patterns", []) or [] if str(item).strip()
    ]
    skill_block.pop("items", None)
    skill_block.pop("guidance", None)

    if not normalized.get("dimension") or not normalized.get("value"):
        dimension, value = infer_dimension_value(normalized["taxonomy"])
        normalized["dimension"] = normalized.get("dimension") or dimension
        normalized["value"] = normalized.get("value") or value

    if not normalized.get("retrieval_text"):
        normalized["retrieval_text"] = build_retrieval_text(
            str(skill_block.get("title") or ""),
            str(skill_block.get("trigger") or ""),
            str(skill_block.get("knowledge") or ""),
            skill_block["anti_patterns"],
            str(normalized.get("dimension") or "general"),
            str(normalized.get("value") or "unknown"),
        )
    normalized.setdefault("status", "active")
    normalized.setdefault("version", 1)
    normalized.setdefault("provenance", {})
    provenance = normalized["provenance"]
    provenance.setdefault("supported_by_cases", provenance.get("supported_by_insights", []))
    provenance.setdefault("supported_by_successes", [])
    provenance.setdefault("supported_by_failures", [])
    provenance.setdefault("created_at", utc_now())
    provenance.setdefault("updated_at", utc_now())
    return normalized
