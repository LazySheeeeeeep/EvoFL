from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


SKILL_TYPES = ("project_skill", "issue_skill", "strategy_skill")
SKILL_STATUSES = ("active", "superseded", "rejected")
MIN_KNOWLEDGE_ITEMS = 2
MAX_KNOWLEDGE_ITEMS = 4
# Three compact statements fit directly into an embedding document and the
# Explorer's mid-trajectory tool result. The prompt targets a smaller range;
# this cap leaves enough provider variance for a retry to succeed.
MAX_KNOWLEDGE_CHARS = 960
# Newly finalized cards are deliberately stricter than legacy records accepted
# by ``validate_atomic_knowledge``. Keeping that compatibility boundary lets
# historical runs remain readable while forcing persisted staged-skill cards to
# remain concise embedding documents.
FINAL_KNOWLEDGE_ITEMS = 3
FINAL_MAX_KNOWLEDGE_CHARS = 960
FINAL_MAX_TRIGGER_CHARS = 320
# A Strategy card needs enough room to state its causal contrast and the
# observation that resolves it. Three bounded bullets and a 960-character
# total keep the embedding document compact without rejecting that evidence.
FINAL_MAX_STATEMENT_CHARS = 360
LEGACY_SKILL_TYPE_MAP = {
    "project_type": "project_skill",
    "strategy_type": "strategy_skill",
    "issue_type": "issue_skill",
}


class UnsupportedLegacySkillError(ValueError):
    """Raised when an obsolete skill type has no safe two-type mapping."""


def slugify(value: str, fallback: str = "skill") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or fallback


@dataclass
class DimensionSkill:
    """A compact Project, Issue, or Strategy skill card.

    The class name is retained for import compatibility. New serialized records
    use ``skill_type`` and contain one of the supported V3 skill types.
    """

    skill_id: str
    status: str
    version: int
    skill_type: str
    value: str
    title: str
    trigger: str
    knowledge: list[Any]

    def __post_init__(self) -> None:
        # Issue knowledge has stable numeric addresses for one-item edits.
        # Other Skill types retain their compact string-list representation.
        self.knowledge = normalize_skill_knowledge(self.knowledge, self.skill_type)

    @property
    def knowledge_texts(self) -> list[str]:
        return knowledge_texts(self.knowledge)

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
        value = str(normalized.get("value") or "unknown")
        skill_block = normalized.get("skill") or {}
        title = str(skill_block.get("title") or "").strip()
        trigger = str(skill_block.get("trigger") or "").strip()
        knowledge = normalize_skill_knowledge(skill_block.get("knowledge"), skill_type)
        if not title or not trigger or not knowledge:
            raise ValueError(f"Skill {skill_id} missing title/trigger/knowledge.")
        return cls(
            skill_id=skill_id,
            status=status,
            version=version,
            skill_type=skill_type,
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
        include_knowledge_ids: bool = False,
    ) -> dict[str, Any]:
        """Return the runtime view used by Explorer and Reflector."""

        del score, match_reasons
        knowledge: list[Any] = self.knowledge
        if self.skill_type == "issue_skill" and not include_knowledge_ids:
            knowledge = self.knowledge_texts
        return {
            "skill_id": self.skill_id,
            "skill_type": self.skill_type,
            "value": self.value,
            "title": self.title,
            "trigger": self.trigger,
            "knowledge": knowledge,
        }

def migrate_legacy_skill(record: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(record)
    skill_block = normalized.setdefault("skill", {})

    knowledge_parts: list[str] = []
    if skill_block.get("knowledge"):
        knowledge_parts.extend(normalize_knowledge(skill_block.get("knowledge")))
    if skill_block.get("guidance"):
        knowledge_parts.extend(normalize_knowledge(skill_block.get("guidance")))
    for item in skill_block.get("items", []) or []:
        if isinstance(item, dict) and item.get("text"):
            knowledge_parts.append(str(item["text"]))
    if knowledge_parts:
        skill_block["knowledge"] = list(
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
    if skill_type == "issue_skill":
        skill_block["knowledge"] = normalize_issue_knowledge(skill_block.get("knowledge"))
    # Historical scope values are accepted on read but are not part of the
    # current two-type Skill schema.
    normalized.pop("scope", None)
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


def normalize_knowledge(value: Any) -> list[str]:
    """Normalize legacy string knowledge and new atomic knowledge lists."""

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            _knowledge_item_text(item).strip()
            for item in value
            if _knowledge_item_text(item).strip()
        )
    )


def _knowledge_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("text") or "")
    return ""


def normalize_issue_knowledge(value: Any) -> list[dict[str, Any]]:
    """Normalize Issue knowledge into stable numeric, unordered list items.

    Historical string lists are accepted and assigned IDs in their original
    order. Persisted Issue cards retain those IDs, so later case reflections
    can replace or remove a proposition without relying on text matching.
    """

    raw_items = value if isinstance(value, list) else [value]
    normalized: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    next_id = 1
    for raw in raw_items:
        text = _knowledge_item_text(raw).strip()
        if not text:
            continue
        raw_id = raw.get("id") if isinstance(raw, dict) else None
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            item_id = 0
        if item_id < 1 or item_id in used_ids:
            while next_id in used_ids:
                next_id += 1
            item_id = next_id
        used_ids.add(item_id)
        next_id = max(next_id, item_id + 1)
        normalized.append({"id": item_id, "text": text})
    return normalized


def normalize_skill_knowledge(value: Any, skill_type: str) -> list[Any]:
    if skill_type == "issue_skill":
        return normalize_issue_knowledge(value)
    return normalize_knowledge(value)


def knowledge_texts(value: Any) -> list[str]:
    """Return text-only knowledge for runtime prompts and embedding documents."""

    return normalize_knowledge(value)


def validate_atomic_knowledge(value: Any) -> list[str]:
    """Validate the compact knowledge contract used by newly evolved skills."""

    knowledge = normalize_knowledge(value)
    if not MIN_KNOWLEDGE_ITEMS <= len(knowledge) <= MAX_KNOWLEDGE_ITEMS:
        raise ValueError(
            f"knowledge requires {MIN_KNOWLEDGE_ITEMS}-{MAX_KNOWLEDGE_ITEMS} atomic statements."
        )
    if sum(len(item) for item in knowledge) > MAX_KNOWLEDGE_CHARS:
        raise ValueError(f"knowledge exceeds {MAX_KNOWLEDGE_CHARS} total characters.")
    return knowledge


def validate_issue_knowledge(value: Any) -> list[dict[str, Any]]:
    """Validate Issue propositions without imposing a fixed item count.

    Item IDs are normalized rather than supplied by the model on creation.
    The per-case update protocol ensures a reflection edits at most one item.
    """

    knowledge = normalize_issue_knowledge(value)
    if not knowledge:
        raise ValueError("Issue Skill requires at least one knowledge proposition.")
    return knowledge


def validate_portable_skill_card(raw: Any, *, skill_type: str) -> dict[str, Any]:
    """Validate the compact card contract used after LLM finalization.

    The check constrains only output shape and length. Semantic center remains
    an LLM judgment, so the code does not impose a brittle architecture or
    diagnostic taxonomy on future skill evolution.
    """

    if skill_type not in SKILL_TYPES:
        raise ValueError(f"Unsupported skill type: {skill_type!r}")
    if not isinstance(raw, dict):
        raise ValueError("Finalized skill must be an object.")
    value = str(raw.get("value") or "").strip()
    title = str(raw.get("title") or "").strip()
    trigger = str(raw.get("trigger") or "").strip()
    if not value or not title or not trigger:
        raise ValueError("Finalized skill requires value, title, and trigger.")
    if len(trigger) > FINAL_MAX_TRIGGER_CHARS:
        raise ValueError(
            f"Finalized skill trigger exceeds {FINAL_MAX_TRIGGER_CHARS} characters."
        )
    knowledge = normalize_knowledge(raw.get("knowledge"))
    if len(knowledge) != FINAL_KNOWLEDGE_ITEMS:
        raise ValueError(
            f"Finalized skill requires exactly {FINAL_KNOWLEDGE_ITEMS} knowledge statements."
        )
    if any(len(statement) > FINAL_MAX_STATEMENT_CHARS for statement in knowledge):
        raise ValueError(
            f"Finalized knowledge statement exceeds {FINAL_MAX_STATEMENT_CHARS} characters."
        )
    if sum(len(statement) for statement in knowledge) > FINAL_MAX_KNOWLEDGE_CHARS:
        raise ValueError(
            f"Finalized knowledge exceeds {FINAL_MAX_KNOWLEDGE_CHARS} total characters."
        )
    return {
        "value": value,
        "title": title,
        "trigger": trigger,
        "knowledge": knowledge,
    }


# Preferred public name for the two-type framework. DimensionSkill remains an
# import-compatible alias for historical callers.
SkillCard = DimensionSkill
