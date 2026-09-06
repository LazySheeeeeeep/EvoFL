from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .fault_taxonomy import validate_fault_family

SKILL_TYPES = ("project_skill", "fault_skill", "strategy_skill")
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
}


class UnsupportedLegacySkillError(ValueError):
    """Raised when an obsolete skill type has no safe two-type mapping."""


def slugify(value: str, fallback: str = "skill") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or fallback


@dataclass
class DimensionSkill:
    """A compact Project, Fault, or Strategy skill card.

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
    repo_id: str | None = None
    fault_family: str | None = None
    fault_subtype: str | None = None

    def __post_init__(self) -> None:
        self.knowledge = normalize_skill_knowledge(self.knowledge, self.skill_type)
        if self.skill_type == "project_skill":
            self.repo_id = normalize_repo_id(self.repo_id) if self.repo_id else None
        if self.skill_type == "fault_skill":
            self.fault_family = validate_fault_family(self.fault_family)
            self.fault_subtype = str(self.fault_subtype or self.value).strip()
            if not self.fault_subtype:
                raise ValueError("Fault Skill requires fault_subtype.")
            self.value = self.fault_subtype

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
        fault_family = normalized.get("fault_family")
        fault_subtype = str(normalized.get("fault_subtype") or "").strip() or None
        repo_id = (
            normalize_repo_id(normalized.get("repo_id"))
            if skill_type == "project_skill" and normalized.get("repo_id")
            else None
        )
        value = str(
            fault_subtype
            if skill_type == "fault_skill"
            else repo_id or normalized.get("value") or "unknown"
        )
        skill_block = normalized.get("skill") or {}
        title = str(skill_block.get("title") or "").strip()
        trigger = str(skill_block.get("trigger") or "").strip()
        knowledge = normalize_skill_knowledge(skill_block.get("knowledge"), skill_type)
        if not title or not knowledge or (skill_type != "project_skill" and not trigger):
            raise ValueError(f"Skill {skill_id} missing required title/trigger/knowledge.")
        return cls(
            skill_id=skill_id,
            status=status,
            version=version,
            skill_type=skill_type,
            value=value,
            title=title,
            trigger=trigger,
            knowledge=knowledge,
            repo_id=repo_id,
            fault_family=str(fault_family or "").strip() or None,
            fault_subtype=fault_subtype,
        )

    def to_dict(self) -> dict[str, Any]:
        skill_payload = {
            "title": self.title,
            "knowledge": self.knowledge,
        }
        if self.skill_type != "project_skill":
            skill_payload["trigger"] = self.trigger
        payload = {
            "skill_id": self.skill_id,
            "status": self.status,
            "version": self.version,
            "skill_type": self.skill_type,
            "skill": skill_payload,
        }
        if self.skill_type == "fault_skill":
            payload["fault_family"] = self.fault_family
            payload["fault_subtype"] = self.fault_subtype
        elif self.skill_type == "project_skill" and self.repo_id:
            payload["repo_id"] = self.repo_id
        else:
            payload["value"] = self.value
        return payload

    def compact_dict(
        self,
        *,
        score: float | None = None,
        match_reasons: dict[str, Any] | None = None,
        include_knowledge_ids: bool = False,
    ) -> dict[str, Any]:
        """Return the runtime view used by Explorer and Reflector."""

        del score, match_reasons
        del include_knowledge_ids
        payload = {
            "skill_id": self.skill_id,
            "skill_type": self.skill_type,
            "title": self.title,
            "knowledge": self.knowledge_texts,
        }
        if self.skill_type == "fault_skill":
            payload["fault_family"] = self.fault_family
            payload["fault_subtype"] = self.fault_subtype
        elif self.skill_type == "project_skill" and self.repo_id:
            payload["repo_id"] = self.repo_id
        else:
            payload["value"] = self.value
        if self.skill_type != "project_skill":
            payload["trigger"] = self.trigger
        return payload


def normalize_repo_id(value: Any) -> str:
    """Return the stable repository identity used by Project Skills."""

    text = str(value or "").strip().replace("\\", "/")
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.removesuffix(".git").strip("/")
    # SWE-smith identifies a repository snapshot as
    # ``swesmith/owner__repository.<commit>``. Project knowledge belongs to the
    # upstream repository, not to one generated snapshot.
    if text.lower().startswith("swesmith/"):
        synthetic = text.split("/", 1)[1]
        owner_repo, separator, _commit = synthetic.rpartition(".")
        if separator and "__" in owner_repo:
            owner, repository = owner_repo.split("__", 1)
            text = f"{owner}/{repository}"
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"repo_id must use owner/repository form, got {value!r}.")
    return "/".join(parts[-2:]).lower()


def validate_project_knowledge(value: Any) -> list[str]:
    """Validate repository-specific static facts without a shared Skill grammar."""

    knowledge = normalize_knowledge(value)
    if not knowledge:
        raise ValueError("Project knowledge requires at least one statement.")
    return knowledge


def validate_project_create(raw: Any, *, expected_repo_id: str) -> dict[str, Any]:
    """Validate the independent Project create protocol."""

    if not isinstance(raw, dict):
        raise ValueError("Project create payload must be an object.")
    repo_id = normalize_repo_id(raw.get("repo_id") or expected_repo_id)
    if repo_id != normalize_repo_id(expected_repo_id):
        raise ValueError("Project create repo_id must match the current repository.")
    title = str(raw.get("title") or "").strip()
    if not title:
        raise ValueError("Project create requires title.")
    return {
        "repo_id": repo_id,
        "title": title,
        "knowledge": validate_project_knowledge(raw.get("knowledge")),
    }


def validate_project_add(value: Any) -> list[str]:
    """Validate knowledge additions in the independent Project add protocol."""

    return validate_project_knowledge(value)

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
            f"Legacy skill type {raw_type!r} is not part of the Project/Fault/Strategy SkillBank."
        )
    normalized["skill_type"] = skill_type
    # Historical scope values are accepted on read but are not part of the
    # current two-type Skill schema.
    normalized.pop("scope", None)
    normalized.pop("dimension", None)

    if skill_type != "fault_skill":
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


def normalize_skill_knowledge(value: Any, skill_type: str) -> list[Any]:
    del skill_type
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


def validate_fault_knowledge(value: Any) -> list[str]:
    knowledge = normalize_knowledge(value)
    if not knowledge:
        raise ValueError("Fault Skill requires at least one knowledge statement.")
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
    title = str(raw.get("title") or "").strip()
    trigger = str(raw.get("trigger") or "").strip()
    if not title or not trigger:
        raise ValueError("Finalized skill requires title and trigger.")
    if skill_type == "fault_skill":
        family = validate_fault_family(raw.get("fault_family"))
        subtype = str(raw.get("fault_subtype") or "").strip()
        if not subtype:
            raise ValueError("Finalized Fault Skill requires fault_subtype.")
        return {
            "fault_family": family,
            "fault_subtype": subtype,
            "title": title,
            "trigger": trigger,
            "knowledge": validate_fault_knowledge(raw.get("knowledge")),
        }
    value = str(raw.get("value") or "").strip()
    if not value:
        raise ValueError("Finalized skill requires value.")
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


# Preferred public name for the staged Skill framework. DimensionSkill remains an
# import-compatible alias for historical callers.
SkillCard = DimensionSkill
