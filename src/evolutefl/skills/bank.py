from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from evolutefl.embedding import EmbeddingClient
from evolutefl.json_utils import read_jsonl, write_jsonl

from .embedding_store import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    EmbeddingRetrievalConfig,
    SkillEmbeddingStore,
    query_embedding_text,
    select_embedding_skills,
)
from .retrieval import DEFAULT_DIMENSION_QUOTA, select_skills
from .schema import DIMENSIONS, DimensionSkill, build_retrieval_text, slugify, utc_now


class SkillBankV0:
    def __init__(
        self,
        path: str | Path,
        *,
        max_matched_skills: int = 5,
        max_per_dimension: dict[str, int] | None = None,
        min_score: float = 4.0,
        project_type_min_score: float = 6.0,
        retrieval_mode: str = "lexical",
        embedding_client: EmbeddingClient | None = None,
        embedding_cache_path: str | Path | None = None,
        embedding_model_id: str = "jina-embeddings-v3",
        embedding_query_task: str = "retrieval.query",
        embedding_document_task: str = "retrieval.passage",
        embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
        embedding_fallback_to_lexical: bool = False,
        embedding_batch_size: int = 16,
    ) -> None:
        self.path = Path(path)
        self.max_matched_skills = max_matched_skills
        self.max_per_dimension = {**DEFAULT_DIMENSION_QUOTA, **(max_per_dimension or {})}
        self.min_score = min_score
        self.project_type_min_score = project_type_min_score
        self.retrieval_mode = _valid_retrieval_mode(retrieval_mode)
        self.embedding_client = embedding_client
        default_cache = self.path.parent / "embeddings" / "jina_v3" / "skill_embeddings.jsonl"
        self.embedding_config = EmbeddingRetrievalConfig(
            cache_path=Path(embedding_cache_path) if embedding_cache_path else default_cache,
            model_id=embedding_model_id,
            query_task=embedding_query_task,
            document_task=embedding_document_task,
            min_score=embedding_min_score,
            fallback_to_lexical=embedding_fallback_to_lexical,
            batch_size=embedding_batch_size,
        )

    def load(self) -> list[DimensionSkill]:
        records = read_jsonl(self.path)
        latest_by_id: dict[str, DimensionSkill] = {}
        ordered_ids: list[str] = []
        for record in records:
            skill = DimensionSkill.from_dict(record)
            if skill.skill_id not in latest_by_id:
                ordered_ids.append(skill.skill_id)
            current = latest_by_id.get(skill.skill_id)
            if current is None or _skill_sort_key(skill) >= _skill_sort_key(current):
                latest_by_id[skill.skill_id] = skill
        return [latest_by_id[skill_id] for skill_id in ordered_ids]

    def active_skills(self) -> list[DimensionSkill]:
        return [skill for skill in self.load() if skill.status == "active"]

    def search_for_explorer(self, repo: str, issue: str) -> dict[str, Any]:
        query = query_embedding_text(repo, issue)
        active = self.active_skills()
        selected, trace = self._select_for_query(
            query,
            active,
            max_matched_skills=self.max_matched_skills,
            max_per_dimension=self.max_per_dimension,
        )
        scores = trace.get("scores", {})
        reasons = trace.get("match_reasons", {})
        matched_skills = [
            skill.compact_dict(score=scores.get(skill.skill_id), match_reasons=reasons.get(skill.skill_id, {}))
            for skill in selected
        ]
        dimension_slots = _build_dimension_slots(active, matched_skills, trace)
        trace["dimension_slot_summary"] = {
            dimension: slot["status"] for dimension, slot in dimension_slots.items()
        }
        return {
            "matched_skills": matched_skills,
            "dimension_slots": dimension_slots,
            "skill_search_trace": trace,
        }

    def search_for_reflector(self, repo: str, issue: str, insight: dict[str, Any]) -> dict[str, Any]:
        analysis = insight.get("analysis", {}) if isinstance(insight, dict) else {}
        dimension = analysis.get("missing_or_reinforced_dimension") or analysis.get("missing_dimension") or "general"
        value = analysis.get("target_value_hint", "unknown")
        query = f"{repo}\n{issue}\n{analysis.get('transferable_lesson', '')}\n{value}"
        active = self.active_skills()
        selected, trace = self._select_for_query(
            query,
            active,
            max_matched_skills=8,
            max_per_dimension={"general": 2, "project_type": 2, "fault_mode": 3, "strategy_type": 2},
        )
        scores = trace.get("scores", {})
        reasons = trace.get("match_reasons", {})
        same_dimension = [skill for skill in selected if skill.dimension == dimension]
        compact = [
            skill.compact_dict(score=scores.get(skill.skill_id), match_reasons=reasons.get(skill.skill_id, {}))
            for skill in selected
        ]
        dimension_slots = _build_dimension_slots(active, compact, trace)
        trace["dimension_slot_summary"] = {
            slot_dimension: slot["status"] for slot_dimension, slot in dimension_slots.items()
        }
        return {
            "dimension_query": {
                "target_dimension": dimension,
                "target_value": value,
            },
            "dimension_slots": dimension_slots,
            "same_dimension_skills": [
                skill.compact_dict(score=scores.get(skill.skill_id), match_reasons=reasons.get(skill.skill_id, {}))
                for skill in same_dimension
            ],
            "candidate_target_skills": compact,
            "search_trace": trace,
        }

    def rebuild_embeddings(self) -> dict[str, Any]:
        if self.embedding_client is None:
            raise RuntimeError("Embedding client is not configured.")
        active = self.active_skills()
        store = SkillEmbeddingStore(self.embedding_config.cache_path, model_id=self.embedding_config.model_id)
        _, cache_trace = store.ensure_skill_embeddings(
            active,
            self.embedding_client,
            task=self.embedding_config.document_task,
            batch_size=self.embedding_config.batch_size,
            prune_to_skills=True,
        )
        return {
            "skill_bank_path": str(self.path),
            "active_skill_count": len(active),
            "embedding_cache": cache_trace,
        }

    def _select_for_query(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_dimension: dict[str, int],
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        if self.retrieval_mode == "lexical":
            return self._select_lexical(query, active, max_matched_skills=max_matched_skills, max_per_dimension=max_per_dimension)
        if self.retrieval_mode == "embedding":
            return self._select_embedding_or_fallback(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_dimension=max_per_dimension,
            )
        embedding_selected, embedding_trace = self._select_embedding_or_fallback(
            query,
            active,
            max_matched_skills=max_matched_skills,
            max_per_dimension=max_per_dimension,
            fallback_to_lexical=False,
        )
        lexical_selected, lexical_trace = self._select_lexical(
            query,
            active,
            max_matched_skills=max_matched_skills,
            max_per_dimension=max_per_dimension,
        )
        selected = _merge_selected_by_quota(
            [*embedding_selected, *lexical_selected],
            max_matched_skills=max_matched_skills,
            max_per_dimension=max_per_dimension,
        )
        trace = _merge_hybrid_trace(query, selected, embedding_trace, lexical_trace)
        return selected, trace

    def _select_lexical(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_dimension: dict[str, int],
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        return select_skills(
            query,
            active,
            max_matched_skills=max_matched_skills,
            max_per_dimension=max_per_dimension,
            min_score=self.min_score,
            project_type_min_score=self.project_type_min_score,
        )

    def _select_embedding_or_fallback(
        self,
        query: str,
        active: list[DimensionSkill],
        *,
        max_matched_skills: int,
        max_per_dimension: dict[str, int],
        fallback_to_lexical: bool | None = None,
    ) -> tuple[list[DimensionSkill], dict[str, Any]]:
        should_fallback = self.embedding_config.fallback_to_lexical if fallback_to_lexical is None else fallback_to_lexical
        if self.embedding_client is None:
            if should_fallback:
                selected, trace = self._select_lexical(
                    query,
                    active,
                    max_matched_skills=max_matched_skills,
                    max_per_dimension=max_per_dimension,
                )
                trace["retrieval_mode"] = "embedding_unconfigured_fallback_to_lexical_v1"
                trace.setdefault("notes", []).append("Embedding client is not configured.")
                return selected, trace
            return [], {
                "query": query,
                "retrieval_mode": "embedding_unconfigured_v1",
                "candidate_skill_ids": [],
                "selected_skill_ids": [],
                "scores": {},
                "match_reasons": {},
                "filtered_below_threshold": [],
                "notes": ["Embedding client is not configured."],
            }
        try:
            return select_embedding_skills(
                query_text=query,
                skills=active,
                client=self.embedding_client,
                config=self.embedding_config,
                max_matched_skills=max_matched_skills,
                max_per_dimension=max_per_dimension,
            )
        except Exception as exc:  # noqa: BLE001 - retrieval should degrade, not break Explorer.
            if not should_fallback:
                return [], {
                    "query": query,
                    "retrieval_mode": "embedding_failed_v1",
                    "candidate_skill_ids": [],
                    "selected_skill_ids": [],
                    "scores": {},
                    "match_reasons": {},
                    "filtered_below_threshold": [],
                    "notes": [f"Embedding retrieval failed: {exc}"],
                }
            selected, trace = self._select_lexical(
                query,
                active,
                max_matched_skills=max_matched_skills,
                max_per_dimension=max_per_dimension,
            )
            trace["retrieval_mode"] = "embedding_failed_fallback_to_lexical_v1"
            trace.setdefault("notes", []).append(f"Embedding retrieval failed: {exc}")
            return selected, trace

    def apply_update(self, edit: dict[str, Any]) -> dict[str, Any]:
        operation = edit.get("operation")
        if operation == "preserve":
            return self._apply_preserve(edit)
        if operation == "add" and not (edit.get("target") or {}).get("skill_id"):
            return self._apply_create_new(edit)
        if operation in ("add", "replace", "delete"):
            return self._apply_modify_existing(edit)
        raise ValueError(f"Unsupported skill edit operation: {operation!r}")

    def _apply_create_new(self, edit: dict[str, Any]) -> dict[str, Any]:
        target = edit.get("target") or {}
        dimension = _valid_dimension(target.get("dimension"))
        value = str(target.get("value") or "unknown")
        content = edit.get("content") or {}
        text = str(content.get("text") or content.get("new_text") or "").strip()
        if not text:
            raise ValueError("add edit requires content.text for new skill.")
        title = str(content.get("title") or _title_from_value(value))
        trigger = str(content.get("trigger") or f"Use when the issue matches {value.replace('_', ' ')} localization patterns.")
        anti_patterns = _as_list(content.get("anti_patterns"))
        retrieval_text = str(
            content.get("retrieval_text")
            or build_retrieval_text(title, trigger, text, anti_patterns, dimension, value)
        )
        existing = self.load()
        skill_id = _unique_skill_id(existing, f"{dimension}_{value}_v1")
        now = utc_now()
        provenance = _provenance_from_edit(edit, created_at=now, updated_at=now)
        skill = DimensionSkill(
            skill_id=skill_id,
            status="active",
            version=1,
            dimension=dimension,
            value=value,
            taxonomy=_taxonomy_for_dimension(dimension, value),
            retrieval_text=retrieval_text,
            title=title,
            trigger=trigger,
            knowledge=text,
            anti_patterns=anti_patterns,
            supported_by_cases=provenance["supported_by_cases"],
            supported_by_successes=provenance["supported_by_successes"],
            supported_by_failures=provenance["supported_by_failures"],
            created_at=now,
            updated_at=now,
            edit_history=provenance["edit_history"],
        )
        records = read_jsonl(self.path)
        records.append(skill.to_dict())
        write_jsonl(self.path, records)
        return {
            "action": "create_new",
            "updated_skill_id": skill.skill_id,
            "version": skill.version,
            "embedding_cache": self._warm_embedding_for_skill(skill),
        }

    def _apply_modify_existing(self, edit: dict[str, Any]) -> dict[str, Any]:
        target = edit.get("target") or {}
        target_skill_id = str(target.get("skill_id") or "")
        if not target_skill_id:
            raise ValueError(f"{edit.get('operation')} edit requires target.skill_id.")
        old = {skill.skill_id: skill for skill in self.active_skills()}.get(target_skill_id)
        if not old:
            raise ValueError(f"No active skill found for target.skill_id={target_skill_id!r}.")

        new_skill = deepcopy(old)
        new_skill.version = old.version + 1
        new_skill.updated_at = utc_now()
        new_skill.parent_skill_id = old.skill_id
        new_skill.supersedes = old.skill_id
        field = str(target.get("field") or "skill.knowledge")
        operation = str(edit.get("operation"))
        _apply_text_edit(new_skill, field, operation, edit.get("content") or {})
        _merge_provenance(new_skill, edit)
        if not new_skill.retrieval_text:
            new_skill.retrieval_text = build_retrieval_text(
                new_skill.title,
                new_skill.trigger,
                new_skill.knowledge,
                new_skill.anti_patterns,
                new_skill.dimension,
                new_skill.value,
            )

        superseded = deepcopy(old)
        superseded.status = "superseded"
        superseded.updated_at = new_skill.updated_at

        records = read_jsonl(self.path)
        records.append(superseded.to_dict())
        records.append(new_skill.to_dict())
        write_jsonl(self.path, records)
        return {
            "action": operation,
            "updated_skill_id": new_skill.skill_id,
            "version": new_skill.version,
            "field": field,
            "embedding_cache": self._warm_embedding_for_skill(new_skill),
        }

    def _apply_preserve(self, edit: dict[str, Any]) -> dict[str, Any]:
        target = edit.get("target") or {}
        target_skill_id = str(target.get("skill_id") or "")
        if not target_skill_id:
            return {"action": "preserve", "updated_skill_id": None, "reason": "preserve edit has no target skill"}
        old = {skill.skill_id: skill for skill in self.active_skills()}.get(target_skill_id)
        if not old:
            return {"action": "preserve", "updated_skill_id": None, "reason": "target skill not active"}
        new_skill = deepcopy(old)
        new_skill.version = old.version + 1
        new_skill.updated_at = utc_now()
        new_skill.parent_skill_id = old.skill_id
        new_skill.supersedes = old.skill_id
        _merge_provenance(new_skill, edit)
        superseded = deepcopy(old)
        superseded.status = "superseded"
        superseded.updated_at = new_skill.updated_at
        records = read_jsonl(self.path)
        records.append(superseded.to_dict())
        records.append(new_skill.to_dict())
        write_jsonl(self.path, records)
        return {
            "action": "preserve",
            "updated_skill_id": new_skill.skill_id,
            "version": new_skill.version,
            "embedding_cache": self._warm_embedding_for_skill(new_skill),
        }

    def _warm_embedding_for_skill(self, skill: DimensionSkill) -> dict[str, Any]:
        if self.embedding_client is None:
            return {"enabled": False}
        store = SkillEmbeddingStore(self.embedding_config.cache_path, model_id=self.embedding_config.model_id)
        try:
            _, trace = store.ensure_skill_embeddings(
                [skill],
                self.embedding_client,
                task=self.embedding_config.document_task,
                batch_size=1,
            )
            return {"enabled": True, **trace}
        except Exception as exc:  # noqa: BLE001 - skill update should not fail because the cache is cold.
            return {"enabled": True, "error": str(exc)}


def _valid_dimension(value: Any) -> str:
    dimension = str(value or "general")
    if dimension not in DIMENSIONS:
        raise ValueError(f"Invalid target dimension: {dimension!r}")
    return dimension


def _valid_retrieval_mode(value: Any) -> str:
    mode = str(value or "lexical").strip().lower()
    if mode not in {"lexical", "embedding", "hybrid"}:
        raise ValueError(f"Invalid skill retrieval_mode: {mode!r}")
    return mode


def _skill_sort_key(skill: DimensionSkill) -> tuple[int]:
    return (int(skill.version),)


def _merge_selected_by_quota(
    candidates: list[DimensionSkill],
    *,
    max_matched_skills: int,
    max_per_dimension: dict[str, int],
) -> list[DimensionSkill]:
    selected: list[DimensionSkill] = []
    seen: set[str] = set()
    used_by_dimension = {dimension: 0 for dimension in DIMENSIONS}
    for skill in candidates:
        if skill.skill_id in seen:
            continue
        dimension = skill.dimension if skill.dimension in DIMENSIONS else "general"
        if used_by_dimension[dimension] >= max_per_dimension.get(dimension, 1):
            continue
        selected.append(skill)
        seen.add(skill.skill_id)
        used_by_dimension[dimension] += 1
        if len(selected) >= max_matched_skills:
            break
    return selected


def _merge_hybrid_trace(
    query: str,
    selected: list[DimensionSkill],
    embedding_trace: dict[str, Any],
    lexical_trace: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    reasons: dict[str, Any] = {}
    for source, trace in (("embedding", embedding_trace), ("lexical", lexical_trace)):
        for skill_id, score in (trace.get("scores") or {}).items():
            try:
                scores[skill_id] = max(float(score), float(scores.get(skill_id, 0.0)))
            except (TypeError, ValueError):
                scores.setdefault(skill_id, 0.0)
        for skill_id, reason in (trace.get("match_reasons") or {}).items():
            reasons.setdefault(skill_id, {})[source] = reason
    filtered = []
    filtered.extend(embedding_trace.get("filtered_below_threshold") or [])
    filtered.extend(lexical_trace.get("filtered_below_threshold") or [])
    notes = ["hybrid retrieval = embedding candidates first, lexical candidates fill remaining slots"]
    notes.extend(embedding_trace.get("notes") or [])
    notes.extend(lexical_trace.get("notes") or [])
    return {
        "query": query,
        "retrieval_mode": "hybrid_dimension_skill_retrieval_v1",
        "candidate_skill_ids": list(
            dict.fromkeys(
                [
                    *(embedding_trace.get("candidate_skill_ids") or []),
                    *(lexical_trace.get("candidate_skill_ids") or []),
                ]
            )
        ),
        "selected_skill_ids": [skill.skill_id for skill in selected],
        "scores": scores,
        "match_reasons": reasons,
        "filtered_below_threshold": filtered,
        "notes": notes,
        "embedding_trace": embedding_trace,
        "lexical_trace": lexical_trace,
    }


def _taxonomy_for_dimension(dimension: str, value: str) -> dict[str, str]:
    taxonomy = {"project_type": "unknown", "fault_mode": "unknown", "strategy_type": "unknown"}
    if dimension in taxonomy:
        taxonomy[dimension] = value
    return taxonomy


def _apply_text_edit(skill: DimensionSkill, field: str, operation: str, content: dict[str, Any]) -> None:
    text = str(content.get("text") or "").strip()
    old_text = str(content.get("old_text") or "").strip()
    new_text = str(content.get("new_text") or text).strip()
    if field == "skill.knowledge":
        skill.knowledge = _edit_string(skill.knowledge, operation, text=text, old_text=old_text, new_text=new_text)
    elif field == "skill.trigger":
        skill.trigger = _edit_string(skill.trigger, operation, text=text, old_text=old_text, new_text=new_text)
    elif field == "retrieval_text":
        skill.retrieval_text = _edit_string(skill.retrieval_text, operation, text=text, old_text=old_text, new_text=new_text)
    elif field == "skill.anti_patterns":
        skill.anti_patterns = _edit_list(skill.anti_patterns, operation, text=text, old_text=old_text, new_text=new_text)
    else:
        raise ValueError(f"Unsupported edit target field: {field!r}")


def _edit_string(current: str, operation: str, *, text: str, old_text: str, new_text: str) -> str:
    if operation == "add":
        addition = text or new_text
        if not addition:
            raise ValueError("add edit requires text.")
        return f"{current.rstrip()}\n\n{addition}".strip() if current else addition
    if operation == "replace":
        if not old_text or not new_text:
            raise ValueError("replace edit requires exact old_text and new_text.")
        if old_text not in current:
            raise ValueError("replace edit old_text was not found in target.")
        return current.replace(old_text, new_text, 1)
    if operation == "delete":
        target = text or old_text
        if not target:
            raise ValueError("delete edit requires text or old_text.")
        if target not in current:
            raise ValueError("delete edit text was not found in target.")
        return current.replace(target, "").strip()
    raise ValueError(f"Unsupported string operation: {operation!r}")


def _edit_list(current: list[str], operation: str, *, text: str, old_text: str, new_text: str) -> list[str]:
    if operation == "add":
        addition = text or new_text
        if not addition:
            raise ValueError("add anti-pattern edit requires text.")
        return list(dict.fromkeys(current + [addition]))
    if operation == "replace":
        source = old_text or text
        replacement = new_text or text
        return [replacement if item == source else item for item in current]
    if operation == "delete":
        target = text or old_text
        return [item for item in current if item != target]
    raise ValueError(f"Unsupported list operation: {operation!r}")


def _provenance_from_edit(edit: dict[str, Any], *, created_at: str, updated_at: str) -> dict[str, Any]:
    source_cases = _as_list(edit.get("source_cases") or edit.get("supported_by_cases"))
    outcome = str(edit.get("outcome_type") or "")
    return {
        "supported_by_cases": source_cases,
        "supported_by_successes": source_cases if outcome == "success" else [],
        "supported_by_failures": source_cases if outcome == "failure" else [],
        "created_at": created_at,
        "updated_at": updated_at,
        "edit_history": [_edit_history_entry(edit)],
    }


def _merge_provenance(skill: DimensionSkill, edit: dict[str, Any]) -> None:
    source_cases = _as_list(edit.get("source_cases") or edit.get("supported_by_cases"))
    outcome = str(edit.get("outcome_type") or "")
    skill.supported_by_cases = list(dict.fromkeys(skill.supported_by_cases + source_cases))
    if outcome == "success":
        skill.supported_by_successes = list(dict.fromkeys(skill.supported_by_successes + source_cases))
    if outcome == "failure":
        skill.supported_by_failures = list(dict.fromkeys(skill.supported_by_failures + source_cases))
    skill.edit_history = list(skill.edit_history) + [_edit_history_entry(edit)]


def _edit_history_entry(edit: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": edit.get("operation"),
        "rationale": edit.get("rationale", ""),
        "evidence": edit.get("evidence", []),
        "risk": edit.get("risk", ""),
        "source_cases": _as_list(edit.get("source_cases") or edit.get("supported_by_cases")),
        "outcome_type": edit.get("outcome_type", ""),
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _title_from_value(value: str) -> str:
    return value.replace("_", " ").strip().title() or "Localization Knowledge"


def _unique_skill_id(existing: list[DimensionSkill], base: str) -> str:
    normalized = slugify(base, fallback="skill_v1")
    existing_ids = {skill.skill_id for skill in existing}
    if normalized not in existing_ids:
        return normalized
    index = 2
    while f"{normalized}_{index}" in existing_ids:
        index += 1
    return f"{normalized}_{index}"


def _build_dimension_slots(
    active_skills: list[DimensionSkill],
    matched_skills: list[dict[str, Any]],
    trace: dict[str, Any],
    *,
    weak_limit_per_dimension: int = 2,
) -> dict[str, Any]:
    active_by_id = {skill.skill_id: skill for skill in active_skills}
    matched_by_dimension: dict[str, list[dict[str, Any]]] = {dimension: [] for dimension in DIMENSIONS}
    for skill in matched_skills:
        dimension = skill.get("dimension") if skill.get("dimension") in DIMENSIONS else "general"
        matched_by_dimension[dimension].append(skill)

    scores = trace.get("scores", {})
    reasons = trace.get("match_reasons", {})
    weak_by_dimension: dict[str, list[dict[str, Any]]] = {dimension: [] for dimension in DIMENSIONS}
    for item in trace.get("filtered_below_threshold", []) or []:
        skill_id = str(item.get("skill_id") or "")
        skill = active_by_id.get(skill_id)
        if not skill:
            continue
        dimension = skill.dimension if skill.dimension in DIMENSIONS else "general"
        weak = skill.compact_dict(score=scores.get(skill_id, item.get("score")), match_reasons=reasons.get(skill_id, {}))
        weak["threshold"] = item.get("threshold")
        weak_by_dimension[dimension].append(weak)

    slots: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        weak_candidates = sorted(
            weak_by_dimension[dimension],
            key=lambda skill: (-float(skill.get("score") or 0.0), skill.get("skill_id", "")),
        )[:weak_limit_per_dimension]
        matched = sorted(
            matched_by_dimension[dimension],
            key=lambda skill: (-float(skill.get("score") or 0.0), skill.get("skill_id", "")),
        )
        if matched:
            status = "matched"
        elif weak_candidates:
            status = "weak"
        else:
            status = "missing"
        slots[dimension] = {
            "dimension": dimension,
            "status": status,
            "matched_skills": matched,
            "weak_candidates": weak_candidates,
            "guidance": (
                "update matched skill if this is the primary knowledge dimension"
                if status == "matched"
                else "update weak candidate only with clear evidence, otherwise create a new dimension skill"
                if status == "weak"
                else "create a new dimension skill only if this is the primary transferable lesson"
            ),
        }
    return slots
