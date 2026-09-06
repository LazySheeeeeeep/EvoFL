from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolutefl.embedding import EmbeddingClient
from evolutefl.json_utils import read_jsonl, write_jsonl

from .retrieval import DEFAULT_SKILL_TYPE_QUOTA
from .schema import SKILL_TYPES, DimensionSkill


DEFAULT_EMBEDDING_CACHE = Path("skill_pools/skill_bank_v0/embeddings/jina_v3/skill_embeddings.jsonl")
DEFAULT_EMBEDDING_MIN_SCORE = 0.48
EMBEDDING_TRACE_LIMIT = 10


@dataclass(frozen=True)
class EmbeddingRetrievalConfig:
    cache_path: Path = DEFAULT_EMBEDDING_CACHE
    model_id: str = "jina-embeddings-v3"
    query_task: str = "retrieval.query"
    document_task: str = "retrieval.passage"
    min_score: float = DEFAULT_EMBEDDING_MIN_SCORE
    fallback_to_lexical: bool = False
    batch_size: int = 16


def skill_embedding_text(skill: DimensionSkill) -> str:
    """Return the complete, bounded Skill card used for semantic retrieval.

    Runtime injection remains selective, but the embedding needs the card's
    role-level knowledge to distinguish similarly named subsystem boundaries.
    Issue cards are maintained as atomic propositions, so their numeric IDs
    remain storage metadata and only proposition text enters this document.
    """

    if skill.skill_type == "project_skill":
        parts = [
            "skill_type: project_skill",
            f"project_type: {skill.value}",
            f"title: {skill.title}",
            f"architecture_signature: {skill.trigger}",
            "knowledge:\n- " + "\n- ".join(skill.knowledge_texts),
        ]
    elif skill.skill_type == "fault_skill":
        parts = [
            "skill_type: fault_skill",
            f"fault_family: {skill.fault_family}",
            f"fault_subtype: {skill.fault_subtype}",
            f"title: {skill.title}",
            f"applicability_trigger: {skill.trigger}",
            "knowledge:\n- " + "\n- ".join(skill.knowledge_texts),
        ]
    else:
        parts = [
            "skill_type: strategy_skill",
            f"value: {skill.value}",
            f"title: {skill.title}",
            f"applicability_trigger: {skill.trigger}",
            "knowledge:\n- " + "\n- ".join(skill.knowledge_texts),
        ]
    return "\n".join(part for part in parts if part.strip())


def query_embedding_text(repo: str, issue: str) -> str:
    return f"repo: {repo}\nissue: {issue}"


class SkillEmbeddingStore:
    def __init__(self, path: str | Path, *, model_id: str = "jina-embeddings-v3") -> None:
        self.path = Path(path)
        self.model_id = model_id

    def load(self) -> dict[tuple[str, int, str, str], dict[str, Any]]:
        records = read_jsonl(self.path)
        cache: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        for record in records:
            try:
                key = (
                    str(record["skill_id"]),
                    int(record["version"]),
                    str(record["text_hash"]),
                    str(record.get("model_id") or self.model_id),
                )
                cache[key] = record
            except (KeyError, TypeError, ValueError):
                continue
        return cache

    def ensure_skill_embeddings(
        self,
        skills: list[DimensionSkill],
        client: EmbeddingClient,
        *,
        task: str = "retrieval.passage",
        batch_size: int = 16,
        prune_to_skills: bool = False,
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        cache = self.load()
        records_by_key = {} if prune_to_skills else dict(cache)
        embeddings: dict[str, list[float]] = {}
        missing: list[tuple[DimensionSkill, str, str]] = []
        requested_keys: set[tuple[str, int, str, str]] = set()
        for skill in skills:
            text = skill_embedding_text(skill)
            text_hash = _hash_text(text)
            key = (skill.skill_id, skill.version, text_hash, self.model_id)
            requested_keys.add(key)
            cached = cache.get(key)
            if cached and isinstance(cached.get("embedding"), list):
                records_by_key[key] = cached
                embeddings[skill.skill_id] = [float(value) for value in cached["embedding"]]
            else:
                missing.append((skill, text, text_hash))

        for start in range(0, len(missing), max(1, batch_size)):
            batch = missing[start : start + max(1, batch_size)]
            vectors = client.embed_texts([item[1] for item in batch], task=task)
            for (skill, text, text_hash), vector in zip(batch, vectors):
                key = (skill.skill_id, skill.version, text_hash, self.model_id)
                record = {
                    "skill_id": skill.skill_id,
                    "version": skill.version,
                    "skill_type": skill.skill_type,
                    "value": skill.value,
                    "model_id": self.model_id,
                    "text_hash": text_hash,
                    "text_preview": text[:400],
                    "embedding": [float(value) for value in vector],
                }
                records_by_key[key] = record
                embeddings[skill.skill_id] = record["embedding"]

        pruned_count = len([key for key in cache if key not in requested_keys]) if prune_to_skills else 0
        if missing or pruned_count:
            write_jsonl(self.path, list(records_by_key.values()))
        return embeddings, {
            "cache_path": str(self.path),
            "cached_count": len(skills) - len(missing),
            "generated_count": len(missing),
            "pruned_count": pruned_count,
            "model_id": self.model_id,
        }


def select_embedding_skills(
    *,
    query_text: str,
    skills: list[DimensionSkill],
    client: EmbeddingClient,
    config: EmbeddingRetrievalConfig,
    max_matched_skills: int = 5,
    max_per_skill_type: dict[str, int] | None = None,
) -> tuple[list[DimensionSkill], dict[str, Any]]:
    if not skills:
        return [], _empty_trace(query_text, config, note="No active skills.")
    store = SkillEmbeddingStore(config.cache_path, model_id=config.model_id)
    skill_embeddings, cache_trace = store.ensure_skill_embeddings(
        skills,
        client,
        task=config.document_task,
        batch_size=config.batch_size,
    )
    query_embedding = client.embed_texts([query_text], task=config.query_task)[0]
    scored: list[tuple[float, DimensionSkill]] = []
    for skill in skills:
        vector = skill_embeddings.get(skill.skill_id)
        if vector is None:
            continue
        scored.append((cosine_similarity(query_embedding, vector), skill))
    scored.sort(key=lambda pair: (-pair[0], pair[1].skill_type, pair[1].skill_id))
    top_scores = [
        {
            "skill_id": skill.skill_id,
            "score": score,
            "skill_type": skill.skill_type,
            "value": skill.value,
        }
        for score, skill in scored[:EMBEDDING_TRACE_LIMIT]
    ]
    quotas = {**DEFAULT_SKILL_TYPE_QUOTA, **(max_per_skill_type or {})}
    selected: list[DimensionSkill] = []
    used_by_skill_type = {skill_type: 0 for skill_type in SKILL_TYPES}
    filtered: list[dict[str, Any]] = []
    for score, skill in scored:
        if score < config.min_score:
            if score > 0:
                filtered.append({"skill_id": skill.skill_id, "score": score, "threshold": config.min_score})
            continue
        skill_type = skill.skill_type
        if skill_type not in SKILL_TYPES or used_by_skill_type[skill_type] >= quotas.get(skill_type, 1):
            continue
        selected.append(skill)
        used_by_skill_type[skill_type] += 1
        if len(selected) >= max_matched_skills:
            break
    trace = {
        "query": query_text,
        "retrieval_mode": "embedding_skill_type_retrieval_v1",
        "candidate_skill_ids": [skill.skill_id for score, skill in scored if score >= config.min_score],
        "selected_skill_ids": [skill.skill_id for skill in selected],
        "top_scores": top_scores,
        "scores": {skill.skill_id: score for score, skill in scored},
        "match_reasons": {
            skill.skill_id: {
                "embedding_score": score,
                "threshold": config.min_score,
                "skill_type": skill.skill_type,
                "value": skill.value,
            }
            for score, skill in scored
        },
        "filtered_below_threshold": filtered,
        "notes": [f"embedding_min_score={config.min_score}", f"cache={cache_trace['cache_path']}"],
        "embedding_cache": cache_trace,
    }
    if not selected:
        trace["notes"].append("No embedding skill match found.")
    return selected, trace


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _empty_trace(query_text: str, config: EmbeddingRetrievalConfig, *, note: str) -> dict[str, Any]:
    return {
        "query": query_text,
        "retrieval_mode": "embedding_skill_type_retrieval_v1",
        "candidate_skill_ids": [],
        "selected_skill_ids": [],
        "scores": {},
        "match_reasons": {},
        "filtered_below_threshold": [],
        "notes": [note, f"embedding_min_score={config.min_score}"],
    }
