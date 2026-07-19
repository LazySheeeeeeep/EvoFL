from __future__ import annotations

from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.embedding.config import make_embedding_client

from .bank import SkillBankV0
from .embedding_store import DEFAULT_EMBEDDING_MIN_SCORE


def make_skill_bank(config: dict[str, Any]) -> SkillBankV0:
    bank_cfg = config.get("skill_bank", {}) or {}
    embedding_cfg = config.get("embedding", {}) or {}
    return SkillBankV0(
        resolve_path(bank_cfg.get("path", "skill_pools/skill_bank_v0/skills.jsonl")),
        max_matched_skills=int(bank_cfg.get("max_matched_skills", 5)),
        max_per_skill_type=bank_cfg.get("max_per_skill_type") or bank_cfg.get("max_per_dimension"),
        min_score=float(bank_cfg.get("min_score", 4.0)),
        project_skill_min_score=float(
            bank_cfg.get("project_skill_min_score", bank_cfg.get("project_type_min_score", 6.0))
        ),
        retrieval_mode=str(bank_cfg.get("retrieval_mode", "lexical")),
        embedding_client=make_embedding_client(config),
        embedding_cache_path=_resolve_optional_path(
            embedding_cfg.get("cache_path") or bank_cfg.get("embedding_cache_path")
        ),
        embedding_model_id=str(embedding_cfg.get("model") or bank_cfg.get("embedding_model_id") or "jina-embeddings-v3"),
        embedding_query_task=str(embedding_cfg.get("query_task") or bank_cfg.get("embedding_query_task") or "retrieval.query"),
        embedding_document_task=str(embedding_cfg.get("document_task") or bank_cfg.get("embedding_document_task") or "retrieval.passage"),
        embedding_min_score=float(
            bank_cfg.get("embedding_min_score", embedding_cfg.get("min_score", DEFAULT_EMBEDDING_MIN_SCORE))
        ),
        embedding_fallback_to_lexical=bool(bank_cfg.get("embedding_fallback_to_lexical", False)),
        embedding_batch_size=int(embedding_cfg.get("batch_size", bank_cfg.get("embedding_batch_size", 16))),
    )


def _resolve_optional_path(path: Any) -> Path | None:
    if not path:
        return None
    return resolve_path(path)
