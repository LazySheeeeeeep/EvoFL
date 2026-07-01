from __future__ import annotations

import os
from typing import Any

from .client import HttpEmbeddingClient


def make_embedding_client(config: dict[str, Any]) -> HttpEmbeddingClient | None:
    embedding_cfg = config.get("embedding", {}) or {}
    if not embedding_cfg.get("enabled", False):
        return None
    base_url = str(embedding_cfg.get("base_url") or "")
    base_url_env = embedding_cfg.get("base_url_env")
    if base_url_env and os.getenv(str(base_url_env)):
        base_url = os.environ[str(base_url_env)]
    if not base_url:
        return None
    return HttpEmbeddingClient(
        base_url=base_url,
        endpoint=str(embedding_cfg.get("endpoint") or "/embed"),
        model=str(embedding_cfg.get("model") or "jina-embeddings-v3"),
        api_key=embedding_cfg.get("api_key"),
        api_key_env=embedding_cfg.get("api_key_env"),
        timeout=float(embedding_cfg.get("timeout", 120)),
    )
