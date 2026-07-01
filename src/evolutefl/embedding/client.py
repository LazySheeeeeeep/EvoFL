from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class EmbeddingClient:
    def embed_texts(self, texts: list[str], *, task: str | None = None) -> list[list[float]]:
        """Return one embedding per input text."""
        raise NotImplementedError


class NullEmbeddingClient:
    def embed_texts(self, texts: list[str], *, task: str | None = None) -> list[list[float]]:
        raise RuntimeError("Embedding client is not configured.")


class HttpEmbeddingClient:
    """Small HTTP client for a local Jina-v3 embedding service.

    The default payload targets a tiny local FastAPI wrapper:
    {"texts": [...], "task": "retrieval.passage", "model": "..."}

    The response parser also accepts OpenAI-compatible embeddings:
    {"data": [{"embedding": [...]}, ...]}.
    """

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str = "/embed",
        model: str = "jina-embeddings-v3",
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.model = model
        self.api_key = api_key or (os.getenv(api_key_env) if api_key_env else None)
        self.timeout = timeout

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def embed_texts(self, texts: list[str], *, task: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any]
        if self.endpoint.endswith("/embeddings"):
            payload = {"input": texts, "model": self.model}
            if task:
                payload["task"] = task
        else:
            payload = {"texts": texts, "model": self.model}
            if task:
                payload["task"] = task
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc
        data = json.loads(raw)
        embeddings = _extract_embeddings(data)
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding service returned {len(embeddings)} embeddings for {len(texts)} texts.")
        return embeddings


def _extract_embeddings(data: Any) -> list[list[float]]:
    if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
        return [_coerce_embedding(item) for item in data["embeddings"]]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [_coerce_embedding(item.get("embedding")) for item in data["data"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [_coerce_embedding(item) for item in data]
    raise RuntimeError("Embedding response must contain 'embeddings' or OpenAI-compatible 'data'.")


def _coerce_embedding(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise RuntimeError("Embedding vector must be a list.")
    return [float(item) for item in value]
