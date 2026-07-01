from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any


TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_attempts: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or (os.getenv(api_key_env or "") if api_key_env else None)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.extra_body = extra_body or {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleClient":
        return cls(
            base_url=config.get("base_url", ""),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env"),
            model=config.get("model", ""),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 4096)),
            timeout=int(config.get("timeout", 120)),
            extra_body=config.get("extra_body") or {},
        )

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise ValueError("LLM base_url is empty. Configure llm.base_url or provider base_url_env.")
        if not self.api_key:
            raise ValueError("LLM API key is missing.")
        if not self.model:
            raise ValueError("LLM model is missing.")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        configured_extra_body = getattr(self, "extra_body", None)
        if configured_extra_body:
            body.update(configured_extra_body)
        if extra_body:
            body.update(extra_body)
        if response_format:
            body["response_format"] = response_format
        if tools:
            body["tools"] = tools
        if tools and tool_choice is not None:
            body["tool_choice"] = tool_choice

        last_error: Exception | None = None
        delays = [4, 8, 10]
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._post_chat(body)
            except Exception as exc:  # noqa: BLE001 - retry decision is explicit below.
                last_error = exc
                if attempt >= self.max_attempts or not _is_retryable(exc):
                    raise
                time.sleep(delays[min(attempt - 1, len(delays) - 1)])
        assert last_error is not None
        raise last_error

    def _post_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTPError {exc.code}: {detail}") from exc
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "raw": raw,
        }


class FakeLLMClient:
    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.responses:
            return {"content": "", "tool_calls": [], "raw": {"fake": True}}
        response = self.responses.pop(0)
        return {
            "content": response.get("content", ""),
            "tool_calls": response.get("tool_calls", []),
            "raw": response.get("raw", response),
        }


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout)):
        return True
    cause = exc.__cause__
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code in TRANSIENT_HTTP_STATUSES
    message = str(exc).lower()
    return any(marker in message for marker in ("timeout", "temporarily unavailable", "rate limit", "empty response"))
