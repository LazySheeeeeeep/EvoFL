from __future__ import annotations

import json
import multiprocessing
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any


TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}


class LLMHTTPError(RuntimeError):
    """OpenAI-compatible HTTP error that retains the provider status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = int(status_code)
        super().__init__(f"LLM HTTPError {self.status_code}: {detail}")


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
        max_attempts: int = 5,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
        supports_tool_choice: bool = True,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or (os.getenv(api_key_env or "") if api_key_env else None)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.seed = seed
        self.extra_body = extra_body or {}
        # Some OpenAI-compatible reasoning endpoints accept ``tools`` but
        # reject an explicit ``tool_choice`` field.  The agent can still
        # constrain those turns by exposing only the required tool.
        self.supports_tool_choice = bool(supports_tool_choice)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleClient":
        return cls(
            base_url=config.get("base_url", ""),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env"),
            model=config.get("model", ""),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(os.getenv("EVOLUTEFL_LLM_MAX_TOKENS", config.get("max_tokens", 4096))),
            timeout=int(os.getenv("EVOLUTEFL_LLM_TIMEOUT", config.get("timeout", 120))),
            max_attempts=int(os.getenv("EVOLUTEFL_LLM_MAX_ATTEMPTS", config.get("max_attempts", 5))),
            seed=_optional_int(config.get("seed")),
            extra_body=config.get("extra_body") or {},
            supports_tool_choice=bool(config.get("supports_tool_choice", True)),
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
        seed: int | None = None,
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
        request_seed = self.seed if seed is None else seed
        if request_seed is not None:
            body["seed"] = request_seed
        configured_extra_body = getattr(self, "extra_body", None)
        if configured_extra_body:
            body.update(configured_extra_body)
        if extra_body:
            body.update(extra_body)
        if response_format:
            body["response_format"] = response_format
        if tools:
            body["tools"] = tools
        if tools and tool_choice is not None and self.supports_tool_choice:
            body["tool_choice"] = tool_choice

        last_error: Exception | None = None
        # Transport retries resend the identical request. They deliberately do
        # not append messages or alter the agent protocol.
        delays = [4, 8, 16, 32]
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
        raw = _post_json_with_deadline(
            base_url=self.base_url,
            api_key=str(self.api_key),
            body=body,
            timeout=self.timeout,
        )
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "raw": raw,
        }


def _post_json_with_deadline(
    *,
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: int | float,
) -> dict[str, Any]:
    """Make one request with a killable POSIX wall-clock deadline.

    ``urllib``'s socket timeout and SIGALRM both proved insufficient with some
    compatible proxies: a half-open TLS response can remain in ``select`` past
    the configured deadline.  A small forked worker gives the WSL parent an
    unambiguous way to terminate only the stuck request. Windows retains the
    direct implementation because ``fork`` is unavailable there.
    """

    # Long-lived local training runs may already own native library worker
    # threads (for example an embedding client). Forking once per HTTP call
    # in that state can fail before the child writes its pipe response. Keep
    # the killable-worker path as the default, but allow batch runners to opt
    # into urllib's normal socket deadline explicitly.
    if os.name != "posix" or os.getenv("EVOLUTEFL_DIRECT_HTTP_TIMEOUT") == "1":
        return _post_json_direct(base_url, api_key, body, timeout)

    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_post_json_worker,
        args=(child_connection, base_url, api_key, body, timeout),
        daemon=True,
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(float(timeout)):
            process.terminate()
            process.join(timeout=2)
            raise TimeoutError(f"LLM request exceeded {float(timeout):g} seconds.")
        status, payload = parent_connection.recv()
    except EOFError as exc:
        raise RuntimeError("LLM transport worker exited without a response.") from exc
    finally:
        parent_connection.close()
        if process.is_alive():
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    if status == "ok":
        return payload
    if status == "http_error":
        raise LLMHTTPError(payload["status_code"], payload["detail"])
    if status == "transport_error":
        raise RuntimeError(f"LLM transport error: {payload}")
    raise RuntimeError(f"LLM request worker failed: {payload}")


def _post_json_worker(
    connection: Any,
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: int | float,
) -> None:
    try:
        connection.send(("ok", _post_json_direct(base_url, api_key, body, timeout)))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        connection.send(("http_error", {"status_code": exc.code, "detail": detail}))
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        connection.send(("transport_error", str(exc)))
    except Exception as exc:  # noqa: BLE001 - parent applies retry policy.
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _post_json_direct(
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout: int | float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
    if isinstance(exc, LLMHTTPError):
        return exc.status_code in TRANSIENT_HTTP_STATUSES
    if isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout)):
        return True
    cause = exc.__cause__
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code in TRANSIENT_HTTP_STATUSES
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timeout",
            "temporarily unavailable",
            "rate limit",
            "empty response",
            "transport error",
        )
    )


def _optional_int(value: Any) -> int | None:
    """Normalize an optional configured seed without treating zero as absent."""
    if value is None or value == "":
        return None
    return int(value)
