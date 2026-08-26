from __future__ import annotations

from evolutefl.llm.client import OpenAICompatibleClient


def test_client_reads_configured_retry_attempts() -> None:
    client = OpenAICompatibleClient.from_config(
        {
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "test-model",
            "max_attempts": 7,
        }
    )

    assert client.max_attempts == 7


def test_client_forwards_configured_seed() -> None:
    client = OpenAICompatibleClient.from_config(
        {
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "test-model",
            "seed": 42,
        }
    )
    captured: dict = {}
    client._post_chat = lambda body: captured.update(body) or {"content": "", "tool_calls": [], "raw": {}}  # type: ignore[method-assign]

    client.chat(messages=[{"role": "user", "content": "ping"}])

    assert captured["seed"] == 42


def test_client_defaults_to_five_retry_attempts() -> None:
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
    )

    assert client.max_attempts == 5


def test_client_allows_process_local_transport_overrides(monkeypatch) -> None:
    monkeypatch.setenv("EVOLUTEFL_LLM_TIMEOUT", "300")
    monkeypatch.setenv("EVOLUTEFL_LLM_MAX_ATTEMPTS", "8")
    monkeypatch.setenv("EVOLUTEFL_LLM_MAX_TOKENS", "1024")

    client = OpenAICompatibleClient.from_config(
        {"base_url": "https://example.invalid/v1", "api_key": "test-key", "model": "test-model"}
    )

    assert client.timeout == 300
    assert client.max_attempts == 8
    assert client.max_tokens == 1024


def test_client_can_omit_explicit_tool_choice_for_compatible_reasoning_provider() -> None:
    client = OpenAICompatibleClient.from_config(
        {
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "test-model",
            "supports_tool_choice": False,
        }
    )
    captured: dict = {}
    client._post_chat = lambda body: captured.update(body) or {"content": "", "tool_calls": [], "raw": {}}  # type: ignore[method-assign]

    client.chat(
        messages=[{"role": "user", "content": "ping"}],
        tools=[{"type": "function", "function": {"name": "probe", "parameters": {}}}],
        tool_choice={"type": "function", "function": {"name": "probe"}},
    )

    assert captured["tools"][0]["function"]["name"] == "probe"
    assert "tool_choice" not in captured
