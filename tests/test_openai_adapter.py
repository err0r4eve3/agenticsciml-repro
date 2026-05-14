from __future__ import annotations

import sys
import types

from agenticsciml.llm.openai_adapter import OpenAIAdapter


class FakeOpenAI:
    last_kwargs: dict[str, str] | None = None

    def __init__(self, **kwargs: str) -> None:
        FakeOpenAI.last_kwargs = kwargs


def test_openai_adapter_supports_base_url_env(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "deepseekv4pro")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "120")
    FakeOpenAI.last_kwargs = None

    adapter = OpenAIAdapter()

    assert adapter.model == "deepseekv4pro"
    assert adapter.base_url == "https://api.deepseek.com"
    assert adapter.timeout_s == 120.0
    assert FakeOpenAI.last_kwargs == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "timeout": 120.0,
    }


def test_openai_adapter_omits_base_url_when_unset(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_TIMEOUT_S", raising=False)
    FakeOpenAI.last_kwargs = None

    adapter = OpenAIAdapter(model="gpt-5-mini")

    assert adapter.base_url is None
    assert adapter.timeout_s == 60.0
    assert FakeOpenAI.last_kwargs == {"api_key": "test-key", "timeout": 60.0}


def test_openai_adapter_rejects_invalid_timeout_env(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "0")

    try:
        OpenAIAdapter()
    except RuntimeError as exc:
        assert "OPENAI_TIMEOUT_S must be positive" in str(exc)
    else:
        raise AssertionError("OpenAIAdapter accepted non-positive timeout")
