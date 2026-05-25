from __future__ import annotations

import sys
import types
from typing import Any

from agenticsciml.llm.openai_adapter import OpenAIAdapter


class FakeOpenAI:
    last_kwargs: dict[str, str] | None = None

    def __init__(self, **kwargs: str) -> None:
        FakeOpenAI.last_kwargs = kwargs


class FakeResponses:
    last_kwargs: dict[str, Any] | None = None

    def parse(self, **kwargs: Any) -> object:
        FakeResponses.last_kwargs = kwargs
        model = kwargs["text_format"]
        return types.SimpleNamespace(
            output_parsed=model(
                title="Native proposal",
                diagnosis="Root underfits.",
                mutation_plan=["Add features."],
                expected_effect="Lower validation MSE.",
                risks=["May overfit."],
            ),
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )


class FakeChatCompletions:
    response_text = "{}"
    last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> object:
        FakeChatCompletions.last_kwargs = kwargs
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.response_text))],
            usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )


class FakeNativeOpenAI(FakeOpenAI):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.responses = FakeResponses()


class FakeChatOpenAI(FakeOpenAI):
    def __init__(self, **kwargs: str) -> None:
        super().__init__(**kwargs)
        self.chat = types.SimpleNamespace(completions=FakeChatCompletions())


def test_openai_adapter_supports_base_url_env(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "deepseekv4pro")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "120")
    FakeOpenAI.last_kwargs = None

    adapter = OpenAIAdapter()

    assert adapter.model == "deepseek-v4-pro"
    assert adapter.base_url == "https://api.deepseek.com"
    assert adapter.timeout_s == 120.0
    assert FakeOpenAI.last_kwargs == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "timeout": 120.0,
    }
    assert adapter.provider_capabilities.adapter_type == "openai_compatible_chat"
    assert adapter.provider_capabilities.supports_structured_outputs is False
    assert adapter.provider_capabilities.supports_image_inputs is False


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
    assert adapter.provider_capabilities.adapter_type == "openai_native_responses"
    assert adapter.provider_capabilities.supports_structured_outputs is True
    assert adapter.provider_capabilities.supports_image_inputs is True


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


def test_openai_adapter_uses_native_structured_outputs_when_available(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeNativeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    FakeResponses.last_kwargs = None

    adapter = OpenAIAdapter(model="gpt-5-mini")
    payload = adapter.complete_json("return proposal", "proposal")

    assert payload["title"] == "Native proposal"
    assert FakeResponses.last_kwargs is not None
    assert FakeResponses.last_kwargs["text_format"].__name__ == "ProposalOutput"
    assert adapter.last_call_metadata is not None
    assert adapter.last_call_metadata["usage"]["completion_tokens"] == 5


def test_openai_adapter_passes_reasoning_effort_to_native_responses(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeNativeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    FakeResponses.last_kwargs = None

    adapter = OpenAIAdapter(model="gpt-5.5")
    adapter.complete_json("return proposal", "proposal", reasoning_effort="xhigh")

    assert FakeResponses.last_kwargs is not None
    assert FakeResponses.last_kwargs["reasoning"] == {"effort": "xhigh"}
    assert adapter.last_call_metadata is not None
    assert adapter.last_call_metadata["reasoning_effort"] == "xhigh"


def test_openai_adapter_compatible_json_fallback_still_rejects_schema_drift(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeChatOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    FakeChatCompletions.response_text = (
        '{"title":"Proposal","diagnosis":"x","mutation_plan":["y"],'
        '"expected_effect":"z","risks":[],"extra":"drift"}'
    )

    adapter = OpenAIAdapter(model="deepseekv4pro")

    try:
        adapter.complete_json("return proposal", "proposal")
    except ValueError as exc:
        assert "extra_forbidden" in str(exc)
    else:
        raise AssertionError("OpenAI-compatible fallback accepted schema drift")


def test_openai_adapter_compatible_json_prompt_includes_schema_types(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeChatOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    FakeChatCompletions.last_kwargs = None
    FakeChatCompletions.response_text = (
        '{"title":"Proposal","diagnosis":"x","mutation_plan":["y"],'
        '"expected_effect":"z","risks":["r"]}'
    )

    adapter = OpenAIAdapter(model="deepseekv4pro")
    payload = adapter.complete_json("return proposal", "proposal", reasoning_effort="xhigh")

    assert payload["risks"] == ["r"]
    assert FakeChatCompletions.last_kwargs is not None
    assert FakeChatCompletions.last_kwargs["reasoning_effort"] == "xhigh"
    message = FakeChatCompletions.last_kwargs["messages"][-1]["content"]
    assert "Required JSON Schema" in message
    assert '"risks"' in message
    assert '"type": "array"' in message
    assert "Arrays must be JSON arrays" in message
    assert adapter.last_call_metadata is not None
    assert adapter.last_call_metadata["reasoning_effort"] == "xhigh"


def test_openai_adapter_compatible_json_extracts_object_from_wrapped_text(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeChatOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    FakeChatCompletions.response_text = (
        "Here is the JSON:\n```json\n"
        '{"summary":"ok","strengths":["s"],"weaknesses":["w"],"next_steps":["n"]}'
        "\n```"
    )

    adapter = OpenAIAdapter(model="deepseek-v4-pro")
    payload = adapter.complete_json("return analysis", "analysis")

    assert payload["next_steps"] == ["n"]
