from __future__ import annotations

import json
import os
from typing import Any

from agenticsciml.agents.output_schemas import (
    json_schema_for,
    output_model_for,
    validate_output_payload,
)
from agenticsciml.llm.capabilities import ProviderCapabilities, capabilities_for_openai_compatible
from agenticsciml.llm.base import LLMClient


def _timeout_from_env() -> float:
    raw = os.environ.get("OPENAI_TIMEOUT_S", "60")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"OPENAI_TIMEOUT_S must be numeric, got {raw!r}.") from exc
    if timeout <= 0:
        raise RuntimeError(f"OPENAI_TIMEOUT_S must be positive, got {raw!r}.")
    return timeout


class OpenAIAdapter(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ):
        raw_model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = _normalize_model_name(raw_model, self.base_url)
        self.timeout_s = timeout_s if timeout_s is not None else _timeout_from_env()
        self.provider_capabilities: ProviderCapabilities = capabilities_for_openai_compatible(self.base_url)
        self.provider_name = self.provider_capabilities.provider
        self.adapter_type = self.provider_capabilities.adapter_type
        self.last_call_metadata: dict[str, Any] | None = None
        if not self.api_key:
            raise RuntimeError("Real LLM mode requires OPENAI_API_KEY.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the real-llm extra to use OpenAIAdapter.") from exc
        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout_s}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort
        response = self.client.chat.completions.create(**request_kwargs)
        self.last_call_metadata = self._metadata(
            method="complete_text",
            schema_name=None,
            response=response,
            reasoning_effort=reasoning_effort,
        )
        return response.choices[0].message.content or ""

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        if (
            self.provider_capabilities.supports_responses
            and self.provider_capabilities.supports_structured_outputs
            and output_model_for(schema_name) is not None
        ):
            return self._complete_json_responses(
                prompt,
                schema_name,
                system=system,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        return self._complete_json_compatible(
            prompt,
            schema_name,
            system=system,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    def _complete_json_responses(
        self,
        prompt: str,
        schema_name: str,
        system: str | None,
        temperature: float,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        model = output_model_for(schema_name)
        if model is None:
            return self._complete_json_compatible(
                prompt,
                schema_name,
                system=system,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": _response_input(prompt, system),
            "temperature": temperature,
            "text_format": model,
        }
        if reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        response = self.client.responses.parse(**request_kwargs)
        self.last_call_metadata = self._metadata(
            method="complete_json",
            schema_name=schema_name,
            response=response,
            reasoning_effort=reasoning_effort,
        )
        parsed = _extract_parsed_response(response)
        if parsed is None:
            raise RuntimeError(f"Model did not return parsed structured output for {schema_name}.")
        if hasattr(parsed, "model_dump"):
            payload = parsed.model_dump(exclude_none=True)
        elif isinstance(parsed, dict):
            payload = parsed
        else:
            raise RuntimeError(f"Parsed structured output for {schema_name} has unsupported type.")
        return validate_output_payload(payload, schema_name=schema_name)

    def _complete_json_compatible(
        self,
        prompt: str,
        schema_name: str,
        system: str | None,
        temperature: float,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        json_prompt = _compatible_json_prompt(prompt, schema_name)
        text = self.complete_text(
            json_prompt,
            system=system,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        json_text = _extract_json_object_text(text)
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON for {schema_name}: {text[:300]}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Model JSON for {schema_name} must be an object.")
        return validate_output_payload(payload, schema_name=schema_name)

    def _metadata(
        self,
        *,
        method: str,
        schema_name: str | None,
        response: Any,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        metadata = {
            "provider": self.provider_name,
            "model": self.model,
            "method": method,
            "schema_name": schema_name,
            "adapter_type": self.adapter_type,
            "provider_capabilities": self.provider_capabilities.to_dict(),
            "usage": _usage_metadata(response),
        }
        if reasoning_effort is not None:
            metadata["reasoning_effort"] = reasoning_effort
        return metadata


def _response_input(prompt: str, system: str | None) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if system:
        items.append({"role": "system", "content": system})
    items.append({"role": "user", "content": prompt})
    return items


def _normalize_model_name(model: str, base_url: str | None) -> str:
    if not base_url or "api.deepseek.com" not in base_url:
        return model
    aliases = {
        "deepseekv4pro": "deepseek-v4-pro",
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseekv4flash": "deepseek-v4-flash",
        "deepseek-v4-flash": "deepseek-v4-flash",
    }
    return aliases.get(model, model)


def _compatible_json_prompt(prompt: str, schema_name: str) -> str:
    schema = json_schema_for(schema_name)
    rules = [
        "Return exactly one valid JSON object and nothing else.",
        "Do not include markdown fences, comments, prose, or trailing text.",
        "Honor every JSON Schema type exactly.",
        "Arrays must be JSON arrays, even when there is only one item.",
        "Use [] for empty arrays; never encode arrays as strings or numbered paragraphs.",
        "Do not include fields that are not listed in the schema.",
    ]
    schema_block = ""
    if schema is not None:
        schema_block = (
            "\n\n## Required JSON Schema\n\n"
            f"{json.dumps(schema, indent=2, sort_keys=True)}"
        )
    return (
        f"{prompt}\n\n"
        f"## JSON Output Rules for schema '{schema_name}'\n\n"
        + "\n".join(f"- {rule}" for rule in rules)
        + schema_block
    )


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    if start == -1:
        return stripped

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped


def _extract_parsed_response(response: Any) -> Any:
    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        return output_parsed
    for output in getattr(response, "output", []) or []:
        for item in getattr(output, "content", []) or []:
            if getattr(item, "type", None) == "refusal":
                refusal = getattr(item, "refusal", "")
                raise RuntimeError(f"Model refused structured output request: {refusal}")
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                return parsed
    return None


def _usage_metadata(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    fields = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    result: dict[str, int] = {}
    for target, candidates in fields.items():
        for candidate in candidates:
            value = getattr(usage, candidate, None)
            if isinstance(value, int) and not isinstance(value, bool):
                result[target] = value
                break
    return result
