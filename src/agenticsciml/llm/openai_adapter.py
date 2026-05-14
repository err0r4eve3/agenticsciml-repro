from __future__ import annotations

import json
import os
from typing import Any

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
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.timeout_s = timeout_s if timeout_s is not None else _timeout_from_env()
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
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        json_prompt = (
            f"{prompt}\n\nReturn only valid JSON for schema '{schema_name}'. "
            "Do not include markdown fences."
        )
        text = self.complete_text(json_prompt, system=system, temperature=temperature)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON for {schema_name}: {text[:300]}") from exc
