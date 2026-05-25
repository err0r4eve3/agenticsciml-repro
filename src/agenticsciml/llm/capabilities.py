from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider: str
    adapter_type: str
    supports_responses: bool
    supports_structured_outputs: bool
    supports_image_inputs: bool
    supports_usage: bool
    supports_trace_export: bool
    supports_prompt_cache: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "adapter_type": self.adapter_type,
            "supports_responses": self.supports_responses,
            "supports_structured_outputs": self.supports_structured_outputs,
            "supports_image_inputs": self.supports_image_inputs,
            "supports_usage": self.supports_usage,
            "supports_trace_export": self.supports_trace_export,
            "supports_prompt_cache": self.supports_prompt_cache,
        }


def capabilities_for_openai_compatible(base_url: str | None) -> ProviderCapabilities:
    if base_url:
        parsed = urlparse(base_url)
        provider = parsed.netloc or base_url
        return ProviderCapabilities(
            provider=provider,
            adapter_type="openai_compatible_chat",
            supports_responses=False,
            supports_structured_outputs=False,
            supports_image_inputs=False,
            supports_usage=True,
            supports_trace_export=False,
            supports_prompt_cache=False,
        )
    return ProviderCapabilities(
        provider="openai",
        adapter_type="openai_native_responses",
        supports_responses=True,
        supports_structured_outputs=True,
        supports_image_inputs=True,
        supports_usage=True,
        supports_trace_export=True,
        supports_prompt_cache=True,
    )
