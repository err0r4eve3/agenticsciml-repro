from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def complete_json_with_images(
        self,
        prompt: str,
        schema_name: str,
        image_paths: list[Path],
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("This LLM client does not support image inputs.")
