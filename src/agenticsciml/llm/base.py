from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        raise NotImplementedError
