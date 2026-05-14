from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class LLMBudgetExceeded(RuntimeError):
    pass


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {raw!r}.")
    return value


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got {raw!r}.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {raw!r}.")
    return value


@dataclass(slots=True)
class LLMBudget:
    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_calls: int | None = None
    max_cost_usd: float | None = None
    cost_per_1k_tokens_usd: float | None = None
    calls_used: int = 0
    prompt_tokens_used: int = 0
    output_tokens_used: int = 0
    estimated_cost_usd: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMBudget":
        budget = cls(
            max_prompt_tokens=_optional_int_env("AGENTICSCIML_MAX_PROMPT_TOKENS"),
            max_output_tokens=_optional_int_env("AGENTICSCIML_MAX_OUTPUT_TOKENS"),
            max_total_tokens=_optional_int_env("AGENTICSCIML_MAX_TOTAL_TOKENS"),
            max_calls=_optional_int_env("AGENTICSCIML_MAX_LLM_CALLS"),
            max_cost_usd=_optional_float_env("AGENTICSCIML_MAX_COST_USD"),
            cost_per_1k_tokens_usd=_optional_float_env("AGENTICSCIML_COST_PER_1K_TOKENS_USD"),
        )
        if budget.max_cost_usd is not None and budget.cost_per_1k_tokens_usd is None:
            raise RuntimeError(
                "AGENTICSCIML_MAX_COST_USD requires AGENTICSCIML_COST_PER_1K_TOKENS_USD "
                "so cost can be enforced."
            )
        return budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_calls": self.max_calls,
            "max_cost_usd": self.max_cost_usd,
            "cost_per_1k_tokens_usd": self.cost_per_1k_tokens_usd,
            "calls_used": self.calls_used,
            "prompt_tokens_used": self.prompt_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    def check_before_call(self, prompt_tokens: int) -> None:
        next_call_count = self.calls_used + 1
        if self.max_calls is not None and next_call_count > self.max_calls:
            raise LLMBudgetExceeded(
                f"LLM call budget exceeded: next={next_call_count}, max={self.max_calls}"
            )
        if self.max_prompt_tokens is not None and self.prompt_tokens_used + prompt_tokens > self.max_prompt_tokens:
            raise LLMBudgetExceeded(
                "LLM prompt token budget exceeded: "
                f"next={self.prompt_tokens_used + prompt_tokens}, max={self.max_prompt_tokens}"
            )
        if self.max_total_tokens is not None and self.prompt_tokens_used + self.output_tokens_used + prompt_tokens > self.max_total_tokens:
            raise LLMBudgetExceeded(
                "LLM total token budget exceeded before call: "
                f"next={self.prompt_tokens_used + self.output_tokens_used + prompt_tokens}, "
                f"max={self.max_total_tokens}"
            )

    def reserve_call(self, prompt_tokens: int) -> None:
        self.check_before_call(prompt_tokens)
        if self.max_cost_usd is not None and self.cost_per_1k_tokens_usd is not None:
            next_total = self.prompt_tokens_used + self.output_tokens_used + prompt_tokens
            next_cost = next_total / 1000.0 * self.cost_per_1k_tokens_usd
            if next_cost > self.max_cost_usd:
                raise LLMBudgetExceeded(
                    f"LLM cost budget exceeded: estimated={next_cost:.6f}, max={self.max_cost_usd:.6f}"
                )
        self.calls_used += 1
        self.prompt_tokens_used += prompt_tokens
        self._refresh_cost()

    def record_response(self, *, output_tokens: int) -> None:
        self.output_tokens_used += output_tokens
        self._check_after_response()

    def record_call(self, *, prompt_tokens: int, output_tokens: int) -> None:
        self.reserve_call(prompt_tokens)
        self.record_response(output_tokens=output_tokens)

    def _refresh_cost(self) -> None:
        if self.cost_per_1k_tokens_usd is not None:
            total = self.prompt_tokens_used + self.output_tokens_used
            self.estimated_cost_usd = total / 1000.0 * self.cost_per_1k_tokens_usd

    def _check_after_response(self) -> None:
        if self.max_output_tokens is not None and self.output_tokens_used > self.max_output_tokens:
            raise LLMBudgetExceeded(
                f"LLM output token budget exceeded: used={self.output_tokens_used}, max={self.max_output_tokens}"
            )
        total = self.prompt_tokens_used + self.output_tokens_used
        if self.max_total_tokens is not None and total > self.max_total_tokens:
            raise LLMBudgetExceeded(f"LLM total token budget exceeded: used={total}, max={self.max_total_tokens}")
        self._refresh_cost()
        if self.max_cost_usd is not None and self.estimated_cost_usd > self.max_cost_usd:
            raise LLMBudgetExceeded(
                f"LLM cost budget exceeded: estimated={self.estimated_cost_usd:.6f}, max={self.max_cost_usd:.6f}"
            )
