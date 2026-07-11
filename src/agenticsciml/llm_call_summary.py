from __future__ import annotations

import math
from typing import Any, Iterable


def summarize_llm_call_events(
    events: Iterable[dict[str, Any]],
    *,
    use_mock: bool,
) -> dict[str, object]:
    """Derive exported LLM call diagnostics from generation trace events."""
    by_role: dict[str, int] = {}
    total = 0
    prompt_token_estimate = 0
    response_token_estimate = 0
    duration_s = 0.0
    generation_attempt_count = 0
    generation_attempt_duration_s = 0.0
    unbound_generation_attempt_count = 0
    pre_provider_rejection_count = 0
    provider_usage_call_count = 0
    provider_prompt_tokens = 0
    provider_completion_tokens = 0
    provider_total_tokens = 0

    for event in events:
        if event.get("event_type") != "generation_span":
            continue
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        generation_attempt_count += 1
        raw_duration_s = metadata.get("duration_s", 0.0)
        if (
            not isinstance(raw_duration_s, (int, float))
            or isinstance(raw_duration_s, bool)
            or not math.isfinite(float(raw_duration_s))
            or raw_duration_s < 0
        ):
            raise ValueError("generation duration_s must be a finite non-negative number")
        attempt_duration_s = float(raw_duration_s)
        generation_attempt_duration_s += attempt_duration_s
        if not use_mock and not isinstance(metadata.get("llm_call_id"), str):
            unbound_generation_attempt_count += 1
            if metadata.get("error_type") == "LLMBudgetExceeded":
                pre_provider_rejection_count += 1
            continue
        role = str(metadata.get("spec_role") or event.get("name") or "unknown")
        by_role[role] = by_role.get(role, 0) + 1
        total += 1
        prompt_token_estimate += _non_negative_trace_int(
            metadata.get("prompt_token_estimate", 0),
            field="prompt_token_estimate",
        )
        response_token_estimate += _non_negative_trace_int(
            metadata.get("response_token_estimate", 0),
            field="response_token_estimate",
        )
        duration_s += attempt_duration_s
        usage = metadata.get("usage")
        if isinstance(usage, dict) and all(
            isinstance(usage.get(field), int)
            and not isinstance(usage.get(field), bool)
            and int(usage[field]) >= 0
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            provider_usage_call_count += 1
            provider_prompt_tokens += int(usage["prompt_tokens"])
            provider_completion_tokens += int(usage["completion_tokens"])
            provider_total_tokens += int(usage["total_tokens"])

    return {
        "total": total,
        "by_role": dict(sorted(by_role.items())),
        "generation_attempt_count": generation_attempt_count,
        "generation_attempt_duration_s": generation_attempt_duration_s,
        "unbound_generation_attempt_count": unbound_generation_attempt_count,
        "pre_provider_rejection_count": pre_provider_rejection_count,
        "prompt_token_estimate": prompt_token_estimate,
        "response_token_estimate": response_token_estimate,
        "provider_usage": {
            "call_count": provider_usage_call_count,
            "complete": total > 0 and provider_usage_call_count == total,
            "prompt_tokens": provider_prompt_tokens,
            "completion_tokens": provider_completion_tokens,
            "total_tokens": provider_total_tokens,
        },
        "duration_s": duration_s,
    }


def _non_negative_trace_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"generation {field} must be a non-negative integer")
    return value
