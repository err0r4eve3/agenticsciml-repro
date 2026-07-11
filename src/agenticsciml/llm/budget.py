from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable

from agenticsciml.llm.base import LLMClient


class LLMBudgetExceeded(RuntimeError):
    pass


class LLMBudgetPreflightError(RuntimeError):
    pass


@dataclass(slots=True)
class _RecordingState:
    ledger_path: Path
    budget: "LLMBudget"
    lock: threading.RLock
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    call_count: int = 0
    budget_calls_offset: int = 0
    budget_prompt_tokens_offset: int = 0
    budget_output_tokens_offset: int = 0


class RecordingLLMClient(LLMClient):
    """Budget-enforcing LLM wrapper with a secret-free JSONL call ledger.

    Child wrappers created with :meth:`wrap_child` share the same budget,
    ledger, lock, and call-id sequence. This lets orchestrators route roles to
    different provider/model adapters without weakening the run-level budget.
    """

    def __init__(
        self,
        inner: LLMClient,
        ledger_path: Path,
        budget: "LLMBudget",
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        _state: _RecordingState | None = None,
    ) -> None:
        self.inner = inner
        self.provider = _llm_provider_name(inner)
        self.provider_name = self.provider
        self.model = getattr(inner, "model", None) or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        self.adapter_type = getattr(inner, "adapter_type", type(inner).__name__)
        self.provider_capabilities = _llm_provider_capabilities(inner)
        self._state = _state or _load_recording_state(
            Path(ledger_path),
            budget,
            progress_callback=progress_callback,
        )
        self.budget = self._state.budget
        self.ledger_path = self._state.ledger_path
        self._local = threading.local()

    @property
    def last_call_metadata(self) -> dict[str, Any] | None:
        metadata = getattr(self._local, "last_call_metadata", None)
        return metadata if isinstance(metadata, dict) else None

    def wrap_child(self, inner: LLMClient) -> "RecordingLLMClient":
        """Wrap a role-specific adapter while preserving one run ledger."""

        return RecordingLLMClient(
            inner,
            self.ledger_path,
            self.budget,
            _state=self._state,
        )

    def refresh_from_ledger(self) -> None:
        """Reload shared counters after the orchestrator acquires its run lock."""

        current = self._state.budget
        refreshed = LLMBudget(
            max_prompt_tokens=current.max_prompt_tokens,
            max_output_tokens=current.max_output_tokens,
            max_total_tokens=current.max_total_tokens,
            max_calls=current.max_calls,
            max_cost_usd=current.max_cost_usd,
            cost_per_1k_tokens_usd=current.cost_per_1k_tokens_usd,
        )
        loaded = _load_recording_state(self.ledger_path, refreshed)
        with self._state.lock:
            current.calls_used = self._state.budget_calls_offset + refreshed.calls_used
            current.prompt_tokens_used = (
                self._state.budget_prompt_tokens_offset + refreshed.prompt_tokens_used
            )
            current.output_tokens_used = (
                self._state.budget_output_tokens_offset + refreshed.output_tokens_used
            )
            current._check_after_response()
            if current.max_calls is not None and current.calls_used > current.max_calls:
                raise LLMBudgetExceeded(
                    "Existing aggregate LLM ledger usage exceeds call budget: "
                    f"used={current.calls_used}, max={current.max_calls}"
                )
            self._state.call_count = loaded.call_count

    def ledger_usage(self) -> dict[str, Any]:
        """Return usage attributable to this wrapper's ledger, excluding shared offsets."""

        with self._state.lock:
            calls = self._state.call_count
            prompt_tokens = (
                self.budget.prompt_tokens_used - self._state.budget_prompt_tokens_offset
            )
            output_tokens = (
                self.budget.output_tokens_used - self._state.budget_output_tokens_offset
            )
            cost_rate = self.budget.cost_per_1k_tokens_usd
            return {
                "schema_version": 1,
                "calls_used": calls,
                "prompt_tokens_used": prompt_tokens,
                "output_tokens_used": output_tokens,
                "total_tokens_used": prompt_tokens + output_tokens,
                "estimated_cost_usd": (
                    (prompt_tokens + output_tokens) / 1000.0 * cost_rate
                    if cost_rate is not None
                    else 0.0
                ),
                "aggregate_offset": {
                    "calls_used": self._state.budget_calls_offset,
                    "prompt_tokens_used": self._state.budget_prompt_tokens_offset,
                    "output_tokens_used": self._state.budget_output_tokens_offset,
                },
            }

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> str:
        return self._record_call(
            method="complete_text",
            schema_name=None,
            prompt=prompt,
            system=system,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            call=lambda: _call_inner_complete_text(
                self.inner,
                prompt,
                system=system,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            ),
        )

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        return self._record_call(
            method="complete_json",
            schema_name=schema_name,
            prompt=prompt,
            system=system,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            call=lambda: _call_inner_complete_json(
                self.inner,
                prompt,
                schema_name,
                system=system,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            ),
        )

    def complete_json_with_images(
        self,
        prompt: str,
        schema_name: str,
        image_paths: list[Path],
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        return self._record_call(
            method="complete_json_with_images",
            schema_name=schema_name,
            prompt=prompt,
            system=system,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            call=lambda: _call_inner_complete_json_with_images(
                self.inner,
                prompt,
                schema_name,
                image_paths,
                system=system,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            ),
        )

    def _record_call(
        self,
        *,
        method: str,
        schema_name: str | None,
        prompt: str,
        system: str | None,
        temperature: float,
        reasoning_effort: str | None,
        call: Any,
    ) -> Any:
        # A pre-provider budget rejection has no call identity or usage of its
        # own. Clear the previous thread-local call before reserve_call() so an
        # outer failure trace cannot inherit stale provider evidence.
        self._local.last_call_metadata = None
        inner_metadata_before = getattr(self.inner, "last_call_metadata", None)
        inner_metadata_reset = _reset_inner_call_metadata(self.inner)
        prompt_tokens = _estimate_tokens(prompt)
        with self._state.lock:
            self.budget.reserve_call(prompt_tokens)
            self._state.call_count += 1
            call_id = f"llm_call_{self._state.call_count:06d}"
        started_wall = time.time()
        started = time.monotonic()
        record: dict[str, Any] = {
            "schema_version": 1,
            "call_id": call_id,
            "provider": self.provider,
            "model": self.model,
            "adapter_type": self.adapter_type,
            "provider_capabilities": self.provider_capabilities,
            "method": method,
            "schema_name": schema_name,
            "span_kind": "generation_span",
            "prompt_hash": _hash_text(prompt),
            "system_hash": _hash_text(system or ""),
            "prompt_token_estimate": prompt_tokens,
            "prompt_tokens_accounted": prompt_tokens,
            "prompt_token_source": "local_estimate",
            "temperature": temperature,
            "started_at_unix": started_wall,
        }
        if reasoning_effort is not None:
            record["reasoning_effort"] = reasoning_effort
        self._emit_progress("llm_call_started", record)
        try:
            response = call()
        except Exception as exc:
            inner_metadata = _current_inner_call_metadata(
                self.inner,
                previous=inner_metadata_before,
                reset_succeeded=inner_metadata_reset,
            )
            provider_prompt_tokens = _usage_token_count(
                inner_metadata,
                "prompt_tokens",
                "input_tokens",
            )
            provider_response_tokens = _usage_token_count(
                inner_metadata,
                "completion_tokens",
                "output_tokens",
            )
            budget_error: LLMBudgetExceeded | None = None
            if provider_prompt_tokens is not None or provider_response_tokens is not None:
                prompt_tokens_accounted = (
                    provider_prompt_tokens
                    if provider_prompt_tokens is not None
                    else prompt_tokens
                )
                record.update(
                    {
                        "prompt_tokens_accounted": prompt_tokens_accounted,
                        "prompt_token_source": (
                            "provider_usage"
                            if provider_prompt_tokens is not None
                            else "local_estimate"
                        ),
                    }
                )
                if provider_response_tokens is not None:
                    record.update(
                        {
                            "response_token_estimate": provider_response_tokens,
                            "response_token_source": "provider_usage",
                        }
                    )
                try:
                    with self._state.lock:
                        self.budget.record_response(
                            output_tokens=provider_response_tokens or 0,
                            prompt_tokens_reserved=prompt_tokens,
                            prompt_tokens_accounted=prompt_tokens_accounted,
                        )
                except LLMBudgetExceeded as accounting_exc:
                    budget_error = accounting_exc
            record.update(
                {
                    "success": False,
                    "error_type": type(budget_error or exc).__name__,
                    "duration_s": time.monotonic() - started,
                }
            )
            if budget_error is not None:
                record["underlying_error_type"] = type(exc).__name__
            self._local.last_call_metadata = _trace_call_metadata(
                record,
                inner_metadata,
            )
            self._append_ledger(record)
            self._emit_progress("llm_call_finished", record)
            if budget_error is not None:
                raise budget_error from exc
            raise
        inner_metadata = _current_inner_call_metadata(
            self.inner,
            previous=inner_metadata_before,
            reset_succeeded=inner_metadata_reset,
        )
        provider_prompt_tokens = _usage_token_count(
            inner_metadata,
            "prompt_tokens",
            "input_tokens",
        )
        prompt_tokens_accounted = (
            provider_prompt_tokens
            if provider_prompt_tokens is not None
            else prompt_tokens
        )
        response_tokens, response_token_source = _response_token_count(
            response,
            inner_metadata,
        )
        record.update(
            {
                "prompt_tokens_accounted": prompt_tokens_accounted,
                "prompt_token_source": (
                    "provider_usage"
                    if provider_prompt_tokens is not None
                    else "local_estimate"
                ),
                "response_token_source": response_token_source,
            }
        )
        try:
            with self._state.lock:
                self.budget.record_response(
                    output_tokens=response_tokens,
                    prompt_tokens_reserved=prompt_tokens,
                    prompt_tokens_accounted=prompt_tokens_accounted,
                )
        except LLMBudgetExceeded as exc:
            # The provider response has already been produced (and may be
            # billable), so persist its usage before propagating the budget
            # failure.  ``success=False`` describes the wrapper call result;
            # ``response_token_estimate`` preserves the provider-side usage.
            record.update(
                {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "response_token_estimate": response_tokens,
                    "duration_s": time.monotonic() - started,
                }
            )
            self._local.last_call_metadata = _trace_call_metadata(
                record,
                inner_metadata,
            )
            self._append_ledger(record)
            self._emit_progress("llm_call_finished", record)
            raise
        record.update(
            {
                "success": True,
                "response_hash": (
                    _hash_payload(response)
                    if isinstance(response, dict)
                    else _hash_text(str(response))
                ),
                "response_token_estimate": response_tokens,
                "duration_s": time.monotonic() - started,
            }
        )
        self._local.last_call_metadata = _trace_call_metadata(
            record,
            inner_metadata,
        )
        self._append_ledger(record)
        self._emit_progress("llm_call_finished", record)
        return response

    def _emit_progress(self, event: str, record: dict[str, Any]) -> None:
        callback = self._state.progress_callback
        if callback is None:
            return
        with self._state.lock:
            budget = {
                "calls_used": self.budget.calls_used,
                "prompt_tokens_used": self.budget.prompt_tokens_used,
                "output_tokens_used": self.budget.output_tokens_used,
                "total_tokens_used": (
                    self.budget.prompt_tokens_used + self.budget.output_tokens_used
                ),
                "estimated_cost_usd": self.budget.estimated_cost_usd,
            }
        payload = {
            "schema_version": 1,
            "event": event,
            "call_id": record["call_id"],
            "provider": record["provider"],
            "model": record["model"],
            "adapter_type": record["adapter_type"],
            "method": record["method"],
            "schema_name": record["schema_name"],
            "budget": budget,
        }
        if event == "llm_call_finished":
            payload.update(
                {
                    "success": record["success"],
                    "duration_s": record["duration_s"],
                    "error_type": record.get("error_type"),
                }
            )
        try:
            callback(payload)
        except Exception:
            return

    def _append_ledger(self, record: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with self._state.lock:
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(line)


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
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be a finite positive number, got {raw!r}.")
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

    def record_response(
        self,
        *,
        output_tokens: int,
        prompt_tokens_reserved: int | None = None,
        prompt_tokens_accounted: int | None = None,
    ) -> None:
        if prompt_tokens_accounted is not None:
            reserved = (
                prompt_tokens_reserved
                if prompt_tokens_reserved is not None
                else prompt_tokens_accounted
            )
            self.prompt_tokens_used += prompt_tokens_accounted - reserved
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
        # Provider usage is already billable at this boundary. Keep cost
        # accounting current even when a token or total limit fails closed.
        self._refresh_cost()
        if self.max_prompt_tokens is not None and self.prompt_tokens_used > self.max_prompt_tokens:
            raise LLMBudgetExceeded(
                "LLM prompt token budget exceeded: "
                f"used={self.prompt_tokens_used}, max={self.max_prompt_tokens}"
            )
        if self.max_output_tokens is not None and self.output_tokens_used > self.max_output_tokens:
            raise LLMBudgetExceeded(
                f"LLM output token budget exceeded: used={self.output_tokens_used}, max={self.max_output_tokens}"
            )
        total = self.prompt_tokens_used + self.output_tokens_used
        if self.max_total_tokens is not None and total > self.max_total_tokens:
            raise LLMBudgetExceeded(f"LLM total token budget exceeded: used={total}, max={self.max_total_tokens}")
        if self.max_cost_usd is not None and self.estimated_cost_usd > self.max_cost_usd:
            raise LLMBudgetExceeded(
                f"LLM cost budget exceeded: estimated={self.estimated_cost_usd:.6f}, max={self.max_cost_usd:.6f}"
            )


def estimate_orchestrator_llm_call_range(
    *,
    max_iterations: int,
    parallel_mutations: int,
) -> dict[str, int]:
    root_calls = 4
    if max_iterations <= 0:
        return {"min": root_calls, "max": root_calls}
    mutation_slots = max(1, max_iterations) * max(1, parallel_mutations)
    return {
        "min": root_calls + mutation_slots * 6,
        "max": root_calls + mutation_slots * 18,
    }


def combine_llm_call_ranges(ranges: list[dict[str, int]]) -> dict[str, int]:
    if not ranges:
        return {"min": 0, "max": 0}
    return {
        "min": sum(int(item.get("min", 0)) for item in ranges),
        "max": sum(int(item.get("max", 0)) for item in ranges),
    }


def llm_call_budget_preflight(
    *,
    budget: LLMBudget,
    expected_llm_call_range: dict[str, int],
) -> dict[str, Any]:
    expected_min = int(expected_llm_call_range.get("min", 0))
    expected_max = int(expected_llm_call_range.get("max", 0))
    blockers: list[str] = []
    projected_max = budget.calls_used + expected_max
    if budget.max_calls is not None and projected_max > budget.max_calls:
        blockers.append(
            "estimated_max_llm_calls exceeds AGENTICSCIML_MAX_LLM_CALLS: "
            f"already_used={budget.calls_used}, estimated_max={expected_max}, "
            f"projected_max={projected_max}, max_calls={budget.max_calls}"
        )
    return {
        "schema_version": 1,
        "status": "blocked_by_budget" if blockers else "ready",
        "passed": not blockers,
        "expected_min_llm_calls": expected_min,
        "expected_max_llm_calls": expected_max,
        "calls_already_used": budget.calls_used,
        "projected_max_llm_calls": projected_max,
        "configured_max_llm_calls": budget.max_calls,
        "blockers": blockers,
        "notes": [
            "Preflight checks deterministic call-count budgets before provider calls.",
            "Prompt, output, total-token, and cost budgets are still enforced by the runtime LLM ledger.",
        ],
    }


def require_llm_call_budget_preflight(preflight: dict[str, Any]) -> None:
    if preflight.get("passed") is True:
        return
    blockers = preflight.get("blockers")
    detail = "; ".join(str(item) for item in blockers) if isinstance(blockers, list) else ""
    raise LLMBudgetPreflightError(f"LLM call budget preflight failed: {detail}")


def load_llm_budget_usage(ledger_path: Path, budget: LLMBudget) -> LLMBudget:
    """Validate an existing call ledger and restore its usage into ``budget``.

    This provider-free entry point is useful for resume preflight checks that
    must account for prior calls before constructing a network adapter.
    """

    _load_recording_state(Path(ledger_path), budget)
    return budget


def _load_recording_state(
    ledger_path: Path,
    budget: LLMBudget,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> _RecordingState:
    state = _RecordingState(
        ledger_path=ledger_path,
        budget=budget,
        lock=threading.RLock(),
        progress_callback=progress_callback,
    )
    if not ledger_path.exists():
        state.budget_calls_offset = budget.calls_used
        state.budget_prompt_tokens_offset = budget.prompt_tokens_used
        state.budget_output_tokens_offset = budget.output_tokens_used
        return state

    records_by_call_number: dict[int, tuple[int, dict[str, Any]]] = {}
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Cannot resume with invalid LLM ledger JSON at {ledger_path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"Cannot resume with non-object LLM ledger row at {ledger_path}:{line_number}")
        call_id = record.get("call_id")
        if not isinstance(call_id, str) or re.fullmatch(r"llm_call_(\d{6})", call_id) is None:
            raise RuntimeError(
                f"Cannot resume with invalid LLM ledger call_id at {ledger_path}:{line_number}"
            )
        call_number = int(call_id.removeprefix("llm_call_"))
        if call_number < 1:
            raise RuntimeError(
                f"Cannot resume with invalid LLM ledger call_id at {ledger_path}:{line_number}"
            )
        if call_number in records_by_call_number:
            previous_line = records_by_call_number[call_number][0]
            raise RuntimeError(
                f"Cannot resume with duplicate LLM ledger call_id {call_id} at "
                f"{ledger_path}:{line_number}; first seen on line {previous_line}"
            )
        records_by_call_number[call_number] = (line_number, record)

    calls = len(records_by_call_number)
    expected_call_numbers = set(range(1, calls + 1))
    actual_call_numbers = set(records_by_call_number)
    if actual_call_numbers != expected_call_numbers:
        missing = sorted(expected_call_numbers - actual_call_numbers)
        unexpected = sorted(actual_call_numbers - expected_call_numbers)
        raise RuntimeError(
            f"Cannot resume with non-contiguous LLM ledger call_id set at {ledger_path}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    prompt_tokens = 0
    output_tokens = 0
    for call_number in range(1, calls + 1):
        line_number, record = records_by_call_number[call_number]
        prompt_tokens += _ledger_prompt_tokens_accounted(
            record,
            ledger_path,
            line_number,
        )
        # Failed provider calls may have no response usage. Any post-response
        # failure does: its response_token_estimate remains billable and is
        # therefore restored just like a successful response.
        if "response_token_estimate" in record:
            output_tokens += _non_negative_ledger_int(
                record.get("response_token_estimate"),
                ledger_path,
                line_number,
                "response_token_estimate",
            )

    if any((budget.calls_used, budget.prompt_tokens_used, budget.output_tokens_used)):
        if (
            budget.calls_used != calls
            or budget.prompt_tokens_used != prompt_tokens
            or budget.output_tokens_used != output_tokens
        ):
            raise RuntimeError("Configured LLM budget usage does not match the existing ledger")
    else:
        budget.calls_used = calls
        budget.prompt_tokens_used = prompt_tokens
        budget.output_tokens_used = output_tokens
        budget._check_after_response()
        if budget.max_calls is not None and calls > budget.max_calls:
            raise LLMBudgetExceeded(
                f"Existing LLM ledger already exceeds call budget: used={calls}, max={budget.max_calls}"
            )
    state.call_count = calls
    return state


def _non_negative_ledger_int(value: Any, path: Path, line_number: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"Cannot resume with invalid {field} at {path}:{line_number}"
        )
    return value


def _ledger_prompt_tokens_accounted(
    record: dict[str, Any],
    path: Path,
    line_number: int,
) -> int:
    prompt_estimate = _non_negative_ledger_int(
        record.get("prompt_token_estimate"),
        path,
        line_number,
        "prompt_token_estimate",
    )
    accounting_fields = {
        "prompt_tokens_accounted",
        "prompt_token_source",
        "response_token_source",
    }
    present_fields = accounting_fields & set(record)
    if not present_fields:
        return prompt_estimate
    for field in ("prompt_tokens_accounted", "prompt_token_source"):
        if field not in record:
            raise RuntimeError(
                f"Cannot resume with partial LLM token accounting at "
                f"{path}:{line_number}; missing={field}"
            )
    prompt_accounted = _non_negative_ledger_int(
        record.get("prompt_tokens_accounted"),
        path,
        line_number,
        "prompt_tokens_accounted",
    )
    prompt_source = record.get("prompt_token_source")
    if prompt_source not in {"local_estimate", "provider_usage"}:
        raise RuntimeError(
            f"Cannot resume with invalid prompt_token_source at {path}:{line_number}"
        )
    has_response_tokens = "response_token_estimate" in record
    has_response_source = "response_token_source" in record
    if record.get("success") is True and not has_response_tokens and not has_response_source:
        raise RuntimeError(
            f"Cannot resume successful LLM call without response token accounting at "
            f"{path}:{line_number}"
        )
    if has_response_tokens != has_response_source:
        missing = (
            "response_token_source"
            if has_response_tokens
            else "response_token_estimate"
        )
        raise RuntimeError(
            f"Cannot resume with partial LLM token accounting at "
            f"{path}:{line_number}; missing={missing}"
        )
    if has_response_source and record.get("response_token_source") not in {
        "local_estimate",
        "provider_usage",
    }:
        raise RuntimeError(
            f"Cannot resume with invalid response_token_source at {path}:{line_number}"
        )
    if prompt_source == "local_estimate" and prompt_accounted != prompt_estimate:
        raise RuntimeError(
            f"Cannot resume with inconsistent local prompt token accounting at "
            f"{path}:{line_number}"
        )
    return prompt_accounted


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trace_call_metadata(record: dict[str, Any], inner_metadata: Any = None) -> dict[str, Any]:
    metadata = {
        "llm_call_id": record["call_id"],
        "span_kind": record["span_kind"],
        "provider": record["provider"],
        "model": record["model"],
        "adapter_type": record["adapter_type"],
        "provider_capabilities": record["provider_capabilities"],
        "method": record["method"],
        "schema_name": record["schema_name"],
    }
    if "reasoning_effort" in record:
        metadata["reasoning_effort"] = record["reasoning_effort"]
    if isinstance(inner_metadata, dict) and isinstance(inner_metadata.get("usage"), dict):
        usage = {}
        prompt_tokens = _usage_token_count(
            inner_metadata,
            "prompt_tokens",
            "input_tokens",
        )
        completion_tokens = _usage_token_count(
            inner_metadata,
            "completion_tokens",
            "output_tokens",
        )
        total_tokens = _usage_token_count(inner_metadata, "total_tokens")
        if prompt_tokens is not None:
            usage["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            usage["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens
        if "total_tokens" not in usage and {
            "prompt_tokens",
            "completion_tokens",
        } <= set(usage):
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        if usage:
            metadata["usage"] = usage
    if isinstance(inner_metadata, dict):
        image_input_count = inner_metadata.get("image_input_count")
        if (
            isinstance(image_input_count, int)
            and not isinstance(image_input_count, bool)
            and image_input_count >= 0
        ):
            metadata["image_input_count"] = image_input_count
        image_input_filenames = inner_metadata.get("image_input_filenames")
        if isinstance(image_input_filenames, list) and all(
            isinstance(item, str) for item in image_input_filenames
        ):
            metadata["image_input_filenames"] = [
                Path(item).name for item in image_input_filenames
            ]
    return metadata


def _reset_inner_call_metadata(inner: LLMClient) -> bool:
    try:
        setattr(inner, "last_call_metadata", None)
    except (AttributeError, TypeError):
        return False
    return True


def _current_inner_call_metadata(
    inner: LLMClient,
    *,
    previous: Any,
    reset_succeeded: bool,
) -> Any:
    current = getattr(inner, "last_call_metadata", None)
    if not reset_succeeded and current is previous:
        return None
    return current


def _call_inner_complete_text(
    inner: LLMClient,
    prompt: str,
    *,
    system: str | None,
    temperature: float,
    reasoning_effort: str | None,
) -> str:
    kwargs: dict[str, Any] = {"system": system, "temperature": temperature}
    if reasoning_effort is not None and _accepts_reasoning_effort(inner.complete_text):
        kwargs["reasoning_effort"] = reasoning_effort
    return inner.complete_text(prompt, **kwargs)


def _call_inner_complete_json(
    inner: LLMClient,
    prompt: str,
    schema_name: str,
    *,
    system: str | None,
    temperature: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"system": system, "temperature": temperature}
    if reasoning_effort is not None and _accepts_reasoning_effort(inner.complete_json):
        kwargs["reasoning_effort"] = reasoning_effort
    return inner.complete_json(prompt, schema_name, **kwargs)


def _call_inner_complete_json_with_images(
    inner: LLMClient,
    prompt: str,
    schema_name: str,
    image_paths: list[Path],
    *,
    system: str | None,
    temperature: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"system": system, "temperature": temperature}
    if reasoning_effort is not None and _accepts_reasoning_effort(
        inner.complete_json_with_images
    ):
        kwargs["reasoning_effort"] = reasoning_effort
    return inner.complete_json_with_images(
        prompt,
        schema_name,
        image_paths,
        **kwargs,
    )


def _accepts_reasoning_effort(method: Any) -> bool:
    try:
        method_signature = signature(method)
    except (TypeError, ValueError):
        return True
    if "reasoning_effort" in method_signature.parameters:
        return True
    return any(
        parameter.kind == Parameter.VAR_KEYWORD
        for parameter in method_signature.parameters.values()
    )


def _llm_provider_name(llm_client: LLMClient) -> str:
    provider_name = getattr(llm_client, "provider_name", None)
    if isinstance(provider_name, str) and provider_name:
        return provider_name
    return type(llm_client).__name__


def _llm_provider_capabilities(llm_client: LLMClient) -> dict[str, object]:
    capabilities = getattr(llm_client, "provider_capabilities", None)
    if hasattr(capabilities, "to_dict"):
        return capabilities.to_dict()
    if isinstance(capabilities, dict):
        return capabilities
    provider = _llm_provider_name(llm_client)
    return {
        "provider": provider,
        "adapter_type": getattr(llm_client, "adapter_type", type(llm_client).__name__),
        "supports_responses": False,
        "supports_structured_outputs": False,
        "supports_image_inputs": False,
        "supports_usage": False,
        "supports_trace_export": False,
        "supports_prompt_cache": False,
    }


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _usage_token_count(metadata: Any, *keys: str) -> int | None:
    if not isinstance(metadata, dict):
        return None
    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _response_token_count(response: Any, metadata: Any = None) -> tuple[int, str]:
    provider_tokens = _usage_token_count(
        metadata,
        "completion_tokens",
        "output_tokens",
    )
    if provider_tokens is not None:
        return provider_tokens, "provider_usage"
    if isinstance(response, dict):
        return (
            _estimate_tokens(json.dumps(response, sort_keys=True, default=str)),
            "local_estimate",
        )
    return _estimate_tokens(str(response)), "local_estimate"
