from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from agenticsciml.llm.base import LLMClient
from agenticsciml.llm.budget import (
    LLMBudget,
    LLMBudgetExceeded,
    RecordingLLMClient,
    load_llm_budget_usage,
)


class _TextLLM(LLMClient):
    model = "budget-test-model"
    adapter_type = "budget-test-adapter"

    def __init__(self, complete_text: Any) -> None:
        self._complete_text = complete_text

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> str:
        return str(self._complete_text(prompt))

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        return {"text": self.complete_text(prompt)}


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_recording_budget_resume_accepts_out_of_order_parallel_ledger_rows(tmp_path: Path) -> None:
    slow_started = threading.Event()
    release_slow = threading.Event()

    def complete(prompt: str) -> str:
        if prompt == "slow":
            slow_started.set()
            if not release_slow.wait(timeout=5):
                raise TimeoutError("test did not release slow call")
        return f"{prompt} response"

    ledger_path = tmp_path / "llm_call_ledger.jsonl"
    recording = RecordingLLMClient(_TextLLM(complete), ledger_path, LLMBudget())
    errors: list[Exception] = []

    def invoke(prompt: str) -> None:
        try:
            recording.complete_text(prompt)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    slow_thread = threading.Thread(target=invoke, args=("slow",))
    slow_thread.start()
    assert slow_started.wait(timeout=5)
    fast_thread = threading.Thread(target=invoke, args=("fast",))
    fast_thread.start()
    fast_thread.join(timeout=5)
    assert not fast_thread.is_alive()
    release_slow.set()
    slow_thread.join(timeout=5)
    assert not slow_thread.is_alive()
    assert errors == []

    rows = _ledger_rows(ledger_path)
    assert [row["call_id"] for row in rows] == ["llm_call_000002", "llm_call_000001"]

    resumed_budget = LLMBudget()
    resumed = RecordingLLMClient(_TextLLM(lambda prompt: f"{prompt} response"), ledger_path, resumed_budget)
    assert resumed_budget.calls_used == 2
    assert resumed_budget.prompt_tokens_used == sum(row["prompt_token_estimate"] for row in rows)
    assert resumed_budget.output_tokens_used == sum(row["response_token_estimate"] for row in rows)

    assert resumed.complete_text("third") == "third response"
    assert _ledger_rows(ledger_path)[-1]["call_id"] == "llm_call_000003"


def test_post_response_budget_failure_is_ledgered_and_reloaded_as_billable_usage(tmp_path: Path) -> None:
    ledger_path = tmp_path / "llm_call_ledger.jsonl"
    response = "billable provider response"
    budget = LLMBudget(max_output_tokens=1)
    recording = RecordingLLMClient(_TextLLM(lambda _prompt: response), ledger_path, budget)

    with pytest.raises(LLMBudgetExceeded, match="output token budget exceeded"):
        recording.complete_text("prompt without secrets")

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt_tokens_accounted"] == row["prompt_token_estimate"]
    assert row["prompt_token_source"] == "local_estimate"
    assert row["response_token_source"] == "local_estimate"
    assert row["call_id"] == "llm_call_000001"
    assert row["success"] is False
    assert row["error_type"] == "LLMBudgetExceeded"
    assert row["response_token_estimate"] == budget.output_tokens_used
    assert row["response_token_estimate"] > budget.max_output_tokens
    assert "response" not in row
    assert "prompt" not in row
    assert "response_hash" not in row

    strict_resume_budget = LLMBudget(max_output_tokens=1)
    with pytest.raises(LLMBudgetExceeded, match="output token budget exceeded"):
        load_llm_budget_usage(ledger_path, strict_resume_budget)
    assert strict_resume_budget.calls_used == 1
    assert strict_resume_budget.output_tokens_used == row["response_token_estimate"]

    relaxed_resume_budget = LLMBudget(max_output_tokens=100)
    assert load_llm_budget_usage(ledger_path, relaxed_resume_budget) is relaxed_resume_budget
    assert relaxed_resume_budget.calls_used == 1
    assert relaxed_resume_budget.prompt_tokens_used == row["prompt_token_estimate"]
    assert relaxed_resume_budget.output_tokens_used == row["response_token_estimate"]


def test_recording_llm_progress_callback_is_secret_free(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    recording = RecordingLLMClient(
        _TextLLM(lambda _prompt: "private response"),
        tmp_path / "llm_call_ledger.jsonl",
        LLMBudget(),
        progress_callback=events.append,
    )

    assert recording.complete_text("private prompt", system="private system") == "private response"

    assert [event["event"] for event in events] == ["llm_call_started", "llm_call_finished"]
    assert all(event["call_id"] == "llm_call_000001" for event in events)
    assert events[-1]["success"] is True
    assert events[-1]["duration_s"] >= 0
    assert set(events[-1]["budget"]) == {
        "calls_used",
        "prompt_tokens_used",
        "output_tokens_used",
        "total_tokens_used",
        "estimated_cost_usd",
    }
    encoded = json.dumps(events, sort_keys=True)
    assert "private prompt" not in encoded
    assert "private system" not in encoded
    assert "private response" not in encoded
    assert "prompt_hash" not in encoded
    assert "system_hash" not in encoded
    assert "response_hash" not in encoded
    assert "content_hash" not in encoded


def test_recording_llm_wrap_child_preserves_progress_callback(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    recording = RecordingLLMClient(
        _TextLLM(lambda _prompt: "parent"),
        tmp_path / "llm_call_ledger.jsonl",
        LLMBudget(),
        progress_callback=events.append,
    )
    child = recording.wrap_child(_TextLLM(lambda _prompt: "child"))

    assert child.complete_text("prompt") == "child"
    assert [event["event"] for event in events] == ["llm_call_started", "llm_call_finished"]
    assert all(event["call_id"] == "llm_call_000001" for event in events)


def test_recording_llm_progress_callback_reports_provider_failure(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []

    def timeout(_prompt: str) -> str:
        raise TimeoutError("private provider detail")

    recording = RecordingLLMClient(
        _TextLLM(timeout),
        tmp_path / "llm_call_ledger.jsonl",
        LLMBudget(),
        progress_callback=events.append,
    )

    with pytest.raises(TimeoutError, match="private provider detail"):
        recording.complete_text("private prompt")

    assert events[-1]["event"] == "llm_call_finished"
    assert events[-1]["success"] is False
    assert events[-1]["error_type"] == "TimeoutError"
    assert "private provider detail" not in json.dumps(events, sort_keys=True)


def test_recording_llm_progress_callback_failure_does_not_break_call(tmp_path: Path) -> None:
    def broken_progress(_event: dict[str, Any]) -> None:
        raise OSError("closed stderr")

    recording = RecordingLLMClient(
        _TextLLM(lambda _prompt: "ok"),
        tmp_path / "llm_call_ledger.jsonl",
        LLMBudget(),
        progress_callback=broken_progress,
    )

    assert recording.complete_text("prompt") == "ok"
    assert _ledger_rows(recording.ledger_path)[0]["success"] is True


def test_recording_llm_trace_metadata_preserves_numeric_provider_usage(tmp_path: Path) -> None:
    class UsageLLM(_TextLLM):
        def complete_text(
            self,
            prompt: str,
            system: str | None = None,
            temperature: float = 0.0,
            reasoning_effort: str | None = None,
        ) -> str:
            self.last_call_metadata = {
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 11,
                    "total_tokens": 18,
                    "private_detail": "must not propagate",
                }
            }
            return "provider response"

    recording = RecordingLLMClient(
        UsageLLM(lambda _prompt: "unused"),
        tmp_path / "llm_call_ledger.jsonl",
        LLMBudget(),
    )

    assert recording.complete_text("prompt") == "provider response"
    assert recording.last_call_metadata is not None
    assert recording.last_call_metadata["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 11,
        "total_tokens": 18,
    }
    assert recording.budget.prompt_tokens_used == 7
    row = _ledger_rows(recording.ledger_path)[0]
    assert row["prompt_token_estimate"] != row["prompt_tokens_accounted"]
    assert row["prompt_tokens_accounted"] == 7
    assert row["prompt_token_source"] == "provider_usage"
    assert row["response_token_source"] == "provider_usage"

    resumed_budget = LLMBudget()
    load_llm_budget_usage(recording.ledger_path, resumed_budget)
    assert resumed_budget.prompt_tokens_used == 7
    assert resumed_budget.output_tokens_used == 11


def test_provider_prompt_usage_can_fail_budget_after_response(tmp_path: Path) -> None:
    class UsageLLM(_TextLLM):
        def complete_text(
            self,
            prompt: str,
            system: str | None = None,
            temperature: float = 0.0,
            reasoning_effort: str | None = None,
        ) -> str:
            self.last_call_metadata = {
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 1,
                    "total_tokens": 8,
                }
            }
            return "ok"

    recording = RecordingLLMClient(
        UsageLLM(lambda _prompt: "unused"),
        tmp_path / "llm_call_ledger.jsonl",
        LLMBudget(max_prompt_tokens=3, cost_per_1k_tokens_usd=1.0),
    )

    with pytest.raises(LLMBudgetExceeded, match="prompt token budget exceeded"):
        recording.complete_text("x")

    row = _ledger_rows(recording.ledger_path)[0]
    assert row["success"] is False
    assert row["error_type"] == "LLMBudgetExceeded"
    assert row["prompt_token_estimate"] == 1
    assert row["prompt_tokens_accounted"] == 7
    assert row["prompt_token_source"] == "provider_usage"
    assert row["response_token_estimate"] == 1
    assert row["response_token_source"] == "provider_usage"
    assert recording.budget.prompt_tokens_used == 7
    assert recording.budget.output_tokens_used == 1
    assert recording.budget.estimated_cost_usd == pytest.approx(0.008)


def test_provider_usage_is_billable_when_response_parsing_fails(tmp_path: Path) -> None:
    class UsageThenErrorLLM(_TextLLM):
        def complete_text(
            self,
            prompt: str,
            system: str | None = None,
            temperature: float = 0.0,
            reasoning_effort: str | None = None,
        ) -> str:
            self.last_call_metadata = {
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 5,
                    "total_tokens": 14,
                }
            }
            raise ValueError("invalid structured response")

    ledger_path = tmp_path / "llm_call_ledger.jsonl"
    recording = RecordingLLMClient(
        UsageThenErrorLLM(lambda _prompt: "unused"),
        ledger_path,
        LLMBudget(cost_per_1k_tokens_usd=1.0),
    )

    with pytest.raises(ValueError, match="invalid structured response"):
        recording.complete_text("x")

    row = _ledger_rows(ledger_path)[0]
    assert row["success"] is False
    assert row["error_type"] == "ValueError"
    assert row["prompt_tokens_accounted"] == 9
    assert row["prompt_token_source"] == "provider_usage"
    assert row["response_token_estimate"] == 5
    assert row["response_token_source"] == "provider_usage"
    assert recording.budget.prompt_tokens_used == 9
    assert recording.budget.output_tokens_used == 5
    assert recording.budget.estimated_cost_usd == pytest.approx(0.014)

    resumed_budget = LLMBudget(cost_per_1k_tokens_usd=1.0)
    load_llm_budget_usage(ledger_path, resumed_budget)
    assert resumed_budget.prompt_tokens_used == 9
    assert resumed_budget.output_tokens_used == 5
    assert resumed_budget.estimated_cost_usd == pytest.approx(0.014)


def test_billable_parse_failure_stops_on_reconciled_budget_excess(tmp_path: Path) -> None:
    class UsageThenErrorLLM(_TextLLM):
        def complete_text(
            self,
            prompt: str,
            system: str | None = None,
            temperature: float = 0.0,
            reasoning_effort: str | None = None,
        ) -> str:
            self.last_call_metadata = {
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 5,
                    "total_tokens": 14,
                }
            }
            raise ValueError("invalid structured response")

    recording = RecordingLLMClient(
        UsageThenErrorLLM(lambda _prompt: "unused"),
        tmp_path / "llm_call_ledger.jsonl",
        LLMBudget(max_prompt_tokens=3, cost_per_1k_tokens_usd=1.0),
    )

    with pytest.raises(LLMBudgetExceeded, match="prompt token budget exceeded") as exc_info:
        recording.complete_text("x")

    assert isinstance(exc_info.value.__cause__, ValueError)
    row = _ledger_rows(recording.ledger_path)[0]
    assert row["error_type"] == "LLMBudgetExceeded"
    assert row["underlying_error_type"] == "ValueError"
    assert row["prompt_tokens_accounted"] == 9
    assert row["response_token_estimate"] == 5
    assert recording.budget.estimated_cost_usd == pytest.approx(0.014)


def test_budget_resume_rejects_partial_or_inconsistent_new_accounting(tmp_path: Path) -> None:
    base_row = {
        "call_id": "llm_call_000001",
        "prompt_token_estimate": 2,
        "prompt_tokens_accounted": 7,
        "prompt_token_source": "provider_usage",
        "response_token_estimate": 3,
        "response_token_source": "provider_usage",
        "success": True,
    }
    ledger_path = tmp_path / "llm_call_ledger.jsonl"

    for mutation, error_pattern in (
        ({"prompt_tokens_accounted": None}, "partial LLM token accounting"),
        ({"prompt_token_source": "invalid"}, "invalid prompt_token_source"),
        ({"response_token_source": None}, "partial LLM token accounting"),
        (
            {
                "response_token_estimate": None,
                "response_token_source": None,
            },
            "successful LLM call without response token accounting",
        ),
        (
            {
                "prompt_tokens_accounted": 7,
                "prompt_token_source": "local_estimate",
            },
            "inconsistent local prompt token accounting",
        ),
    ):
        row = dict(base_row)
        for field, value in mutation.items():
            if value is None:
                row.pop(field)
            else:
                row[field] = value
        ledger_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match=error_pattern):
            load_llm_budget_usage(ledger_path, LLMBudget())


def test_budget_resume_accepts_legacy_accounting_fields_absent(tmp_path: Path) -> None:
    ledger_path = tmp_path / "llm_call_ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "call_id": "llm_call_000001",
                "prompt_token_estimate": 2,
                "response_token_estimate": 3,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    budget = load_llm_budget_usage(ledger_path, LLMBudget())

    assert budget.calls_used == 1
    assert budget.prompt_tokens_used == 2
    assert budget.output_tokens_used == 3


@pytest.mark.parametrize(
    ("call_ids", "error_pattern"),
    [
        (["llm_call_000001", "llm_call_000001"], "duplicate LLM ledger call_id"),
        (["llm_call_000001", "llm_call_000003"], "non-contiguous LLM ledger call_id set"),
    ],
)
def test_budget_resume_rejects_duplicate_or_gapped_call_ids(
    tmp_path: Path,
    call_ids: list[str],
    error_pattern: str,
) -> None:
    ledger_path = tmp_path / "llm_call_ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "call_id": call_id,
                    "prompt_token_estimate": 1,
                    "response_token_estimate": 1,
                    "success": True,
                }
            )
            for call_id in call_ids
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=error_pattern):
        load_llm_budget_usage(ledger_path, LLMBudget())
