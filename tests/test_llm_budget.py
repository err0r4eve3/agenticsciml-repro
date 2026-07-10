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
