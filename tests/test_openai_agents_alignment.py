import json
from pathlib import Path
from typing import Any

import pytest

from agenticsciml.agents.base import AgentBase, StructuredOutputError
from agenticsciml.agents.proposer import ProposerAgent
from agenticsciml.agents.output_schemas import validate_output_payload
from agenticsciml.config import EvaluationContract
from agenticsciml.execution.sandbox import prepare_solution_workspace, train_and_evaluate
from agenticsciml.llm.base import LLMClient
from agenticsciml.llm.budget import LLMBudget, LLMBudgetExceeded, RecordingLLMClient
from agenticsciml.reporting import write_sdk_trace_export
from agenticsciml.storage import ExperimentStorage


def test_unknown_structured_output_model_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown structured output model"):
        validate_output_payload({}, schema_name="unregistered_schema")


class FlakyJsonLLM(LLMClient):
    def __init__(self):
        self.calls = 0

    def complete_text(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        return "text"

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {"title": "missing required fields"}
        return {
            "title": "Valid proposal",
            "diagnosis": "Root underfits.",
            "mutation_plan": ["Add features."],
            "expected_effect": "Lower validation MSE.",
            "risks": ["May overfit."],
        }


class AlwaysInvalidJsonLLM(FlakyJsonLLM):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls += 1
        return {"title": "still invalid"}


class RaisingThenValidJsonLLM(FlakyJsonLLM):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("invalid json payload")
        return {
            "title": "Valid proposal",
            "diagnosis": "Root underfits.",
            "mutation_plan": ["Add features."],
            "expected_effect": "Lower validation MSE.",
            "risks": ["May overfit."],
        }


class APITimeoutError(RuntimeError):
    pass


class TimeoutThenValidJsonLLM(FlakyJsonLLM):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            raise APITimeoutError("Request timed out.")
        return {
            "title": "Valid proposal",
            "diagnosis": "Root underfits.",
            "mutation_plan": ["Add features."],
            "expected_effect": "Lower validation MSE.",
            "risks": ["May overfit."],
        }


class BudgetExceededJsonLLM(FlakyJsonLLM):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.calls += 1
        raise LLMBudgetExceeded("LLM prompt token budget exceeded")


class RaisingTextLLM(FlakyJsonLLM):
    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        raise RuntimeError("provider text failure")


class ExtraFieldProposalLLM(FlakyJsonLLM):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "title": "Valid proposal",
            "diagnosis": "Root underfits.",
            "mutation_plan": ["Add features."],
            "expected_effect": "Lower validation MSE.",
            "risks": ["May overfit."],
            "unknown": "schema drift",
        }


class ReasoningCaptureLLM(LLMClient):
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "method": "complete_text",
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
        )
        return "ok"

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "complete_json",
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
        )
        return {
            "title": "Valid proposal",
            "diagnosis": "Root underfits.",
            "mutation_plan": ["Add features."],
            "expected_effect": "Lower validation MSE.",
            "risks": ["May overfit."],
        }


class WrongTypeProposalLLM(FlakyJsonLLM):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "title": "Invalid proposal",
            "diagnosis": "Root underfits.",
            "mutation_plan": "not a list",
            "expected_effect": "Lower validation MSE.",
            "risks": ["May overfit."],
        }


class RecordingProposalLLM(LLMClient):
    def __init__(self):
        self.final_prompt = ""

    def complete_text(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        if "Critic round" in prompt:
            return "critic-specific-risk: verify the jump feature does not leak validation facts."
        return "draft proposal summary"

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.final_prompt = prompt
        return {
            "title": "Valid proposal",
            "diagnosis": "Root underfits.",
            "mutation_plan": ["Add Fourier features."],
            "expected_effect": "Lower validation MSE.",
            "risks": ["May overfit."],
        }


def test_agent_json_output_is_retried_and_schema_checked(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = ProposerAgent(FlakyJsonLLM(), storage)

    data = agent.complete_json_checked(
        prompt="return a proposal",
        schema_name="proposal",
        retries=1,
    )

    assert data["diagnosis"] == "Root underfits."


def test_agent_base_routes_default_reasoning_effort_to_llm(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = ReasoningCaptureLLM()
    agent = AgentBase(
        llm,
        storage,
        default_temperature=0.3,
        default_reasoning_effort="xhigh",
    )

    assert agent.complete_text(prompt="summarize") == "ok"
    data = agent.complete_json_checked(
        prompt="return a proposal",
        schema_name="proposal",
        required_fields=("title", "diagnosis", "mutation_plan", "expected_effect", "risks"),
        retries=0,
    )

    assert data["title"] == "Valid proposal"
    assert llm.calls == [
        {"method": "complete_text", "temperature": 0.3, "reasoning_effort": "xhigh"},
        {"method": "complete_json", "temperature": 0.3, "reasoning_effort": "xhigh"},
    ]


def test_agent_text_failure_keeps_ledger_and_generation_trace_bound(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    ledger_path = storage.run_dir / "llm_call_ledger.jsonl"
    llm = RecordingLLMClient(
        RaisingTextLLM(),
        ledger_path,
        LLMBudget(max_calls=2),
    )
    agent = AgentBase(llm, storage)

    with pytest.raises(RuntimeError, match="provider text failure"):
        agent.complete_text(prompt="analyze the training data")

    ledger = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    traces = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    generation = [
        event
        for event in traces
        if event.get("event_type") == "generation_span"
    ]
    assert len(ledger) == len(generation) == 1
    assert ledger[0]["success"] is False
    assert generation[0]["metadata"]["llm_call_id"] == ledger[0]["call_id"]
    assert generation[0]["metadata"]["error_type"] == "RuntimeError"


def test_agent_json_output_fails_closed_after_retry_budget(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = AgentBase(AlwaysInvalidJsonLLM(), storage)

    with pytest.raises(StructuredOutputError):
        agent.complete_json_checked(
            prompt="return a proposal",
            schema_name="proposal",
            required_fields=("title", "diagnosis", "mutation_plan", "expected_effect", "risks"),
            retries=1,
        )


def test_agent_json_output_rejects_unknown_fields(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = ProposerAgent(ExtraFieldProposalLLM(), storage)

    with pytest.raises(StructuredOutputError) as exc:
        agent.complete_json_checked(prompt="return a proposal", schema_name="proposal", retries=0)

    assert "typed schema validation" in str(exc.value)


def test_agent_json_output_rejects_wrong_field_types(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = ProposerAgent(WrongTypeProposalLLM(), storage)

    with pytest.raises(StructuredOutputError) as exc:
        agent.complete_json_checked(prompt="return a proposal", schema_name="proposal", retries=0)

    assert "typed schema validation" in str(exc.value)


def test_agent_json_exception_is_retried_and_traced(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = ProposerAgent(RaisingThenValidJsonLLM(), storage)

    data = agent.complete_json_checked(
        prompt="return a proposal",
        schema_name="proposal",
        retries=1,
    )
    events = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert data["diagnosis"] == "Root underfits."
    assert any(
        event["event_type"] == "guardrail_span"
        and event["metadata"].get("passed") is False
        and "invalid json payload" in event["metadata"].get("error", "")
        for event in events
    )


def test_agent_json_timeout_is_not_schema_retried(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = TimeoutThenValidJsonLLM()
    agent = ProposerAgent(llm, storage)

    with pytest.raises(StructuredOutputError) as exc:
        agent.complete_json_checked(
            prompt="return a proposal",
            schema_name="proposal",
            retries=1,
        )
    events = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert llm.calls == 1
    assert "APITimeoutError" in str(exc.value)
    failed_generations = [
        event
        for event in events
        if event["event_type"] == "generation_span"
        and event["name"] == "proposer"
        and event["metadata"].get("error_type") == "APITimeoutError"
    ]
    assert len(failed_generations) == 1


def test_agent_json_budget_excess_is_not_retried_or_reclassified(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = BudgetExceededJsonLLM()
    agent = ProposerAgent(llm, storage)

    with pytest.raises(LLMBudgetExceeded, match="prompt token budget exceeded"):
        agent.complete_json_checked(
            prompt="return a proposal",
            schema_name="proposal",
            retries=1,
        )

    assert llm.calls == 1
    guardrail_events = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "guardrail_span"
    ]
    assert not any(
        event["name"] == "proposer:proposal:structured_output"
        for event in guardrail_events
    )


def test_agent_budget_rejection_trace_does_not_reuse_previous_llm_call_id(
    tmp_path: Path,
) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    recording = RecordingLLMClient(
        FlakyJsonLLM(),
        storage.run_dir / "llm_call_ledger.jsonl",
        LLMBudget(max_calls=1),
    )
    agent = ProposerAgent(recording, storage)

    assert agent.complete_text("first call") == "text"
    with pytest.raises(LLMBudgetExceeded, match="LLM call budget exceeded"):
        agent.complete_json_checked(
            prompt="return a proposal",
            schema_name="proposal",
            retries=1,
        )

    generation_events = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "generation_span"
    ]
    assert generation_events[0]["metadata"]["llm_call_id"] == "llm_call_000001"
    assert generation_events[-1]["metadata"]["error_type"] == "LLMBudgetExceeded"
    assert "llm_call_id" not in generation_events[-1]["metadata"]
    assert "usage" not in generation_events[-1]["metadata"]


def test_proposer_final_round_uses_critic_feedback(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingProposalLLM()

    ProposerAgent(llm, storage).debate(
        solution_id="solution_001",
        parent_summary="root score is weak",
        kb_entry="Fourier features help oscillation",
        related_reports=[],
    )

    assert "critic-specific-risk" in llm.final_prompt


def test_trace_records_agent_generation_and_guardrail_events(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingProposalLLM()

    ProposerAgent(llm, storage).debate(
        solution_id="solution_001",
        parent_summary="root score is weak",
        kb_entry=None,
        related_reports=[],
    )

    events = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = {event["event_type"] for event in events}
    assert {"generation_span", "agent_span", "guardrail_span"} <= event_types
    generation_events = [event for event in events if event["event_type"] == "generation_span"]
    assert generation_events
    for event in generation_events:
        metadata = event["metadata"]
        assert metadata["duration_s"] >= 0
        assert metadata["prompt_token_estimate"] > 0
        assert metadata["response_token_estimate"] > 0


def test_sdk_trace_export_maps_spans_and_redacts_raw_fields(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    storage.record_trace(
        "generation_span",
        "proposer",
        {
            "provider": "mock",
            "model": "mock",
            "prompt": "raw prompt should not leave local trace",
            "response_hash": "abc",
        },
    )

    path = write_sdk_trace_export(storage.run_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["span_count"] == 1
    assert payload["spans"][0]["span_kind"] == "generation"
    assert payload["spans"][0]["metadata"]["prompt"] == "<redacted>"
    assert payload["spans"][0]["metadata"]["response_hash"] == "<redacted>"


def test_sandbox_guardrail_detects_evaluator_mutation(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "bad_solution"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(
        """
import argparse
from pathlib import Path

class MODEL:
    def predict(self, x):
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    if args.mode == "train":
        Path("evaluate.py").write_text("print('tampered')")
        Path("model.pkl").write_bytes(b"bad")

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=10)

    assert result.exit_code != 0
    assert "guardrail" in result.stderr.lower()


def test_sandbox_guardrail_blocks_network_import_before_execution(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "network_solution"
    prepare_solution_workspace(benchmark, workspace)
    marker = workspace / "executed.txt"
    (workspace / "solution.py").write_text(
        f"""
import argparse
import socket
from pathlib import Path

class MODEL:
    def predict(self, x):
        return x

def main():
    Path({str(marker)!r}).write_text("ran")

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=10)

    assert result.exit_code != 0
    assert "guardrail" in result.stderr.lower()
    assert "socket" in result.stderr
    assert not marker.exists()


def test_sandbox_guardrail_blocks_absolute_path_writes(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "absolute_path_solution"
    prepare_solution_workspace(benchmark, workspace)
    outside = tmp_path / "outside.txt"
    (workspace / "solution.py").write_text(
        f"""
import argparse
from pathlib import Path

class MODEL:
    def predict(self, x):
        return x

def main():
    Path({str(outside)!r}).write_text("outside")

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=10)

    assert result.exit_code != 0
    assert "blocked absolute path" in result.stderr
    assert not outside.exists()


def test_agents_md_requires_openai_agents_sdk_alignment() -> None:
    text = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "OpenAI Agents SDK Alignment" in text
    assert "structured outputs" in text
    assert "guardrails" in text
    assert "tracing" in text


def test_agenticsciml_assistant_governance_is_documented() -> None:
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    skill = Path(".agents/skills/agenticsciml-chatui-operator/SKILL.md").read_text(encoding="utf-8")
    contract = Path("docs/agenticsciml_assistant.md").read_text(encoding="utf-8")
    index = Path("docs/index.md").read_text(encoding="utf-8")

    assert "AgenticSciML Assistant Boundaries" in agents
    assert "Scientific Claim Policy" in agents
    assert "ChatUI And Tool Policy" in agents
    assert "`/api/solver/chat` is an internal algorithm-tool endpoint, not an OpenAI Apps" in agents
    assert "Prompt Injection Boundary" in agents

    assert "version: 0.4.0" in skill
    assert "not an MCP server" in skill
    assert "account_id" in skill
    assert "assistant_mode" in skill
    assert "reasoning_effort=high" in skill
    assert "GET /api/solver/settings" in skill
    assert "GET /api/algorithms" in skill
    assert "GET /api/paper-tasks" in skill
    assert "GET /api/agent-roles" in skill
    assert "Valid Action Categories" in skill
    assert "Real LLM Mode" in skill
    assert "Prompt Injection Handling" in skill

    assert "Future MCP Wrapper Contract" in contract
    assert "readOnlyHint" in contract
    assert "destructiveHint" in contract
    assert "openWorldHint" in contract
    assert "agenticsciml.validate_claim" in contract
    assert "Do not make ChatUI a free-form group chat controller" in contract

    assert "agenticsciml_assistant.md" in index
