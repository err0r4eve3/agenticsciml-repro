import json
from pathlib import Path
from typing import Any

import pytest

from agenticsciml.agents.base import AgentBase, StructuredOutputError
from agenticsciml.agents.proposer import ProposerAgent
from agenticsciml.config import EvaluationContract
from agenticsciml.execution.sandbox import prepare_solution_workspace, train_and_evaluate
from agenticsciml.llm.base import LLMClient
from agenticsciml.storage import ExperimentStorage


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
