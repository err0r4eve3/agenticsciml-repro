from pathlib import Path

import pytest

from agenticsciml.agents import CriticAgent, ProposerAgent, SelectorAgent
from agenticsciml.agents.base import ArtifactMissingError, InputContractError
from agenticsciml.agents.specs import AGENT_SPECS, AgentSpec, PromptTemplate
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.storage import ExperimentStorage


def test_mock_json_outputs_are_stable() -> None:
    llm = MockLLMClient()

    first = llm.complete_json("select parent", "selector")
    second = llm.complete_json("select parent", "selector")

    assert first == second
    assert first["selected_parent_ids"] == ["solution_000"]


def test_agents_save_transcripts_and_structured_outputs(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = MockLLMClient()

    proposal = ProposerAgent(llm, storage).debate(
        solution_id="solution_001",
        parent_summary="score=1.0",
        kb_entry="Fourier features",
        related_reports=[],
    )
    selection = SelectorAgent(llm, storage).select(
        candidates=[{"node_id": "solution_000", "score": 1.0}],
        best_node_id="solution_000",
        max_to_select=1,
    )

    assert proposal.title
    assert selection == ["solution_000"]
    assert (storage.run_dir / "solutions" / "solution_001" / "proposal.md").exists()
    assert (storage.run_dir / "solutions" / "solution_001" / "critic.md").exists()
    assert (storage.run_dir / "transcripts" / "selector.json").exists()


def test_agent_specs_document_roles_and_context_boundaries() -> None:
    proposer = AGENT_SPECS["proposer"]

    assert proposer.role == "proposer"
    assert "write code" in proposer.non_role
    assert "parent_summary" in proposer.input_schema
    assert "mutation_plan" in proposer.output_schema
    assert proposer.artifacts == ("proposal.md", "critic.md")
    assert proposer.to_dict()["budget"]["rounds"] == 4


def test_agent_spec_round_trips() -> None:
    spec = AgentSpec(
        role="test_agent",
        state_node="test node",
        purpose="Check a test behavior.",
        non_role=("do unrelated work",),
        input_schema=("input",),
        output_schema=("output",),
        visible_context=("local context",),
        tools=("llm",),
        budget={"calls": 1},
        artifacts=("artifact.md",),
        failure_policy="fail closed",
    )

    assert AgentSpec.from_dict(spec.to_dict()) == spec


def test_prompt_template_requires_all_values() -> None:
    template = PromptTemplate("Hello {name}, use {artifact}.")

    assert template.render({"name": "agent", "artifact": "proposal.md"}) == "Hello agent, use proposal.md."
    with pytest.raises(KeyError):
        template.render({"name": "agent"})


def test_agent_artifact_guard_reports_missing_files(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = ProposerAgent(MockLLMClient(), storage)
    storage.create_solution_workspace("solution_002")

    with pytest.raises(ArtifactMissingError) as exc:
        agent.require_artifacts("solution_002", ("proposal.md",))

    assert "proposal.md" in str(exc.value)


def test_agent_instances_bind_runtime_specs_and_validate_inputs(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    proposer = ProposerAgent(MockLLMClient(), storage)

    assert proposer.spec == AGENT_SPECS["proposer"]
    proposer.require_inputs(
        {
            "solution_id": "solution_001",
            "parent_summary": "score=1.0",
            "kb_entry": None,
            "related_reports": [],
            "branch_context": {},
        }
    )
    with pytest.raises(InputContractError) as exc:
        proposer.require_inputs({"parent_summary": "score=1.0"})

    assert "solution_id" in str(exc.value)


def test_agent_span_trace_includes_spec_metadata(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")

    ProposerAgent(MockLLMClient(), storage).debate(
        solution_id="solution_001",
        parent_summary="score=1.0",
        kb_entry=None,
        related_reports=[],
    )

    trace_text = (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert '"state_node": "mutation proposal"' in trace_text
    assert '"spec_role": "proposer"' in trace_text


def test_critic_agent_writes_structured_artifact(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    critic = CriticAgent(MockLLMClient(), storage)

    critique = critic.critique(
        solution_id="solution_003",
        proposal_summary="Add Fourier features.",
        context="validation_mse root baseline is high",
        round_index=1,
    )

    critic_path = storage.run_dir / "solutions" / "solution_003" / "critic.md"
    assert "Critic Round 1" in critic_path.read_text(encoding="utf-8")
    assert critique
