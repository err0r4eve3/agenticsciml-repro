import json
from pathlib import Path

import numpy as np
import pytest

from agenticsciml.agents import (
    CriticAgent,
    DataAnalystAgent,
    EvaluatorAgent,
    ProposerAgent,
    ResultAnalystAgent,
    RootEngineerAgent,
    SelectorAgent,
)
from agenticsciml.agents.base import ArtifactMissingError, InputContractError
from agenticsciml.agents.specs import AGENT_SPECS, AgentSpec, PromptTemplate
from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.llm.base import LLMClient
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.storage import ExperimentStorage


class RecordingLLM(LLMClient):
    def __init__(self) -> None:
        self.last_prompt = ""

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.last_prompt = prompt
        return "ok"

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        self.last_prompt = prompt
        if schema_name in {"analysis", "result_analyst"}:
            return {
                "summary": "analyzed observations",
                "strengths": ["score recorded"],
                "weaknesses": ["prediction-only plots do not expose labels"],
                "next_steps": ["compare with sibling observations"],
            }
        if schema_name == "root_engineer":
            return {"proposal": "plain isolated baseline", "code": "class MODEL:\n    pass\n"}
        return {
            "metric_name": "validation_mse",
            "higher_is_better": False,
            "checkpoint_path": "model.pkl",
        }


class VotingLLM(LLMClient):
    def __init__(self) -> None:
        self.responses = [
            {"selected_parent_ids": ["worse_loss", "promising"], "rationale": "worse has an idea"},
            {"selected_parent_ids": ["promising"], "rationale": "promising has lower loss"},
            {"selected_parent_ids": ["worse_loss"], "rationale": "worse is underexplored"},
        ]
        self.index = 0

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return "ok"

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        response = self.responses[self.index]
        self.index += 1
        return response


def test_mock_json_outputs_are_stable() -> None:
    llm = MockLLMClient()

    first = llm.complete_json("select parent", "selector")
    second = llm.complete_json("select parent", "selector")

    assert first == second
    assert first["selected_parent_ids"] == ["solution_000"]


def test_evaluator_prompt_names_required_json_keys(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingLLM()
    bundle = ProblemBundle.load(Path("examples/function_approx"))

    EvaluatorAgent(llm, storage).create_contract(bundle, data_report="short report")

    assert "Return exactly one JSON object" in llm.last_prompt
    assert "`metric_name`, `higher_is_better`, and `checkpoint_path`" in llm.last_prompt
    assert "model.pkl" in llm.last_prompt


def test_data_analyst_writes_training_observation_artifacts(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingLLM()

    DataAnalystAgent(llm, storage).analyze(Path("examples/function_approx"))

    manifest_path = storage.run_dir / "reports" / "data_observations.json"
    svg_path = storage.run_dir / "reports" / "data_overview.svg"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["benchmark_name"] == "function_approx"
    assert manifest["source_mode"] in {"generated_seed0", "repo_existing"}
    assert manifest["plots"][0]["path"] == "reports/data_overview.svg"
    assert manifest["arrays"]["x_train"]["shape"] == [200, 1]
    assert manifest["arrays"]["u_train"]["shape"] == [200, 1]
    assert "val_data" not in manifest_path.read_text(encoding="utf-8")
    assert svg_path.exists()
    assert "Observation manifest:" in llm.last_prompt
    assert "reports/data_overview.svg" in llm.last_prompt


def test_result_analyst_writes_prediction_only_observation_artifacts(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    workspace = storage.create_solution_workspace("solution_001")
    (workspace / "eval.json").write_text(
        json.dumps({"metric": "validation_mse", "score": 0.25, "higher_is_better": False}),
        encoding="utf-8",
    )
    (workspace / "train.log").write_text("train completed\n", encoding="utf-8")
    np.savez(workspace / "predict_input.npz", x_val=np.linspace(-1.0, 1.0, 5).reshape(-1, 1))
    np.savez(workspace / "predictions.npz", predictions=np.linspace(0.0, 1.0, 5).reshape(-1, 1))
    llm = RecordingLLM()

    ResultAnalystAgent(llm, storage).analyze("solution_001", workspace)

    manifest_path = workspace / "solution_observations.json"
    svg_path = workspace / "prediction_overview.svg"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["solution_id"] == "solution_001"
    assert manifest["privacy_boundary"] == "prediction_only_no_validation_labels"
    assert manifest["plots"][0]["path"] == "solutions/solution_001/prediction_overview.svg"
    assert manifest["arrays"]["predict_input.x_val"]["shape"] == [5, 1]
    assert manifest["arrays"]["predictions.predictions"]["shape"] == [5, 1]
    assert svg_path.exists()
    assert "Observation manifest:" in llm.last_prompt
    assert "prediction_overview.svg" in llm.last_prompt
    assert "val_data" not in llm.last_prompt
    assert "u_val" not in llm.last_prompt


def test_root_engineer_prompt_excludes_strategy_seed_context(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingLLM()
    bundle = ProblemBundle.load(Path("examples/function_approx"))
    contract = BenchmarkContractFactory.create_contract(bundle)

    RootEngineerAgent(llm, storage).generate(
        "solution_000",
        problem_bundle=bundle,
        contract=contract,
        guidelines="Follow local evaluator only.",
        data_report="Data analyst saw smooth x_train observations.",
        problem_intake_context="User wants sparse sensors and no future leakage.",
    )

    assert "User wants sparse sensors" in llm.last_prompt
    assert "strategy seed catalogs" in llm.last_prompt
    assert "Human/Planner Selected Strategy Seeds" not in llm.last_prompt
    assert "paper_cylinder_bandlimited_filter" not in llm.last_prompt


def test_selector_votes_always_include_best_and_tie_break_by_loss(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    selector = SelectorAgent(VotingLLM(), storage)
    candidates = [
        {
            "node_id": "best",
            "score": {"metric": "validation_mse", "value": 0.1, "higher_is_better": False},
        },
        {
            "node_id": "promising",
            "score": {"metric": "validation_mse", "value": 0.2, "higher_is_better": False},
        },
        {
            "node_id": "worse_loss",
            "score": {"metric": "validation_mse", "value": 0.8, "higher_is_better": False},
        },
    ]

    result = selector.select_with_votes(
        candidates=candidates,
        best_node_id="best",
        max_to_select=2,
        vote_count=3,
    )

    assert result.selected_parent_ids == ["best", "promising"]
    assert result.vote_counts == {"promising": 2, "worse_loss": 2}
    assert (storage.run_dir / "reports" / "selector_votes.json").exists()


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
