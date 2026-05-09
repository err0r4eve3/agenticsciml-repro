from pathlib import Path

from agenticsciml.config import EvaluationContract, ExperimentConfig
from agenticsciml.state import AgentMessage, Proposal, SolutionNode, SolutionScore
from agenticsciml.storage import ExperimentStorage


def test_solution_node_round_trips_through_json() -> None:
    node = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        score=SolutionScore(metric="validation_mse", value=0.125, higher_is_better=False),
        children=["solution_001"],
        status="evaluated",
    )

    restored = SolutionNode.from_dict(node.to_dict())

    assert restored == node
    assert restored.score is not None
    assert restored.score.value == 0.125


def test_storage_creates_solution_workspace_and_transcript(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo-run")
    workspace = storage.create_solution_workspace("solution_000")
    message = AgentMessage(role="proposer", prompt="p", response="r", metadata={"round": 1})

    storage.save_json("config.json", ExperimentConfig(experiment_id="demo-run").to_dict())
    storage.save_transcript("solution_000", "proposer", [message])

    assert workspace.exists()
    assert (storage.run_dir / "config.json").exists()
    assert (workspace / "transcripts" / "proposer.json").exists()


def test_contract_serialization_preserves_commands() -> None:
    contract = EvaluationContract(
        metric_name="validation_mse",
        higher_is_better=False,
        validate_command=["python", "solution.py", "--mode=validate"],
        train_command=["python", "solution.py", "--mode=train"],
        evaluate_command=["python", "evaluate.py"],
    )

    assert EvaluationContract.from_dict(contract.to_dict()).evaluate_command == ["python", "evaluate.py"]


def test_proposal_serialization() -> None:
    proposal = Proposal(
        title="Fourier features",
        diagnosis="The target oscillates.",
        mutation_plan=["Add Fourier feature expansion."],
        expected_effect="Lower MSE",
        risks=["May overfit"],
    )

    assert Proposal.from_dict(proposal.to_dict()).title == "Fourier features"
