import json
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


def test_storage_atomic_writes_do_not_leave_temp_files(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo-run")

    storage.save_json("config.json", {"value": 1})
    storage.save_json("config.json", {"value": 2})
    storage.save_text("reports/report.md", "report")
    storage.save_solution_text("solution_000", "analysis.md", "analysis")

    assert storage.load_json("config.json") == {"value": 2}
    assert not list(storage.run_dir.rglob("*.tmp"))


def test_storage_trace_events_get_monotonic_event_sequence(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo-run")

    storage.record_trace("workflow_span", "start", {})
    storage.record_trace("workflow_span", "end", {})

    events = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_seq"] for event in events] == [1, 2]


def test_contract_serialization_preserves_commands() -> None:
    contract = EvaluationContract.default_function_approx()
    contract.evaluate_command = ["python", "evaluate.py"]
    contract.with_computed_hash()

    assert EvaluationContract.from_dict(contract.to_dict()).evaluate_command == ["python", "evaluate.py"]


def test_contract_serialization_rejects_missing_fidelity_metadata() -> None:
    contract = EvaluationContract(
        metric_name="validation_mse",
        higher_is_better=False,
        validate_command=["python", "solution.py", "--mode=validate"],
        train_command=["python", "solution.py", "--mode=train"],
        predict_command=["python", "solution.py", "--mode=predict"],
        evaluate_command=["python", "evaluate.py"],
    )

    try:
        EvaluationContract.from_dict(contract.to_dict())
    except ValueError as exc:
        assert "benchmark_fidelity" in str(exc)
    else:
        raise AssertionError("Expected missing benchmark_fidelity to fail.")


def test_proposal_serialization() -> None:
    proposal = Proposal(
        title="Fourier features",
        diagnosis="The target oscillates.",
        mutation_plan=["Add Fourier feature expansion."],
        expected_effect="Lower MSE",
        risks=["May overfit"],
    )

    assert Proposal.from_dict(proposal.to_dict()).title == "Fourier features"
