import json
from pathlib import Path

import pytest

from agenticsciml.config import EvaluationContract, EvolutionConfig, ExperimentConfig
from agenticsciml.state import (
    AgentMessage,
    Proposal,
    SOLUTION_TREE_SCHEMA_VERSION,
    SolutionNode,
    SolutionScore,
    validate_solution_tree_artifact_payload,
    validate_solution_tree_payload,
)
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


def test_solution_node_from_dict_rejects_missing_required_field() -> None:
    node = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        status="created",
    )
    payload = node.to_dict()
    del payload["status"]

    with pytest.raises(ValueError, match="missing required field status"):
        SolutionNode.from_dict(payload)


def test_solution_node_from_dict_rejects_invalid_status() -> None:
    node = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        status="created",
    )
    payload = node.to_dict()
    payload["status"] = "done"

    with pytest.raises(ValueError, match="invalid status"):
        SolutionNode.from_dict(payload)


def test_solution_node_from_dict_rejects_invalid_score_shape() -> None:
    node = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        status="created",
    )
    payload = node.to_dict()
    payload["score"] = {"metric": "", "value": "bad", "higher_is_better": "no"}

    with pytest.raises(ValueError, match="score.value must be a finite number"):
        SolutionNode.from_dict(payload)


def test_solution_node_from_dict_rejects_non_finite_score_values() -> None:
    node = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        score=SolutionScore(metric="validation_mse", value=0.125, higher_is_better=False),
        status="evaluated",
    )
    payload = node.to_dict()
    payload["score"]["value"] = float("nan")

    with pytest.raises(ValueError, match="score.value must be a finite number"):
        SolutionNode.from_dict(payload)

    payload = node.to_dict()
    payload["score_delta_from_parent"] = float("inf")

    with pytest.raises(ValueError, match="score_delta_from_parent must be a finite number or null"):
        SolutionNode.from_dict(payload)


def test_solution_node_from_dict_rejects_invalid_status_semantics() -> None:
    evaluated = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        status="evaluated",
    ).to_dict()

    with pytest.raises(ValueError, match="evaluated node must have a score"):
        SolutionNode.from_dict(evaluated)

    failed = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        score=SolutionScore(metric="validation_mse", value=1.0, higher_is_better=False),
        status="failed",
    ).to_dict()

    with pytest.raises(ValueError, match="failed node must not have a score"):
        SolutionNode.from_dict(failed)

    created = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        score=SolutionScore(metric="validation_mse", value=1.0, higher_is_better=False),
        status="created",
    ).to_dict()

    with pytest.raises(ValueError, match="created node must not have a score"):
        SolutionNode.from_dict(created)


def test_solution_node_from_dict_rejects_unknown_fields() -> None:
    node = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="runs/demo/solutions/solution_000",
        score=SolutionScore(metric="validation_mse", value=1.0, higher_is_better=False),
        status="evaluated",
    )
    payload = node.to_dict()
    payload["surprise"] = True

    with pytest.raises(ValueError, match="unknown fields: surprise"):
        SolutionNode.from_dict(payload)

    payload = node.to_dict()
    payload["score"]["surprise"] = True

    with pytest.raises(ValueError, match="score has unknown fields: surprise"):
        SolutionNode.from_dict(payload)


def test_solution_tree_payload_rejects_empty_node_list() -> None:
    issues = validate_solution_tree_payload([], context="checkpoint.json")

    assert "checkpoint.json must contain at least one node" in issues


def test_solution_tree_artifact_payload_rejects_unsupported_schema_version() -> None:
    issues = validate_solution_tree_artifact_payload(
        {"schema_version": "solution_tree.v999", "nodes": []},
        context="checkpoint.json",
    )

    assert (
        "checkpoint.json has unsupported schema_version: "
        f"'solution_tree.v999'; expected '{SOLUTION_TREE_SCHEMA_VERSION}'"
    ) in issues


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


def test_storage_rejects_non_finite_json_artifacts(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo-run")

    with pytest.raises(ValueError):
        storage.save_json("bad.json", {"value": float("nan")})

    with pytest.raises(ValueError):
        storage.record_trace("workflow_span", "bad", {"value": float("inf")})


def test_storage_trace_events_get_monotonic_event_sequence(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo-run")

    storage.record_trace("workflow_span", "start", {})
    storage.record_trace("workflow_span", "end", {})

    events = [
        json.loads(line)
        for line in (storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_seq"] for event in events] == [1, 2]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_iterations", -1),
        ("parallel_mutations", 0),
        ("max_children_per_node", 0),
        ("max_debug_retries", -1),
        ("timeout_s", 0),
        ("selector_vote_count", 0),
    ],
)
def test_evolution_config_rejects_out_of_range_values(field_name: str, value: int) -> None:
    with pytest.raises(ValueError, match=field_name):
        EvolutionConfig(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("parallel_mutations", True, "parallel_mutations must be an integer"),
        ("random_seed", False, "random_seed must be an integer"),
        ("use_kb", "false", "use_kb must be a boolean"),
        ("use_debugger", 0, "use_debugger must be a boolean"),
    ],
)
def test_evolution_config_from_dict_rejects_coercible_wrong_types(
    field_name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EvolutionConfig.from_dict({field_name: value})


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
