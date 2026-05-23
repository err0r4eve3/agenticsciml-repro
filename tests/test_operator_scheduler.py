from __future__ import annotations

import json
from pathlib import Path

from agenticsciml.operator_scheduler import OperatorScheduler
from agenticsciml.state import SolutionNode, SolutionScore


def _node(
    node_id: str,
    *,
    workspace: Path,
    score: float = 1.0,
    delta: float | None = None,
    children: list[str] | None = None,
) -> SolutionNode:
    workspace.mkdir(parents=True, exist_ok=True)
    return SolutionNode(
        node_id=node_id,
        parent_id=None,
        workspace=str(workspace),
        score=SolutionScore(metric="validation_mse", value=score, higher_is_better=False),
        children=children or [],
        status="evaluated",
        method_tags=[],
        score_delta_from_parent=delta,
    )


def test_operator_scheduler_uses_selected_compatible_algorithms_only(tmp_path: Path) -> None:
    parent = _node("solution_000", workspace=tmp_path / "solutions" / "solution_000")
    scheduler = OperatorScheduler(
        benchmark_name="function_approx",
        benchmark_family="function approximation",
        selected_algorithm_ids=["piecewise_local_basis", "pinn_residual_minimizer"],
        nodes=[parent],
        run_dir=tmp_path,
    )

    assignment = scheduler.assign(
        solution_id="solution_001",
        parent=parent,
        branch_context={"branch_intent": "features_or_architecture"},
    ).to_dict()

    assert assignment["operator_id"] == "piecewise_local_basis"
    assert assignment["selection_source"] == "manual_selected"
    assert assignment["candidate_operator_ids"] == ["piecewise_local_basis"]
    assert "pinn_residual_minimizer" in assignment["warnings"][0]


def test_operator_scheduler_auto_selects_benchmark_compatible_operator(tmp_path: Path) -> None:
    parent = _node("solution_000", workspace=tmp_path / "solutions" / "solution_000")
    scheduler = OperatorScheduler(
        benchmark_name="cylinder_wake_reconstruction_faithful_small",
        benchmark_family="inverse reconstruction",
        selected_algorithm_ids=[],
        nodes=[parent],
        run_dir=tmp_path,
    )

    assignment = scheduler.assign(solution_id="solution_001", parent=parent, branch_context={}).to_dict()

    assert assignment["selection_source"] == "auto_compatible"
    assert assignment["operator_id"] in assignment["candidate_operator_ids"]
    assert "sensor_reconstruction" in assignment["compatibility_reason"] or "benchmark example" in assignment[
        "compatibility_reason"
    ]


def test_operator_scheduler_diversifies_same_parent_fanout_axes(tmp_path: Path) -> None:
    parent = _node("solution_000", workspace=tmp_path / "solutions" / "solution_000")
    scheduler = OperatorScheduler(
        benchmark_name="burgers_pinn_faithful_small",
        benchmark_family="PINN",
        selected_algorithm_ids=["fourier_feature_mlp", "pinn_residual_minimizer"],
        nodes=[parent],
        run_dir=tmp_path,
    )
    used_axes: set[str] = set()

    first = scheduler.assign(
        solution_id="solution_001",
        parent=parent,
        branch_context={"branch_intent": "features_or_architecture"},
        used_axes_for_parent=used_axes,
    )
    used_axes.add(first.mutation_axis)
    second = scheduler.assign(
        solution_id="solution_002",
        parent=parent,
        branch_context={"branch_intent": "training_stability"},
        used_axes_for_parent=used_axes,
    )

    assert first.operator_id != second.operator_id
    assert first.mutation_axis != second.mutation_axis


def test_operator_scheduler_penalizes_plateaued_operator_history(tmp_path: Path) -> None:
    previous_workspace = tmp_path / "solutions" / "solution_001"
    previous = _node("solution_001", workspace=previous_workspace, delta=0.0)
    (previous_workspace / "operator_assignment.json").write_text(
        json.dumps({"operator_id": "fourier_feature_mlp", "mutation_axis": "representation_or_features"}),
        encoding="utf-8",
    )
    (previous_workspace / "mutation_effect_report.json").write_text(
        json.dumps({"status": "changed_but_score_plateau", "operator_id": "fourier_feature_mlp"}),
        encoding="utf-8",
    )
    parent = _node("solution_000", workspace=tmp_path / "solutions" / "solution_000")
    scheduler = OperatorScheduler(
        benchmark_name="function_approx_faithful_small",
        benchmark_family="function approximation",
        selected_algorithm_ids=["fourier_feature_mlp", "piecewise_local_basis"],
        nodes=[parent, previous],
        run_dir=tmp_path,
    )

    assignment = scheduler.assign(solution_id="solution_001", parent=parent, branch_context={}).to_dict()

    assert assignment["operator_id"] == "piecewise_local_basis"
    assert assignment["penalized_operator_ids"] == ["fourier_feature_mlp"]
