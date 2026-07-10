from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenticsciml.audit_reports import build_evolution_health_report
from agenticsciml.operator_scheduler import OperatorScheduler
from agenticsciml.state import SolutionNode, SolutionScore


def _node(
    node_id: str,
    *,
    workspace: Path,
    score: float = 1.0,
    delta: float | None = None,
    parent_id: str | None = None,
    children: list[str] | None = None,
    method_tags: list[str] | None = None,
) -> SolutionNode:
    workspace.mkdir(parents=True, exist_ok=True)
    return SolutionNode(
        node_id=node_id,
        parent_id=parent_id,
        workspace=str(workspace),
        score=SolutionScore(metric="validation_mse", value=score, higher_is_better=False),
        children=children or [],
        status="evaluated",
        method_tags=method_tags or [],
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


def test_operator_scheduler_keeps_selected_seeds_for_custom_ks_proxy(tmp_path: Path) -> None:
    parent = _node("solution_000", workspace=tmp_path / "solutions" / "solution_000")
    selected = [
        "finite_difference_residual_probe",
        "deeponet_operator",
        "fno_lite_operator",
        "paper_reaction_diffusion_fno_helpers",
    ]
    scheduler = OperatorScheduler(
        benchmark_name="custom_benchmark_custom_proxy_kuramoto_sivashinsky_893c0a7f2e",
        benchmark_family="custom operator learning",
        selected_algorithm_ids=selected,
        nodes=[parent],
        run_dir=tmp_path,
    )

    assignment = scheduler.assign(solution_id="solution_001", parent=parent, branch_context={}).to_dict()

    assert assignment["selection_source"] == "manual_selected"
    assert assignment["operator_id"] in selected
    assert "baseline_mlp_regressor" not in assignment["candidate_operator_ids"]
    assert assignment["warnings"] == []


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


def test_evolution_health_audits_operator_assignment_consistency(tmp_path: Path) -> None:
    root = _node("solution_000", workspace=tmp_path / "solutions" / "solution_000", children=["solution_001"])
    child_missing = _node(
        "solution_001",
        workspace=tmp_path / "solutions" / "solution_001",
        parent_id="solution_000",
        method_tags=[],
    )
    child_mismatch = _node(
        "solution_002",
        workspace=tmp_path / "solutions" / "solution_002",
        parent_id="solution_000",
        method_tags=["operator:wrong", "axis:wrong"],
    )
    for node in (root, child_missing, child_mismatch):
        Path(node.workspace, "solution.py").write_text(f"# {node.node_id}\n", encoding="utf-8")
    Path(child_mismatch.workspace, "operator_assignment.json").write_text(
        json.dumps(
            {
                "operator_id": "fourier_feature_mlp",
                "mutation_axis": "representation_or_features",
                "warnings": ["fixture warning"],
            }
        ),
        encoding="utf-8",
    )

    report = build_evolution_health_report([root, child_missing, child_mismatch], tmp_path)

    assert report["operator_scheduler_mode"] == "auto-audited"
    assert report["operator_assignment_expected_count"] == 2
    assert report["operator_assignment_count"] == 1
    assert report["missing_operator_assignment_nodes"] == ["solution_001"]
    assert report["operator_method_tag_mismatch_nodes"] == ["solution_002"]
    assert report["operator_assignment_warning_count"] == 1
    assert any("Operator assignment missing" in warning for warning in report["warnings"])
    assert any("method_tags are inconsistent" in warning for warning in report["warnings"])


def test_operator_scheduler_rejects_unknown_selected_algorithm(tmp_path: Path) -> None:
    parent = _node("solution_000", workspace=tmp_path / "solutions" / "solution_000")

    with pytest.raises(ValueError, match="Unknown selected algorithm id.*not-a-real-operator"):
        OperatorScheduler(
            benchmark_name="function_approx",
            benchmark_family="function approximation",
            selected_algorithm_ids=["not-a-real-operator"],
            nodes=[parent],
            run_dir=tmp_path,
        )


def test_evolution_health_does_not_credit_unverified_operator_implementation(tmp_path: Path) -> None:
    root = _node(
        "solution_000",
        workspace=tmp_path / "solutions" / "solution_000",
        score=1.0,
        children=["solution_001"],
    )
    child = _node(
        "solution_001",
        workspace=tmp_path / "solutions" / "solution_001",
        score=0.4,
        delta=-0.6,
        parent_id="solution_000",
        method_tags=["operator:kernel_surrogate_regression", "axis:optimization_or_schedule"],
    )
    for node in (root, child):
        Path(node.workspace, "solution.py").write_text("MODEL = 'fourier_ridge'\n", encoding="utf-8")
    Path(child.workspace, "operator_assignment.json").write_text(
        json.dumps(
            {
                "operator_id": "kernel_surrogate_regression",
                "mutation_axis": "optimization_or_schedule",
                "operator_expected_terms": ["kernel", "surrogate"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    Path(child.workspace, "mutation_effect_report.json").write_text(
        json.dumps(
            {
                "status": "changed_score_moved",
                "operator_id": "kernel_surrogate_regression",
                "operator_implementation_verified": False,
                "operator_static_evidence": {
                    "kernel": {"proposal_signal": False, "engineer_signal": False, "code_signal": False},
                    "surrogate": {"proposal_signal": False, "engineer_signal": False, "code_signal": False},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_evolution_health_report([root, child], tmp_path)
    health = report["operator_health"]["kernel_surrogate_regression"]

    assert health["assigned"] == 1
    assert health["evaluated"] == 1
    assert health["improved"] == 0
    assert health["best_improvement"] is None
    assert health["implementation_verified"] == 0
    assert health["implementation_unverified"] == 1
    assert report["operator_implementation_unverified_nodes"] == ["solution_001"]
    assert any("not verified" in warning for warning in report["warnings"])
