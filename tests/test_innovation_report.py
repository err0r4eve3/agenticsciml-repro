import json
from pathlib import Path

from agenticsciml.audit_reports import (
    build_innovation_report,
    build_scientific_result_card,
    render_innovation_report_markdown,
    render_scientific_result_card_markdown,
)
from agenticsciml.emergence_audit import audit_solution_emergence
from agenticsciml.state import SolutionNode, SolutionScore


def _write_solution_artifacts(workspace: Path, *, proposal: str, engineering: str, code: str) -> None:
    workspace.mkdir(parents=True)
    (workspace / "proposal.md").write_text(proposal, encoding="utf-8")
    (workspace / "engineering_summary.md").write_text(engineering, encoding="utf-8")
    (workspace / "solution.py").write_text(code, encoding="utf-8")


def test_innovation_report_records_workflow_exploration_without_scientific_claims(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    root_workspace = run_dir / "solutions" / "solution_000"
    child_workspace = run_dir / "solutions" / "solution_001"
    _write_solution_artifacts(
        root_workspace,
        proposal="Use a baseline ridge regressor.",
        engineering="Implemented deterministic ridge baseline.",
        code="PREDICTION_MODE = 'ridge'\n",
    )
    _write_solution_artifacts(
        child_workspace,
        proposal=(
            "Combine Fourier feature lifting with sparse sensor smoothing and a residual-style "
            "regularization schedule adapted from parent score analysis."
        ),
        engineering="Implemented Fourier features, bandlimited filtering, and regularization schedule.",
        code="FEATURES = ['fourier', 'bandlimited_filter']\nREGULARIZATION = 1e-3\n",
    )
    (child_workspace / "mutation_effect_report.json").write_text(
        json.dumps({"status": "changed_score_moved", "diff_line_count": 12}),
        encoding="utf-8",
    )
    (child_workspace / "kb_application_report.json").write_text(
        json.dumps({"status": "implemented", "warnings": []}),
        encoding="utf-8",
    )
    (child_workspace / "emergence_report.json").write_text(
        json.dumps({"claim_level": "candidate_emergent", "blocking_gaps": []}),
        encoding="utf-8",
    )
    nodes = [
        SolutionNode(
            node_id="solution_000",
            parent_id=None,
            workspace=str(root_workspace),
            score=SolutionScore(metric="relative_l2", value=0.9, higher_is_better=False),
            children=["solution_001"],
            status="evaluated",
            method_tags=["root_baseline"],
        ),
        SolutionNode(
            node_id="solution_001",
            parent_id="solution_000",
            workspace=str(child_workspace),
            score=SolutionScore(metric="relative_l2", value=0.6, higher_is_better=False),
            status="evaluated",
            method_tags=["fourier_feature", "bandlimited_filter"],
            score_delta_from_parent=-0.3,
        ),
    ]

    report = build_innovation_report(
        nodes=nodes,
        run_dir=run_dir,
        benchmark_name="cylinder_wake_reconstruction_faithful_small",
        strategy_seed_ids=["paper_cylinder_bandlimited_filter", "fourier_feature_mlp"],
        problem_intake={"problem_statement": "Recover vorticity from sparse sensors."},
        planner_snapshot={"selected_algorithm_ids": ["paper_cylinder_bandlimited_filter"]},
        evolution_health={
            "solution_count": 2,
            "unique_code_count": 2,
            "duplicate_code_count": 0,
            "best_improvement": 0.3,
            "warnings": [],
        },
        claim_gate={
            "status": "allowed",
            "scientific_claim_supported": False,
            "paper_level_claim_supported": False,
        },
    )
    markdown = render_innovation_report_markdown(report)

    assert report["schema_version"] == 1
    assert report["innovation_claim_level"] == "workflow_exploration_only"
    assert report["scientific_novelty_supported"] is False
    assert report["paper_level_discovery_supported"] is False
    assert report["evidence_summary"]["solution_count"] == 2
    assert report["evidence_summary"]["candidate_emergent_count"] == 1
    assert {axis["axis_id"] for axis in report["novelty_axes"]} >= {
        "representation_or_features",
        "data_or_sensor_processing",
        "optimization_or_schedule",
    }
    child = next(item for item in report["solution_innovation"] if item["node_id"] == "solution_001")
    assert child["innovation_signal_count"] >= 3
    assert child["emergence_claim_level"] == "candidate_emergent"
    assert any("not scientific novelty evidence" in warning for warning in report["warnings"])
    assert "workflow_exploration_only" in markdown
    assert "not scientific novelty evidence" in markdown

    scientific_card = build_scientific_result_card(
        nodes=nodes,
        champion=nodes[1],
        run_dir=run_dir,
        benchmark_name="cylinder_wake_reconstruction_faithful_small",
        evidence_metadata={
            "llm_mode": "mock",
            "evidence_mode": "mock_workflow_shape",
            "benchmark_fidelity_level": "faithful-small",
            "claim_gate": {
                "status": "allowed",
                "scientific_claim_supported": False,
                "paper_level_claim_supported": False,
                "evaluator_trust_level": "trusted_local_proxy",
            },
        },
        evolution_health={
            "solution_count": 2,
            "unique_code_count": 2,
            "duplicate_code_count": 0,
            "max_plateau_length": 1,
            "best_improvement": 0.3,
        },
        innovation_report=report,
        scientific_readiness={
            "status": "blocked",
            "scientific_claim_supported": False,
            "blockers": [{"check_id": "real_llm", "message": "run used real LLM mode"}],
        },
    )
    card_markdown = render_scientific_result_card_markdown(scientific_card)

    assert scientific_card["evidence_grade"] == "workflow_evidence_only"
    assert scientific_card["score"]["metric"] == "relative_l2"
    assert round(scientific_card["score"]["improvement_over_root"], 6) == 0.3
    assert scientific_card["claim_support"]["scientific_claim_supported"] is False
    assert "mock LLM mode validates workflow shape only" in scientific_card["uncertainty_flags"]
    assert any(item.startswith("resolve real_llm") for item in scientific_card["minimum_next_validation"])
    assert "Scientific Result Card" in card_markdown
    assert "scientific claim supported: False" in card_markdown


def test_emergence_audit_rejects_incompatible_score_metric_and_direction(
    tmp_path: Path,
) -> None:
    root_workspace = tmp_path / "solutions" / "solution_000"
    parent_workspace = tmp_path / "solutions" / "solution_001"
    child_workspace = tmp_path / "solutions" / "solution_002"
    for workspace in (root_workspace, parent_workspace, child_workspace):
        workspace.mkdir(parents=True)
    (child_workspace / "proposal.md").write_text(
        "Parent analysis found a residual plateau; adapt the boundary correction.",
        encoding="utf-8",
    )
    (child_workspace / "analysis.md").write_text(
        "The parent residual and score changed after the correction.",
        encoding="utf-8",
    )
    (child_workspace / "policy_fidelity_report.json").write_text(
        json.dumps({"execution_allowed": True, "status": "passed"}),
        encoding="utf-8",
    )
    root = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace=str(root_workspace),
        score=SolutionScore("relative_l2", 1.0, higher_is_better=False),
        status="evaluated",
    )
    parent = SolutionNode(
        node_id="solution_001",
        parent_id="solution_000",
        workspace=str(parent_workspace),
        score=SolutionScore("accuracy", 0.9, higher_is_better=True),
        status="evaluated",
    )
    child = SolutionNode(
        node_id="solution_002",
        parent_id="solution_001",
        workspace=str(child_workspace),
        score=SolutionScore("relative_l2", 0.2, higher_is_better=False),
        status="evaluated",
        score_delta_from_parent=-0.7,
    )

    report = audit_solution_emergence(
        node=child,
        parent_node=parent,
        root_node=root,
        benchmark_dir=Path("examples/function_approx"),
        strategy_seed_ids=[],
    )

    comparison = report["score_evidence"]["parent_comparison"]
    assert comparison["comparable"] is False
    assert comparison["metric_matches"] is False
    assert comparison["direction_matches"] is False
    assert report["score_evidence"]["improved_over_parent"] is False
    assert "parent_score_metric_mismatch" in report["blocking_gaps"]
    assert "parent_score_direction_mismatch" in report["blocking_gaps"]
    assert report["claim_level"] != "candidate_emergent"


def test_innovation_report_ignores_probe_variable_and_bookkeeping_tags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    workspace = run_dir / "solutions" / "solution_000"
    _write_solution_artifacts(
        workspace,
        proposal="Use a plain baseline.",
        engineering="Implemented the plain baseline.",
        code="probe = model.predict(inputs)\n",
    )
    root = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace=str(workspace),
        score=SolutionScore(metric="relative_l2", value=1.0, higher_is_better=False),
        status="evaluated",
        method_tags=["root_baseline", "operator:baseline_mlp_regressor", "axis:representation_or_features"],
    )

    report = build_innovation_report(
        nodes=[root],
        run_dir=run_dir,
        benchmark_name="function_approx",
        strategy_seed_ids=["fourier_feature_mlp", "kernel_surrogate_regression"],
        problem_intake={},
        planner_snapshot={},
        evolution_health={"unique_code_count": 1, "duplicate_code_count": 0, "warnings": []},
        claim_gate={"status": "allowed", "scientific_claim_supported": False},
    )

    assert report["novelty_axes"] == []
    assert report["solution_innovation"][0]["novelty_axis_ids"] == []


def test_scientific_result_card_lists_every_readiness_blocker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    workspace = run_dir / "solutions" / "solution_000"
    _write_solution_artifacts(workspace, proposal="baseline", engineering="baseline", code="MODEL = 1\n")
    root = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace=str(workspace),
        score=SolutionScore(metric="relative_l2", value=1.0, higher_is_better=False),
        status="evaluated",
        method_tags=["root_baseline"],
    )
    blockers = [
        {"check_id": f"gate_{index}", "message": f"missing gate {index}"}
        for index in range(8)
    ]

    card = build_scientific_result_card(
        nodes=[root],
        champion=root,
        run_dir=run_dir,
        benchmark_name="function_approx",
        evidence_metadata={
            "llm_mode": "mock",
            "evidence_mode": "mock_workflow_shape",
            "benchmark_fidelity_level": "proxy",
            "claim_gate": {"status": "allowed", "scientific_claim_supported": False},
        },
        evolution_health={"unique_code_count": 1, "duplicate_code_count": 0, "max_plateau_length": 1},
        innovation_report={"evidence_summary": {"candidate_emergent_count": 0}},
        scientific_readiness={"status": "blocked", "scientific_claim_supported": False, "blockers": blockers},
    )

    assert [item for item in card["minimum_next_validation"] if item.startswith("resolve gate_")] == [
        f"resolve gate_{index}: missing gate {index}"
        for index in range(8)
    ]
