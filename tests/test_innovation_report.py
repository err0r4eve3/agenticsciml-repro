import json
from pathlib import Path

from agenticsciml.audit_reports import (
    build_innovation_report,
    build_scientific_result_card,
    render_innovation_report_markdown,
    render_scientific_result_card_markdown,
)
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
