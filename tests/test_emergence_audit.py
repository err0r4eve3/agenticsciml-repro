from __future__ import annotations

import json
from pathlib import Path

from agenticsciml.emergence_audit import audit_solution_emergence
from agenticsciml.state import SolutionNode, SolutionScore


def _node(
    node_id: str,
    workspace: Path,
    *,
    parent_id: str | None = None,
    score: float | None = None,
    score_delta: float | None = None,
    method_tags: list[str] | None = None,
) -> SolutionNode:
    return SolutionNode(
        node_id=node_id,
        parent_id=parent_id,
        workspace=str(workspace),
        score=SolutionScore("validation_mse", score) if score is not None else None,
        status="evaluated" if score is not None else "failed",
        proposal_path=str(workspace / "proposal.md"),
        analysis_path=str(workspace / "analysis.md"),
        benchmark_name="function_approx",
        contract_hash="a" * 64,
        method_tags=method_tags or [],
        score_delta_from_parent=score_delta,
    )


def test_emergence_audit_marks_missing_artifacts_not_assessed(tmp_path: Path) -> None:
    workspace = tmp_path / "solutions" / "solution_000"
    workspace.mkdir(parents=True)
    node = _node("solution_000", workspace, score=1.0)

    report = audit_solution_emergence(
        node=node,
        parent_node=None,
        root_node=node,
        benchmark_dir=Path("examples/function_approx"),
        strategy_seed_ids=[],
    )

    assert report["auditor_version"] == "emergence_audit.v1"
    assert report["claim_level"] == "not_assessed"
    assert "proposal_missing" in report["blocking_gaps"]


def test_emergence_audit_does_not_promote_direct_kb_match(tmp_path: Path) -> None:
    workspace = tmp_path / "solutions" / "solution_001"
    workspace.mkdir(parents=True)
    (workspace / "proposal.md").write_text(
        "# Proposal\n\nUse mixture of experts with gated experts for piecewise regime changes.\n",
        encoding="utf-8",
    )
    (workspace / "analysis.md").write_text("Improves discontinuity fit.", encoding="utf-8")
    (workspace / "policy_fidelity_report.json").write_text(
        json.dumps({"execution_allowed": True, "status": "passed"}),
        encoding="utf-8",
    )
    root = _node("solution_000", tmp_path / "solutions" / "solution_000", score=1.0)
    parent = _node("solution_000", tmp_path / "solutions" / "solution_000", score=1.0)
    node = _node(
        "solution_001",
        workspace,
        parent_id="solution_000",
        score=0.5,
        score_delta=-0.5,
        method_tags=["mixture_of_experts"],
    )

    report = audit_solution_emergence(
        node=node,
        parent_node=parent,
        root_node=root,
        benchmark_dir=Path("examples/function_approx"),
        strategy_seed_ids=[],
    )

    assert report["claim_level"] == "kb_direct"
    assert report["kb_overlap"]["direct_match_entry_ids"]
    assert "direct_kb_overlap" in report["blocking_gaps"]


def test_emergence_audit_allows_candidate_only_with_prior_score_and_policy_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "solutions" / "solution_002"
    workspace.mkdir(parents=True)
    (workspace / "proposal.md").write_text(
        "\n".join(
            [
                "# Proposal",
                "Parent analysis shows boundary residual spikes and score plateau.",
                "Create an asymmetric residual bridge with a local interface correction term.",
                "This adapts prior failure evidence rather than copying a stored method.",
            ]
        ),
        encoding="utf-8",
    )
    (workspace / "analysis.md").write_text(
        "The interface correction reduced parent residual errors.",
        encoding="utf-8",
    )
    (workspace / "policy_fidelity_report.json").write_text(
        json.dumps(
            {
                "execution_allowed": True,
                "status": "passed",
                "summary": {"failed_blocker_count": 0, "failed_warning_count": 0},
            }
        ),
        encoding="utf-8",
    )
    root_workspace = tmp_path / "solutions" / "solution_000"
    root_workspace.mkdir(parents=True)
    parent_workspace = tmp_path / "solutions" / "solution_001"
    parent_workspace.mkdir(parents=True)
    root = _node("solution_000", root_workspace, score=1.2)
    parent = _node("solution_001", parent_workspace, parent_id="solution_000", score=0.9)
    node = _node(
        "solution_002",
        workspace,
        parent_id="solution_001",
        score=0.4,
        score_delta=-0.5,
        method_tags=["interface_bridge"],
    )

    report = audit_solution_emergence(
        node=node,
        parent_node=parent,
        root_node=root,
        benchmark_dir=Path("examples/function_approx"),
        strategy_seed_ids=[],
    )

    assert report["claim_level"] == "candidate_emergent"
    assert report["score_evidence"]["improved_over_parent"] is True
    assert report["prior_result_evidence"]["present"] is True
    assert report["policy_fidelity_evidence"]["passing"] is True
    assert report["blocking_gaps"] == []
