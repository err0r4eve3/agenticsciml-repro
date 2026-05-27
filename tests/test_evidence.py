from __future__ import annotations

from pathlib import Path

import pytest

from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    LLM_MODE_MOCK,
    LLM_MODE_REAL,
    SCIENTIFIC_DISCOVERY_READINESS_REQUIREMENTS,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
    SCIENTIFIC_CLAIM_NOT_VALIDATED,
    SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY,
    SCIENTIFIC_CLAIMS,
    claim_gate_for_run,
    evidence_metadata_for_run,
)
from agenticsciml.benchmarks import benchmark_for_path
from agenticsciml.scientific_readiness import build_scientific_discovery_readiness_report


@pytest.mark.parametrize("fidelity_level", ["proxy", "faithful-small", "paper-like"])
def test_mock_runs_never_support_scientific_claims(fidelity_level: str) -> None:
    metadata = evidence_metadata_for_run(use_mock=True, fidelity_level=fidelity_level)

    assert metadata["llm_mode"] == LLM_MODE_MOCK
    assert metadata["benchmark_fidelity_level"] == fidelity_level
    assert metadata["evidence_mode"] == EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE
    assert metadata["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED
    assert metadata["scientific_claim"] in SCIENTIFIC_CLAIMS


def test_real_proxy_runs_are_limited_to_proxy_workflow_claims() -> None:
    metadata = evidence_metadata_for_run(use_mock=False, fidelity_level="proxy")

    assert metadata["llm_mode"] == LLM_MODE_REAL
    assert metadata["evidence_mode"] == "real_llm_proxy_benchmark"
    assert metadata["scientific_claim"] == SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY
    assert metadata["scientific_claim"] in SCIENTIFIC_CLAIMS


@pytest.mark.parametrize("fidelity_level", ["faithful-small", "paper-like"])
def test_real_non_proxy_runs_start_as_not_validated(fidelity_level: str) -> None:
    metadata = evidence_metadata_for_run(use_mock=False, fidelity_level=fidelity_level)

    assert metadata["llm_mode"] == LLM_MODE_REAL
    assert metadata["evidence_mode"] == f"real_llm_{fidelity_level}_benchmark"
    assert metadata["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_VALIDATED
    assert metadata["scientific_claim"] in SCIENTIFIC_CLAIMS


def test_workflow_proxy_claim_gate_allows_launch_without_scientific_claims() -> None:
    gate = claim_gate_for_run(
        claim_level="workflow_proxy",
        use_mock=True,
        fidelity_level="proxy",
        is_custom_proxy=True,
    )

    assert gate["status"] == "allowed"
    assert gate["paper_level_claim_supported"] is False
    assert gate["scientific_claim_supported"] is False
    assert gate["evaluator_trust_level"] == "synthetic_proxy"


def test_paper_workflow_claim_gate_blocks_missing_paper_evidence() -> None:
    gate = claim_gate_for_run(
        claim_level="paper_workflow",
        use_mock=True,
        fidelity_level="faithful-small",
        domain_evaluator_approved=False,
        paper_benchmark_approved=False,
        selector_heterogeneous=False,
        kb_paper_equivalent=False,
        actual_multimodal_evidence=False,
    )

    assert gate["status"] == "blocked"
    assert gate["paper_level_claim_supported"] is False
    assert "paper_workflow requires real LLM mode" in gate["reasons"]
    assert "paper_workflow requires paper-like benchmark fidelity, got faithful-small" in gate["reasons"]
    assert "paper_workflow requires domain reviewer and review notes" in gate["reasons"]


def test_scientific_discovery_readiness_fails_closed_for_mock_runs() -> None:
    benchmark = benchmark_for_path(Path("examples/cylinder_wake_reconstruction_faithful_small").resolve())
    assert benchmark is not None
    gate = claim_gate_for_run(
        claim_level="paper_workflow",
        use_mock=True,
        fidelity_level=benchmark.fidelity_level,
        paper_benchmark_approved=False,
        selector_heterogeneous=False,
        kb_paper_equivalent=False,
        actual_multimodal_evidence=False,
    )

    report = build_scientific_discovery_readiness_report(
        benchmark=benchmark,
        use_mock=True,
        claim_level="paper_workflow",
        claim_gate=gate,
        selector_diversity={"heterogeneous_selector_evidence": False},
        kb_manifest={"paper_kb_equivalent": False},
        visual_audit_manifest={"actual_image_inputs_used": False, "audited_solution_count": 1},
        method_experience_manifest={
            "record_count": 1,
            "failure_attribution_present": True,
            "classification_counts": {"success": 1},
        },
        domain_evaluator_approved=False,
        domain_reviewer=None,
        domain_review_notes=None,
        paper_benchmark_approved=False,
        resource_constraints={"cpu": "local"},
        expert_blueprint_id="fluid_pde",
        problem_intake={"data_source": "synthetic fixture"},
        planner_snapshot={},
    )

    assert report["status"] == "blocked"
    assert report["scientific_claim_supported"] is False
    blocker_ids = {blocker["check_id"] for blocker in report["blockers"]}
    assert "real_llm" in blocker_ids
    assert "actual_image_inputs" in blocker_ids
    assert "multi_seed_ablation" in blocker_ids
    assert "actual_multimodal_evidence" in SCIENTIFIC_DISCOVERY_READINESS_REQUIREMENTS
