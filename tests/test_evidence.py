from __future__ import annotations

import pytest

from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    LLM_MODE_MOCK,
    LLM_MODE_REAL,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
    SCIENTIFIC_CLAIM_NOT_VALIDATED,
    SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY,
    SCIENTIFIC_CLAIMS,
    claim_gate_for_run,
    evidence_metadata_for_run,
)


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
