from __future__ import annotations

from typing import Any


LLM_MODE_MOCK = "mock"
LLM_MODE_REAL = "real"

EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE = "mock_workflow_shape"
EVIDENCE_MODE_REAL_LLM_SMOKE = "real_llm_smoke"
EVIDENCE_MODE_REAL_LLM_ABLATION = "real_llm_ablation"
SCIENTIFIC_CLAIM_NOT_SUPPORTED = "not_supported"
SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY = "proxy_workflow_only"
SCIENTIFIC_CLAIM_NOT_VALIDATED = "not_validated"
CLAIM_LEVEL_WORKFLOW_PROXY = "workflow_proxy"
CLAIM_LEVEL_PAPER_WORKFLOW = "paper_workflow"
CLAIM_LEVELS = {CLAIM_LEVEL_WORKFLOW_PROXY, CLAIM_LEVEL_PAPER_WORKFLOW}
CLAIM_GATE_ALLOWED = "allowed"
CLAIM_GATE_BLOCKED = "blocked"
CLAIM_GATE_DOWNGRADED = "downgraded"
EVALUATOR_TRUST_SYNTHETIC_PROXY = "synthetic_proxy"
EVALUATOR_TRUST_LOCAL_PROXY = "trusted_local_proxy"
EVALUATOR_TRUST_DOMAIN_REVIEWED = "domain_reviewed"

SCIENTIFIC_CLAIMS = {
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
    SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY,
    SCIENTIFIC_CLAIM_NOT_VALIDATED,
}
PAPER_WORKFLOW_REQUIREMENTS = (
    "real_llm_mode",
    "paper_like_benchmark",
    "domain_evaluator_approval",
    "paper_benchmark_approval",
    "heterogeneous_selector_panel",
    "paper_equivalent_kb",
    "actual_multimodal_evidence",
)
SCIENTIFIC_DISCOVERY_READINESS_REQUIREMENTS = (
    *PAPER_WORKFLOW_REQUIREMENTS,
    "multi_seed_ablation",
    "failure_attribution",
    "expert_blueprint",
    "resource_constraints",
)


def evidence_metadata_for_run(*, use_mock: bool, fidelity_level: str) -> dict[str, object]:
    if use_mock:
        return {
            "llm_mode": LLM_MODE_MOCK,
            "benchmark_fidelity_level": fidelity_level,
            "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
            "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        }
    return {
        "llm_mode": LLM_MODE_REAL,
        "benchmark_fidelity_level": fidelity_level,
        "evidence_mode": f"real_llm_{fidelity_level}_benchmark",
        "scientific_claim": (
            SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY
            if fidelity_level == "proxy"
            else SCIENTIFIC_CLAIM_NOT_VALIDATED
        ),
    }


def claim_gate_for_run(
    *,
    claim_level: str = CLAIM_LEVEL_WORKFLOW_PROXY,
    use_mock: bool,
    fidelity_level: str,
    is_custom_proxy: bool = False,
    domain_evaluator_approved: bool = False,
    domain_reviewer: str | None = None,
    domain_review_notes: str | None = None,
    paper_benchmark_approved: bool = False,
    selector_heterogeneous: bool = False,
    kb_paper_equivalent: bool = False,
    actual_multimodal_evidence: bool = False,
) -> dict[str, Any]:
    normalized_claim_level = claim_level if claim_level in CLAIM_LEVELS else CLAIM_LEVEL_WORKFLOW_PROXY
    evaluator_trust_level = (
        EVALUATOR_TRUST_DOMAIN_REVIEWED
        if domain_evaluator_approved
        else EVALUATOR_TRUST_SYNTHETIC_PROXY
        if is_custom_proxy
        else EVALUATOR_TRUST_LOCAL_PROXY
    )
    paper_benchmark_equivalent = fidelity_level == "paper-like" and paper_benchmark_approved
    domain_evaluator_present = domain_evaluator_approved
    metric_validated_by_domain_expert = domain_evaluator_approved
    common: dict[str, Any] = {
        "schema_version": 1,
        "claim_level": normalized_claim_level,
        "paper_level_claim_supported": False,
        "scientific_claim_supported": False,
        "evaluator_trust_level": evaluator_trust_level,
        "paper_benchmark_equivalent": paper_benchmark_equivalent,
        "domain_evaluator_present": domain_evaluator_present,
        "metric_validated_by_domain_expert": metric_validated_by_domain_expert,
        "domain_reviewer": (domain_reviewer or "").strip() or None,
        "domain_review_notes": (domain_review_notes or "").strip() or None,
        "required_for_paper_workflow": list(PAPER_WORKFLOW_REQUIREMENTS),
    }
    if normalized_claim_level == CLAIM_LEVEL_WORKFLOW_PROXY:
        return {
            **common,
            "status": CLAIM_GATE_ALLOWED,
            "reasons": [
                "workflow_proxy runs are allowed only as workflow evidence",
                "paper-level and scientific claims remain disabled by default",
            ],
        }

    reasons: list[str] = []
    if use_mock:
        reasons.append("paper_workflow requires real LLM mode")
    if fidelity_level != "paper-like":
        reasons.append(f"paper_workflow requires paper-like benchmark fidelity, got {fidelity_level}")
    if not domain_evaluator_approved:
        reasons.append("paper_workflow requires human domain evaluator approval")
    if not ((domain_reviewer or "").strip() and (domain_review_notes or "").strip()):
        reasons.append("paper_workflow requires domain reviewer and review notes")
    if not paper_benchmark_approved:
        reasons.append("paper_workflow requires explicit paper benchmark approval")
    if not selector_heterogeneous:
        reasons.append("paper_workflow requires heterogeneous selector panel evidence")
    if not kb_paper_equivalent:
        reasons.append("paper_workflow requires paper-equivalent KB provenance")
    if not actual_multimodal_evidence:
        reasons.append("paper_workflow requires actual multimodal image evidence use")
    if reasons:
        return {
            **common,
            "status": CLAIM_GATE_BLOCKED,
            "reasons": reasons,
        }
    return {
        **common,
        "status": CLAIM_GATE_ALLOWED,
        "paper_level_claim_supported": True,
        "scientific_claim_supported": True,
        "reasons": ["paper_workflow claim gate requirements passed"],
    }
