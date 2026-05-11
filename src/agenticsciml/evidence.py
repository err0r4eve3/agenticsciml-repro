from __future__ import annotations


LLM_MODE_MOCK = "mock"
LLM_MODE_REAL = "real"

EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE = "mock_workflow_shape"
SCIENTIFIC_CLAIM_NOT_SUPPORTED = "not_supported"
SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY = "proxy_workflow_only"
SCIENTIFIC_CLAIM_NOT_VALIDATED = "not_validated"

SCIENTIFIC_CLAIMS = {
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
    SCIENTIFIC_CLAIM_PROXY_WORKFLOW_ONLY,
    SCIENTIFIC_CLAIM_NOT_VALIDATED,
}


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
