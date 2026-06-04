from __future__ import annotations

from typing import Any

from agenticsciml.benchmarks import BenchmarkSpec


SCIENTIFIC_DISCOVERY_READINESS_SCHEMA_VERSION = 1


def build_scientific_discovery_readiness_report(
    *,
    benchmark: BenchmarkSpec,
    use_mock: bool,
    claim_level: str,
    claim_gate: dict[str, Any],
    selector_diversity: dict[str, Any],
    kb_manifest: dict[str, Any],
    visual_audit_manifest: dict[str, Any],
    method_experience_manifest: dict[str, Any],
    domain_evaluator_approved: bool,
    domain_reviewer: str | None,
    domain_review_notes: str | None,
    paper_benchmark_approved: bool,
    resource_constraints: dict[str, Any],
    expert_blueprint_id: str | None,
    problem_intake: dict[str, Any],
    planner_snapshot: dict[str, Any],
    multi_seed_ablation: dict[str, Any] | None = None,
    domain_approval_report: dict[str, Any] | None = None,
    paper_like_benchmark_dossier: dict[str, Any] | None = None,
    selector_heterogeneity_report: dict[str, Any] | None = None,
    multi_seed_ablation_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    domain_report = domain_approval_report or {}
    paper_dossier = paper_like_benchmark_dossier or {}
    selector_report = selector_heterogeneity_report or {}
    multi_seed_manifest = multi_seed_ablation_evidence or {}
    checks = [
        _check(
            "paper_like_benchmark",
            "paper-like benchmark is selected and explicitly approved",
            (
                benchmark.fidelity_level == "paper-like"
                and paper_benchmark_approved
                and paper_dossier.get("paper_like_ready", True) is True
            ),
            {
                "fidelity_level": benchmark.fidelity_level,
                "paper_benchmark_approved": paper_benchmark_approved,
                "paper_like_benchmark_dossier": paper_dossier,
            },
        ),
        _check(
            "real_llm",
            "run used real LLM mode",
            not use_mock,
            {"use_mock": use_mock},
        ),
        _check(
            "heterogeneous_selector",
            "selector evidence comes from at least two non-mock heterogeneous provider/model paths",
            bool(
                selector_report.get("heterogeneous_selector_evidence")
                if selector_report
                else selector_diversity.get("heterogeneous_selector_evidence")
            ),
            selector_report or selector_diversity,
        ),
        _check(
            "paper_equivalent_kb",
            "knowledge base provenance is marked paper-equivalent",
            bool(kb_manifest.get("paper_kb_equivalent")),
            kb_manifest,
        ),
        _check(
            "actual_image_inputs",
            "real visual provider received actual image inputs",
            bool(visual_audit_manifest.get("actual_image_inputs_used")),
            {
                "visual_audit_mode": visual_audit_manifest.get("visual_audit_mode"),
                "actual_image_inputs_used": visual_audit_manifest.get("actual_image_inputs_used"),
                "audited_solution_count": visual_audit_manifest.get("audited_solution_count"),
            },
        ),
        _check(
            "domain_review",
            "human domain evaluator approved evaluator and claim boundary",
            bool(
                domain_report.get("approved")
                if domain_report
                else domain_evaluator_approved and domain_reviewer and domain_review_notes
            ),
            {
                "domain_evaluator_approved": domain_evaluator_approved,
                "domain_reviewer": domain_reviewer,
                "domain_review_notes_present": bool(domain_review_notes),
                "domain_approval_report": domain_report,
            },
        ),
        _check(
            "multi_seed_ablation",
            "multi-seed or ablation evidence is attached to the run",
            _multi_seed_or_ablation_present(
                planner_snapshot,
                multi_seed_ablation or {},
                multi_seed_manifest,
            ),
            {
                "planner_snapshot_keys": sorted(planner_snapshot),
                "multi_seed_ablation": planner_snapshot.get("multi_seed_ablation"),
                "ablation_manifest": planner_snapshot.get("ablation_manifest"),
                "configured_multi_seed_ablation": multi_seed_ablation or {},
                "multi_seed_ablation_evidence": multi_seed_manifest,
            },
        ),
        _check(
            "failure_attribution",
            "method experience records include success/failure/plateau attribution for solution artifacts",
            bool(method_experience_manifest.get("failure_attribution_present")),
            {
                "record_count": method_experience_manifest.get("record_count", 0),
                "classification_counts": method_experience_manifest.get("classification_counts", {}),
            },
        ),
        _check(
            "expert_blueprint",
            "fluid/PDE problem intake records an expert blueprint and resource constraints",
            bool(expert_blueprint_id and resource_constraints),
            {
                "expert_blueprint_id": expert_blueprint_id,
                "resource_constraints": resource_constraints,
                "problem_intake_data_source": problem_intake.get("data_source")
                or problem_intake.get("data_description"),
            },
        ),
    ]
    blockers = [check for check in checks if not check["passed"]]
    scientific_claim_supported = (
        not blockers
        and claim_gate.get("scientific_claim_supported") is True
        and claim_gate.get("status") == "allowed"
    )
    return {
        "schema_version": SCIENTIFIC_DISCOVERY_READINESS_SCHEMA_VERSION,
        "readiness_version": "scientific_discovery_readiness.v1",
        "claim_level": claim_level,
        "benchmark": benchmark.to_dict(),
        "scientific_claim_supported": scientific_claim_supported,
        "status": "ready" if scientific_claim_supported else "blocked",
        "blockers": [
            {
                "check_id": check["check_id"],
                "message": check["message"],
                "evidence": check["evidence"],
            }
            for check in blockers
        ],
        "checks": checks,
        "claim_gate": claim_gate,
        "visual_audit_manifest": visual_audit_manifest,
        "method_experience_manifest": method_experience_manifest,
        "domain_approval_report": domain_report,
        "paper_like_benchmark_dossier": paper_dossier,
        "selector_heterogeneity_report": selector_report,
        "multi_seed_ablation_evidence": multi_seed_manifest,
        "resource_constraints": dict(resource_constraints),
        "expert_blueprint_id": expert_blueprint_id,
        "claim_boundary": (
            "scientific_discovery_readiness is a fail-closed evidence checklist. It does not convert mock, "
            "proxy, faithful-small, or text-only artifacts into scientific discovery support."
        ),
    }


def render_scientific_discovery_readiness_markdown(report: dict[str, Any]) -> str:
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    lines = [
        "# Scientific Discovery Readiness",
        "",
        f"- status: {report.get('status')}",
        f"- scientific_claim_supported: {report.get('scientific_claim_supported')}",
        f"- claim_level: {report.get('claim_level')}",
        "",
        "## Checks",
        "",
    ]
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = "pass" if check.get("passed") else "blocked"
        lines.append(f"- {check.get('check_id')}: {status} - {check.get('message')}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            str(report.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _check(
    check_id: str,
    message: str,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "message": message,
        "evidence": evidence,
    }


def _multi_seed_or_ablation_present(
    planner_snapshot: dict[str, Any],
    configured_multi_seed: dict[str, Any],
    evidence_manifest: dict[str, Any],
) -> bool:
    if evidence_manifest:
        return evidence_manifest.get("verified_multi_seed_ablation") is True
    for multi_seed in (
        configured_multi_seed,
        planner_snapshot.get("multi_seed_ablation"),
    ):
        if isinstance(multi_seed, dict) and _multi_seed_manifest_satisfies_minimum(multi_seed):
            return True
    ablation_manifest = planner_snapshot.get("ablation_manifest")
    return isinstance(ablation_manifest, dict) and bool(ablation_manifest.get("verified"))


def _multi_seed_manifest_satisfies_minimum(manifest: dict[str, Any]) -> bool:
    seed_count = _count_from_manifest(manifest, "seed_count", "seeds")
    ablation_count = _count_from_manifest(manifest, "ablation_count", "variants")
    return seed_count >= 2 and ablation_count >= 1 and manifest.get("verified") is True


def _count_from_manifest(manifest: dict[str, Any], count_key: str, list_key: str) -> int:
    count = manifest.get(count_key)
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    values = manifest.get(list_key)
    if isinstance(values, list):
        return len(values)
    return 0
