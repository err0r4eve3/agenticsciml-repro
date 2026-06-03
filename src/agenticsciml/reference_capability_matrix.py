from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticsciml.storage import _atomic_write_text


REFERENCE_CAPABILITY_MATRIX_SCHEMA_VERSION = 1
REFERENCE_CAPABILITY_MATRIX_JSON = "reference_capability_matrix.json"
REFERENCE_CAPABILITY_MATRIX_MD = "reference_capability_matrix.md"
REFERENCE_MATRIX_CLAIM_BOUNDARY = (
    "This matrix is an offline planning and audit rubric. It does not validate results, "
    "does not prove a real-world solution, and does not change claim gates."
)

REFERENCE_MECHANISMS: tuple[dict[str, object], ...] = (
    {
        "mechanism_id": "scientific_problem_decomposition",
        "source_family": "AgenticSciML / AI Fluid Scientist / materials discovery",
        "reference_mechanism": (
            "Decompose a scientific request into hypothesis, observable, metric, constraints, "
            "failure modes, and review checklist before launching agents."
        ),
        "project_module_mapping": "Problem Intake, paper_workflow_readiness, real_problem_closure",
        "required_artifact": "planning/problem_intake.json with hypothesis/observable/metric/failure fields",
        "fail_closed_blocker": "missing_problem_intake_rubric_fields",
        "no_key_local_implementation_path": (
            "Run build-reference-capability-matrix with a local problem_intake JSON; no provider call is needed."
        ),
        "future_real_run_requirement": (
            "A completed run must still provide evaluator scores, trace summary, domain review, and failure sample audit."
        ),
    },
    {
        "mechanism_id": "fluid_pde_visual_physics_audit",
        "source_family": "AI Fluid Scientist / ALL-FEM / PDE benchmarks",
        "reference_mechanism": (
            "Audit PDE/fluid outputs through prediction-only field, residual proxy, boundary, smoothness, "
            "and conservation-oriented diagnostics."
        ),
        "project_module_mapping": "observations.py visual_audit_report and visual_audit_manifest",
        "required_artifact": "solutions/<id>/visual_audit_report.json plus visual diagnostic images",
        "fail_closed_blocker": "actual_multimodal_or_physics_audit_missing",
        "no_key_local_implementation_path": (
            "Generate mock/offline visual diagnostic artifacts from predictions without reading private labels."
        ),
        "future_real_run_requirement": (
            "For paper workflow, an image-capable provider must actually receive PNG/JPEG inputs and domain review must approve limits."
        ),
    },
    {
        "mechanism_id": "athena_graft_action_reward_trace",
        "source_family": "ATHENA / GRAFT-ATHENA",
        "reference_mechanism": (
            "Represent method evolution as action path, solution artifact, reward, fingerprint, and local experience record."
        ),
        "project_module_mapping": "method_templates, method_substrate, method_experience_record",
        "required_artifact": "solutions/<id>/method_experience_record.json and reports/method_experience_cache.json",
        "fail_closed_blocker": "method_failure_attribution_missing",
        "no_key_local_implementation_path": (
            "Instantiate inert method templates and validate blueprint/fingerprint/reward records locally."
        ),
        "future_real_run_requirement": (
            "Positive and negative experience must be tied to trusted evaluator artifacts across seeds or ablations."
        ),
    },
    {
        "mechanism_id": "code_orchestrated_agent_boundary",
        "source_family": "agent-systems scaling / OpenAI Agents SDK alignment",
        "reference_mechanism": (
            "Use agents for bounded generation, critique, selection, and summary while deterministic Python owns state, "
            "contracts, sorting, execution, and claim gates."
        ),
        "project_module_mapping": "orchestrator, trace_summary, selector_evidence, sandbox",
        "required_artifact": "trace.jsonl, trace_summary.json, selector_votes.json, evaluation_contract.json",
        "fail_closed_blocker": "orchestrator_trace_or_contract_gate_failed",
        "no_key_local_implementation_path": (
            "Run mock/offline tests that assert trace, contract, selector, and sandbox invariants."
        ),
        "future_real_run_requirement": (
            "Real providers may contribute structured outputs only; completed trace and evaluator gates remain authoritative."
        ),
    },
    {
        "mechanism_id": "domain_negative_result_closure",
        "source_family": "materials discovery / experimental fluid mechanics",
        "reference_mechanism": (
            "Preserve domain approval, negative results, failed samples, ablation coverage, and experiment-loop gaps."
        ),
        "project_module_mapping": "domain_approval, multi_seed_ablation_evidence, real_problem_closure",
        "required_artifact": "reports/domain_approval.json and reports/multi_seed_ablation_evidence.json",
        "fail_closed_blocker": "domain_or_ablation_review_missing",
        "no_key_local_implementation_path": (
            "Generate domain approval templates and verify ablation CSV/manifests without model calls."
        ),
        "future_real_run_requirement": (
            "A domain reviewer must approve evidence limits and failed samples; multi-seed ablation must be verified."
        ),
    },
)


def build_reference_capability_matrix(
    *,
    problem_intake: dict[str, Any] | None = None,
    expert_blueprint_id: str | None = None,
) -> dict[str, Any]:
    rubric = build_problem_intake_rubric(
        problem_intake or {},
        expert_blueprint_id=expert_blueprint_id,
    )
    return {
        "schema_version": REFERENCE_CAPABILITY_MATRIX_SCHEMA_VERSION,
        "matrix_version": "reference_capability_matrix.v1",
        "status": "ready_for_offline_planning" if rubric["passed"] is True else "blocked",
        "scientific_claim_supported": False,
        "mechanism_count": len(REFERENCE_MECHANISMS),
        "mechanisms": [dict(item) for item in REFERENCE_MECHANISMS],
        "problem_intake_rubric": rubric,
        "offline_only": True,
        "claim_boundary": REFERENCE_MATRIX_CLAIM_BOUNDARY,
    }


def build_problem_intake_rubric(
    problem_intake: dict[str, Any],
    *,
    expert_blueprint_id: str | None = None,
) -> dict[str, Any]:
    checks = [
        _rubric_check("hypothesis", problem_intake, aliases=("hypothesis", "scientific_hypothesis")),
        _rubric_check("observable", problem_intake, aliases=("observable", "observables", "measured_quantities")),
        _rubric_check("metric", problem_intake, aliases=("metric", "success_metric", "evaluation_metric")),
        _rubric_check("failure_modes", problem_intake, aliases=("failure_modes", "known_failure_modes")),
        _rubric_check(
            "physical_constraints",
            problem_intake,
            aliases=("physical_constraints", "physics_constraints", "boundary_conditions"),
        ),
        _rubric_check(
            "domain_review_checklist",
            problem_intake,
            aliases=("domain_review_checklist", "review_checklist", "domain_review_requirements"),
        ),
        {
            "field_id": "expert_blueprint",
            "passed": bool(expert_blueprint_id),
            "matched_key": "expert_blueprint_id" if expert_blueprint_id else None,
            "message": "expert_blueprint_id is recorded" if expert_blueprint_id else "expert_blueprint_id is missing",
        },
    ]
    missing = [
        {
            "field_id": check["field_id"],
            "message": check["message"],
        }
        for check in checks
        if check["passed"] is not True
    ]
    return {
        "status": "ready" if not missing else "blocked",
        "passed": not missing,
        "checks": checks,
        "missing_fields": missing,
        "claim_boundary": "Problem intake completeness is planning evidence only; evaluator artifacts still decide outcomes.",
    }


def write_reference_capability_matrix(
    *,
    output_dir: Path,
    problem_intake: dict[str, Any] | None = None,
    expert_blueprint_id: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_reference_capability_matrix(
        problem_intake=problem_intake,
        expert_blueprint_id=expert_blueprint_id,
    )
    matrix_json = output_dir / REFERENCE_CAPABILITY_MATRIX_JSON
    matrix_md = output_dir / REFERENCE_CAPABILITY_MATRIX_MD
    _atomic_write_text(matrix_json, json.dumps(matrix, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(matrix_md, render_reference_capability_matrix_markdown(matrix))
    return {
        "matrix": matrix,
        "paths": {
            "matrix_json": str(matrix_json),
            "matrix_md": str(matrix_md),
        },
    }


def render_reference_capability_matrix_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Reference Capability Matrix",
        "",
        f"- status: {matrix.get('status')}",
        f"- scientific_claim_supported: {matrix.get('scientific_claim_supported')}",
        f"- mechanism_count: {matrix.get('mechanism_count')}",
        "",
        "## Problem Intake Rubric",
        "",
    ]
    rubric = matrix.get("problem_intake_rubric") if isinstance(matrix.get("problem_intake_rubric"), dict) else {}
    lines.append(f"- status: {rubric.get('status')}")
    for check in rubric.get("checks", []):
        if isinstance(check, dict):
            status = "pass" if check.get("passed") else "blocked"
            lines.append(f"- {check.get('field_id')}: {status}")
    lines.extend(["", "## Mechanisms", ""])
    for item in matrix.get("mechanisms", []):
        if not isinstance(item, dict):
            continue
        lines.append(f"### {item.get('mechanism_id')}")
        lines.append("")
        lines.append(f"- reference_mechanism: {item.get('reference_mechanism')}")
        lines.append(f"- project_module_mapping: {item.get('project_module_mapping')}")
        lines.append(f"- required_artifact: {item.get('required_artifact')}")
        lines.append(f"- fail_closed_blocker: {item.get('fail_closed_blocker')}")
        lines.append(f"- no_key_local_implementation_path: {item.get('no_key_local_implementation_path')}")
        lines.append(f"- future_real_run_requirement: {item.get('future_real_run_requirement')}")
        lines.append("")
    lines.extend(["## Claim Boundary", "", str(matrix.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _rubric_check(
    field_id: str,
    problem_intake: dict[str, Any],
    *,
    aliases: tuple[str, ...],
) -> dict[str, Any]:
    matched_key = None
    for key in aliases:
        if _has_content(problem_intake.get(key)):
            matched_key = key
            break
    return {
        "field_id": field_id,
        "passed": matched_key is not None,
        "matched_key": matched_key,
        "message": f"{field_id} is recorded" if matched_key else f"{field_id} is missing",
    }


def _has_content(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set):
        return any(_has_content(item) for item in value)
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    return True
