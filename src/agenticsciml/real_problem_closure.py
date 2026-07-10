from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agenticsciml.evidence import (
    CLAIM_GATE_ALLOWED,
    CLAIM_GATE_BLOCKED,
    CLAIM_GATE_DOWNGRADED,
)
from agenticsciml.paper_workflow_readiness import (
    build_paper_workflow_readiness_bundle,
    write_paper_workflow_readiness_bundle,
)
from agenticsciml.reporting.trace_summary import summarize_trace
from agenticsciml.storage import _atomic_write_text


REAL_PROBLEM_CLOSURE_SCHEMA_VERSION = 1
REAL_PROBLEM_CLOSURE_VERSION = "real_problem_closure.v1"
REAL_PROBLEM_CLAIM_BOUNDARY = (
    "This closure plan is a fail-closed real-problem claim gate. It is not evidence that "
    "multi-agent solving works on real scientific problems until every required proof artifact exists."
)

REAL_PROBLEM_MODULES = (
    {
        "module_id": "real_llm_execution",
        "title": "Real LLM execution and budget",
        "readiness_check_ids": ("real_llm_credentials", "real_llm_budget"),
        "proof_artifacts": (
            "real provider credentials configured without storing secrets",
            "explicit LLM call/token/cost budget",
            "completed run_metadata.json with llm_mode=real_llm",
        ),
        "completion_mode": "external_runtime_required",
    },
    {
        "module_id": "real_multimodal_input",
        "title": "Actual multimodal image input",
        "readiness_check_ids": ("real_multimodal_provider",),
        "proof_artifacts": (
            "reports/visual_audit_manifest.json",
            "solutions/<id>/visual_audit_report.json",
            "actual_image_inputs_used=true from a real image-capable provider call",
        ),
        "completion_mode": "external_runtime_required",
    },
    {
        "module_id": "heterogeneous_selector",
        "title": "Heterogeneous real selector evidence",
        "readiness_check_ids": ("heterogeneous_real_selector",),
        "proof_artifacts": (
            "selector_votes.json with runtime votes",
            "reports/selector_heterogeneity.json with at least two real provider/model paths",
        ),
        "completion_mode": "external_runtime_required",
    },
    {
        "module_id": "paper_like_benchmark",
        "title": "Paper-like benchmark and evaluator",
        "readiness_check_ids": ("paper_like_benchmark",),
        "proof_artifacts": (
            "paper_benchmark_manifest.json",
            "reports/paper_like_benchmark_dossier.json",
            "paper-equivalent data/evaluator provenance with private-label protocol",
        ),
        "completion_mode": "paper_asset_required",
    },
    {
        "module_id": "paper_equivalent_kb",
        "title": "Paper-equivalent knowledge base",
        "readiness_check_ids": ("paper_equivalent_kb",),
        "proof_artifacts": (
            "benchmark kb manifest with paper_kb_equivalent=true",
            "paper/code/data source digests",
        ),
        "completion_mode": "paper_asset_required",
    },
    {
        "module_id": "domain_approval",
        "title": "Domain approval and failure review",
        "readiness_check_ids": ("domain_approval_packet",),
        "proof_artifacts": (
            "reports/domain_approval.json",
            "reviewer identity",
            "review notes and checklist including failure samples",
        ),
        "completion_mode": "human_domain_review_required",
    },
    {
        "module_id": "multi_seed_ablation",
        "title": "Verified multi-seed ablation",
        "readiness_check_ids": ("multi_seed_ablation_output",),
        "proof_artifacts": (
            "ablation_runs.csv",
            "ablation_summary.csv",
            "multi_seed_ablation_verified_manifest.json",
        ),
        "completion_mode": "local_or_external_experiment_required",
    },
    {
        "module_id": "resource_blueprint",
        "title": "Resource constraints and expert blueprint",
        "readiness_check_ids": ("resource_and_blueprint",),
        "proof_artifacts": (
            "expert_blueprint_id",
            "resource_constraints with CPU/GPU/timeout/dependency/data limits",
        ),
        "completion_mode": "configuration_required",
    },
    {
        "module_id": "completed_run_audit",
        "title": "Completed run audit and claim gate",
        "readiness_check_ids": ("completed_run_artifacts",),
        "proof_artifacts": (
            "reports/scientific_discovery_readiness.json",
            "trace_summary.json quality_gate=true",
            "run_metadata.json and claim_gate showing claim support only after all checks pass",
        ),
        "completion_mode": "validated_completed_run_required",
    },
)


def build_real_problem_closure_plan(
    *,
    benchmark_dir: Path,
    selector_panel: list[dict[str, object]] | None = None,
    selector_evidence_path: Path | None = None,
    problem_intake: dict[str, Any] | None = None,
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    completed_run_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    paper_bundle = build_paper_workflow_readiness_bundle(
        benchmark_dir=benchmark_dir,
        selector_panel=selector_panel,
        selector_evidence_path=selector_evidence_path,
        problem_intake=problem_intake,
        resource_constraints=resource_constraints,
        expert_blueprint_id=expert_blueprint_id,
        domain_approval_path=domain_approval_path,
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds,
        expected_variants=expected_variants,
        env=env,
    )
    checks_by_id = {
        str(check.get("check_id")): dict(check)
        for check in paper_bundle.get("checks", [])
        if isinstance(check, dict) and check.get("check_id")
    }
    completed_run_audit = _completed_run_audit_readiness(completed_run_dir)
    checks_by_id["completed_run_artifacts"] = {
        "check_id": "completed_run_artifacts",
        "category": "completed_run_audit",
        "passed": completed_run_audit["ready"],
        "message": "Completed run metadata, trace summary, and scientific readiness artifact are present.",
        "next_action": "Run a real paper_workflow experiment and verify scientific_discovery_readiness plus trace_summary.",
        "evidence": completed_run_audit,
    }
    modules = [
        _module_status(module, checks_by_id)
        for module in REAL_PROBLEM_MODULES
    ]
    blocked_modules = [module for module in modules if module["status"] != "ready"]
    return {
        "schema_version": REAL_PROBLEM_CLOSURE_SCHEMA_VERSION,
        "closure_version": REAL_PROBLEM_CLOSURE_VERSION,
        "status": "blocked" if blocked_modules else "ready",
        "multi_agent_real_problem_claim_supported": False,
        "ready_to_rely_on_multi_agent_problem_solving": False,
        "benchmark": paper_bundle.get("benchmark", {}),
        "paper_workflow_readiness_status": paper_bundle.get("status"),
        "paper_workflow_blockers": paper_bundle.get("blockers", []),
        "completed_run_audit": completed_run_audit,
        "reference_capability_matrix": paper_bundle.get("reference_capability_matrix", {}),
        "llm_problem_context_pack": paper_bundle.get("llm_problem_context_pack", {}),
        "modules": modules,
        "blocked_modules": blocked_modules,
        "local_engineering_completed": [
            "paper_workflow_readiness_gate",
            "iteration_campaign_gate",
            "real_problem_closure_plan",
        ],
        "non_autofillable_requirements": [
            module["module_id"]
            for module in modules
            if module["status"] != "ready" and module["completion_mode"] != "configuration_required"
        ],
        "claim_boundary": REAL_PROBLEM_CLAIM_BOUNDARY,
    }


def write_real_problem_closure_plan(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    selector_panel: list[dict[str, object]] | None = None,
    selector_evidence_path: Path | None = None,
    problem_intake: dict[str, Any] | None = None,
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    completed_run_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_result = write_paper_workflow_readiness_bundle(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        selector_panel=selector_panel,
        selector_evidence_path=selector_evidence_path,
        problem_intake=problem_intake,
        resource_constraints=resource_constraints,
        expert_blueprint_id=expert_blueprint_id,
        domain_approval_path=domain_approval_path,
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds,
        expected_variants=expected_variants,
        env=env,
    )
    plan = build_real_problem_closure_plan(
        benchmark_dir=benchmark_dir,
        selector_panel=selector_panel,
        selector_evidence_path=selector_evidence_path,
        problem_intake=problem_intake,
        resource_constraints=resource_constraints,
        expert_blueprint_id=expert_blueprint_id,
        domain_approval_path=domain_approval_path,
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds,
        expected_variants=expected_variants,
        completed_run_dir=completed_run_dir,
        env=env,
    )
    plan_json = output_dir / "real_problem_closure_plan.json"
    plan_md = output_dir / "real_problem_closure_plan.md"
    _atomic_write_text(plan_json, json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(plan_md, render_real_problem_closure_markdown(plan))
    return {
        "plan": plan,
        "paths": {
            "plan_json": str(plan_json),
            "plan_md": str(plan_md),
            "paper_workflow_readiness_json": paper_result["paths"]["plan_json"],
            "paper_workflow_readiness_md": paper_result["paths"]["plan_md"],
            "llm_problem_context_pack_json": paper_result["paths"]["llm_problem_context_pack_json"],
            "llm_problem_context_pack_md": paper_result["paths"]["llm_problem_context_pack_md"],
        },
    }


def render_real_problem_closure_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Real Problem Evidence Closure",
        "",
        f"- status: {plan.get('status')}",
        "- multi_agent_real_problem_claim_supported: "
        f"{plan.get('multi_agent_real_problem_claim_supported')}",
        "- ready_to_rely_on_multi_agent_problem_solving: "
        f"{plan.get('ready_to_rely_on_multi_agent_problem_solving')}",
        f"- paper_workflow_readiness_status: {plan.get('paper_workflow_readiness_status')}",
        f"- llm_problem_context_status: {plan.get('llm_problem_context_pack', {}).get('status')}",
        "",
        "## Modules",
        "",
    ]
    for module in plan.get("modules", []):
        if not isinstance(module, dict):
            continue
        lines.append(f"- {module.get('module_id')}: {module.get('status')} ({module.get('completion_mode')})")
        for blocker in module.get("blocking_check_ids", []):
            lines.append(f"  - blocker: {blocker}")
    lines.extend(["", "## Claim Boundary", "", str(plan.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _module_status(
    module: dict[str, object],
    checks_by_id: dict[str, dict[str, object]],
) -> dict[str, Any]:
    readiness_check_ids = [str(item) for item in module.get("readiness_check_ids", ())]
    blocking_check_ids: list[str] = []
    next_actions: list[str] = []
    for check_id in readiness_check_ids:
        check = checks_by_id.get(check_id)
        if check is None or check.get("passed") is not True:
            blocking_check_ids.append(check_id)
            if check and check.get("next_action"):
                next_actions.append(str(check["next_action"]))
    status = "ready" if not blocking_check_ids else "blocked"
    return {
        "module_id": module["module_id"],
        "title": module["title"],
        "status": status,
        "completion_mode": module["completion_mode"],
        "readiness_check_ids": readiness_check_ids,
        "blocking_check_ids": blocking_check_ids,
        "proof_artifacts": list(module["proof_artifacts"]),
        "next_actions": next_actions,
        "claim_boundary": "Module readiness is only claimable when its proof artifacts are present and verified.",
    }


def _completed_run_audit_readiness(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {
            "ready": False,
            "path": None,
            "issues": ["completed_run_dir is missing"],
        }
    path = Path(run_dir)
    issues: list[str] = []
    run_metadata = _read_json_artifact(path / "run_metadata.json", issues)
    _read_json_artifact(path / "evaluation_contract.json", issues)
    _read_json_artifact(path / "tree.json", issues)
    _read_json_artifact(path / "checkpoint.json", issues)
    stored_trace_summary = _read_json_artifact(path / "trace_summary.json", issues)
    readiness = _read_json_artifact(path / "reports" / "scientific_discovery_readiness.json", issues)
    trace_path = path / "trace.jsonl"
    if not trace_path.exists():
        issues.append(f"trace.jsonl is missing at {trace_path}")
        recomputed_trace_summary: dict[str, Any] = {}
    else:
        try:
            recomputed_trace_summary = summarize_trace(path)
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"trace.jsonl could not be recomputed: {type(exc).__name__}: {exc}")
            recomputed_trace_summary = {}
    trace_summary_stale = bool(
        stored_trace_summary
        and recomputed_trace_summary
        and stored_trace_summary != recomputed_trace_summary
    )
    if trace_summary_stale:
        issues.append(
            "trace_summary.json is stale or inconsistent with trace.jsonl and current run artifacts"
        )

    run_state = run_metadata.get("run_state") if isinstance(run_metadata, dict) else None
    if run_state not in {"completed", "exported", "finalized"}:
        issues.append("run_metadata.run_state must be completed, exported, or finalized")
    llm_mode = run_metadata.get("llm_mode") if isinstance(run_metadata, dict) else None
    if llm_mode != "real":
        issues.append("run_metadata.llm_mode must be real")
    quality_gate = (
        recomputed_trace_summary.get("quality_gate")
        if isinstance(recomputed_trace_summary, dict)
        else {}
    )
    if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
        issues.append("recomputed trace_summary.quality_gate.passed must be true")
    readiness_status = readiness.get("status") if isinstance(readiness, dict) else None
    if readiness_status not in {"ready", "blocked"}:
        issues.append("scientific_discovery_readiness.status must be ready or blocked")
    readiness_claim = readiness.get("scientific_claim_supported") if isinstance(readiness, dict) else None
    if readiness_claim is not True and readiness_claim is not False:
        issues.append("scientific_discovery_readiness.scientific_claim_supported must be boolean")
    elif (readiness_status == "ready") != readiness_claim:
        issues.append(
            "scientific_discovery_readiness.status must agree with scientific_claim_supported"
        )
    trace_claim_gate = (
        recomputed_trace_summary.get("claim_gate")
        if isinstance(recomputed_trace_summary, dict)
        else {}
    )
    if not isinstance(trace_claim_gate, dict):
        issues.append("recomputed trace_summary.claim_gate must be an object")
        trace_claim_gate = {}
    trace_claim_status = trace_claim_gate.get("status")
    if trace_claim_status not in {
        CLAIM_GATE_ALLOWED,
        CLAIM_GATE_BLOCKED,
        CLAIM_GATE_DOWNGRADED,
    }:
        issues.append("recomputed trace_summary.claim_gate.status is missing or invalid")
    trace_claim_supported = (
        trace_claim_gate.get("scientific_claim_supported")
        if isinstance(trace_claim_gate, dict)
        else None
    )
    if trace_claim_supported is not True and trace_claim_supported is not False:
        issues.append(
            "recomputed trace_summary.claim_gate.scientific_claim_supported must be boolean"
        )
    elif readiness_claim is True or readiness_claim is False:
        if trace_claim_supported != readiness_claim:
            issues.append(
                "recomputed trace_summary claim_gate must agree with scientific discovery readiness"
            )
    metadata_claim_gate = run_metadata.get("claim_gate") if isinstance(run_metadata, dict) else None
    if not isinstance(metadata_claim_gate, dict):
        issues.append("run_metadata.claim_gate must be an object")
    else:
        for key in ("status", "scientific_claim_supported"):
            if metadata_claim_gate.get(key) != trace_claim_gate.get(key):
                issues.append(
                    f"run_metadata.claim_gate.{key} must agree with recomputed trace_summary.claim_gate"
                )

    return {
        "ready": not issues,
        "path": str(path),
        "issues": issues,
        "run_state": run_state,
        "llm_mode": llm_mode,
        "trace_quality_gate_passed": quality_gate.get("passed") if isinstance(quality_gate, dict) else None,
        "trace_summary_recomputed": bool(recomputed_trace_summary),
        "trace_summary_stale": trace_summary_stale,
        "scientific_readiness_status": readiness_status,
        "scientific_claim_supported": readiness_claim,
        "claim_gate_status": trace_claim_status,
        "claim_gate_scientific_claim_supported": trace_claim_supported,
        "claim_boundary": (
            "Completed run audit proves the run artifacts are present and internally consistent. "
            "It does not by itself prove paper-score reproduction, scientific discovery, or real-problem closure."
        ),
    }


def _read_json_artifact(path: Path, issues: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"{path.name} is missing at {path}")
        return {}
    except json.JSONDecodeError as exc:
        issues.append(f"{path.name} is invalid JSON: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{path.name} must contain a JSON object")
        return {}
    return payload
