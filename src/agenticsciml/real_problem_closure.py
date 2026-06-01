from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agenticsciml.paper_workflow_readiness import (
    build_paper_workflow_readiness_bundle,
    write_paper_workflow_readiness_bundle,
)
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
        "readiness_check_ids": (),
        "proof_artifacts": (
            "reports/scientific_discovery_readiness.json",
            "trace_summary.json quality_gate=true",
            "run_metadata.json and claim_gate showing claim support only after all checks pass",
        ),
        "completion_mode": "validated_completed_run_required",
        "always_blocked_until_completed_run": True,
    },
)


def build_real_problem_closure_plan(
    *,
    benchmark_dir: Path,
    selector_panel: list[dict[str, object]] | None = None,
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    paper_bundle = build_paper_workflow_readiness_bundle(
        benchmark_dir=benchmark_dir,
        selector_panel=selector_panel,
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
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_result = write_paper_workflow_readiness_bundle(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        selector_panel=selector_panel,
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
        resource_constraints=resource_constraints,
        expert_blueprint_id=expert_blueprint_id,
        domain_approval_path=domain_approval_path,
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds,
        expected_variants=expected_variants,
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
    if module.get("always_blocked_until_completed_run") is True:
        blocking_check_ids.append("completed_run_artifacts")
        next_actions.append("Run a real paper_workflow experiment and verify scientific_discovery_readiness plus trace_summary.")
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
