from __future__ import annotations

import hashlib
import json
from typing import Any

from agenticsciml.algorithm_catalog import AlgorithmSpec
from agenticsciml.benchmarks import BenchmarkSpec
from agenticsciml.evidence import (
    CLAIM_GATE_BLOCKED,
    CLAIM_LEVEL_WORKFLOW_PROXY,
    claim_gate_for_run,
)
from agenticsciml.retrieval.kb_store import kb_manifest_for_dir


ARTIFACT_CAPTURE_REQUIREMENTS = (
    "config.json",
    "planning/problem_intake.json",
    "planning/readiness_report.json",
    "evaluation_contract.json",
    "trace.jsonl",
    "trace_summary.json",
    "run_metadata.json",
    "checkpoint.json",
    "tree.json",
    "leaderboard.csv",
    "champion/claim_gate.json",
    "run_inputs/manifest.json",
    "solutions/",
    "solutions/*/kb_application_report.json",
    "solutions/*/mutation_effect_report.json",
    "solutions/*/policy_fidelity_report.json",
    "solutions/*/emergence_report.json",
    "solutions/*/visual_audit_report.json",
    "solutions/*/method_experience_record.json",
    "reports/data_analysis_structured.json",
    "reports/evolution_health.json",
    "reports/visual_audit_manifest.json",
    "reports/scientific_discovery_readiness.json",
    "reports/scientific_discovery_readiness.md",
    "reports/method_experience_cache.json",
    "reports/",
)
STRATEGY_LOCK_INSPECTION_KEYS = (
    "required_terms",
    "forbidden_terms",
    "required_imports",
    "forbidden_imports",
    "required_call_names",
    "forbidden_call_names",
)


def build_readiness_report(
    *,
    benchmark: BenchmarkSpec,
    algorithms: list[AlgorithmSpec],
    selected_algorithm_ids: list[str],
    mode: str,
    run_budget: dict[str, object],
    problem_intake: dict[str, Any],
    planner_snapshot: dict[str, Any],
    manual_strategy_locks: list[dict[str, Any]] | None = None,
    branch_context: dict[str, Any] | None = None,
    selector_panel: list[dict[str, Any]] | None = None,
    claim_level: str = CLAIM_LEVEL_WORKFLOW_PROXY,
    domain_evaluator_approved: bool = False,
    domain_reviewer: str | None = None,
    domain_review_notes: str | None = None,
    paper_benchmark_approved: bool = False,
    real_confirmed: bool = False,
    real_mode_enabled: bool = False,
    visual_audit_mode: str = "off",
    resource_constraints: dict[str, Any] | None = None,
    expert_blueprint_id: str | None = None,
) -> dict[str, object]:
    normalized_ids = _dedupe_strings(selected_algorithm_ids)
    locks = [_normalize_strategy_lock(item, index) for index, item in enumerate(manual_strategy_locks or [])]
    branch = dict(branch_context or {})
    selector_panel_items = [dict(item) for item in selector_panel or []]
    kb_manifest = kb_manifest_for_dir(benchmark.path / "kb")
    selector_config_heterogeneous = _selector_panel_config_heterogeneous(selector_panel_items)
    claim_gate = claim_gate_for_run(
        claim_level=claim_level,
        use_mock=mode != "real",
        fidelity_level=benchmark.fidelity_level,
        is_custom_proxy=benchmark.paper_section == "custom",
        domain_evaluator_approved=domain_evaluator_approved,
        domain_reviewer=domain_reviewer,
        domain_review_notes=domain_review_notes,
        paper_benchmark_approved=paper_benchmark_approved,
        selector_heterogeneous=selector_config_heterogeneous,
        kb_paper_equivalent=bool(kb_manifest.get("paper_kb_equivalent")),
        actual_multimodal_evidence=False,
    )
    checks: list[dict[str, object]] = []

    _append_claim_boundary_checks(checks, benchmark)
    _append_claim_gate_checks(checks, claim_gate)
    _append_algorithm_checks(checks, algorithms, normalized_ids)
    _append_strategy_lock_checks(checks, locks, branch)
    _append_real_mode_checks(checks, mode=mode, real_confirmed=real_confirmed, real_mode_enabled=real_mode_enabled)
    _append_scientific_readiness_intake_checks(
        checks,
        visual_audit_mode=visual_audit_mode,
        resource_constraints=resource_constraints or {},
        expert_blueprint_id=expert_blueprint_id,
    )
    checks.append(
        _check(
            "artifact-capture.required",
            "artifact_capture",
            "info",
            True,
            "Run launch will persist the planning context, readiness report, traces, scores, and solution artifacts under the run directory.",
        )
    )
    checks.append(
        _check(
            "sandbox.generated-code",
            "sandbox",
            "info",
            True,
            "Generated solution code remains under the existing orchestrator sandbox and timeout policy.",
        )
    )

    blockers = [check for check in checks if check["severity"] == "blocker" and not check["passed"]]
    warnings = [check for check in checks if check["severity"] == "warning"]
    status = "blocked" if blockers else "ready_with_warnings" if warnings else "ready"
    report_payload = {
        "schema_version": 1,
        "readiness_version": "run_readiness.v1",
        "benchmark": benchmark.to_dict(),
        "mode": mode,
        "run_budget": dict(run_budget),
        "problem_intake_summary": _first_text(
            problem_intake.get("problem_summary"),
            problem_intake.get("problem_statement"),
        ),
        "planner_version": _first_text(planner_snapshot.get("planner_version")),
        "selected_algorithm_ids": normalized_ids,
        "manual_strategy_locks": locks,
        "branch_context": branch,
        "claim_level": claim_gate["claim_level"],
        "domain_evaluator_approved": domain_evaluator_approved,
        "domain_reviewer": claim_gate.get("domain_reviewer"),
        "domain_review_notes": claim_gate.get("domain_review_notes"),
        "paper_benchmark_approved": paper_benchmark_approved,
        "visual_audit_mode": visual_audit_mode,
        "resource_constraints": dict(resource_constraints or {}),
        "expert_blueprint_id": expert_blueprint_id,
    }
    report_id = "readiness_" + hashlib.sha256(
        json.dumps(report_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "readiness_id": report_id,
        **report_payload,
        "status": status,
        "launch_allowed": not blockers,
        "summary": {
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "info_count": sum(1 for check in checks if check["severity"] == "info"),
            "check_count": len(checks),
        },
        "checks": checks,
        "claim_gate": claim_gate,
        "benchmark_fidelity_preview": _benchmark_fidelity_preview(benchmark),
        "kb_manifest": kb_manifest,
        "selector_panel_preview": {
            "configured_member_count": len(selector_panel_items),
            "configured_heterogeneous": selector_config_heterogeneous,
        },
        "algorithm_seed_preview": _algorithm_seed_preview(algorithms, normalized_ids),
        "strategy_lock_preview": _strategy_lock_preview(locks, branch),
        "real_mode_gates": {
            "real_mode_requested": mode == "real",
            "real_confirmed": real_confirmed,
            "server_enabled": real_mode_enabled,
            "blocked_reasons": [str(check["check_id"]) for check in blockers if check["category"] == "real_mode"],
        },
        "artifact_capture_requirements": list(ARTIFACT_CAPTURE_REQUIREMENTS),
        "scientific_discovery_readiness_preview": {
            "expected_artifacts": [
                "reports/scientific_discovery_readiness.json",
                "reports/visual_audit_manifest.json",
                "solutions/*/visual_audit_report.json",
                "solutions/*/method_experience_record.json",
            ],
            "claim_boundary": (
                "The completed run must still pass its fail-closed readiness report before any "
                "scientific discovery claim is supported."
            ),
        },
        "claim_boundary": (
            "Run readiness is a pre-run audit of catalog alignment, launch gates, and artifact expectations. "
            "It is not scientific validation, not evaluator synthesis, and not paper-score evidence."
        ),
    }


def readiness_summary(report: dict[str, Any]) -> dict[str, object]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "readiness_id": report.get("readiness_id"),
        "status": report.get("status"),
        "launch_allowed": report.get("launch_allowed"),
        "blocker_count": summary.get("blocker_count", 0),
        "warning_count": summary.get("warning_count", 0),
        "check_count": summary.get("check_count", 0),
        "claim_gate": report.get("claim_gate") if isinstance(report.get("claim_gate"), dict) else {},
    }


def _append_claim_boundary_checks(checks: list[dict[str, object]], benchmark: BenchmarkSpec) -> None:
    if benchmark.fidelity_level == "proxy":
        checks.append(
            _check(
                "claim-boundary.proxy-warning",
                "claim_boundary",
                "warning",
                True,
                "Selected benchmark is proxy-scoped; outputs must not be described as paper-score or paper-level reproduction evidence.",
                source_refs=[{"kind": "benchmark_catalog", "id": benchmark.name}],
            )
        )
        return
    checks.append(
        _check(
            "claim-boundary.explicit",
            "claim_boundary",
            "info",
            True,
            f"Selected benchmark fidelity is {benchmark.fidelity_level}; claim boundaries still come from completed run artifacts.",
            source_refs=[{"kind": "benchmark_catalog", "id": benchmark.name}],
        )
    )


def _append_claim_gate_checks(checks: list[dict[str, object]], claim_gate: dict[str, Any]) -> None:
    blocked = claim_gate.get("status") == CLAIM_GATE_BLOCKED
    checks.append(
        _check(
            "claim-gate.paper-workflow" if blocked else "claim-gate.workflow-proxy",
            "claim_gate",
            "blocker" if blocked else "info",
            not blocked,
            (
                "paper_workflow claim gate blocked launch: "
                + "; ".join(str(reason) for reason in claim_gate.get("reasons", []))
                if blocked
                else "Claim gate allows launch only within the recorded evidence boundary."
            ),
        )
    )


def _append_algorithm_checks(
    checks: list[dict[str, object]],
    algorithms: list[AlgorithmSpec],
    selected_algorithm_ids: list[str],
) -> None:
    algorithms_by_id = {algorithm.algorithm_id: algorithm for algorithm in algorithms}
    if not selected_algorithm_ids:
        checks.append(
            _check(
                "algorithm-selection.empty",
                "algorithm_seed",
                "warning",
                True,
                "No algorithm strategy seeds are selected; the run can proceed, but the generated strategy will be less auditable.",
            )
        )
        return
    unknown = [algorithm_id for algorithm_id in selected_algorithm_ids if algorithm_id not in algorithms_by_id]
    if unknown:
        checks.append(
            _check(
                "algorithm-selection.unknown",
                "algorithm_seed",
                "blocker",
                False,
                "Unknown selected algorithm id(s): " + ", ".join(unknown),
            )
        )
    checks.append(
        _check(
            "algorithm-selection.catalog-seeds",
            "algorithm_seed",
            "info",
            True,
            "Selected algorithms are catalog strategy seeds and prompt context, not evaluated implementations.",
        )
    )


def _append_strategy_lock_checks(
    checks: list[dict[str, object]],
    locks: list[dict[str, object]],
    branch_context: dict[str, Any],
) -> None:
    if not locks:
        checks.append(
            _check(
                "strategy-locks.empty",
                "strategy_lock",
                "warning",
                True,
                "No manual strategy locks are recorded; expert intervention can still be added before launch.",
            )
        )
    expected = _dedupe_strings(branch_context.get("expected_inherited_lock_ids", []))
    if expected:
        known = {str(lock["lock_id"]) for lock in locks}
        missing = [lock_id for lock_id in expected if lock_id not in known]
        checks.append(
            _check(
                "strategy-locks.branch-inheritance",
                "strategy_lock",
                "warning" if missing else "info",
                not missing,
                (
                    "Branch context expects lock inheritance for missing lock id(s): " + ", ".join(missing)
                    if missing
                    else "Branch lock inheritance expectations resolve to recorded manual locks."
                ),
            )
        )


def _append_real_mode_checks(
    checks: list[dict[str, object]],
    *,
    mode: str,
    real_confirmed: bool,
    real_mode_enabled: bool,
) -> None:
    if mode != "real":
        checks.append(
            _check(
                "real-mode.not-requested",
                "real_mode",
                "info",
                True,
                "Real LLM mode is not requested for this planned launch.",
            )
        )
        return
    if not real_confirmed:
        checks.append(
            _check(
                "real-mode.requires-confirmation",
                "real_mode",
                "blocker",
                False,
                "Real LLM mode requires explicit user confirmation before launch.",
            )
        )
    if not real_mode_enabled:
        checks.append(
            _check(
                "real-mode.server-disabled",
                "real_mode",
                "blocker",
                False,
                "Real LLM mode is disabled on the Web API server.",
            )
        )


def _append_scientific_readiness_intake_checks(
    checks: list[dict[str, object]],
    *,
    visual_audit_mode: str,
    resource_constraints: dict[str, Any],
    expert_blueprint_id: str | None,
) -> None:
    checks.append(
        _check(
            "visual-audit.mode",
            "scientific_readiness",
            "info" if visual_audit_mode != "off" else "warning",
            True,
            (
                f"visual_audit_mode={visual_audit_mode}; completed runs will record visual readiness artifacts. "
                "Only actual real image-provider use can satisfy the paper_workflow multimodal gate."
            ),
        )
    )
    checks.append(
        _check(
            "resource-constraints.present",
            "scientific_readiness",
            "info" if resource_constraints else "warning",
            True,
            "Resource constraints are recorded for scientific readiness review."
            if resource_constraints
            else "Resource constraints are missing; readiness will remain blocked for real scientific claims.",
        )
    )
    checks.append(
        _check(
            "expert-blueprint.present",
            "scientific_readiness",
            "info" if expert_blueprint_id else "warning",
            True,
            "Expert blueprint is recorded for controlled method search."
            if expert_blueprint_id
            else "Expert blueprint is missing; fluid/PDE readiness will remain blocked for real scientific claims.",
        )
    )


def _benchmark_fidelity_preview(benchmark: BenchmarkSpec) -> list[dict[str, object]]:
    return [
        {
            "benchmark": benchmark.name,
            "paper_section": benchmark.paper_section,
            "paper_task_name": benchmark.paper_task_name,
            "family": benchmark.family,
            "fidelity_level": benchmark.fidelity_level,
            "expected_runtime_s": benchmark.expected_runtime_s,
            "allowed_claims": [
                f"local {benchmark.fidelity_level} workflow evidence",
                "completed evaluator artifacts when a run finishes",
            ],
            "disallowed_claims": [
                "paper-score reproduction",
                "SOTA or scientific discovery claim without completed run artifacts",
            ],
            "notes": benchmark.paper_gap_notes,
        }
    ]


def _algorithm_seed_preview(
    algorithms: list[AlgorithmSpec],
    selected_algorithm_ids: list[str],
) -> list[dict[str, object]]:
    algorithms_by_id = {algorithm.algorithm_id: algorithm for algorithm in algorithms}
    preview = []
    for algorithm_id in selected_algorithm_ids:
        algorithm = algorithms_by_id.get(algorithm_id)
        if algorithm is None:
            preview.append(
                {
                    "algorithm_id": algorithm_id,
                    "status": "unknown",
                    "catalog_role": "unresolved",
                    "is_evaluated_implementation": False,
                    "expected_use": "blocked_until_catalog_entry_exists",
                }
            )
            continue
        preview.append(
            {
                "algorithm_id": algorithm.algorithm_id,
                "name": algorithm.name,
                "family": algorithm.family,
                "status": algorithm.status,
                "catalog_role": "strategy_seed",
                "is_evaluated_implementation": False,
                "expected_use": "planning_seed_only",
                "claim_boundary": algorithm.claim_boundary,
                "safety_notes": algorithm.safety_notes,
            }
        )
    return preview


def _strategy_lock_preview(
    locks: list[dict[str, object]],
    branch_context: dict[str, Any],
) -> list[dict[str, object]]:
    expected = set(_dedupe_strings(branch_context.get("expected_inherited_lock_ids", [])))
    return [
        {
            "lock_id": lock["lock_id"],
            "kind": lock["kind"],
            "scope": lock["scope"],
            "required": lock["required"],
            "status": "recorded",
            "inheritance_expected": lock["lock_id"] in expected or lock["scope"] == "all_branches",
            "auditable_criteria_count": _strategy_lock_criteria_count(lock),
            "notes": [
                "Manual strategy locks are user hypotheses or constraints; they are not validated scientific facts."
            ],
        }
        for lock in locks
    ]


def _normalize_strategy_lock(item: dict[str, Any], index: int) -> dict[str, object]:
    lock_id = _first_text(item.get("lock_id"), item.get("id")) or f"manual_lock_{index + 1:03d}"
    normalized: dict[str, object] = {
        "lock_id": lock_id,
        "kind": _first_text(item.get("kind")) or "constraint",
        "text": _first_text(item.get("text"), item.get("description")) or "",
        "scope": _first_text(item.get("scope")) or "all_branches",
        "required": bool(item.get("required", True)),
    }
    inspection = _normalized_inspection_mapping(item.get("inspection"))
    if inspection:
        normalized["inspection"] = inspection
    for key in STRATEGY_LOCK_INSPECTION_KEYS:
        values = _dedupe_text_values(item.get(key))
        if values:
            normalized[key] = values
    return normalized


def _normalized_inspection_mapping(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key in STRATEGY_LOCK_INSPECTION_KEYS:
        values = _dedupe_text_values(value.get(key))
        if values:
            normalized[key] = values
    return normalized


def _strategy_lock_criteria_count(lock: dict[str, object]) -> int:
    count = 0
    inspection = lock.get("inspection")
    if isinstance(inspection, dict):
        count += sum(len(value) for value in inspection.values() if isinstance(value, list))
    for key in STRATEGY_LOCK_INSPECTION_KEYS:
        value = lock.get(key)
        if isinstance(value, list):
            count += len(value)
    return count


def _selector_panel_config_heterogeneous(selector_panel: list[dict[str, Any]]) -> bool:
    if len(selector_panel) < 2:
        return False
    signatures = {
        (
            str(item.get("model", "")).strip(),
            str(item.get("provider", item.get("base_url", ""))).strip(),
        )
        for item in selector_panel
    }
    return len(signatures) > 1


def _check(
    check_id: str,
    category: str,
    severity: str,
    passed: bool,
    message: str,
    *,
    source_refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "category": category,
        "severity": severity,
        "passed": passed,
        "message": message,
        "source_refs": source_refs or [],
    }


def _dedupe_strings(values: object) -> list[str]:
    normalized: list[str] = []
    if not isinstance(values, list):
        return normalized
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _dedupe_text_values(values: object) -> list[str]:
    if isinstance(values, str):
        values = [values]
    return _dedupe_strings(values)


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
