from __future__ import annotations

import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from agenticsciml.llm_problem_context import build_llm_problem_context_pack
from agenticsciml.reference_capability_matrix import build_reference_capability_matrix
from agenticsciml.storage import _atomic_write_text


AUDIT_JSON = "agenticsciml_paper_problem_loop_audit.json"
AUDIT_MD = "summary.md"


def write_paper_problem_loop_audit(
    *,
    output_dir: Path,
    case_limit: int | None = None,
    source_collection_path: Path | None = None,
    source_candidate_limit: int = 3,
) -> dict[str, Any]:
    from agenticsciml.web.app import CURATED_PAPER_PROBLEM_CASES, ProblemIntakeRequest, _problem_intake_plan_payload

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list(CURATED_PAPER_PROBLEM_CASES[:case_limit])
    results = []
    for index, case in enumerate(cases, start=1):
        tags = [str(tag) for tag in case["tags"]]
        expert_blueprint = _expert_blueprint(tags)
        intake = _problem_intake(case, tags)
        plan = _problem_intake_plan_payload(
            ProblemIntakeRequest(
                problem_statement=(
                    f"{case['title']}\n\n{case['summary']}\n\n"
                    f"Real-world problem: {case['real_problem']}"
                ),
                requirements=(
                    "Use local deterministic planning only; preserve evaluator, sandbox, artifact, "
                    "and claim-gate boundaries."
                ),
                evaluation_criteria=str(intake["metric"]),
                data_description=str(intake["data_source"]),
                mode="mock",
                target_solution_count=6,
                parallel_mutations=2,
                selector_vote_count=3,
                max_children_per_node=3,
                visual_audit_mode="off",
                resource_constraints=_resource_constraints(),
                expert_blueprint_id=expert_blueprint,
            )
        )
        source_review = _source_review(case)
        matrix = build_reference_capability_matrix(
            problem_intake=intake,
            expert_blueprint_id=expert_blueprint,
        )
        context = build_llm_problem_context_pack(
            problem_intake=intake,
            expert_blueprint_id=expert_blueprint,
            resource_constraints=_resource_constraints(),
            reference_capability_matrix=matrix,
        )
        artifacts = _write_case_artifacts(
            output_dir=output_dir,
            index=index,
            paper_id=str(case["id"]),
            source_review=source_review,
            plan=plan,
            matrix=matrix,
            context=context,
        )
        benchmark_candidates = plan.get("benchmark_candidates") if isinstance(plan.get("benchmark_candidates"), list) else []
        top_candidate = benchmark_candidates[0] if benchmark_candidates and isinstance(benchmark_candidates[0], dict) else {}
        results.append(
            {
                "index": index,
                "paper_id": case["id"],
                "title": case["title"],
                "title_zh": case["title_zh"],
                "url": case["url"],
                "real_problem": case["real_problem"],
                "real_problem_zh": case["real_problem_zh"],
                "expert_blueprint_id": expert_blueprint,
                "planner_status": plan.get("status"),
                "recommended_benchmark": _nested_text(plan, "recommended_benchmark", "name"),
                "recommended_benchmark_score": top_candidate.get("score"),
                "selected_algorithm_ids": plan.get("selected_algorithm_ids"),
                "run_allowed": plan.get("run_allowed"),
                "source_review_status": source_review.get("status"),
                "reference_matrix_status": matrix.get("status"),
                "llm_context_status": context.get("status"),
                "prompt_quality_control_ids": [
                    item.get("control_id")
                    for item in context.get("prompt_quality_controls", [])
                    if isinstance(item, dict)
                ],
                "artifacts": artifacts,
            }
        )
    source_collection = _read_source_collection(source_collection_path)
    source_candidate_results = _source_candidate_results(
        output_dir=output_dir,
        source_collection=source_collection,
        limit=source_candidate_limit,
    )
    audit = _audit_payload(
        results,
        source_collection=source_collection,
        source_candidate_results=source_candidate_results,
    )
    _atomic_write_text(output_dir / AUDIT_JSON, json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False))
    _atomic_write_text(output_dir / AUDIT_MD, _render_markdown(audit))
    return {
        "audit": audit,
        "paths": {
            "audit_json": str((output_dir / AUDIT_JSON).resolve()),
            "audit_md": str((output_dir / AUDIT_MD).resolve()),
        },
    }


def _audit_payload(
    results: list[dict[str, Any]],
    *,
    source_collection: dict[str, Any] | None = None,
    source_candidate_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("planner_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    scores = [item["recommended_benchmark_score"] for item in results if isinstance(item.get("recommended_benchmark_score"), int)]
    issues = _audit_issues(results)
    payload = {
        "artifact_type": "agenticsciml_paper_problem_loop_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_count": len(results),
        "planner_status_counts": status_counts,
        "mean_recommended_benchmark_score": mean(scores) if scores else None,
        "source_review_ready_count": sum(
            1 for item in results if item.get("source_review_status") == "ready_for_source_audit"
        ),
        "llm_context_ready_count": sum(1 for item in results if item.get("llm_context_status") == "ready_for_llm_context"),
        "reference_matrix_ready_count": sum(
            1 for item in results if item.get("reference_matrix_status") == "ready_for_offline_planning"
        ),
        "issues": issues,
        "passed": not issues,
        "claim_boundary": (
            "Deterministic planning evidence only; no real LLM calls and no scientific claim support."
        ),
        "results": results,
    }
    if source_collection is not None:
        payload["source_collection"] = {
            "status": source_collection.get("status"),
            "candidate_count": source_collection.get("candidate_count"),
            "issue_count": source_collection.get("issue_count"),
            "query": source_collection.get("query"),
            "created_at": source_collection.get("created_at"),
        }
    if source_candidate_results is not None:
        payload["source_candidate_count"] = len(source_candidate_results)
        payload["source_candidate_context_ready_count"] = sum(
            1 for item in source_candidate_results if item.get("llm_context_status") == "ready_for_llm_context"
        )
        payload["source_candidate_manual_wiki_review_count"] = sum(
            1 for item in source_candidate_results if item.get("wiki_promotion_status") == "manual_review_required"
        )
        payload["source_candidate_results"] = source_candidate_results
    return payload


def _audit_issues(results: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if len(results) < 10:
        issues.append("expected at least 10 paper-problem cases")
    for item in results:
        paper_id = str(item.get("paper_id"))
        if item.get("llm_context_status") != "ready_for_llm_context":
            issues.append(f"{paper_id} llm context is not ready")
        if item.get("reference_matrix_status") != "ready_for_offline_planning":
            issues.append(f"{paper_id} reference matrix is not ready")
        if not item.get("title_zh") or not item.get("real_problem_zh"):
            issues.append(f"{paper_id} missing bilingual wiki fields")
        if item.get("source_review_status") != "ready_for_source_audit":
            issues.append(f"{paper_id} source metadata is incomplete")
    fno = next((item for item in results if item.get("paper_id") == "paper:fourier_neural_operator_parametric_pdes"), None)
    if fno:
        selected = fno.get("selected_algorithm_ids") if isinstance(fno.get("selected_algorithm_ids"), list) else []
        if fno.get("recommended_benchmark") != "reaction_diffusion_operator_faithful_small":
            issues.append("FNO paper should map to reaction_diffusion_operator_faithful_small")
        if "fno_lite_operator" not in selected:
            issues.append("FNO paper should select fno_lite_operator")
    return issues


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# AgenticSciML Paper Problem Loop Audit",
        "",
        f"- case_count: {audit.get('case_count')}",
        f"- passed: {audit.get('passed')}",
        f"- planner_status_counts: {audit.get('planner_status_counts')}",
        f"- source_review_ready_count: {audit.get('source_review_ready_count')}",
        f"- llm_context_ready_count: {audit.get('llm_context_ready_count')}",
        f"- reference_matrix_ready_count: {audit.get('reference_matrix_ready_count')}",
        f"- claim_boundary: {audit.get('claim_boundary')}",
        "",
    ]
    if isinstance(audit.get("source_collection"), dict):
        lines.extend([f"- source_collection: {audit['source_collection']}", ""])
    if "source_candidate_count" in audit:
        lines.extend(
            [
                f"- source_candidate_count: {audit.get('source_candidate_count')}",
                f"- source_candidate_context_ready_count: {audit.get('source_candidate_context_ready_count')}",
                f"- source_candidate_manual_wiki_review_count: {audit.get('source_candidate_manual_wiki_review_count')}",
                "",
            ]
        )
    if audit.get("issues"):
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in audit["issues"])
        lines.append("")
    lines.extend(["## Cases", ""])
    for item in audit.get("results", []):
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                f"### {item.get('index')}. {item.get('title')}",
                "",
                f"- title_zh: {item.get('title_zh')}",
                f"- url: {item.get('url')}",
                f"- real_problem_zh: {item.get('real_problem_zh')}",
                f"- source_review_status: {item.get('source_review_status')}",
                f"- planner_status: {item.get('planner_status')}",
                f"- recommended_benchmark: {item.get('recommended_benchmark')} (score={item.get('recommended_benchmark_score')})",
                f"- selected_algorithm_ids: {', '.join(item.get('selected_algorithm_ids') or [])}",
                f"- llm_context_status: {item.get('llm_context_status')}",
                f"- artifacts: {', '.join((item.get('artifacts') or {}).values())}",
                "",
            ]
        )
    return "\n".join(lines)


def _source_candidate_results(
    *,
    output_dir: Path,
    source_collection: dict[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not isinstance(source_collection, dict):
        return []
    candidates = source_collection.get("candidates") if isinstance(source_collection.get("candidates"), list) else []
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        if not isinstance(candidate, dict):
            continue
        case = _source_candidate_case(candidate)
        tags = [str(tag) for tag in case["tags"]]
        expert_blueprint = _expert_blueprint(tags)
        intake = _problem_intake(case, tags)
        plan = _plan_problem(intake, expert_blueprint)
        matrix = build_reference_capability_matrix(problem_intake=intake, expert_blueprint_id=expert_blueprint)
        context = build_llm_problem_context_pack(
            problem_intake=intake,
            expert_blueprint_id=expert_blueprint,
            resource_constraints=_resource_constraints(),
            reference_capability_matrix=matrix,
        )
        artifacts = _write_case_artifacts(
            output_dir=output_dir,
            index=index,
            paper_id=str(case["id"]),
            source_review=_source_review(case),
            plan=plan,
            matrix=matrix,
            context=context,
            group="source_candidates",
        )
        results.append(
            {
                "index": index,
                "paper_id": case["id"],
                "title": case["title"],
                "url": case["url"],
                "real_problem": case["real_problem"],
                "real_problem_zh": case["real_problem_zh"],
                "wiki_promotion_status": "manual_review_required",
                "expert_blueprint_id": expert_blueprint,
                "planner_status": plan.get("status"),
                "recommended_benchmark": _nested_text(plan, "recommended_benchmark", "name"),
                "source_review_status": _source_review(case).get("status"),
                "reference_matrix_status": matrix.get("status"),
                "llm_context_status": context.get("status"),
                "artifacts": artifacts,
            }
        )
    return results


def _plan_problem(intake: dict[str, object], expert_blueprint: str) -> dict[str, Any]:
    from agenticsciml.web.app import ProblemIntakeRequest, _problem_intake_plan_payload

    return _problem_intake_plan_payload(
        ProblemIntakeRequest(
            problem_statement=str(intake["problem_summary"]),
            requirements=(
                "Use local deterministic planning only; preserve evaluator, sandbox, artifact, "
                "and claim-gate boundaries."
            ),
            evaluation_criteria=str(intake["metric"]),
            data_description=str(intake["data_source"]),
            mode="mock",
            target_solution_count=6,
            parallel_mutations=2,
            selector_vote_count=3,
            max_children_per_node=3,
            visual_audit_mode="off",
            resource_constraints=_resource_constraints(),
            expert_blueprint_id=expert_blueprint,
        )
    )


def _write_case_artifacts(
    *,
    output_dir: Path,
    index: int,
    paper_id: str,
    source_review: dict[str, Any],
    plan: dict[str, Any],
    matrix: dict[str, Any],
    context: dict[str, Any],
    group: str = "cases",
) -> dict[str, str]:
    case_dir = output_dir / group / f"{index:02d}-{_slug(paper_id.split(':')[-1])}"
    payloads = {
        "source_review": source_review,
        "planner": plan,
        "reference_matrix": matrix,
        "llm_context_pack": context,
    }
    paths: dict[str, str] = {}
    for name, payload in payloads.items():
        path = case_dir / f"{name}.json"
        _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        paths[name] = path.relative_to(output_dir).as_posix()
    return paths


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "case"


def _read_source_collection(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_source_collection_json", "candidate_count": 0, "issue_count": 1}
    return payload if isinstance(payload, dict) else {"status": "invalid_source_collection_json", "candidate_count": 0, "issue_count": 1}


def _source_review(case: dict[str, object]) -> dict[str, Any]:
    missing = [
        key
        for key in ("id", "title", "url", "authors", "submitted", "real_problem", "real_problem_zh")
        if not case.get(key)
    ]
    source_type = "arxiv" if str(case.get("url", "")).startswith("https://arxiv.org/abs/") else "publisher"
    return {
        "status": "incomplete_source_metadata" if missing else "ready_for_source_audit",
        "source_type": source_type,
        "missing_fields": missing,
        "paper_id": case.get("id"),
        "title": case.get("title"),
        "url": case.get("url"),
        "authors": case.get("authors"),
        "submitted": case.get("submitted"),
        "published": case.get("published"),
        "real_problem": case.get("real_problem"),
        "real_problem_zh": case.get("real_problem_zh"),
    }


def _source_candidate_case(candidate: dict[str, Any]) -> dict[str, object]:
    tags = _source_candidate_tags(candidate)
    published = candidate.get("published")
    return {
        "id": f"source:{candidate.get('id') or _slug(str(candidate.get('title', 'candidate')))}",
        "title": candidate.get("title") or "Untitled source candidate",
        "summary": candidate.get("summary") or "",
        "url": candidate.get("url") or "",
        "authors": candidate.get("authors") or [],
        "submitted": published,
        "published": published,
        "real_problem": candidate.get("real_problem") or "Track this source candidate as planning context.",
        "real_problem_zh": candidate.get("real_problem_zh") or "把该候选论文作为规划上下文跟踪。",
        "tags": tags,
    }


def _source_candidate_tags(candidate: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(candidate.get(key) or "")
        for key in ("title", "summary", "real_problem")
    ).lower()
    tags = ["paper", "source-candidate"]
    if "agent" in text or "llm" in text:
        tags += ["agent-benchmark", "llm-evaluation"]
    if "operator" in text:
        tags += ["operator-learning", "neural-operator"]
    if "turbulence" in text or "fluid" in text or "closure" in text:
        tags += ["fluid-dynamics", "closure-modeling"]
    if "inverse" in text or "physics-informed" in text or "pinn" in text:
        tags += ["inverse-problem", "pinn"]
    return tags


def _problem_intake(case: dict[str, object], tags: list[str]) -> dict[str, object]:
    observable: list[str] = ["trusted_score", "trace_artifact", "failure_report"]
    metric = "artifact-backed readiness and task-specific error"
    constraints = ["no private-label leakage", "claim gate remains blocked without run evidence"]
    failure_modes = ["low-confidence benchmark mapping", "unsupported scientific claim", "missing domain review"]
    if any(tag in tags for tag in ("fluid-dynamics", "closure-modeling")):
        observable = ["wake_velocity", "drag", "closure_residual", "solver_stability"]
        metric = "leave-one-shape-out relative error plus solver-stability audit"
        constraints += ["RANS residual consistency", "solver-agnostic closure stability"]
    elif any(tag in tags for tag in ("operator-learning", "neural-operator", "deeponet", "fno")):
        observable = ["input_function", "output_field", "spectral_response", "residual_proxy"]
        metric = "relative L2 plus operator-fidelity diagnostic"
        constraints += ["operator input/output contract", "spectral or sensitivity audit when available"]
    elif "inverse-problem" in tags:
        observable = ["observed_field", "unknown_parameter", "residual", "optimization_trace"]
        metric = "parameter recovery error and PDE residual consistency"
        constraints += ["inverse parameter identifiability", "PDE residual consistency"]
    elif any(tag in tags for tag in ("agent-benchmark", "llm-evaluation", "multi-agent")):
        observable = ["prompt", "response", "tool_trace", "verified_outcome"]
        metric = "verifiable task success with artifact-backed failure attribution"
        constraints += ["no hidden chain-of-thought", "controlled action boundary"]
    return {
        "problem_summary": f"Paper case: {case['title']}. Real problem: {case['real_problem']}",
        "hypothesis": f"AgenticSciML can plan a bounded workflow proxy for: {case['real_problem']}",
        "observable": observable,
        "metric": metric,
        "failure_modes": failure_modes,
        "physical_constraints": constraints,
        "domain_review_checklist": [
            "source paper URL reviewed",
            "benchmark fidelity boundary reviewed",
            "manual Wiki edits do not become run evidence",
        ],
        "data_source": "public paper-derived problem statement plus local deterministic proxy fixtures",
    }


def _expert_blueprint(tags: list[str]) -> str:
    by_tag = {
        "fluid-dynamics": "fluid_pde",
        "closure-modeling": "fluid_pde",
        "pde": "piml",
        "pinn": "piml",
        "inverse-problem": "inverse_reconstruction",
        "operator-learning": "operator_learning",
        "neural-operator": "operator_learning",
        "deeponet": "operator_learning",
        "fno": "operator_learning",
        "agent-benchmark": "numerical_methods",
        "llm-evaluation": "numerical_methods",
        "multi-agent": "numerical_methods",
    }
    return next((by_tag[tag] for tag in tags if tag in by_tag), "numerical_methods")


def _resource_constraints() -> dict[str, object]:
    return {
        "cpu": "local",
        "gpu": False,
        "timeout_s": 120,
        "dependency_limits": ["numpy", "scipy"],
        "data_limits": "public metadata plus local proxy only",
    }


def _nested_text(payload: dict[str, Any], key: str, child_key: str) -> str | None:
    child = payload.get(key)
    if isinstance(child, dict) and isinstance(child.get(child_key), str):
        return child[child_key]
    return None
