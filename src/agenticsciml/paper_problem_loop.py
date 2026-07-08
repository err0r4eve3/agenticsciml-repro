from __future__ import annotations

import calendar
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from agenticsciml.llm_problem_context import PROMPT_QUALITY_CONTROLS, build_llm_problem_context_pack
from agenticsciml.reference_capability_matrix import build_reference_capability_matrix
from agenticsciml.storage import _atomic_write_text


AUDIT_JSON = "agenticsciml_paper_problem_loop_audit.json"
AUDIT_MD = "summary.md"
LOOP_INDEX_JSON = "paper_problem_loop_index.json"
LOOP_INDEX_MD = "paper_problem_loop_index.md"
LOOP_HEALTH_JSON = "paper_problem_loop_health.json"


def write_paper_problem_loop_audit(
    *,
    output_dir: Path,
    case_limit: int | None = None,
    source_collection_path: Path | None = None,
    source_candidate_limit: int = 3,
    source_candidate_offset: int = 0,
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
                "prompt_quality_control_ids": _prompt_quality_control_ids(context),
                "artifacts": artifacts,
            }
        )
    source_collection = _read_source_collection(source_collection_path)
    source_candidate_results = _source_candidate_results(
        output_dir=output_dir,
        source_collection=source_collection,
        limit=source_candidate_limit,
        offset=source_candidate_offset,
    )
    llm_wiki_audit = _write_llm_wiki_audit(output_dir)
    audit = _audit_payload(
        results,
        source_collection=source_collection,
        source_candidate_results=source_candidate_results,
        source_candidate_offset=source_candidate_offset,
        llm_wiki_audit=llm_wiki_audit,
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


def update_paper_problem_loop_index(*, index_path: Path, audit_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    existing = _read_loop_index(index_path)
    entry = _loop_index_entry(index_path.parent, audit_path, audit)
    rounds = [
        item
        for item in existing.get("rounds", [])
        if isinstance(item, dict) and item.get("audit_json") != entry["audit_json"]
    ]
    rounds.append(entry)
    index = {
        "artifact_type": "agenticsciml_paper_problem_loop_index",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "round_count": len(rounds),
        "latest": rounds[-1],
        "failed_round_count": sum(1 for item in rounds if item.get("passed") is not True),
        "rounds": rounds,
    }
    _atomic_write_text(index_path, json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False))
    _atomic_write_text(index_path.with_name(LOOP_INDEX_MD), _render_loop_index_markdown(index))
    return index


def verify_paper_problem_loop(*, output_dir: Path, max_age_s: float | None = None) -> dict[str, Any]:
    issues: list[str] = []
    index_path = output_dir / LOOP_INDEX_JSON
    index_md_path = output_dir / LOOP_INDEX_MD
    index = _read_json_file(index_path, issues)
    index_md = index_md_path.read_text(encoding="utf-8") if index_md_path.exists() else ""
    if not index_md:
        issues.append(f"missing {LOOP_INDEX_MD}")
    latest = index.get("latest") if isinstance(index.get("latest"), dict) else {}
    latest_round_id = latest.get("round_id") if isinstance(latest.get("round_id"), str) else ""
    audit_ref = latest.get("audit_json") if isinstance(latest.get("audit_json"), str) else ""
    if not latest_round_id:
        issues.append("index missing latest round")
    if not audit_ref:
        issues.append("index missing latest audit path")
    audit_path = (output_dir / audit_ref).resolve(strict=False) if audit_ref else output_dir / "__missing_audit__"
    if audit_ref and not audit_path.is_relative_to(output_dir.resolve(strict=False)):
        issues.append("latest audit path escapes output dir")
        audit_path = output_dir / "__escaped_audit__"
    audit = _read_json_file(audit_path, issues)
    if index_md and latest_round_id not in index_md:
        issues.append("index markdown missing latest round")
    if index_md and audit_ref not in index_md:
        issues.append("index markdown missing latest audit path")
    if audit:
        _verify_latest_audit(latest, audit_path, audit, issues, max_age_s=max_age_s)
    status = "passed" if not issues else "failed"
    return {
        "artifact_type": "agenticsciml_paper_problem_loop_health",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
        "index_json": str(index_path.resolve(strict=False)),
        "index_md": str(index_md_path.resolve(strict=False)),
        "latest_audit_json": str(audit_path),
        "round_count": index.get("round_count"),
        "failed_round_count": index.get("failed_round_count"),
        "latest_round_id": latest.get("round_id"),
    }


def _verify_latest_audit(
    latest: dict[str, Any],
    audit_path: Path,
    audit: dict[str, Any],
    issues: list[str],
    *,
    max_age_s: float | None,
) -> None:
    audit_issues = audit.get("issues")
    if audit.get("passed") is not True:
        issues.append("latest audit did not pass")
    if audit_issues:
        issues.append("latest audit has issues")
    expected = {
        "passed": audit.get("passed"),
        "issue_count": len(audit_issues) if isinstance(audit_issues, list) else None,
        "case_count": audit.get("case_count"),
        "source_candidate_count": audit.get("source_candidate_count"),
        "source_candidate_offset": audit.get("source_candidate_offset"),
        "source_candidate_selection_ids": audit.get("source_candidate_selection_ids"),
        "prompt_quality_control_ready_count": audit.get("prompt_quality_control_ready_count"),
        "source_candidate_prompt_quality_control_ready_count": audit.get(
            "source_candidate_prompt_quality_control_ready_count"
        ),
    }
    for key, value in expected.items():
        if latest.get(key) != value:
            issues.append(f"latest index mismatch: {key}")
    wiki = audit.get("llm_wiki_audit") if isinstance(audit.get("llm_wiki_audit"), dict) else {}
    if latest.get("llm_wiki_status") != wiki.get("status"):
        issues.append("latest index mismatch: llm_wiki_status")
    if wiki.get("status") != "passed":
        issues.append("llm wiki audit did not pass")
    if wiki.get("languages") != ["en", "zh-CN"]:
        issues.append("llm wiki audit missing bilingual languages")
    if wiki.get("manual_editing") is not True:
        issues.append("llm wiki manual editing is not enabled")
    if (wiki.get("manual_edit_roundtrip") or {}).get("status") != "passed":
        issues.append("llm wiki manual edit roundtrip did not pass")
    artifacts = wiki.get("artifacts") if isinstance(wiki.get("artifacts"), dict) else {}
    for name, rel_path in artifacts.items():
        if not isinstance(rel_path, str) or not (audit_path.parent / rel_path).exists():
            issues.append(f"missing llm wiki artifact: {name}")
    if max_age_s is not None:
        _verify_audit_age(audit.get("created_at"), max_age_s, issues)


def _verify_audit_age(created_at: Any, max_age_s: float, issues: list[str]) -> None:
    if not isinstance(created_at, str):
        issues.append("latest audit missing created_at")
        return
    try:
        created_epoch = calendar.timegm(time.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        issues.append("latest audit created_at is invalid")
        return
    if time.time() - created_epoch > max_age_s:
        issues.append("latest audit is stale")


def _read_json_file(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing {path.name}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        issues.append(f"invalid json: {path.name}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"json root is not object: {path.name}")
        return {}
    return payload


def _read_loop_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rounds": _backfill_loop_rounds(path.parent)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"rounds": _backfill_loop_rounds(path.parent)}
    if not isinstance(payload, dict) or not isinstance(payload.get("rounds"), list):
        return {"rounds": _backfill_loop_rounds(path.parent)}
    by_path = {
        item.get("audit_json"): item
        for item in payload["rounds"]
        if isinstance(item, dict) and item.get("audit_json")
    }
    for item in _backfill_loop_rounds(path.parent):
        by_path.setdefault(item.get("audit_json"), item)
    return {"rounds": sorted(by_path.values(), key=lambda item: str(item.get("audit_json")))}


def _backfill_loop_rounds(root: Path) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for audit_path in sorted(root.glob(f"round-*/{AUDIT_JSON}")):
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(audit, dict):
            rounds.append(_loop_index_entry(root, audit_path, audit))
    return rounds


def _loop_index_entry(root: Path, audit_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "round_id": audit_path.parent.name,
        "audit_json": audit_path.relative_to(root).as_posix() if audit_path.is_relative_to(root) else str(audit_path),
        "created_at": audit.get("created_at"),
        "passed": audit.get("passed"),
        "issue_count": len(audit.get("issues", [])) if isinstance(audit.get("issues"), list) else None,
        "case_count": audit.get("case_count"),
        "source_candidate_count": audit.get("source_candidate_count"),
        "source_candidate_offset": audit.get("source_candidate_offset"),
        "source_candidate_selection_ids": audit.get("source_candidate_selection_ids"),
        "prompt_quality_control_ready_count": audit.get("prompt_quality_control_ready_count"),
        "source_candidate_prompt_quality_control_ready_count": audit.get(
            "source_candidate_prompt_quality_control_ready_count"
        ),
        "llm_wiki_status": (audit.get("llm_wiki_audit") or {}).get("status")
        if isinstance(audit.get("llm_wiki_audit"), dict)
        else None,
    }


def _render_loop_index_markdown(index: dict[str, Any]) -> str:
    latest = index.get("latest") if isinstance(index.get("latest"), dict) else {}
    lines = [
        "# Paper Problem Loop Index",
        "",
        f"- Updated at: {index.get('updated_at')}",
        f"- Round count: {index.get('round_count')}",
        f"- Failed round count: {index.get('failed_round_count')}",
        f"- Latest round: {latest.get('round_id')}",
        f"- Latest audit: `{latest.get('audit_json')}`",
        f"- Latest status: passed={latest.get('passed')} issues={latest.get('issue_count')}",
        f"- Latest cases/source candidates: {latest.get('case_count')}/{latest.get('source_candidate_count')}",
        f"- Latest source candidate offset: {latest.get('source_candidate_offset')}",
        f"- Latest source candidate ids: {', '.join(latest.get('source_candidate_selection_ids') or [])}",
        f"- Latest prompt gates: curated={latest.get('prompt_quality_control_ready_count')} source={latest.get('source_candidate_prompt_quality_control_ready_count')}",
        f"- Latest LLM Wiki status: {latest.get('llm_wiki_status')}",
        "",
        "## Recent Rounds",
        "",
        "| round | passed | issues | source candidates | wiki | audit |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    rounds = index.get("rounds") if isinstance(index.get("rounds"), list) else []
    for item in rounds[-20:]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            f"{item.get('round_id')} | {item.get('passed')} | {item.get('issue_count')} | "
            f"{item.get('source_candidate_count')} | {item.get('llm_wiki_status')} | "
            f"`{item.get('audit_json')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _audit_payload(
    results: list[dict[str, Any]],
    *,
    source_collection: dict[str, Any] | None = None,
    source_candidate_results: list[dict[str, Any]] | None = None,
    source_candidate_offset: int = 0,
    llm_wiki_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("planner_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    scores = [item["recommended_benchmark_score"] for item in results if isinstance(item.get("recommended_benchmark_score"), int)]
    issues = _audit_issues(results)
    issues += _prompt_quality_issues(results, prefix="case")
    if source_candidate_results is not None:
        issues += _prompt_quality_issues(source_candidate_results, prefix="source candidate")
    if llm_wiki_audit is not None:
        issues += [f"llm wiki: {issue}" for issue in llm_wiki_audit.get("issues", [])]
    payload = {
        "artifact_type": "agenticsciml_paper_problem_loop_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_count": len(results),
        "planner_status_counts": status_counts,
        "mean_recommended_benchmark_score": mean(scores) if scores else None,
        "source_review_ready_count": sum(
            1 for item in results if item.get("source_review_status") == "ready_for_source_audit"
        ),
        "prompt_quality_control_ready_count": sum(
            1 for item in results if _has_required_prompt_controls(item.get("prompt_quality_control_ids"))
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
        payload["source_candidate_offset"] = source_candidate_offset
        payload["source_candidate_count"] = len(source_candidate_results)
        payload["source_candidate_selection_ids"] = [
            item.get("paper_id") for item in source_candidate_results if item.get("paper_id")
        ]
        payload["source_candidate_prompt_quality_control_ready_count"] = sum(
            1 for item in source_candidate_results if _has_required_prompt_controls(item.get("prompt_quality_control_ids"))
        )
        payload["source_candidate_context_ready_count"] = sum(
            1 for item in source_candidate_results if item.get("llm_context_status") == "ready_for_llm_context"
        )
        payload["source_candidate_manual_wiki_review_count"] = sum(
            1 for item in source_candidate_results if item.get("wiki_promotion_status") == "manual_review_required"
        )
        payload["source_candidate_results"] = source_candidate_results
    if llm_wiki_audit is not None:
        payload["llm_wiki_audit"] = {
            "status": llm_wiki_audit.get("status"),
            "issue_count": llm_wiki_audit.get("issue_count"),
            "paper_problem_case_count": llm_wiki_audit.get("paper_problem_case_count"),
            "languages": llm_wiki_audit.get("languages"),
            "manual_editing": llm_wiki_audit.get("manual_editing"),
            "persistence": llm_wiki_audit.get("persistence"),
            "manual_edit_roundtrip": llm_wiki_audit.get("manual_edit_roundtrip"),
            "artifacts": llm_wiki_audit.get("artifacts"),
        }
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
        f"- prompt_quality_control_ready_count: {audit.get('prompt_quality_control_ready_count')}",
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
                f"- source_candidate_offset: {audit.get('source_candidate_offset')}",
                f"- source_candidate_selection_ids: {', '.join(audit.get('source_candidate_selection_ids') or [])}",
                f"- source_candidate_prompt_quality_control_ready_count: {audit.get('source_candidate_prompt_quality_control_ready_count')}",
                f"- source_candidate_context_ready_count: {audit.get('source_candidate_context_ready_count')}",
                f"- source_candidate_manual_wiki_review_count: {audit.get('source_candidate_manual_wiki_review_count')}",
                "",
            ]
        )
    if isinstance(audit.get("llm_wiki_audit"), dict):
        lines.extend([f"- llm_wiki_audit: {audit['llm_wiki_audit']}", ""])
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


def _write_llm_wiki_audit(output_dir: Path) -> dict[str, Any]:
    from agenticsciml.web.app import _llm_wiki_okf_payload

    graph = _llm_wiki_okf_payload()
    graph_path = output_dir / "llm_wiki" / "llm_wiki_okf.json"
    audit_path = output_dir / "llm_wiki" / "llm_wiki_audit.json"
    _atomic_write_text(graph_path, json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False))
    issues = _llm_wiki_issues(graph)
    manual_roundtrip = _manual_wiki_edit_roundtrip(graph, output_dir)
    if manual_roundtrip["status"] != "passed":
        issues.append("manual edit roundtrip failed")
    paper_nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("type") == "paper_problem_case"]
    edit_policy = graph.get("edit_policy") if isinstance(graph.get("edit_policy"), dict) else {}
    audit = {
        "artifact_type": "agenticsciml_llm_wiki_okf_audit",
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues,
        "paper_problem_case_count": len(paper_nodes),
        "languages": graph.get("languages"),
        "manual_editing": edit_policy.get("manual_editing"),
        "persistence": edit_policy.get("persistence"),
        "manual_edit_roundtrip": manual_roundtrip,
        "artifacts": {
            "graph": graph_path.relative_to(output_dir).as_posix(),
            "audit": audit_path.relative_to(output_dir).as_posix(),
            "manual_edit_roundtrip": manual_roundtrip["path"],
        },
    }
    _atomic_write_text(audit_path, json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False))
    return audit


def _manual_wiki_edit_roundtrip(graph: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from agenticsciml.web.app import _validated_llm_wiki_payload

    edited = dict(graph)
    edited["title"] = f"{graph.get('title', 'AgenticSciML LLM Wiki Knowledge Graph')} - manual edit audit"
    edited["manual_edit_audit"] = {
        "status": "simulated_manual_edit",
        "claim_boundary": "Roundtrip validates editability only; it is not benchmark evidence.",
    }
    path = output_dir / "llm_wiki" / "manual_edit_roundtrip.json"
    try:
        validated = _validated_llm_wiki_payload(edited)
        status = (
            "passed"
            if validated.get("title") != graph.get("title")
            and isinstance(validated.get("edit_policy"), dict)
            and validated["edit_policy"].get("manual_editing") is True
            and validated["edit_policy"].get("persistence") == "account_scoped_json"
            else "failed"
        )
        error = None
    except Exception as exc:
        validated = edited
        status = "failed"
        error = str(exc)
    _atomic_write_text(path, json.dumps(validated, indent=2, ensure_ascii=False, allow_nan=False))
    return {
        "status": status,
        "error": error,
        "path": path.relative_to(output_dir).as_posix(),
        "title_changed": validated.get("title") != graph.get("title"),
        "manual_editing": (validated.get("edit_policy") or {}).get("manual_editing")
        if isinstance(validated.get("edit_policy"), dict)
        else None,
        "persistence": (validated.get("edit_policy") or {}).get("persistence")
        if isinstance(validated.get("edit_policy"), dict)
        else None,
    }


def _llm_wiki_issues(graph: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if graph.get("type") != "llm_wiki_knowledge_graph":
        issues.append("graph type must be llm_wiki_knowledge_graph")
    if graph.get("okf_version") != "0.1":
        issues.append("okf_version must be 0.1")
    if graph.get("languages") != ["en", "zh-CN"]:
        issues.append("languages must be ['en', 'zh-CN']")
    edit_policy = graph.get("edit_policy") if isinstance(graph.get("edit_policy"), dict) else {}
    if edit_policy.get("manual_editing") is not True:
        issues.append("manual_editing must be true")
    if edit_policy.get("persistence") != "account_scoped_json":
        issues.append("persistence must be account_scoped_json")
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    node_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    if len(node_ids) != len(nodes):
        issues.append("node ids must be unique")
    paper_nodes = [node for node in nodes if isinstance(node, dict) and node.get("type") == "paper_problem_case"]
    if len(paper_nodes) < 10:
        issues.append("expected at least 10 paper_problem_case nodes")
    for node in nodes:
        if not isinstance(node, dict):
            issues.append("node must be an object")
            continue
        for key in ("id", "type", "title", "description", "tags", "timestamp"):
            if not node.get(key):
                issues.append(f"{node.get('id') or 'unknown'} missing {key}")
        if node.get("type") == "paper_problem_case":
            for key in ("title_zh", "description_zh", "real_problem", "real_problem_zh"):
                if not node.get(key):
                    issues.append(f"{node.get('id') or 'unknown'} missing {key}")
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            issues.append("edge must be an object")
            continue
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            issues.append(f"edge {edge.get('source')}->{edge.get('target')} references missing node")
    return issues


def _source_candidate_results(
    *,
    output_dir: Path,
    source_collection: dict[str, Any] | None,
    limit: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if limit <= 0 or not isinstance(source_collection, dict):
        return []
    candidates = source_collection.get("candidates") if isinstance(source_collection.get("candidates"), list) else []
    candidates = _rotated_source_candidates(candidates, offset)[:limit]
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
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
                "prompt_quality_control_ids": _prompt_quality_control_ids(context),
                "artifacts": artifacts,
            }
        )
    return results


def _rotated_source_candidates(candidates: list[Any], offset: int) -> list[Any]:
    if not candidates:
        return []
    start = offset % len(candidates)
    return candidates[start:] + candidates[:start]


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


def _prompt_quality_control_ids(context: dict[str, Any]) -> list[str]:
    return [
        str(item.get("control_id"))
        for item in context.get("prompt_quality_controls", [])
        if isinstance(item, dict) and item.get("control_id")
    ]


def _required_prompt_control_ids() -> set[str]:
    return {
        str(item["control_id"])
        for item in PROMPT_QUALITY_CONTROLS
        if isinstance(item.get("control_id"), str)
    }


def _has_required_prompt_controls(value: object) -> bool:
    return isinstance(value, list) and set(value) >= _required_prompt_control_ids()


def _prompt_quality_issues(items: list[dict[str, Any]], *, prefix: str) -> list[str]:
    required = _required_prompt_control_ids()
    issues: list[str] = []
    for item in items:
        controls = set(item.get("prompt_quality_control_ids") if isinstance(item.get("prompt_quality_control_ids"), list) else [])
        missing = sorted(required - controls)
        if missing:
            item_id = item.get("paper_id") or item.get("title") or "unknown"
            issues.append(f"{prefix} {item_id} missing prompt controls: {','.join(missing)}")
    return issues


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
