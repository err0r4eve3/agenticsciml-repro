from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agenticsciml.operator_scheduler import OPERATOR_SCHEDULER_MODE
from agenticsciml.retrieval.kb_store import KnowledgeBaseEntry
from agenticsciml.state import Proposal, SolutionNode


KB_APPLICATION_SCHEMA_VERSION = 1
MUTATION_EFFECT_SCHEMA_VERSION = 1
EVOLUTION_HEALTH_SCHEMA_VERSION = 1
INNOVATION_REPORT_SCHEMA_VERSION = 1

NOVELTY_AXIS_TERMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "representation_or_features",
        "Representation / feature construction",
        ("fourier", "feature", "spectral", "basis", "kernel", "latent", "embedding", "deeponet", "fno"),
    ),
    (
        "physics_or_residual",
        "Physics, residual, or constraint structure",
        ("pinn", "physics", "residual", "collocation", "boundary", "initial condition", "pde", "poisson", "burgers"),
    ),
    (
        "optimization_or_schedule",
        "Optimization, schedule, or loss weighting",
        ("schedule", "epoch", "learning rate", "lr", "weight", "regularization", "ridge", "sampling"),
    ),
    (
        "data_or_sensor_processing",
        "Data, sensor, or field-processing strategy",
        ("sensor", "lagged", "history", "filter", "bandlimit", "smooth", "denoise", "probe", "field"),
    ),
    (
        "algorithm_composition",
        "Composition of multiple strategy seeds or method tags",
        ("combine", "hybrid", "compose", "ensemble", "mixture", "stack", "plus", "multi"),
    ),
    (
        "debugging_or_robustness",
        "Debugging, guardrail, or robustness adaptation",
        ("debug", "repair", "retry", "guardrail", "failure", "robust", "fallback", "stability"),
    ),
)


def build_kb_application_report(
    *,
    solution_id: str,
    kb_entry: KnowledgeBaseEntry | None,
    proposal: Proposal | None,
    workspace: Path,
) -> dict[str, Any]:
    if kb_entry is None:
        return {
            "schema_version": KB_APPLICATION_SCHEMA_VERSION,
            "solution_id": solution_id,
            "retrieved_entry_id": None,
            "retrieved_title": None,
            "actionable_points": [],
            "proposal_adopted_points": [],
            "engineer_implemented_points": [],
            "static_evidence": {},
            "status": "not_retrieved",
            "warnings": [],
        }

    actionable_points = _actionable_kb_points(kb_entry)
    proposal_adopted_points = _proposal_adopted_points(proposal, actionable_points)
    engineering_response = _read_json_object(workspace / "engineering_response.json")
    engineer_implemented_points = [
        str(item)
        for item in engineering_response.get("implemented_kb_points", [])
        if isinstance(item, str) and item.strip()
    ]
    proposal_text = proposal.to_markdown() if proposal else _read_text(workspace / "proposal.md")
    engineering_text = _read_text(workspace / "engineering_summary.md")
    code_text = _read_text(workspace / "solution.py")
    static_evidence = {
        point["id"]: _static_point_evidence(point, proposal_text, engineering_text, code_text)
        for point in actionable_points
    }
    code_verified_ids = {
        point_id
        for point_id, evidence in static_evidence.items()
        if evidence.get("code_signal")
    }
    engineer_signal_ids = {
        point_id
        for point_id, evidence in static_evidence.items()
        if evidence.get("engineer_signal")
    }
    adopted_ids = {
        point_id
        for point_id, evidence in static_evidence.items()
        if evidence.get("proposal_signal")
    }
    engineer_claimed_ids = _claimed_point_ids(engineer_implemented_points, actionable_points)
    unverified_claimed_ids = sorted(engineer_claimed_ids - code_verified_ids)
    missing_static_evidence = sorted(
        point_id
        for point_id in engineer_claimed_ids
        if point_id not in code_verified_ids
    )

    if unverified_claimed_ids or (engineer_implemented_points and not code_verified_ids):
        status = "unverified"
    elif code_verified_ids:
        status = "implemented"
    elif adopted_ids or proposal_adopted_points or engineer_signal_ids:
        status = "proposed"
    else:
        status = "retrieved_only"
    warnings: list[str] = []
    if status == "retrieved_only":
        warnings.append("KB was retrieved but no proposal or code adoption evidence was found.")
    if status == "unverified":
        warnings.append(
            "Engineer claimed KB implementation, but lightweight static code evidence did not verify those claims."
        )
    return {
        "schema_version": KB_APPLICATION_SCHEMA_VERSION,
        "solution_id": solution_id,
        "retrieved_entry_id": kb_entry.entry_id,
        "retrieved_title": kb_entry.title,
        "actionable_points": actionable_points,
        "proposal_adopted_points": proposal_adopted_points or sorted(adopted_ids),
        "engineer_implemented_points": engineer_implemented_points
        or sorted(code_verified_ids | engineer_signal_ids),
        "engineer_claimed_point_ids": sorted(engineer_claimed_ids),
        "code_verified_point_ids": sorted(code_verified_ids),
        "unverified_implemented_points": unverified_claimed_ids,
        "missing_static_evidence": missing_static_evidence,
        "static_evidence": static_evidence,
        "status": status,
        "warnings": warnings,
        "claim_boundary": (
            "KB application audit records workflow evidence only. Static evidence is a lightweight "
            "signal, not a proof of scientific validity or benchmark improvement."
        ),
    }


def build_mutation_effect_report(
    *,
    node: SolutionNode,
    parent_node: SolutionNode | None,
    parent_code: str | None,
    child_code: str,
    workspace: Path,
    operator_assignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent_digest = _sha256_text(parent_code) if parent_code is not None else None
    child_digest = _sha256_text(child_code)
    proposal_text = _read_text(workspace / "proposal.md")
    parent_changed = parent_digest is not None and parent_digest != child_digest
    diff_line_count = _diff_line_count(parent_code or "", child_code) if parent_code is not None else 0
    score_delta = node.score_delta_from_parent
    duplicate_of = parent_node.node_id if parent_digest is not None and parent_digest == child_digest and parent_node else None
    if node.status == "failed":
        status = "failed"
    elif duplicate_of:
        status = "duplicate_parent"
    elif score_delta is not None and abs(score_delta) <= 1e-12:
        status = "changed_but_score_plateau" if parent_changed else "duplicate_parent"
    else:
        status = "changed_score_moved" if parent_changed else "unclassified"
    assignment = operator_assignment if isinstance(operator_assignment, dict) else _read_json_object(
        workspace / "operator_assignment.json"
    )
    operator_expected_terms = [
        str(item)
        for item in assignment.get("operator_expected_terms", [])
        if isinstance(item, str) and item.strip()
    ]
    operator_static_evidence = _operator_static_evidence(
        operator_expected_terms,
        proposal_text=proposal_text,
        engineering_text=_read_text(workspace / "engineering_summary.md"),
        code_text=child_code,
    )
    return {
        "schema_version": MUTATION_EFFECT_SCHEMA_VERSION,
        "solution_id": node.node_id,
        "parent_id": node.parent_id,
        "status": status,
        "operator_id": assignment.get("operator_id"),
        "mutation_axis": assignment.get("mutation_axis"),
        "operator_expected_terms": operator_expected_terms,
        "operator_static_evidence": operator_static_evidence,
        "code_digest": child_digest,
        "parent_code_digest": parent_digest,
        "proposal_digest": _sha256_text(proposal_text) if proposal_text else None,
        "diff_line_count": diff_line_count,
        "code_changed_from_parent": parent_changed,
        "duplicate_of": duplicate_of,
        "score": node.score.to_dict() if node.score else None,
        "score_delta_from_parent": score_delta,
        "method_tags": list(node.method_tags),
        "claim_boundary": (
            "Mutation effect audit explains workflow evolution health. It does not change evaluator score."
        ),
    }


def build_evolution_health_report(nodes: list[SolutionNode], run_dir: Path) -> dict[str, Any]:
    digests: dict[str, str] = {}
    duplicate_nodes: list[dict[str, str]] = []
    score_plateau_nodes: list[str] = []
    mutation_status_counts: dict[str, int] = {}
    operator_health: dict[str, dict[str, Any]] = {}
    missing_operator_assignment_nodes: list[str] = []
    operator_method_tag_mismatch_nodes: list[str] = []
    operator_assignment_warnings: list[dict[str, Any]] = []
    operator_assignment_count = 0
    best_improvement: float | None = None

    for node in nodes:
        workspace = Path(node.workspace)
        digest = _file_sha256(workspace / "solution.py")
        if digest:
            duplicate_of = next((node_id for node_id, seen_digest in digests.items() if seen_digest == digest), None)
            if duplicate_of:
                duplicate_nodes.append({"node_id": node.node_id, "duplicate_of": duplicate_of})
            digests[node.node_id] = digest
        report = _read_json_object(workspace / "mutation_effect_report.json")
        status = str(report.get("status") or ("root" if node.parent_id is None else "missing"))
        mutation_status_counts[status] = mutation_status_counts.get(status, 0) + 1
        assignment = _read_json_object(workspace / "operator_assignment.json")
        if node.parent_id is not None:
            if assignment:
                operator_assignment_count += 1
                if _operator_method_tag_mismatch(node, assignment):
                    operator_method_tag_mismatch_nodes.append(node.node_id)
                warnings = assignment.get("warnings")
                if isinstance(warnings, list) and warnings:
                    operator_assignment_warnings.append(
                        {
                            "node_id": node.node_id,
                            "operator_id": assignment.get("operator_id"),
                            "warnings": [str(warning) for warning in warnings],
                        }
                    )
            else:
                missing_operator_assignment_nodes.append(node.node_id)
        _update_operator_health(operator_health, node, report, workspace)
        if status == "changed_but_score_plateau":
            score_plateau_nodes.append(node.node_id)
        if node.score and node.parent_id and node.score_delta_from_parent is not None:
            improvement = (
                node.score_delta_from_parent
                if node.score.higher_is_better
                else -node.score_delta_from_parent
            )
            best_improvement = improvement if best_improvement is None else max(best_improvement, improvement)
    plateau_lengths = _score_plateau_lengths(nodes)
    report = {
        "schema_version": EVOLUTION_HEALTH_SCHEMA_VERSION,
        "solution_count": len(nodes),
        "unique_code_count": len(set(digests.values())),
        "duplicate_code_count": len(duplicate_nodes),
        "duplicate_nodes": duplicate_nodes,
        "mutation_status_counts": dict(sorted(mutation_status_counts.items())),
        "operator_scheduler_mode": OPERATOR_SCHEDULER_MODE,
        "operator_assignment_count": operator_assignment_count,
        "operator_assignment_expected_count": sum(1 for node in nodes if node.parent_id is not None),
        "missing_operator_assignment_nodes": missing_operator_assignment_nodes,
        "operator_method_tag_mismatch_nodes": operator_method_tag_mismatch_nodes,
        "operator_assignment_warning_count": len(operator_assignment_warnings),
        "operator_assignment_warnings": operator_assignment_warnings,
        "operator_health": dict(sorted(operator_health.items())),
        "score_plateau_nodes": score_plateau_nodes,
        "max_plateau_length": max(plateau_lengths, default=0),
        "best_improvement": best_improvement,
        "warnings": [],
        "claim_boundary": (
            "Evolution health reports duplicate code and score plateau diagnostics; evaluator scores remain authoritative."
        ),
    }
    if duplicate_nodes:
        report["warnings"].append("Duplicate solution code detected; mutation may not be changing effective code.")
    if score_plateau_nodes or report["max_plateau_length"] >= 3:
        report["warnings"].append("Score plateau detected; mutation may be ineffective or evaluator may lack resolution.")
    if missing_operator_assignment_nodes:
        report["warnings"].append("Operator assignment missing for one or more child nodes.")
    if operator_method_tag_mismatch_nodes:
        report["warnings"].append("Operator assignment and solution method_tags are inconsistent.")
    if operator_assignment_warnings:
        report["warnings"].append("Operator scheduler warnings were recorded for one or more child nodes.")
    return report


def _operator_method_tag_mismatch(node: SolutionNode, assignment: dict[str, Any]) -> bool:
    operator_id = assignment.get("operator_id")
    mutation_axis = assignment.get("mutation_axis")
    tags = {tag for tag in node.method_tags if isinstance(tag, str)}
    if isinstance(operator_id, str) and operator_id and f"operator:{operator_id}" not in tags:
        return True
    if isinstance(mutation_axis, str) and mutation_axis and f"axis:{mutation_axis}" not in tags:
        return True
    return False


def _update_operator_health(
    operator_health: dict[str, dict[str, Any]],
    node: SolutionNode,
    mutation_report: dict[str, Any],
    workspace: Path,
) -> None:
    assignment = _read_json_object(workspace / "operator_assignment.json")
    operator_id = assignment.get("operator_id") or mutation_report.get("operator_id")
    if not isinstance(operator_id, str) or not operator_id:
        return
    entry = operator_health.setdefault(
        operator_id,
        {
            "assigned": 0,
            "evaluated": 0,
            "duplicate": 0,
            "plateau": 0,
            "improved": 0,
            "best_improvement": None,
            "axes": {},
        },
    )
    entry["assigned"] += 1
    if node.status == "evaluated":
        entry["evaluated"] += 1
    status = mutation_report.get("status")
    if status == "duplicate_parent":
        entry["duplicate"] += 1
    if status == "changed_but_score_plateau":
        entry["plateau"] += 1
    axis = assignment.get("mutation_axis") or mutation_report.get("mutation_axis")
    if isinstance(axis, str) and axis:
        axes = entry["axes"]
        axes[axis] = int(axes.get(axis, 0)) + 1
    if node.score and node.parent_id and node.score_delta_from_parent is not None:
        improvement = (
            node.score_delta_from_parent
            if node.score.higher_is_better
            else -node.score_delta_from_parent
        )
        if improvement > 0:
            entry["improved"] += 1
            current_best = entry.get("best_improvement")
            entry["best_improvement"] = (
                improvement
                if current_best is None
                else max(float(current_best), improvement)
            )


def _operator_static_evidence(
    expected_terms: list[str],
    *,
    proposal_text: str,
    engineering_text: str,
    code_text: str,
) -> dict[str, dict[str, bool]]:
    return {
        term: {
            "proposal_signal": _has_any_term(proposal_text, [term]),
            "engineer_signal": _has_any_term(engineering_text, [term]),
            "code_signal": _has_any_term(code_text, [term]),
        }
        for term in expected_terms
    }


def build_innovation_report(
    *,
    nodes: list[SolutionNode],
    run_dir: Path,
    benchmark_name: str,
    strategy_seed_ids: list[str],
    problem_intake: dict[str, Any],
    planner_snapshot: dict[str, Any],
    evolution_health: dict[str, Any],
    claim_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    method_tag_counts = Counter(
        tag
        for node in nodes
        for tag in node.method_tags
        if isinstance(tag, str) and tag.strip()
    )
    solution_innovation = [
        _solution_innovation_summary(node, run_dir=run_dir, strategy_seed_ids=strategy_seed_ids)
        for node in nodes
    ]
    novelty_axes = _run_novelty_axes(solution_innovation)
    candidate_emergent_count = sum(
        1 for item in solution_innovation if item.get("emergence_claim_level") == "candidate_emergent"
    )
    warnings = _innovation_warnings(
        solution_count=len(nodes),
        method_tag_counts=method_tag_counts,
        novelty_axis_count=len(novelty_axes),
        candidate_emergent_count=candidate_emergent_count,
        evolution_health=evolution_health,
        claim_gate=claim_gate,
    )
    evidence_summary = {
        "solution_count": len(nodes),
        "strategy_seed_count": len(strategy_seed_ids),
        "unique_method_tag_count": len(method_tag_counts),
        "novelty_axis_count": len(novelty_axes),
        "candidate_emergent_count": candidate_emergent_count,
        "unique_code_count": evolution_health.get("unique_code_count"),
        "duplicate_code_count": evolution_health.get("duplicate_code_count"),
        "best_improvement": evolution_health.get("best_improvement"),
        "operator_count": len(evolution_health.get("operator_health", {}))
        if isinstance(evolution_health.get("operator_health"), dict)
        else 0,
        "warning_count": len(warnings),
    }
    return {
        "schema_version": INNOVATION_REPORT_SCHEMA_VERSION,
        "report_type": "agenticsciml_run_innovation_audit",
        "benchmark_name": benchmark_name,
        "innovation_claim_level": "workflow_exploration_only",
        "scientific_novelty_supported": False,
        "paper_level_discovery_supported": False,
        "strategy_seed_ids": list(strategy_seed_ids),
        "method_tag_counts": dict(sorted(method_tag_counts.items())),
        "problem_summary": _innovation_problem_summary(problem_intake, planner_snapshot),
        "evidence_summary": evidence_summary,
        "operator_coverage": evolution_health.get("operator_health", {}),
        "novelty_axes": novelty_axes,
        "solution_innovation": solution_innovation,
        "warnings": warnings,
        "next_experiment_suggestions": _innovation_next_steps(
            novelty_axis_count=len(novelty_axes),
            method_tag_counts=method_tag_counts,
            candidate_emergent_count=candidate_emergent_count,
            claim_gate=claim_gate,
        ),
        "claim_gate": claim_gate or {},
        "claim_boundary": (
            "Innovation audit records workflow exploration signals only. It is not scientific novelty evidence, "
            "not paper-level emergent discovery evidence, and not a substitute for trusted domain evaluation."
        ),
    }


def render_innovation_report_markdown(report: dict[str, Any]) -> str:
    evidence = report.get("evidence_summary") if isinstance(report.get("evidence_summary"), dict) else {}
    axes = report.get("novelty_axes") if isinstance(report.get("novelty_axes"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    suggestions = (
        report.get("next_experiment_suggestions")
        if isinstance(report.get("next_experiment_suggestions"), list)
        else []
    )
    lines = [
        "# Innovation Report",
        "",
        f"- benchmark: {report.get('benchmark_name')}",
        f"- claim level: {report.get('innovation_claim_level')}",
        f"- scientific novelty supported: {report.get('scientific_novelty_supported')}",
        f"- paper-level discovery supported: {report.get('paper_level_discovery_supported')}",
        f"- solution count: {evidence.get('solution_count')}",
        f"- novelty axes: {evidence.get('novelty_axis_count')}",
        f"- candidate emergent count: {evidence.get('candidate_emergent_count')}",
        "",
        "## Novelty Axes",
    ]
    if axes:
        for axis in axes:
            lines.append(
                f"- {axis.get('axis_id')}: {axis.get('label')} "
                f"(solutions: {', '.join(str(item) for item in axis.get('solution_ids', []))})"
            )
    else:
        lines.append("- none detected")
    lines.extend(["", "## Warnings"])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Next Experiment Suggestions"])
    if suggestions:
        lines.extend(f"- {suggestion}" for suggestion in suggestions)
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _actionable_kb_points(entry: KnowledgeBaseEntry) -> list[dict[str, Any]]:
    text = f"{entry.entry_id}\n{entry.title}\n{entry.description}\n{entry.content}".lower()
    if "pinn" in text or "collocation" in text:
        return [
            _point("sample_count", "Set or discuss training/sample count", ["sample", "n_train", "num_samples"]),
            _point("collocation_count", "Set or discuss collocation count", ["collocation", "n_colloc", "residual_points"]),
            _point("depth_width", "Set or discuss network depth/width", ["depth", "width", "hidden", "layers"]),
            _point("residual_weight", "Set or discuss residual loss weight", ["residual", "lambda", "weight"]),
            _point("training_schedule", "Set or discuss training schedule", ["epoch", "steps", "schedule", "lr"]),
        ]
    lines = [
        line.strip("-*# \t")
        for line in entry.content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    points = []
    for index, line in enumerate(lines[:5], start=1):
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line.lower())
        points.append(_point(f"kb_point_{index}", line[:140], words[:6] or [line[:40].lower()]))
    return points


def _solution_innovation_summary(
    node: SolutionNode,
    *,
    run_dir: Path,
    strategy_seed_ids: list[str],
) -> dict[str, Any]:
    workspace = Path(node.workspace)
    text_blob = "\n".join(
        [
            _read_text(workspace / "proposal.md"),
            _read_text(workspace / "engineering_summary.md"),
            _read_text(workspace / "analysis.md"),
            _read_text(workspace / "solution.py"),
            " ".join(node.method_tags),
            " ".join(strategy_seed_ids),
        ]
    )
    axis_ids = _novelty_axis_ids(text_blob, node.method_tags, strategy_seed_ids)
    mutation = _read_json_object(workspace / "mutation_effect_report.json")
    operator_assignment = _read_json_object(workspace / "operator_assignment.json")
    kb_application = _read_json_object(workspace / "kb_application_report.json")
    emergence = _read_json_object(workspace / "emergence_report.json")
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "status": node.status,
        "score": node.score.to_dict() if node.score else None,
        "score_delta_from_parent": node.score_delta_from_parent,
        "method_tags": list(node.method_tags),
        "operator_id": operator_assignment.get("operator_id"),
        "mutation_axis": operator_assignment.get("mutation_axis"),
        "novelty_axis_ids": axis_ids,
        "innovation_signal_count": len(axis_ids)
        + _truthy_signal_count(
            [
                mutation.get("status") in {"changed_score_moved", "changed_but_score_plateau"},
                kb_application.get("status") in {"proposed", "implemented", "unverified"},
                emergence.get("claim_level") == "candidate_emergent",
                bool(node.score_delta_from_parent),
            ]
        ),
        "mutation_status": mutation.get("status"),
        "kb_application_status": kb_application.get("status"),
        "emergence_claim_level": emergence.get("claim_level"),
        "artifact_refs": _innovation_artifact_refs(workspace, run_dir),
    }


def _run_novelty_axes(solution_innovation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    for axis_id, label, _terms in NOVELTY_AXIS_TERMS:
        solution_ids = [
            str(item["node_id"])
            for item in solution_innovation
            if axis_id in item.get("novelty_axis_ids", [])
        ]
        if solution_ids:
            axes.append(
                {
                    "axis_id": axis_id,
                    "label": label,
                    "solution_ids": solution_ids,
                    "evidence": "Detected from proposal, engineering summary, code text, method tags, or strategy seed ids.",
                }
            )
    return axes


def _novelty_axis_ids(
    text: str,
    method_tags: list[str],
    strategy_seed_ids: list[str],
) -> list[str]:
    lowered = text.lower()
    axis_ids: list[str] = []
    for axis_id, _label, terms in NOVELTY_AXIS_TERMS:
        if any(term in lowered for term in terms):
            axis_ids.append(axis_id)
    if len({tag for tag in method_tags if tag}) >= 2 or len(strategy_seed_ids) >= 2:
        if "algorithm_composition" not in axis_ids:
            axis_ids.append("algorithm_composition")
    return axis_ids


def _innovation_artifact_refs(workspace: Path, run_dir: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for name in (
        "proposal.md",
        "engineering_summary.md",
        "analysis.md",
        "solution.py",
        "mutation_effect_report.json",
        "operator_assignment.json",
        "kb_application_report.json",
        "emergence_report.json",
    ):
        path = workspace / name
        if path.exists():
            refs[name] = _relative_artifact_path(path, run_dir)
    return refs


def _relative_artifact_path(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _truthy_signal_count(values: list[bool]) -> int:
    return sum(1 for value in values if value)


def _innovation_problem_summary(
    problem_intake: dict[str, Any],
    planner_snapshot: dict[str, Any],
) -> dict[str, Any]:
    generated = planner_snapshot.get("generated_custom_benchmark")
    return {
        "problem_statement": _compact_text(str(problem_intake.get("problem_statement", "")), 360),
        "benchmark_mapping_status": (
            "custom_proxy_benchmark_scaffolded"
            if isinstance(generated, dict)
            else "catalog_benchmark"
        ),
        "custom_proxy_benchmark": generated if isinstance(generated, dict) else None,
    }


def _innovation_warnings(
    *,
    solution_count: int,
    method_tag_counts: Counter[str],
    novelty_axis_count: int,
    candidate_emergent_count: int,
    evolution_health: dict[str, Any],
    claim_gate: dict[str, Any] | None,
) -> list[str]:
    warnings = [
        "This report is not scientific novelty evidence; it is workflow exploration evidence only."
    ]
    if solution_count < 3:
        warnings.append("Search breadth is small; novelty signals may reflect prompt variation rather than robust discovery.")
    if len(method_tag_counts) <= 1:
        warnings.append("Low method-tag diversity; consider increasing target_solution_count or strategy seed breadth.")
    if novelty_axis_count == 0:
        warnings.append("No novelty axis was detected from available proposal/code/report text.")
    if candidate_emergent_count == 0:
        warnings.append("No per-solution emergence audit reached candidate_emergent.")
    if int(evolution_health.get("duplicate_code_count") or 0) > 0:
        warnings.append("Duplicate code was detected; innovation evidence is weakened.")
    if claim_gate and claim_gate.get("scientific_claim_supported") is not True:
        warnings.append("Claim gate does not support scientific claims for this run.")
    return warnings


def _innovation_next_steps(
    *,
    novelty_axis_count: int,
    method_tag_counts: Counter[str],
    candidate_emergent_count: int,
    claim_gate: dict[str, Any] | None,
) -> list[str]:
    suggestions = [
        "Run at least three seeds and compare innovation axes against score movement.",
        "Ask a domain reviewer to replace or approve any synthetic/custom evaluator before scientific claims.",
    ]
    if novelty_axis_count < 2:
        suggestions.append("Broaden selected algorithm seeds to force representation, physics, and optimization contrasts.")
    if len(method_tag_counts) < 3:
        suggestions.append("Increase target_solution_count or parallel_mutations to improve method-tag diversity.")
    if candidate_emergent_count == 0:
        suggestions.append("Inspect proposal, policy_fidelity, and emergence reports to identify missing prior-result adaptation evidence.")
    if claim_gate and claim_gate.get("paper_level_claim_supported") is not True:
        suggestions.append("Keep reporting as workflow_proxy unless paper_workflow claim gate requirements are satisfied.")
    return suggestions


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _point(point_id: str, description: str, terms: list[str]) -> dict[str, Any]:
    return {"id": point_id, "description": description, "terms": terms}


def _proposal_adopted_points(proposal: Proposal | None, points: list[dict[str, Any]]) -> list[str]:
    if proposal is None:
        return []
    application = proposal.kb_application if isinstance(proposal.kb_application, dict) else {}
    explicit = [
        str(item)
        for values in (
            application.get("proposal_adopted_points"),
            application.get("adopted_points"),
            application.get("actionable_points"),
        )
        if isinstance(values, list)
        for item in values
        if isinstance(item, str)
    ]
    if explicit:
        return explicit
    proposal_text = proposal.to_markdown()
    return [
        str(point["id"])
        for point in points
        if _has_any_term(proposal_text, [str(term) for term in point.get("terms", [])])
    ]


def _claimed_point_ids(claims: list[str], points: list[dict[str, Any]]) -> set[str]:
    if not claims:
        return set()
    claim_text = "\n".join(claims).lower()
    claimed_ids: set[str] = set()
    for point in points:
        point_id = str(point.get("id", ""))
        candidates = [
            point_id,
            point_id.replace("_", " "),
            str(point.get("description", "")),
            *[str(term) for term in point.get("terms", [])],
        ]
        if _has_any_term(claim_text, candidates):
            claimed_ids.add(point_id)
    return claimed_ids


def _static_point_evidence(
    point: dict[str, Any],
    proposal_text: str,
    engineering_text: str,
    code_text: str,
) -> dict[str, Any]:
    terms = [str(term) for term in point.get("terms", [])]
    return {
        "description": point.get("description"),
        "terms": terms,
        "proposal_signal": _has_any_term(proposal_text, terms),
        "engineer_signal": _has_any_term(engineering_text, terms),
        "code_signal": _has_any_term(code_text, terms),
    }


def _has_any_term(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def _diff_line_count(before: str, after: str) -> int:
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="")
    return sum(1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def _score_plateau_lengths(nodes: list[SolutionNode]) -> list[int]:
    lengths: list[int] = []
    current_score: tuple[str, float] | None = None
    current_length = 0
    for node in nodes:
        if not node.score:
            continue
        score_key = (node.score.metric, round(node.score.value, 12))
        if score_key == current_score:
            current_length += 1
        else:
            if current_length:
                lengths.append(current_length)
            current_score = score_key
            current_length = 1
    if current_length:
        lengths.append(current_length)
    return lengths


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
