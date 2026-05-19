from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agenticsciml.retrieval.kb_store import KnowledgeBaseEntry
from agenticsciml.state import Proposal, SolutionNode


KB_APPLICATION_SCHEMA_VERSION = 1
MUTATION_EFFECT_SCHEMA_VERSION = 1
EVOLUTION_HEALTH_SCHEMA_VERSION = 1


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
    implemented_ids = {
        point_id
        for point_id, evidence in static_evidence.items()
        if evidence.get("code_signal") or evidence.get("engineer_signal")
    }
    adopted_ids = {
        point_id
        for point_id, evidence in static_evidence.items()
        if evidence.get("proposal_signal")
    }
    if implemented_ids or engineer_implemented_points:
        status = "implemented"
    elif adopted_ids or proposal_adopted_points:
        status = "proposed"
    else:
        status = "retrieved_only"
    warnings: list[str] = []
    if status == "retrieved_only":
        warnings.append("KB was retrieved but no proposal or code adoption evidence was found.")
    return {
        "schema_version": KB_APPLICATION_SCHEMA_VERSION,
        "solution_id": solution_id,
        "retrieved_entry_id": kb_entry.entry_id,
        "retrieved_title": kb_entry.title,
        "actionable_points": actionable_points,
        "proposal_adopted_points": proposal_adopted_points or sorted(adopted_ids),
        "engineer_implemented_points": engineer_implemented_points or sorted(implemented_ids),
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
    return {
        "schema_version": MUTATION_EFFECT_SCHEMA_VERSION,
        "solution_id": node.node_id,
        "parent_id": node.parent_id,
        "status": status,
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
    return report


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
