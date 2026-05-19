from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOLUTION_TREE_SCHEMA_VERSION = "solution_tree.v1"
SOLUTION_ID_PATTERN = re.compile(r"^solution_(\d{3,})$")
SOLUTION_NODE_REQUIRED_FIELDS = {
    "analysis_path",
    "benchmark_name",
    "children",
    "contract_hash",
    "error",
    "failure_kind",
    "method_tags",
    "node_id",
    "num_debug_attempts",
    "parent_id",
    "proposal_path",
    "score",
    "score_delta_from_parent",
    "status",
    "workspace",
}
SOLUTION_NODE_ALLOWED_FIELDS = SOLUTION_NODE_REQUIRED_FIELDS
SOLUTION_NODE_STATUSES = {"created", "evaluated", "failed"}
SOLUTION_SCORE_ALLOWED_FIELDS = {"metric", "value", "higher_is_better"}


def format_solution_id(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"Solution index must be a non-negative integer: {index!r}")
    return f"solution_{index:03d}"


def solution_id_index(solution_id: str) -> int | None:
    match = SOLUTION_ID_PATTERN.fullmatch(solution_id)
    if not match:
        return None
    return int(match.group(1))


def validate_solution_score_payload(score: Any, *, context: str) -> list[str]:
    issues: list[str] = []
    if score is None:
        return issues
    if not isinstance(score, dict):
        return [f"{context} score must be an object or null"]
    unknown_fields = sorted(set(score) - SOLUTION_SCORE_ALLOWED_FIELDS)
    if unknown_fields:
        issues.append(f"{context} score has unknown fields: {', '.join(unknown_fields)}")
    metric = score.get("metric")
    if not isinstance(metric, str) or not metric:
        issues.append(f"{context} score.metric must be a non-empty string")
    value = score.get("value")
    if not _is_finite_number(value):
        issues.append(f"{context} score.value must be a finite number")
    higher_is_better = score.get("higher_is_better")
    if not isinstance(higher_is_better, bool):
        issues.append(f"{context} score.higher_is_better must be a boolean")
    return issues


def validate_solution_node_payload(data: Any, *, context: str = "SolutionNode") -> list[str]:
    if not isinstance(data, dict):
        return [f"{context} must be an object"]

    issues: list[str] = []
    unknown_fields = sorted(set(data) - SOLUTION_NODE_ALLOWED_FIELDS)
    if unknown_fields:
        issues.append(f"{context} has unknown fields: {', '.join(unknown_fields)}")
    missing_fields = sorted(SOLUTION_NODE_REQUIRED_FIELDS - set(data))
    for field_name in missing_fields:
        issues.append(f"{context} missing required field {field_name}")

    _check_required_str_field(data, "node_id", issues, context)
    _check_required_str_field(data, "workspace", issues, context)
    _check_nullable_str_field(data, "parent_id", issues, context)
    _check_nullable_str_field(data, "proposal_path", issues, context)
    _check_nullable_str_field(data, "analysis_path", issues, context)
    _check_nullable_str_field(data, "error", issues, context)
    _check_nullable_str_field(data, "benchmark_name", issues, context)
    _check_nullable_str_field(data, "contract_hash", issues, context)
    _check_nullable_str_field(data, "failure_kind", issues, context)

    status = data.get("status")
    if not isinstance(status, str) or status not in SOLUTION_NODE_STATUSES:
        issues.append(f"{context} has invalid status: {status!r}")

    children = data.get("children")
    if not isinstance(children, list) or any(not isinstance(item, str) for item in children):
        issues.append(f"{context} children must be a list of strings")

    method_tags = data.get("method_tags")
    if not isinstance(method_tags, list) or any(not isinstance(item, str) for item in method_tags):
        issues.append(f"{context} method_tags must be a list of strings")

    num_debug_attempts = data.get("num_debug_attempts")
    if (
        not isinstance(num_debug_attempts, int)
        or isinstance(num_debug_attempts, bool)
        or num_debug_attempts < 0
    ):
        issues.append(f"{context} num_debug_attempts must be a non-negative integer")

    score_delta = data.get("score_delta_from_parent")
    if score_delta is not None and not _is_finite_number(score_delta):
        issues.append(f"{context} score_delta_from_parent must be a finite number or null")

    issues.extend(validate_solution_score_payload(data.get("score"), context=context))
    issues.extend(validate_solution_node_status_semantics(data, context=context))
    return issues


def validate_solution_node_status_semantics(data: dict[str, Any], *, context: str) -> list[str]:
    issues: list[str] = []
    status = data.get("status")
    score = data.get("score")
    error = data.get("error")
    failure_kind = data.get("failure_kind")
    if status == "evaluated":
        if score is None:
            issues.append(f"{context} evaluated node must have a score")
        if error is not None:
            issues.append(f"{context} evaluated node must not have error")
        if failure_kind is not None:
            issues.append(f"{context} evaluated node must not have failure_kind")
    elif status == "failed":
        if score is not None:
            issues.append(f"{context} failed node must not have a score")
        if not error and not failure_kind:
            issues.append(f"{context} failed node must have error or failure_kind")
    elif status == "created":
        if score is not None:
            issues.append(f"{context} created node must not have a score")
        if error is not None:
            issues.append(f"{context} created node must not have error")
        if failure_kind is not None:
            issues.append(f"{context} created node must not have failure_kind")
    return issues


def validate_solution_tree_payload(nodes: Any, *, context: str = "Solution tree") -> list[str]:
    if not isinstance(nodes, list):
        return [f"{context} nodes must be a list"]

    issues: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            issues.append(f"{context} node at index {index} must be an object")
            continue
        node_id = node.get("node_id")
        node_context = f"{context} node {node_id}" if isinstance(node_id, str) and node_id else f"{context} node at index {index}"
        issues.extend(validate_solution_node_payload(node, context=node_context))
        if not isinstance(node_id, str) or not node_id:
            issues.append(f"{context} node at index {index} has invalid node_id")
            continue
        if node_id in by_id:
            issues.append(f"{context} has duplicate node_id: {node_id}")
            continue
        by_id[node_id] = node

    if not by_id:
        issues.append(f"{context} must contain at least one node")
    if by_id:
        issues.extend(validate_solution_tree_graph_payload(by_id, context=context))
    return issues


def validate_solution_tree_artifact_payload(
    payload: Any,
    *,
    context: str = "Solution tree artifact",
) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{context} must be an object"]
    issues: list[str] = []
    schema_version = payload.get("schema_version")
    if schema_version != SOLUTION_TREE_SCHEMA_VERSION:
        issues.append(
            f"{context} has unsupported schema_version: "
            f"{schema_version!r}; expected {SOLUTION_TREE_SCHEMA_VERSION!r}"
        )
    issues.extend(validate_solution_tree_payload(payload.get("nodes"), context=context))
    return issues


def validate_solution_tree_graph_payload(
    nodes: dict[str, dict[str, Any]],
    *,
    context: str = "Solution tree",
) -> list[str]:
    issues: list[str] = []
    root_ids = [node_id for node_id, node in nodes.items() if node.get("parent_id") is None]
    if len(root_ids) != 1:
        issues.append(f"{context} must have exactly one root node: roots={sorted(root_ids)!r}")

    for node_id, node in sorted(nodes.items()):
        parent_id = node.get("parent_id")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                issues.append(f"{context} node {node_id} has invalid parent_id: {parent_id!r}")
            elif parent_id not in nodes:
                issues.append(f"{context} node {node_id} parent_id references missing node: {parent_id}")

        if "children" not in node:
            issues.append(f"{context} node {node_id} children is missing")
            children = []
        else:
            children = node.get("children")
        if not isinstance(children, list):
            issues.append(f"{context} node {node_id} children must be a list")
            continue
        seen_children: set[str] = set()
        for child_id in children:
            if not isinstance(child_id, str) or not child_id:
                issues.append(f"{context} node {node_id} children has invalid node id: {child_id!r}")
                continue
            if child_id in seen_children:
                issues.append(f"{context} node {node_id} children contains duplicate node id: {child_id}")
                continue
            seen_children.add(child_id)
            child_node = nodes.get(child_id)
            if child_node is None:
                issues.append(f"{context} node {node_id} children references missing node: {child_id}")
                continue
            if child_node.get("parent_id") != node_id:
                issues.append(
                    f"{context} node {node_id} children includes {child_id}, "
                    f"but child parent_id is {child_node.get('parent_id')!r}"
                )

    for node_id, node in sorted(nodes.items()):
        parent_id = node.get("parent_id")
        if not isinstance(parent_id, str) or parent_id not in nodes:
            continue
        parent_children = nodes[parent_id].get("children")
        if not isinstance(parent_children, list):
            continue
        child_count = sum(1 for child_id in parent_children if child_id == node_id)
        if child_count != 1:
            issues.append(
                f"{context} node {node_id} parent children does not include node exactly once: "
                f"parent_id={parent_id}, count={child_count}"
            )

    cycle_nodes = _solution_tree_cycle_nodes(nodes)
    if cycle_nodes:
        issues.append(f"{context} parent links contain a cycle: {cycle_nodes!r}")
    return issues


def validate_solution_node_artifact_paths(
    data: dict[str, Any],
    *,
    run_dir: Path,
    context: str,
) -> list[str]:
    issues: list[str] = []
    run_root = run_dir.resolve(strict=False)
    node_id = data.get("node_id")
    workspace_value = data.get("workspace")
    if not isinstance(node_id, str) or not node_id or not isinstance(workspace_value, str):
        return issues

    workspace = _resolve_artifact_path(workspace_value, run_root)
    solutions_dir = (run_root / "solutions").resolve(strict=False)
    if not _is_relative_to(workspace, solutions_dir):
        issues.append(f"{context} workspace must be under run solutions directory")
    if workspace.name != node_id:
        issues.append(f"{context} workspace basename must match node_id")
    if not workspace.exists():
        issues.append(f"{context} workspace path does not exist")

    for field_name in ("proposal_path", "analysis_path"):
        value = data.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            continue
        artifact_path = _resolve_artifact_path(value, run_root)
        if not _is_relative_to(artifact_path, workspace):
            issues.append(f"{context} {field_name} must be inside node workspace")
        if not artifact_path.exists():
            issues.append(f"{context} {field_name} does not exist")
    return issues


def _resolve_artifact_path(value: str, run_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = run_root / path
    return path.resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _solution_tree_cycle_nodes(nodes: dict[str, dict[str, Any]]) -> list[str]:
    cycle_nodes: set[str] = set()
    for node_id in nodes:
        seen: set[str] = set()
        current_id: str | None = node_id
        while current_id is not None:
            if current_id in seen:
                cycle_nodes.update(seen)
                break
            seen.add(current_id)
            current_node = nodes.get(current_id)
            if current_node is None:
                break
            parent_id = current_node.get("parent_id")
            current_id = parent_id if isinstance(parent_id, str) and parent_id else None
    return sorted(cycle_nodes)


def _check_required_str_field(
    data: dict[str, Any],
    field_name: str,
    issues: list[str],
    context: str,
) -> None:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        issues.append(f"{context} {field_name} must be a non-empty string")


def _check_nullable_str_field(
    data: dict[str, Any],
    field_name: str,
    issues: list[str],
    context: str,
) -> None:
    value = data.get(field_name)
    if value is not None and not isinstance(value, str):
        issues.append(f"{context} {field_name} must be a string or null")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass(slots=True)
class SolutionScore:
    metric: str
    value: float
    higher_is_better: bool = False

    def better_than(self, other: "SolutionScore | None") -> bool:
        if other is None:
            return True
        if self.higher_is_better:
            return self.value > other.value
        return self.value < other.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolutionScore":
        issues = validate_solution_score_payload(data, context="SolutionScore")
        if issues:
            raise ValueError("Invalid SolutionScore payload: " + "; ".join(issues))
        return cls(
            metric=data["metric"],
            value=float(data["value"]),
            higher_is_better=data["higher_is_better"],
        )


@dataclass(slots=True)
class SolutionNode:
    node_id: str
    parent_id: str | None
    workspace: str
    score: SolutionScore | None = None
    children: list[str] = field(default_factory=list)
    status: str = "created"
    proposal_path: str | None = None
    analysis_path: str | None = None
    error: str | None = None
    benchmark_name: str | None = None
    contract_hash: str | None = None
    method_tags: list[str] = field(default_factory=list)
    failure_kind: str | None = None
    score_delta_from_parent: float | None = None
    num_debug_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "workspace": self.workspace,
            "score": self.score.to_dict() if self.score else None,
            "children": self.children,
            "status": self.status,
            "proposal_path": self.proposal_path,
            "analysis_path": self.analysis_path,
            "error": self.error,
            "benchmark_name": self.benchmark_name,
            "contract_hash": self.contract_hash,
            "method_tags": self.method_tags,
            "failure_kind": self.failure_kind,
            "score_delta_from_parent": self.score_delta_from_parent,
            "num_debug_attempts": self.num_debug_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolutionNode":
        issues = validate_solution_node_payload(data)
        if issues:
            raise ValueError("Invalid SolutionNode payload: " + "; ".join(issues))
        return cls(
            node_id=data["node_id"],
            parent_id=data.get("parent_id"),
            workspace=data["workspace"],
            score=SolutionScore.from_dict(data["score"]) if data.get("score") else None,
            children=list(data["children"]),
            status=data["status"],
            proposal_path=data.get("proposal_path"),
            analysis_path=data.get("analysis_path"),
            error=data.get("error"),
            benchmark_name=data.get("benchmark_name"),
            contract_hash=data.get("contract_hash"),
            method_tags=list(data["method_tags"]),
            failure_kind=data.get("failure_kind"),
            score_delta_from_parent=(
                float(data["score_delta_from_parent"])
                if data.get("score_delta_from_parent") is not None
                else None
            ),
            num_debug_attempts=data["num_debug_attempts"],
        )


@dataclass(slots=True)
class AgentMessage:
    role: str
    prompt: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        return cls(
            role=str(data["role"]),
            prompt=str(data["prompt"]),
            response=str(data["response"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class Proposal:
    title: str
    diagnosis: str
    mutation_plan: list[str]
    expected_effect: str
    risks: list[str]
    kb_application: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "diagnosis": self.diagnosis,
            "mutation_plan": self.mutation_plan,
            "expected_effect": self.expected_effect,
            "risks": self.risks,
            "kb_application": self.kb_application,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Proposal":
        return cls(
            title=str(data.get("title", "Untitled proposal")),
            diagnosis=str(data.get("diagnosis", "")),
            mutation_plan=[str(item) for item in data.get("mutation_plan", [])],
            expected_effect=str(data.get("expected_effect", "")),
            risks=[str(item) for item in data.get("risks", [])],
            kb_application=dict(data.get("kb_application", {}))
            if isinstance(data.get("kb_application"), dict)
            else {},
        )

    def to_markdown(self) -> str:
        plan = "\n".join(f"- {item}" for item in self.mutation_plan) or "- No mutation steps."
        risks = "\n".join(f"- {item}" for item in self.risks) or "- No known risks."
        kb_application = self._kb_application_markdown()
        return (
            f"# {self.title}\n\n"
            f"## Diagnosis\n\n{self.diagnosis}\n\n"
            f"## Mutation Plan\n\n{plan}\n\n"
            f"## KB Application\n\n{kb_application}\n\n"
            f"## Expected Effect\n\n{self.expected_effect}\n\n"
            f"## Risks\n\n{risks}\n"
        )

    def _kb_application_markdown(self) -> str:
        if not self.kb_application:
            return "- No KB application summary provided."
        lines = []
        for key, value in sorted(self.kb_application.items()):
            if isinstance(value, list):
                formatted = ", ".join(str(item) for item in value) if value else "none"
            else:
                formatted = str(value)
            lines.append(f"- {key}: {formatted}")
        return "\n".join(lines)


@dataclass(slots=True)
class AnalysisReport:
    node_id: str
    summary: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "next_steps": self.next_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisReport":
        return cls(
            node_id=str(data["node_id"]),
            summary=str(data.get("summary", "")),
            strengths=[str(item) for item in data.get("strengths", [])],
            weaknesses=[str(item) for item in data.get("weaknesses", [])],
            next_steps=[str(item) for item in data.get("next_steps", [])],
        )

    def to_markdown(self) -> str:
        strengths = "\n".join(f"- {item}" for item in self.strengths) or "- None recorded."
        weaknesses = "\n".join(f"- {item}" for item in self.weaknesses) or "- None recorded."
        next_steps = "\n".join(f"- {item}" for item in self.next_steps) or "- None recorded."
        return (
            f"# Analysis: {self.node_id}\n\n"
            f"{self.summary}\n\n"
            f"## Strengths\n\n{strengths}\n\n"
            f"## Weaknesses\n\n{weaknesses}\n\n"
            f"## Next Steps\n\n{next_steps}\n"
        )
