from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
SOLUTION_NODE_STATUSES = {"created", "evaluated", "failed"}


def validate_solution_score_payload(score: Any, *, context: str) -> list[str]:
    issues: list[str] = []
    if score is None:
        return issues
    if not isinstance(score, dict):
        return [f"{context} score must be an object or null"]
    metric = score.get("metric")
    if not isinstance(metric, str) or not metric:
        issues.append(f"{context} score.metric must be a non-empty string")
    value = score.get("value")
    if not isinstance(value, int | float) or isinstance(value, bool):
        issues.append(f"{context} score.value must be a number")
    higher_is_better = score.get("higher_is_better")
    if not isinstance(higher_is_better, bool):
        issues.append(f"{context} score.higher_is_better must be a boolean")
    return issues


def validate_solution_node_payload(data: Any, *, context: str = "SolutionNode") -> list[str]:
    if not isinstance(data, dict):
        return [f"{context} must be an object"]

    issues: list[str] = []
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
    if score_delta is not None and (
        not isinstance(score_delta, int | float) or isinstance(score_delta, bool)
    ):
        issues.append(f"{context} score_delta_from_parent must be a number or null")

    issues.extend(validate_solution_score_payload(data.get("score"), context=context))
    return issues


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "diagnosis": self.diagnosis,
            "mutation_plan": self.mutation_plan,
            "expected_effect": self.expected_effect,
            "risks": self.risks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Proposal":
        return cls(
            title=str(data.get("title", "Untitled proposal")),
            diagnosis=str(data.get("diagnosis", "")),
            mutation_plan=[str(item) for item in data.get("mutation_plan", [])],
            expected_effect=str(data.get("expected_effect", "")),
            risks=[str(item) for item in data.get("risks", [])],
        )

    def to_markdown(self) -> str:
        plan = "\n".join(f"- {item}" for item in self.mutation_plan) or "- No mutation steps."
        risks = "\n".join(f"- {item}" for item in self.risks) or "- No known risks."
        return (
            f"# {self.title}\n\n"
            f"## Diagnosis\n\n{self.diagnosis}\n\n"
            f"## Mutation Plan\n\n{plan}\n\n"
            f"## Expected Effect\n\n{self.expected_effect}\n\n"
            f"## Risks\n\n{risks}\n"
        )


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
