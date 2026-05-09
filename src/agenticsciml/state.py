from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        return cls(
            metric=str(data["metric"]),
            value=float(data["value"]),
            higher_is_better=bool(data.get("higher_is_better", False)),
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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolutionNode":
        return cls(
            node_id=str(data["node_id"]),
            parent_id=data.get("parent_id"),
            workspace=str(data["workspace"]),
            score=SolutionScore.from_dict(data["score"]) if data.get("score") else None,
            children=list(data.get("children", [])),
            status=str(data.get("status", "created")),
            proposal_path=data.get("proposal_path"),
            analysis_path=data.get("analysis_path"),
            error=data.get("error"),
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
