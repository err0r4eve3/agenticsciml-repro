from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


@dataclass(slots=True)
class AgentConfig:
    role: str
    model: str = "mock"
    temperature: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "model": self.model, "temperature": self.temperature}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        return cls(
            role=str(data["role"]),
            model=str(data.get("model", "mock")),
            temperature=float(data.get("temperature", 0.0)),
        )


@dataclass(slots=True)
class EvolutionConfig:
    max_iterations: int = 1
    parallel_mutations: int = 2
    max_children_per_node: int = 10
    max_debug_retries: int = 2
    timeout_s: int = 60
    use_kb: bool = True
    random_kb: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "parallel_mutations": self.parallel_mutations,
            "max_children_per_node": self.max_children_per_node,
            "max_debug_retries": self.max_debug_retries,
            "timeout_s": self.timeout_s,
            "use_kb": self.use_kb,
            "random_kb": self.random_kb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionConfig":
        return cls(
            max_iterations=int(data.get("max_iterations", 1)),
            parallel_mutations=int(data.get("parallel_mutations", 2)),
            max_children_per_node=int(data.get("max_children_per_node", 10)),
            max_debug_retries=int(data.get("max_debug_retries", 2)),
            timeout_s=int(data.get("timeout_s", 60)),
            use_kb=bool(data.get("use_kb", True)),
            random_kb=bool(data.get("random_kb", False)),
        )


@dataclass(slots=True)
class EvaluationContract:
    metric_name: str
    higher_is_better: bool
    validate_command: list[str]
    train_command: list[str]
    evaluate_command: list[str]
    checkpoint_path: str = "model.pkl"

    @classmethod
    def default_function_approx(cls) -> "EvaluationContract":
        return cls(
            metric_name="validation_mse",
            higher_is_better=False,
            validate_command=["python", "solution.py", "--mode=validate"],
            train_command=["python", "solution.py", "--mode=train"],
            evaluate_command=["python", "evaluate.py"],
            checkpoint_path="model.pkl",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "validate_command": self.validate_command,
            "train_command": self.train_command,
            "evaluate_command": self.evaluate_command,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationContract":
        return cls(
            metric_name=str(data["metric_name"]),
            higher_is_better=bool(data["higher_is_better"]),
            validate_command=list(data["validate_command"]),
            train_command=list(data["train_command"]),
            evaluate_command=list(data["evaluate_command"]),
            checkpoint_path=str(data.get("checkpoint_path", "model.pkl")),
        )


@dataclass(slots=True)
class DataConfig:
    train_path: str | None = None
    validation_path: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        return cls(
            train_path=data.get("train_path"),
            validation_path=data.get("validation_path"),
            description=str(data.get("description", "")),
        )


@dataclass(slots=True)
class ExperimentConfig:
    experiment_id: str = "agenticsciml-run"
    benchmark_dir: Path | None = None
    output_dir: Path = Path("runs")
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    use_mock: bool = True
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    auto_approve_evaluation: bool = True
    resume: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "benchmark_dir": str(self.benchmark_dir) if self.benchmark_dir else None,
            "output_dir": str(self.output_dir),
            "evolution": self.evolution.to_dict(),
            "use_mock": self.use_mock,
            "agents": {role: cfg.to_dict() for role, cfg in self.agents.items()},
            "auto_approve_evaluation": self.auto_approve_evaluation,
            "resume": self.resume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_id=str(data.get("experiment_id", "agenticsciml-run")),
            benchmark_dir=_path_or_none(data.get("benchmark_dir")),
            output_dir=Path(data.get("output_dir", "runs")),
            evolution=EvolutionConfig.from_dict(data.get("evolution", {})),
            use_mock=bool(data.get("use_mock", True)),
            agents={
                role: AgentConfig.from_dict(agent_data)
                for role, agent_data in data.get("agents", {}).items()
            },
            auto_approve_evaluation=bool(data.get("auto_approve_evaluation", True)),
            resume=bool(data.get("resume", False)),
        )
