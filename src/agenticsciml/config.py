from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _hash_json_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hex_digest(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"BenchmarkSourceManifest artifact has invalid sha256 digest: {name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"BenchmarkSourceManifest artifact has non-hex digest: {name}") from exc


def _manifest_artifact_name(path: str) -> str:
    normalized = path.replace("\\", "/")
    if "{solution_id}" in normalized:
        normalized = normalized.replace("{solution_id}", "solution")
    return Path(normalized).name


def _validate_benchmark_source_manifest(
    manifest: dict[str, Any],
    *,
    allowed_train_files: list[str],
    evaluator_only_files: list[str],
) -> None:
    required_keys = {
        "schema_version",
        "digest_algorithm",
        "artifacts",
        "data_source_mode",
        "data_generated",
        "data_seed",
        "generator_command",
    }
    missing = sorted(required_keys - set(manifest))
    if missing:
        raise ValueError("BenchmarkSourceManifest missing required field(s): " + ", ".join(missing))
    if manifest.get("schema_version") != 1:
        raise ValueError("BenchmarkSourceManifest schema_version must be 1")
    if manifest.get("digest_algorithm") != "sha256":
        raise ValueError("BenchmarkSourceManifest digest_algorithm must be sha256")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("BenchmarkSourceManifest artifacts must be a non-empty object")
    for name, digest in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("BenchmarkSourceManifest artifact names must be non-empty strings")
        if Path(name).is_absolute() or name == ".." or name.startswith("../") or "/../" in name:
            raise ValueError(f"BenchmarkSourceManifest artifact path is not allowed: {name}")
        _validate_hex_digest(name, digest)

    required_artifacts = {
        "Problem.md",
        "Requirements.md",
        "Evaluation.md",
        "Data_config.json",
        "evaluate.py",
        "generate_data.py",
        "guidelines.md",
    }
    required_artifacts.update(_manifest_artifact_name(path) for path in allowed_train_files)
    required_artifacts.update(
        _manifest_artifact_name(path)
        for path in evaluator_only_files
        if path.endswith(".npz") or path.endswith(".py")
    )
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    if missing_artifacts:
        raise ValueError("BenchmarkSourceManifest missing artifact digest(s): " + ", ".join(missing_artifacts))

    data_source_mode = manifest.get("data_source_mode")
    if data_source_mode not in {"repo_existing", "generated_seed0"}:
        raise ValueError("BenchmarkSourceManifest data_source_mode is invalid")
    expected_generated = data_source_mode == "generated_seed0"
    if manifest.get("data_generated") is not expected_generated:
        raise ValueError("BenchmarkSourceManifest data_generated does not match data_source_mode")
    data_seed = manifest.get("data_seed")
    if not isinstance(data_seed, int) or isinstance(data_seed, bool):
        raise ValueError("BenchmarkSourceManifest data_seed must be an integer")

    generator_command = manifest.get("generator_command")
    if not isinstance(generator_command, list) or not all(isinstance(item, str) for item in generator_command):
        raise ValueError("BenchmarkSourceManifest generator_command must be a list of strings")
    expected_command = ["python", "generate_data.py", "--seed", "0", "--output-dir", "<tmp>"]
    if data_source_mode == "generated_seed0" and generator_command != expected_command:
        raise ValueError("BenchmarkSourceManifest generated_seed0 command is invalid")
    if data_source_mode == "repo_existing" and generator_command:
        raise ValueError("BenchmarkSourceManifest repo_existing mode must not include generator_command")


def _validate_benchmark_fidelity(metadata: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "paper_task_name",
        "paper_section",
        "fidelity_level",
        "expected_runtime_s",
        "requires_torch",
        "requires_gpu",
        "paper_gap_notes",
    }
    unknown = sorted(set(metadata) - required)
    if unknown:
        raise ValueError("Benchmark fidelity metadata unknown field(s): " + ", ".join(unknown))
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError("Benchmark fidelity metadata missing required field(s): " + ", ".join(missing))
    if metadata.get("schema_version") != 1:
        raise ValueError("Benchmark fidelity metadata schema_version must be 1")
    for key in ("paper_task_name", "paper_section", "paper_gap_notes"):
        if not isinstance(metadata.get(key), str):
            raise ValueError(f"Benchmark fidelity metadata {key} must be a string")
        if not metadata[key].strip():
            raise ValueError(f"Benchmark fidelity metadata {key} is required")
    if metadata.get("fidelity_level") not in {"proxy", "faithful-small", "paper-like"}:
        raise ValueError("Benchmark fidelity metadata fidelity_level is invalid")
    expected_runtime_s = metadata.get("expected_runtime_s")
    if not isinstance(expected_runtime_s, int) or isinstance(expected_runtime_s, bool) or expected_runtime_s <= 0:
        raise ValueError("Benchmark fidelity metadata expected_runtime_s must be a positive integer")
    for key in ("requires_torch", "requires_gpu"):
        if not isinstance(metadata.get(key), bool):
            raise ValueError(f"Benchmark fidelity metadata {key} must be a boolean")
    if metadata.get("fidelity_level") == "proxy" and not metadata["paper_gap_notes"].strip():
        raise ValueError("Benchmark fidelity metadata paper_gap_notes is required for proxy benchmarks")


@dataclass(slots=True)
class AgentConfig:
    role: str
    model: str = "mock"
    temperature: float = 0.0
    reasoning_effort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        return cls(
            role=str(data["role"]),
            model=str(data.get("model", "mock")),
            temperature=float(data.get("temperature", 0.0)),
            reasoning_effort=(
                str(data["reasoning_effort"])
                if data.get("reasoning_effort") is not None
                else None
            ),
        )


DEFAULT_AGENT_ROLE_MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "data_analyst": {
        "temperature": 0.35,
        "reasoning_effort": "high",
        "rationale": "Analysis needs synthesis across benchmark files while staying evidence-bound.",
    },
    "evaluator": {
        "temperature": 0.0,
        "reasoning_effort": "high",
        "rationale": "Evaluation contracts must be deterministic, but contract validation is high-stakes.",
    },
    "root_engineer": {
        "temperature": 0.1,
        "reasoning_effort": "xhigh",
        "rationale": "Root solution generation should be stable and deeply reasoned.",
    },
    "retriever": {
        "temperature": 0.0,
        "reasoning_effort": "medium",
        "rationale": "Retrieval selection should be deterministic and comparatively lightweight.",
    },
    "proposer": {
        "temperature": 0.55,
        "reasoning_effort": "xhigh",
        "rationale": "Proposal generation is the main creative search step and benefits from deeper reasoning.",
    },
    "critic": {
        "temperature": 0.35,
        "reasoning_effort": "high",
        "rationale": "Critique needs alternative hypotheses without drifting away from constraints.",
    },
    "engineer": {
        "temperature": 0.1,
        "reasoning_effort": "xhigh",
        "rationale": "Patch generation must be reproducible and carefully reasoned.",
    },
    "debugger": {
        "temperature": 0.05,
        "reasoning_effort": "xhigh",
        "rationale": "Repair work should be conservative and inspect failure evidence deeply.",
    },
    "result_analyst": {
        "temperature": 0.3,
        "reasoning_effort": "high",
        "rationale": "Result summaries need interpretation while preserving evidence boundaries.",
    },
    "selector": {
        "temperature": 0.05,
        "reasoning_effort": "high",
        "rationale": "Parent selection should be stable but still reason over tradeoffs.",
    },
}


def agent_role_default_model_settings(role: str) -> dict[str, Any]:
    settings = DEFAULT_AGENT_ROLE_MODEL_SETTINGS.get(role)
    if settings is None:
        return {"temperature": 0.0, "reasoning_effort": "medium", "rationale": "Unknown role default."}
    return dict(settings)


@dataclass(slots=True)
class EvolutionConfig:
    max_iterations: int = 1
    parallel_mutations: int = 2
    max_children_per_node: int = 10
    max_debug_retries: int = 2
    timeout_s: int = 60
    use_kb: bool = True
    random_kb: bool = False
    random_seed: int = 0
    use_critic: bool = True
    use_debugger: bool = True
    use_branch_context: bool = True
    selector_vote_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "parallel_mutations": self.parallel_mutations,
            "max_children_per_node": self.max_children_per_node,
            "max_debug_retries": self.max_debug_retries,
            "timeout_s": self.timeout_s,
            "use_kb": self.use_kb,
            "random_kb": self.random_kb,
            "random_seed": self.random_seed,
            "use_critic": self.use_critic,
            "use_debugger": self.use_debugger,
            "use_branch_context": self.use_branch_context,
            "selector_vote_count": self.selector_vote_count,
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
            random_seed=int(data.get("random_seed", 0)),
            use_critic=bool(data.get("use_critic", True)),
            use_debugger=bool(data.get("use_debugger", True)),
            use_branch_context=bool(data.get("use_branch_context", True)),
            selector_vote_count=int(data.get("selector_vote_count", 3)),
        )


@dataclass(slots=True)
class EvaluationContract:
    metric_name: str
    higher_is_better: bool
    validate_command: list[str]
    train_command: list[str]
    predict_command: list[str]
    evaluate_command: list[str]
    checkpoint_path: str = "model.pkl"
    benchmark_name: str = "function_approx"
    allowed_train_files: list[str] = field(default_factory=lambda: ["train_data.npz"])
    evaluator_only_files: list[str] = field(
        default_factory=lambda: ["private_eval/{solution_id}/evaluate.py", "private_eval/{solution_id}/val_data.npz"]
    )
    evaluator_digest: str = ""
    data_config_digest: str = ""
    problem_bundle_digest: str = ""
    benchmark_fidelity: dict[str, Any] = field(default_factory=dict)
    benchmark_source_manifest: dict[str, Any] = field(default_factory=dict)
    benchmark_source_manifest_digest: str = ""
    contract_hash: str = ""

    @classmethod
    def default_function_approx(cls) -> "EvaluationContract":
        return cls(
            metric_name="validation_mse",
            higher_is_better=False,
            validate_command=["python", "solution.py", "--mode=validate"],
            train_command=["python", "solution.py", "--mode=train"],
            predict_command=[
                "python",
                "solution.py",
                "--mode=predict",
                "--input",
                "predict_input.npz",
                "--output",
                "predictions.npz",
            ],
            evaluate_command=["python", "<private_eval>/evaluate.py", "--predictions", "predictions.npz"],
            checkpoint_path="model.pkl",
            benchmark_name="function_approx",
            benchmark_fidelity={
                "schema_version": 1,
                "paper_task_name": "Discontinuous function fitting",
                "paper_section": "S1.1",
                "fidelity_level": "proxy",
                "expected_runtime_s": 20,
                "requires_torch": False,
                "requires_gpu": False,
                "paper_gap_notes": "Default lightweight function approximation proxy.",
            },
        ).with_computed_hash()

    def hash_payload(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "validate_command": self.validate_command,
            "train_command": self.train_command,
            "predict_command": self.predict_command,
            "evaluate_command": self.evaluate_command,
            "checkpoint_path": self.checkpoint_path,
            "benchmark_name": self.benchmark_name,
            "allowed_train_files": self.allowed_train_files,
            "evaluator_only_files": self.evaluator_only_files,
            "evaluator_digest": self.evaluator_digest,
            "data_config_digest": self.data_config_digest,
            "problem_bundle_digest": self.problem_bundle_digest,
            "benchmark_fidelity": self.benchmark_fidelity,
            "benchmark_source_manifest_digest": self.benchmark_source_manifest_digest,
        }

    def compute_hash(self) -> str:
        return _hash_json_payload(self.hash_payload())

    def with_computed_hash(self) -> "EvaluationContract":
        self.contract_hash = self.compute_hash()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "validate_command": self.validate_command,
            "train_command": self.train_command,
            "predict_command": self.predict_command,
            "evaluate_command": self.evaluate_command,
            "checkpoint_path": self.checkpoint_path,
            "benchmark_name": self.benchmark_name,
            "allowed_train_files": self.allowed_train_files,
            "evaluator_only_files": self.evaluator_only_files,
            "evaluator_digest": self.evaluator_digest,
            "data_config_digest": self.data_config_digest,
            "problem_bundle_digest": self.problem_bundle_digest,
            "benchmark_fidelity": self.benchmark_fidelity,
            "benchmark_source_manifest": self.benchmark_source_manifest,
            "benchmark_source_manifest_digest": self.benchmark_source_manifest_digest,
            "contract_hash": self.contract_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationContract":
        contract = cls(
            metric_name=str(data["metric_name"]),
            higher_is_better=bool(data["higher_is_better"]),
            validate_command=list(data["validate_command"]),
            train_command=list(data["train_command"]),
            predict_command=list(
                data.get(
                    "predict_command",
                    [
                        "python",
                        "solution.py",
                        "--mode=predict",
                        "--input",
                        "predict_input.npz",
                        "--output",
                        "predictions.npz",
                    ],
                )
            ),
            evaluate_command=list(data["evaluate_command"]),
            checkpoint_path=str(data.get("checkpoint_path", "model.pkl")),
            benchmark_name=str(data.get("benchmark_name", "function_approx")),
            allowed_train_files=list(data.get("allowed_train_files", ["train_data.npz"])),
            evaluator_only_files=list(
                data.get(
                    "evaluator_only_files",
                    ["private_eval/{solution_id}/evaluate.py", "private_eval/{solution_id}/val_data.npz"],
                )
            ),
            evaluator_digest=str(data.get("evaluator_digest", "")),
            data_config_digest=str(data.get("data_config_digest", "")),
            problem_bundle_digest=str(data.get("problem_bundle_digest", "")),
            benchmark_fidelity=dict(data.get("benchmark_fidelity", {})),
            benchmark_source_manifest=dict(data.get("benchmark_source_manifest", {})),
            benchmark_source_manifest_digest=str(data.get("benchmark_source_manifest_digest", "")),
            contract_hash=str(data.get("contract_hash", "")),
        )
        if not contract.benchmark_fidelity:
            raise ValueError("EvaluationContract benchmark_fidelity missing")
        _validate_benchmark_fidelity(contract.benchmark_fidelity)
        if contract.benchmark_source_manifest_digest:
            if not contract.benchmark_source_manifest:
                raise ValueError("EvaluationContract benchmark source manifest missing")
            _validate_benchmark_source_manifest(
                contract.benchmark_source_manifest,
                allowed_train_files=contract.allowed_train_files,
                evaluator_only_files=contract.evaluator_only_files,
            )
            computed_manifest_digest = _hash_json_payload(contract.benchmark_source_manifest)
            if computed_manifest_digest != contract.benchmark_source_manifest_digest:
                raise ValueError(
                    "EvaluationContract benchmark source manifest hash mismatch: "
                    f"stored {contract.benchmark_source_manifest_digest}, "
                    f"computed {computed_manifest_digest}"
                )
        elif contract.benchmark_source_manifest:
            raise ValueError("EvaluationContract benchmark source manifest digest missing")
        if contract.contract_hash:
            computed = contract.compute_hash()
            if contract.contract_hash != computed:
                raise ValueError(
                    "EvaluationContract hash mismatch: "
                    f"stored {contract.contract_hash}, computed {computed}"
                )
            return contract
        return contract.with_computed_hash()


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
    selector_panel: list[AgentConfig] = field(default_factory=list)
    strategy_seed_ids: list[str] = field(default_factory=list)
    problem_intake: dict[str, Any] = field(default_factory=dict)
    planner_snapshot: dict[str, Any] = field(default_factory=dict)
    readiness_report: dict[str, Any] = field(default_factory=dict)
    claim_level: str = "workflow_proxy"
    domain_evaluator_approved: bool = False
    domain_reviewer: str | None = None
    domain_review_notes: str | None = None
    paper_benchmark_approved: bool = False
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
            "selector_panel": [cfg.to_dict() for cfg in self.selector_panel],
            "strategy_seed_ids": list(self.strategy_seed_ids),
            "problem_intake": dict(self.problem_intake),
            "planner_snapshot": dict(self.planner_snapshot),
            "readiness_report": dict(self.readiness_report),
            "claim_level": self.claim_level,
            "domain_evaluator_approved": self.domain_evaluator_approved,
            "domain_reviewer": self.domain_reviewer,
            "domain_review_notes": self.domain_review_notes,
            "paper_benchmark_approved": self.paper_benchmark_approved,
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
            selector_panel=[
                AgentConfig.from_dict(agent_data)
                for agent_data in data.get("selector_panel", [])
            ],
            strategy_seed_ids=[str(item) for item in data.get("strategy_seed_ids", [])],
            problem_intake=(
                dict(data["problem_intake"])
                if isinstance(data.get("problem_intake"), dict)
                else {}
            ),
            planner_snapshot=(
                dict(data["planner_snapshot"])
                if isinstance(data.get("planner_snapshot"), dict)
                else {}
            ),
            readiness_report=(
                dict(data["readiness_report"])
                if isinstance(data.get("readiness_report"), dict)
                else {}
            ),
            claim_level=str(data.get("claim_level", "workflow_proxy")),
            domain_evaluator_approved=bool(data.get("domain_evaluator_approved", False)),
            domain_reviewer=(
                str(data["domain_reviewer"])
                if data.get("domain_reviewer") is not None
                else None
            ),
            domain_review_notes=(
                str(data["domain_review_notes"])
                if data.get("domain_review_notes") is not None
                else None
            ),
            paper_benchmark_approved=bool(data.get("paper_benchmark_approved", False)),
            auto_approve_evaluation=bool(data.get("auto_approve_evaluation", True)),
            resume=bool(data.get("resume", False)),
        )
