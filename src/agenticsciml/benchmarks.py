from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from agenticsciml.config import DataConfig, EvaluationContract


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    path: Path
    paper_section: str
    family: str
    metric: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": str(self.path),
            "paper_section": self.paper_section,
            "family": self.family,
            "metric": self.metric,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ProblemBundle:
    benchmark_name: str
    benchmark_spec: BenchmarkSpec
    problem_md: str
    requirements_md: str
    evaluation_md: str
    data_config: DataConfig
    benchmark_dir: Path

    @classmethod
    def load(cls, benchmark_dir: Path) -> "ProblemBundle":
        path = benchmark_dir.resolve()
        spec = benchmark_for_path(path)
        if spec is None:
            raise ValueError(f"Unknown benchmark: {benchmark_dir}")
        data_config_path = path / "Data_config.json"
        return cls(
            benchmark_name=spec.name,
            benchmark_spec=spec,
            problem_md=(path / "Problem.md").read_text(encoding="utf-8"),
            requirements_md=(path / "Requirements.md").read_text(encoding="utf-8"),
            evaluation_md=(path / "Evaluation.md").read_text(encoding="utf-8"),
            data_config=DataConfig.from_dict(json.loads(data_config_path.read_text(encoding="utf-8"))),
            benchmark_dir=path,
        )

    def summary(self) -> str:
        return (
            f"benchmark_name: {self.benchmark_name}\n"
            f"family: {self.benchmark_spec.family}\n"
            f"metric: {self.benchmark_spec.metric}\n"
            f"description: {self.benchmark_spec.description}\n"
        )


class BenchmarkContractFactory:
    @staticmethod
    def create_contract(problem_bundle: ProblemBundle) -> EvaluationContract:
        data_config = problem_bundle.data_config
        train_path = data_config.train_path or "train_data.npz"
        validation_path = data_config.validation_path or "val_data.npz"
        contract = EvaluationContract(
            metric_name=problem_bundle.benchmark_spec.metric,
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
            benchmark_name=problem_bundle.benchmark_name,
            allowed_train_files=[
                "Problem.md",
                "Requirements.md",
                "Evaluation.md",
                "Data_config.json",
                "generate_data.py",
                "guidelines.md",
                train_path,
            ],
            evaluator_only_files=[
                "private_eval/{solution_id}/evaluate.py",
                f"private_eval/{{solution_id}}/{validation_path}",
            ],
            evaluator_digest=_file_digest(problem_bundle.benchmark_dir / "evaluate.py"),
            data_config_digest=_file_digest(problem_bundle.benchmark_dir / "Data_config.json"),
            problem_bundle_digest=_problem_bundle_digest(problem_bundle),
        )
        return contract.with_computed_hash()

    @staticmethod
    def create_guidelines(problem_bundle: ProblemBundle, contract: EvaluationContract) -> str:
        return (
            "# Evaluation Contract\n\n"
            f"- benchmark: {contract.benchmark_name}\n"
            f"- metric: {contract.metric_name}\n"
            f"- higher_is_better: {contract.higher_is_better}\n"
            f"- checkpoint: {contract.checkpoint_path}\n"
            f"- contract_hash: `{contract.contract_hash}`\n"
            f"- evaluator_digest: `{contract.evaluator_digest}`\n"
            f"- data_config_digest: `{contract.data_config_digest}`\n"
            f"- problem_bundle_digest: `{contract.problem_bundle_digest}`\n"
            f"- predict_command: `{' '.join(contract.predict_command)}`\n"
            f"- allowed_train_files: {', '.join(contract.allowed_train_files)}\n"
            f"- evaluator_only_files: {', '.join(contract.evaluator_only_files)}\n\n"
            "## Prediction-Only Evaluation\n\n"
            "- `validate` and `train` must not read validation data.\n"
            "- `predict` receives only public validation features in `predict_input.npz`.\n"
            "- `predict` must write `predictions.npz` with a `predictions` array.\n"
            "- `evaluate.py` computes the metric from private labels and must not import `solution.py`.\n\n"
            "## Benchmark Summary\n\n"
            f"{problem_bundle.summary()}"
        )

    @staticmethod
    def verify_contract(problem_bundle: ProblemBundle, contract: EvaluationContract) -> None:
        fresh_bundle = ProblemBundle.load(problem_bundle.benchmark_dir)
        expected = BenchmarkContractFactory.create_contract(fresh_bundle)
        if contract.contract_hash != expected.contract_hash:
            raise ValueError(
                "EvaluationContract is stale for current benchmark files: "
                f"stored {contract.contract_hash}, expected {expected.contract_hash}"
            )


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "function_approx": BenchmarkSpec(
        name="function_approx",
        path=EXAMPLES_DIR / "function_approx",
        paper_section="S1.1",
        family="function approximation",
        metric="validation_mse",
        description="Discontinuous oscillatory one-dimensional function approximation.",
    ),
    "poisson_lshape": BenchmarkSpec(
        name="poisson_lshape",
        path=EXAMPLES_DIR / "poisson_lshape",
        paper_section="S1.2",
        family="PINN",
        metric="relative_l2",
        description="Poisson equation surrogate on an L-shaped domain.",
    ),
    "burgers_pinn": BenchmarkSpec(
        name="burgers_pinn",
        path=EXAMPLES_DIR / "burgers_pinn",
        paper_section="S1.3",
        family="PINN",
        metric="relative_l2",
        description="Time-dependent viscous Burgers equation surrogate.",
    ),
    "antiderivative_operator": BenchmarkSpec(
        name="antiderivative_operator",
        path=EXAMPLES_DIR / "antiderivative_operator",
        paper_section="S1.4",
        family="operator learning",
        metric="relative_l2",
        description="Operator learning from input functions to antiderivatives.",
    ),
    "reaction_diffusion_operator": BenchmarkSpec(
        name="reaction_diffusion_operator",
        path=EXAMPLES_DIR / "reaction_diffusion_operator",
        paper_section="S1.5",
        family="operator learning",
        metric="relative_l2",
        description="Multiple-input reaction-diffusion operator surrogate.",
    ),
    "cylinder_wake_reconstruction": BenchmarkSpec(
        name="cylinder_wake_reconstruction",
        path=EXAMPLES_DIR / "cylinder_wake_reconstruction",
        paper_section="S1.6",
        family="inverse reconstruction",
        metric="relative_l2",
        description="2D cylinder wake vorticity reconstruction from sparse noisy sensors.",
    ),
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return list(BENCHMARKS.values())


def benchmark_for_path(path: Path) -> BenchmarkSpec | None:
    resolved = path.resolve()
    for spec in BENCHMARKS.values():
        if resolved == spec.path.resolve():
            return spec
    return BENCHMARKS.get(path.name)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _problem_bundle_digest(problem_bundle: ProblemBundle) -> str:
    spec_payload = problem_bundle.benchmark_spec.to_dict()
    spec_payload.pop("path", None)
    payload = {
        "benchmark_spec": spec_payload,
        "Problem.md": _text_digest(problem_bundle.problem_md),
        "Requirements.md": _text_digest(problem_bundle.requirements_md),
        "Evaluation.md": _text_digest(problem_bundle.evaluation_md),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
