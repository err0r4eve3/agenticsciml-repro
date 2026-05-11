from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticsciml.config import DataConfig, EvaluationContract
from agenticsciml.execution.runner import run_command


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


@dataclass(frozen=True, slots=True)
class BenchmarkSourceManifest:
    artifacts: dict[str, str]
    data_source_mode: str
    data_seed: int = 0
    generator_command: list[str] | None = None
    schema_version: int = 1
    digest_algorithm: str = "sha256"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digest_algorithm": self.digest_algorithm,
            "artifacts": dict(sorted(self.artifacts.items())),
            "data_source_mode": self.data_source_mode,
            "data_generated": self.data_source_mode == "generated_seed0",
            "data_seed": self.data_seed,
            "generator_command": list(self.generator_command or []),
        }

    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class BenchmarkContractFactory:
    @staticmethod
    def create_contract(problem_bundle: ProblemBundle) -> EvaluationContract:
        data_config = problem_bundle.data_config
        train_path = data_config.train_path or "train_data.npz"
        validation_path = data_config.validation_path or "val_data.npz"
        manifest = _benchmark_source_manifest(problem_bundle, train_path, validation_path)
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
            benchmark_source_manifest=manifest.to_dict(),
            benchmark_source_manifest_digest=manifest.digest(),
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
            f"- benchmark_source_manifest_digest: `{contract.benchmark_source_manifest_digest}`\n"
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


_GENERATED_DATA_DIGEST_CACHE: dict[tuple[str, str, str, str, str], dict[str, str]] = {}


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


def _benchmark_source_manifest(
    problem_bundle: ProblemBundle,
    train_path: str,
    validation_path: str,
) -> BenchmarkSourceManifest:
    benchmark_dir = problem_bundle.benchmark_dir
    artifacts = {
        "Problem.md": _text_digest(problem_bundle.problem_md),
        "Requirements.md": _text_digest(problem_bundle.requirements_md),
        "Evaluation.md": _text_digest(problem_bundle.evaluation_md),
        "Data_config.json": _file_digest(benchmark_dir / "Data_config.json"),
        "evaluate.py": _file_digest(benchmark_dir / "evaluate.py"),
        "generate_data.py": _file_digest(benchmark_dir / "generate_data.py"),
        "guidelines.md": _file_digest(benchmark_dir / "guidelines.md"),
    }

    train_file = benchmark_dir / train_path
    validation_file = benchmark_dir / validation_path
    if train_file.exists() and validation_file.exists():
        artifacts[train_path] = _file_digest(train_file)
        artifacts[validation_path] = _file_digest(validation_file)
        return BenchmarkSourceManifest(
            artifacts=artifacts,
            data_source_mode="repo_existing",
            generator_command=[],
        )
    if train_file.exists() != validation_file.exists():
        existing = train_path if train_file.exists() else validation_path
        missing = validation_path if train_file.exists() else train_path
        raise ValueError(
            "Partial benchmark data artifacts are not allowed: "
            f"found {existing}, missing {missing}."
        )

    generated_data = _generated_data_digests(problem_bundle, train_path, validation_path)
    artifacts.update(generated_data)
    return BenchmarkSourceManifest(
        artifacts=artifacts,
        data_source_mode="generated_seed0",
        generator_command=["python", "generate_data.py", "--seed", "0", "--output-dir", "<tmp>"],
    )


def _generated_data_digests(
    problem_bundle: ProblemBundle,
    train_path: str,
    validation_path: str,
) -> dict[str, str]:
    benchmark_dir = problem_bundle.benchmark_dir
    generate_digest = _file_digest(benchmark_dir / "generate_data.py")
    data_config_digest = _file_digest(benchmark_dir / "Data_config.json")
    cache_key = (
        str(benchmark_dir.resolve()),
        generate_digest,
        data_config_digest,
        train_path,
        validation_path,
    )
    if cache_key in _GENERATED_DATA_DIGEST_CACHE:
        return dict(_GENERATED_DATA_DIGEST_CACHE[cache_key])

    with tempfile.TemporaryDirectory(prefix="agenticsciml-manifest-") as tmp:
        tmp_path = Path(tmp)
        result = run_command(
            tmp_path,
            [
                sys.executable,
                str(benchmark_dir / "generate_data.py"),
                "--seed",
                "0",
                "--output-dir",
                str(tmp_path),
            ],
            timeout_s=20,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                "Failed to generate deterministic benchmark data for contract manifest: "
                f"{result.stderr or result.stdout}"
            )
        train_file = tmp_path / train_path
        validation_file = tmp_path / validation_path
        if not train_file.exists() or not validation_file.exists():
            raise RuntimeError(
                "Benchmark data generator did not produce expected files: "
                f"{train_path}, {validation_path}"
            )
        digests = {
            train_path: _file_digest(train_file),
            validation_path: _file_digest(validation_file),
        }
        _GENERATED_DATA_DIGEST_CACHE[cache_key] = digests
        return dict(digests)
