import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from agenticsciml.benchmarks import BENCHMARKS
from agenticsciml.config import EvaluationContract, EvolutionConfig, ExperimentConfig
from agenticsciml.execution.sandbox import prepare_solution_workspace, train_and_evaluate
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


EXPECTED_BENCHMARKS = {
    "function_approx",
    "poisson_lshape",
    "burgers_pinn",
    "antiderivative_operator",
    "reaction_diffusion_operator",
    "cylinder_wake_reconstruction",
}


GENERIC_BASELINE = '''
import argparse
import pickle
import numpy as np

MODEL_CHECKPOINT = "model.pkl"

class MODEL:
    def __init__(self):
        self.mean = None

    def fit(self, x, y):
        self.mean = np.mean(y, axis=0, keepdims=True)

    def predict(self, x):
        if self.mean is None:
            raise RuntimeError("Model has not been trained.")
        return np.repeat(self.mean, len(x), axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train"], required=True)
    args = parser.parse_args()
    if args.mode == "validate":
        model = MODEL()
        model.mean = np.zeros((1, 1))
        assert model.predict(np.zeros((2, 1))).shape[0] == 2
        return
    data = np.load("train_data.npz")
    model = MODEL()
    model.fit(data["x_train"], data["u_train"])
    with open(MODEL_CHECKPOINT, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
'''


def _load_generate_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.parent.name}_generate_data", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_catalog_lists_all_paper_benchmarks() -> None:
    assert set(BENCHMARKS) == EXPECTED_BENCHMARKS
    for name, spec in BENCHMARKS.items():
        assert spec.path.exists(), name
        assert spec.metric
        assert spec.paper_section.startswith("S1.")


def test_all_benchmarks_have_required_artifacts() -> None:
    for name, spec in BENCHMARKS.items():
        required = [
            "Problem.md",
            "Requirements.md",
            "Evaluation.md",
            "Data_config.json",
            "generate_data.py",
            "evaluate.py",
            "guidelines.md",
        ]
        for filename in required:
            assert (spec.path / filename).exists(), f"{name} missing {filename}"
        data_config = json.loads((spec.path / "Data_config.json").read_text(encoding="utf-8"))
        assert data_config["train_path"] == "train_data.npz"
        assert data_config["validation_path"] == "val_data.npz"


def test_all_benchmark_data_generators_are_deterministic(tmp_path: Path) -> None:
    for name, spec in BENCHMARKS.items():
        out_a = tmp_path / name / "a"
        out_b = tmp_path / name / "b"
        module = _load_generate_module(spec.path / "generate_data.py")

        module.generate(seed=7, output_dir=out_a)
        module.generate(seed=7, output_dir=out_b)

        train_a = np.load(out_a / "train_data.npz")
        train_b = np.load(out_b / "train_data.npz")
        val_a = np.load(out_a / "val_data.npz")

        assert train_a["x_train"].ndim == 2
        assert train_a["u_train"].ndim == 2
        assert val_a["x_val"].ndim == 2
        assert val_a["u_val"].ndim == 2
        assert train_a["x_train"].shape[0] == train_a["u_train"].shape[0]
        assert val_a["x_val"].shape[0] == val_a["u_val"].shape[0]
        np.testing.assert_allclose(train_a["x_train"], train_b["x_train"])
        np.testing.assert_allclose(train_a["u_train"], train_b["u_train"])


def test_all_benchmark_evaluators_accept_generic_baseline(tmp_path: Path) -> None:
    for name, spec in BENCHMARKS.items():
        workspace = tmp_path / name
        prepare_solution_workspace(spec.path, workspace)
        (workspace / "solution.py").write_text(GENERIC_BASELINE, encoding="utf-8")

        result = train_and_evaluate(
            workspace,
            EvaluationContract.default_function_approx(),
            timeout_s=20,
        )

        assert result.exit_code == 0, f"{name}: {result.stderr}"
        eval_data = json.loads((workspace / "eval.json").read_text(encoding="utf-8"))
        assert eval_data["metric"] == spec.metric
        assert isinstance(eval_data["score"], float)


def test_cli_lists_benchmarks() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agenticsciml.cli", "benchmarks", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert {item["name"] for item in payload["benchmarks"]} == EXPECTED_BENCHMARKS


def test_mock_orchestrator_root_only_runs_all_benchmarks(tmp_path: Path) -> None:
    for name, spec in BENCHMARKS.items():
        config = ExperimentConfig(
            experiment_id=f"{name}-root",
            benchmark_dir=spec.path,
            output_dir=tmp_path,
            evolution=EvolutionConfig(max_iterations=0, max_debug_retries=0),
            use_mock=True,
        )

        run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

        eval_data = json.loads(
            (run_dir / "solutions" / "solution_000" / "eval.json").read_text(encoding="utf-8")
        )
        assert eval_data["metric"] == spec.metric


def test_mock_orchestrator_one_iteration_runs_all_benchmarks(tmp_path: Path) -> None:
    for name, spec in BENCHMARKS.items():
        config = ExperimentConfig(
            experiment_id=f"{name}-evolution",
            benchmark_dir=spec.path,
            output_dir=tmp_path,
            evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=0),
            use_mock=True,
        )

        run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
        tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))

        assert len(tree["nodes"]) >= 2
        assert (run_dir / "solutions" / "solution_001" / "eval.json").exists()
