import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from agenticsciml.benchmarks import BENCHMARKS, BenchmarkContractFactory, BenchmarkSpec, ProblemBundle
from agenticsciml.config import DataConfig, EvaluationContract, EvolutionConfig, ExperimentConfig
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
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    if args.mode == "validate":
        model = MODEL()
        model.mean = np.zeros((1, 1))
        assert model.predict(np.zeros((2, 1))).shape[0] == 2
        return
    if args.mode == "predict":
        with open(MODEL_CHECKPOINT, "rb") as f:
            model = pickle.load(f)
        data = np.load(args.input)
        np.savez(args.output, predictions=model.predict(data["x_val"]))
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


def _problem_bundle_from_dir(path: Path, spec: BenchmarkSpec) -> ProblemBundle:
    data_config_path = path / "Data_config.json"
    return ProblemBundle(
        benchmark_name=spec.name,
        benchmark_spec=spec,
        problem_md=(path / "Problem.md").read_text(encoding="utf-8"),
        requirements_md=(path / "Requirements.md").read_text(encoding="utf-8"),
        evaluation_md=(path / "Evaluation.md").read_text(encoding="utf-8"),
        data_config=DataConfig.from_dict(json.loads(data_config_path.read_text(encoding="utf-8"))),
        benchmark_dir=path,
    )


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


def test_problem_bundle_and_contract_are_benchmark_aware() -> None:
    hashes = set()
    for name, spec in BENCHMARKS.items():
        bundle = ProblemBundle.load(spec.path)
        contract = BenchmarkContractFactory.create_contract(bundle)

        assert bundle.benchmark_name == name
        assert contract.benchmark_name == name
        assert contract.metric_name == spec.metric
        assert contract.contract_hash
        assert contract.evaluator_digest
        assert contract.data_config_digest
        assert contract.problem_bundle_digest
        assert "train_data.npz" in contract.allowed_train_files
        assert any(path.endswith("val_data.npz") for path in contract.evaluator_only_files)
        assert contract.contract_hash == BenchmarkContractFactory.create_contract(bundle).contract_hash
        hashes.add(contract.contract_hash)

    assert len(hashes) == len(BENCHMARKS)


def test_contract_from_dict_rejects_hash_mismatch() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["metric_name"] = "tampered_metric"

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Expected tampered EvaluationContract hash to fail.")


def test_contract_verify_detects_stale_evaluator_source(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    bundle = _problem_bundle_from_dir(benchmark_dir, spec)
    contract = BenchmarkContractFactory.create_contract(bundle)

    evaluate_path = benchmark_dir / "evaluate.py"
    evaluate_path.write_text(
        evaluate_path.read_text(encoding="utf-8") + "\n# stale contract detector\n",
        encoding="utf-8",
    )
    changed_bundle = _problem_bundle_from_dir(benchmark_dir, spec)

    try:
        BenchmarkContractFactory.verify_contract(changed_bundle, contract)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("Expected stale contract verification to fail.")


def test_unknown_benchmark_bundle_fails_clearly(tmp_path: Path) -> None:
    for filename in ["Problem.md", "Requirements.md", "Evaluation.md", "Data_config.json"]:
        (tmp_path / filename).write_text("{}", encoding="utf-8")

    try:
        ProblemBundle.load(tmp_path)
    except ValueError as exc:
        assert "Unknown benchmark" in str(exc)
    else:
        raise AssertionError("Expected unknown benchmark to fail.")


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
            BenchmarkContractFactory.create_contract(ProblemBundle.load(spec.path)),
            timeout_s=20,
        )

        assert result.exit_code == 0, f"{name}: {result.stderr}"
        eval_data = json.loads((workspace / "eval.json").read_text(encoding="utf-8"))
        assert eval_data["metric"] == spec.metric
        assert isinstance(eval_data["score"], float)


def test_cli_lists_benchmarks(cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agenticsciml.cli", "benchmarks", "--json"],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
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
        tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
        assert tree["nodes"][0]["benchmark_name"] == name
        assert tree["nodes"][0]["contract_hash"]


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
