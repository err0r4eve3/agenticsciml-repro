import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agenticsciml.benchmarks import (
    BENCHMARKS,
    BenchmarkContractFactory,
    BenchmarkSpec,
    ProblemBundle,
    benchmark_for_path,
)
from agenticsciml.config import DataConfig, EvaluationContract, EvolutionConfig, ExperimentConfig
from agenticsciml.execution.sandbox import prepare_solution_workspace, train_and_evaluate
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


EXPECTED_BENCHMARKS = {
    "function_approx",
    "function_approx_faithful_small",
    "poisson_lshape",
    "poisson_lshape_faithful_small",
    "burgers_pinn",
    "burgers_pinn_faithful_small",
    "antiderivative_operator",
    "antiderivative_operator_faithful_small",
    "reaction_diffusion_operator",
    "reaction_diffusion_operator_faithful_small",
    "cylinder_wake_reconstruction",
    "cylinder_wake_reconstruction_faithful_small",
}

ORCHESTRATOR_SMOKE_BENCHMARKS = (
    "function_approx",
    "function_approx_faithful_small",
    "poisson_lshape",
    "poisson_lshape_faithful_small",
    "burgers_pinn",
    "burgers_pinn_faithful_small",
    "antiderivative_operator",
    "antiderivative_operator_faithful_small",
    "reaction_diffusion_operator",
    "reaction_diffusion_operator_faithful_small",
    "cylinder_wake_reconstruction",
    "cylinder_wake_reconstruction_faithful_small",
)


CONTRACT_BOUND_BENCHMARK_FIELDS = [
    "name",
    "paper_section",
    "paper_task_name",
    "family",
    "metric",
    "description",
    "fidelity_level",
    "expected_runtime_s",
    "requires_torch",
    "requires_gpu",
    "paper_gap_notes",
]


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
    assert len(ORCHESTRATOR_SMOKE_BENCHMARKS) == len(set(ORCHESTRATOR_SMOKE_BENCHMARKS))
    assert set(ORCHESTRATOR_SMOKE_BENCHMARKS) == EXPECTED_BENCHMARKS
    for name, spec in BENCHMARKS.items():
        assert spec.path.exists(), name
        assert spec.metric
        assert spec.paper_section.startswith("S1.")
        assert spec.fidelity_level in {"proxy", "faithful-small", "paper-like"}
        assert spec.expected_runtime_s > 0
        assert isinstance(spec.requires_torch, bool)
        assert isinstance(spec.requires_gpu, bool)
        assert spec.paper_task_name
        assert spec.paper_gap_notes
    assert any(spec.fidelity_level == "faithful-small" for spec in BENCHMARKS.values())


def test_benchmark_spec_rejects_invalid_fidelity_metadata() -> None:
    try:
        BenchmarkSpec(
            name="bad",
            path=Path("examples/function_approx"),
            paper_section="S1.x",
            paper_task_name="Bad task",
            family="bad",
            metric="bad_metric",
            description="bad",
            fidelity_level="paper",
            expected_runtime_s=0,
            requires_torch=False,
            requires_gpu=False,
            paper_gap_notes="",
        )
    except ValueError as exc:
        assert "fidelity_level" in str(exc)
    else:
        raise AssertionError("Expected invalid fidelity metadata to fail.")


def test_benchmark_spec_rejects_empty_paper_section() -> None:
    try:
        BenchmarkSpec(
            name="bad",
            path=Path("examples/function_approx"),
            paper_section="",
            paper_task_name="Bad task",
            family="bad",
            metric="bad_metric",
            description="bad",
            fidelity_level="proxy",
            expected_runtime_s=1,
            requires_torch=False,
            requires_gpu=False,
            paper_gap_notes="proxy gap",
        )
    except ValueError as exc:
        assert "paper_section" in str(exc)
    else:
        raise AssertionError("Expected empty paper_section to fail.")


def test_function_approx_faithful_small_matches_paper_shape_and_hides_formula_prompt(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx_faithful_small"]
    module = _load_generate_module(spec.path / "generate_data.py")
    module.generate(seed=0, output_dir=tmp_path)

    train = np.load(tmp_path / "train_data.npz")
    val = np.load(tmp_path / "val_data.npz")

    assert train["x_train"].shape == (200, 1)
    assert train["u_train"].shape == (200, 1)
    assert val["x_val"].shape == (500, 1)
    assert val["u_val"].shape == (500, 1)

    points = np.array([[-0.5], [-0.25], [0.25]])
    expected = np.array([
        [(-0.5) ** 2 - 2.0],
        [(-0.25) ** 2 - 2.0 * np.exp(-3000.0 * (0.25) ** 2)],
        [np.sin(10.0 * np.pi * 0.25) + 1.0],
    ])
    assert np.allclose(module.target_function(points), expected)

    prompt_text = "\n".join(
        (spec.path / filename).read_text(encoding="utf-8")
        for filename in ["Problem.md", "Requirements.md", "Evaluation.md", "guidelines.md"]
    )
    assert "3000" not in prompt_text
    assert "sin(10" not in prompt_text


def test_antiderivative_operator_faithful_small_has_operator_shape_and_linear_structure(tmp_path: Path) -> None:
    spec = BENCHMARKS["antiderivative_operator_faithful_small"]
    module = _load_generate_module(spec.path / "generate_data.py")
    module.generate(seed=0, output_dir=tmp_path)

    train = np.load(tmp_path / "train_data.npz")
    val = np.load(tmp_path / "val_data.npz")

    assert train["x_train"].shape == (320, 100)
    assert train["u_train"].shape == (320, 100)
    assert train["x_grid"].shape == (100,)
    assert val["x_val"].shape == (160, 100)
    assert val["u_val"].shape == (160, 100)
    assert val["length_scales"].shape == (160,)

    constant = np.ones((1, 100), dtype=float)
    integral = module.antiderivative_samples(constant, train["x_grid"])
    np.testing.assert_allclose(integral[0], train["x_grid"], atol=1e-12)

    prompt_text = "\n".join(
        (spec.path / filename).read_text(encoding="utf-8")
        for filename in ["Problem.md", "Requirements.md", "Evaluation.md", "guidelines.md"]
    )
    assert "length_scales" not in prompt_text
    assert "100" in prompt_text


def test_burgers_pinn_faithful_small_has_ic_bc_and_collocation_shape(tmp_path: Path) -> None:
    spec = BENCHMARKS["burgers_pinn_faithful_small"]
    module = _load_generate_module(spec.path / "generate_data.py")
    module.generate(seed=0, output_dir=tmp_path)

    train = np.load(tmp_path / "train_data.npz")
    val = np.load(tmp_path / "val_data.npz")

    assert train["x_train"].shape == (900, 2)
    assert train["u_train"].shape == (900, 1)
    assert train["x_initial"].shape == (160, 2)
    assert train["u_initial"].shape == (160, 1)
    assert train["x_boundary"].shape == (240, 2)
    assert train["u_boundary"].shape == (240, 1)
    assert train["x_collocation"].shape == (1200, 2)
    assert val["x_val"].shape == (3600, 2)
    assert val["u_val"].shape == (3600, 1)
    assert val["n_initial"].reshape(-1)[0] == 120
    assert val["n_boundary"].reshape(-1)[0] == 160

    xs = np.array([[-0.75, 0.0], [0.25, 0.0], [1.0, 0.5], [-1.0, 0.5]])
    values = module.target_solution(xs)
    assert values.shape == (4, 1)
    np.testing.assert_allclose(values[2], values[3], atol=1e-12)


def test_reaction_diffusion_operator_faithful_small_has_multi_input_spacetime_shape(tmp_path: Path) -> None:
    spec = BENCHMARKS["reaction_diffusion_operator_faithful_small"]
    module = _load_generate_module(spec.path / "generate_data.py")
    module.generate(seed=0, output_dir=tmp_path)

    train = np.load(tmp_path / "train_data.npz")
    val = np.load(tmp_path / "val_data.npz")

    assert train["x_train"].shape == (64, 150)
    assert train["u_train"].shape == (64, 2000)
    assert train["x_grid"].shape == (50,)
    assert train["t_grid"].shape == (40,)
    assert train["input_channel_count"].reshape(-1)[0] == 3
    assert val["x_val"].shape == (24, 150)
    assert val["u_val"].shape == (24, 2000)
    assert val["n_x"].reshape(-1)[0] == 50
    assert val["n_t"].reshape(-1)[0] == 40

    diffusion, source, initial = module.unpack_features(train["x_train"][:2])
    assert diffusion.shape == source.shape == initial.shape == (2, 50)
    assert np.all(diffusion > 0.0)
    reshaped = train["u_train"].reshape(64, 40, 50)
    np.testing.assert_allclose(reshaped[:2, 0, :], initial, atol=1e-12)
    np.testing.assert_allclose(reshaped[:, :, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(reshaped[:, :, -1], 0.0, atol=1e-12)

    prompt_text = "\n".join(
        (spec.path / filename).read_text(encoding="utf-8")
        for filename in ["Problem.md", "Requirements.md", "Evaluation.md", "guidelines.md"]
    )
    assert "50" in prompt_text
    assert "paper-score" in prompt_text


def test_cylinder_wake_reconstruction_faithful_small_has_sensor_history_shape_and_no_future_window(
    tmp_path: Path,
) -> None:
    spec = BENCHMARKS["cylinder_wake_reconstruction_faithful_small"]
    module = _load_generate_module(spec.path / "generate_data.py")
    module.generate(seed=0, output_dir=tmp_path)

    train = np.load(tmp_path / "train_data.npz")
    val = np.load(tmp_path / "val_data.npz")

    assert train["x_train"].shape == (272, 41)
    assert train["u_train"].shape == (272, 576)
    assert train["sensor_points"].shape == (8, 2)
    assert train["window_size"].reshape(-1)[0] == 5
    assert train["sensor_count"].reshape(-1)[0] == 8
    assert train["n_x"].reshape(-1)[0] == 24
    assert train["n_y"].reshape(-1)[0] == 24
    assert val["x_val"].shape == (136, 41)
    assert val["u_val"].shape == (136, 576)

    simple_sensors = np.arange(6 * 8, dtype=float).reshape(6, 8)
    simple_times = np.linspace(0.0, 1.0, 6)
    features = module.build_lagged_sensor_features(simple_sensors, simple_times, window_size=5)
    assert features.shape == (2, 41)
    np.testing.assert_allclose(features[0, :40], simple_sensors[:5].reshape(-1))
    np.testing.assert_allclose(features[1, :40], simple_sensors[1:6].reshape(-1))
    assert features[0, -1] == simple_times[4]
    assert features[1, -1] == simple_times[5]

    prompt_text = "\n".join(
        (spec.path / filename).read_text(encoding="utf-8")
        for filename in ["Problem.md", "Requirements.md", "Evaluation.md", "guidelines.md"]
    )
    assert "SHRED-style" in prompt_text
    assert "paper-score" in prompt_text


def test_cylinder_wake_faithful_small_evaluator_mean_per_sample_relative_l2() -> None:
    spec = BENCHMARKS["cylinder_wake_reconstruction_faithful_small"]
    module = _load_generate_module(spec.path / "evaluate.py")
    target = np.array([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]])

    assert module._mean_per_sample_relative_l2(target, target) == 0.0
    assert np.isclose(module._mean_per_sample_relative_l2(np.zeros_like(target), target), 1.0)


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
        assert contract.benchmark_source_manifest_digest
        assert contract.benchmark_fidelity["fidelity_level"] == spec.fidelity_level
        assert contract.benchmark_fidelity["paper_task_name"] == spec.paper_task_name
        manifest = contract.benchmark_source_manifest
        assert manifest["schema_version"] == 1
        assert manifest["digest_algorithm"] == "sha256"
        assert manifest["data_source_mode"] in {"repo_existing", "generated_seed0"}
        if manifest["data_source_mode"] == "generated_seed0":
            assert manifest["generator_command"] == [
                "python",
                "generate_data.py",
                "--seed",
                "0",
                "--output-dir",
                "<tmp>",
            ]
        assert manifest["artifacts"]["generate_data.py"]
        assert manifest["artifacts"]["guidelines.md"]
        assert manifest["artifacts"]["train_data.npz"]
        assert manifest["artifacts"]["val_data.npz"]
        assert "train_data.npz" in contract.allowed_train_files
        assert "generate_data.py" not in contract.allowed_train_files
        assert any(path.endswith("val_data.npz") for path in contract.evaluator_only_files)
        assert contract.contract_hash == BenchmarkContractFactory.create_contract(bundle).contract_hash
        hashes.add(contract.contract_hash)

    assert len(hashes) == len(BENCHMARKS)


def test_contract_hash_binds_benchmark_fidelity_metadata() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_fidelity"]["fidelity_level"] = "faithful-small"

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Expected tampered fidelity metadata to fail contract hash.")


def test_claim_boundary_wording_does_not_change_contract_hash(monkeypatch) -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    before = BenchmarkContractFactory.create_contract(bundle)

    def changed_claim_boundaries(self: BenchmarkSpec) -> dict[str, object]:
        return {
            "mock": {"scientific_claim": "changed_mock_claim_wording"},
            "real_llm": {"scientific_claim": "changed_real_claim_wording"},
            "paper_score_reproduction": "changed_wording",
            "notes": "changed public catalog wording",
        }

    monkeypatch.setattr(BenchmarkSpec, "claim_boundaries", changed_claim_boundaries)
    assert bundle.benchmark_spec.to_dict()["claim_boundaries"]["paper_score_reproduction"] == "changed_wording"

    after = BenchmarkContractFactory.create_contract(bundle)
    assert after.problem_bundle_digest == before.problem_bundle_digest
    assert after.contract_hash == before.contract_hash


def test_public_catalog_only_fields_do_not_change_contract_hash(monkeypatch) -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    before = BenchmarkContractFactory.create_contract(bundle)
    original_to_dict = BenchmarkSpec.to_dict

    def to_dict_with_display_badge(self: BenchmarkSpec) -> dict[str, object]:
        payload = original_to_dict(self)
        payload["display_badge"] = "public catalog label only"
        return payload

    monkeypatch.setattr(BenchmarkSpec, "to_dict", to_dict_with_display_badge)
    assert bundle.benchmark_spec.to_dict()["display_badge"] == "public catalog label only"

    after = BenchmarkContractFactory.create_contract(bundle)
    assert after.problem_bundle_digest == before.problem_bundle_digest
    assert after.contract_hash == before.contract_hash


def test_contract_bound_field_list_matches_digest_metadata() -> None:
    spec = BENCHMARKS["function_approx"]

    assert set(CONTRACT_BOUND_BENCHMARK_FIELDS) == set(spec.contract_digest_metadata())


def _changed_contract_bound_spec(spec: BenchmarkSpec, field_name: str) -> BenchmarkSpec:
    changes: dict[str, object] = {
        "name": "changed_function_approx",
        "paper_section": "S1.changed",
        "paper_task_name": "Changed paper task",
        "family": "changed family",
        "metric": "changed_contract_metric",
        "description": "Changed benchmark description.",
        "fidelity_level": "faithful-small",
        "expected_runtime_s": spec.expected_runtime_s + 1,
        "requires_torch": not spec.requires_torch,
        "requires_gpu": not spec.requires_gpu,
        "paper_gap_notes": "Changed contract-bound paper gap notes.",
    }
    return replace(spec, **{field_name: changes[field_name]})


@pytest.mark.parametrize("field_name", CONTRACT_BOUND_BENCHMARK_FIELDS)
def test_contract_bound_benchmark_fields_change_contract_hash(field_name: str) -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    before = BenchmarkContractFactory.create_contract(bundle)
    changed_bundle = replace(bundle, benchmark_spec=_changed_contract_bound_spec(bundle.benchmark_spec, field_name))

    after = BenchmarkContractFactory.create_contract(changed_bundle)
    assert after.problem_bundle_digest != before.problem_bundle_digest
    assert after.contract_hash != before.contract_hash


def test_contract_from_dict_requires_benchmark_fidelity_metadata() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    del payload["benchmark_fidelity"]
    payload["contract_hash"] = ""

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "benchmark_fidelity" in str(exc)
    else:
        raise AssertionError("Expected missing benchmark_fidelity to fail.")


def test_contract_from_dict_rejects_invalid_benchmark_fidelity_strings() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_fidelity"]["paper_section"] = ""

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "paper_section" in str(exc)
    else:
        raise AssertionError("Expected invalid benchmark_fidelity paper_section to fail.")


def test_contract_from_dict_rejects_unknown_benchmark_fidelity_fields() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_fidelity"]["unexpected_field"] = "must fail closed"
    payload["contract_hash"] = ""

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "unknown field" in str(exc)
    else:
        raise AssertionError("Expected unknown benchmark_fidelity field to fail.")


def test_contract_from_dict_rejects_future_benchmark_fidelity_schema() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_fidelity"]["schema_version"] = 2
    payload["contract_hash"] = ""

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("Expected future benchmark_fidelity schema_version to fail.")


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


def test_contract_from_dict_rejects_manifest_digest_mismatch() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_source_manifest"]["artifacts"]["guidelines.md"] = "0" * 64

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "manifest hash mismatch" in str(exc)
    else:
        raise AssertionError("Expected tampered manifest payload to fail.")


def test_contract_from_dict_rejects_invalid_manifest_schema() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_source_manifest"]["schema_version"] = 999

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("Expected invalid manifest schema_version to fail.")


def test_contract_from_dict_rejects_invalid_manifest_semantics() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_source_manifest"]["data_generated"] = False

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "data_generated" in str(exc)
    else:
        raise AssertionError("Expected inconsistent manifest data_generated flag to fail.")


def test_catalog_requires_pretraining_safe_validation() -> None:
    for name, spec in BENCHMARKS.items():
        requirements = (spec.path / "Requirements.md").read_text(encoding="utf-8")
        assert "--mode=validate" in requirements, name
        assert "must run without training" in requirements, name
        if name == "function_approx":
            assert "Prediction shape must be `(n, 1)` or `(n,)`" in requirements


def test_contract_from_dict_rejects_invalid_generated_command() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    payload["benchmark_source_manifest"]["generator_command"] = ["python", "generate_data.py"]

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "generated_seed0 command" in str(exc)
    else:
        raise AssertionError("Expected invalid manifest generator command to fail.")


def test_contract_from_dict_rejects_missing_required_manifest_artifact() -> None:
    bundle = ProblemBundle.load(BENCHMARKS["function_approx"].path)
    payload = BenchmarkContractFactory.create_contract(bundle).to_dict()
    del payload["benchmark_source_manifest"]["artifacts"]["evaluate.py"]

    try:
        EvaluationContract.from_dict(payload)
    except ValueError as exc:
        assert "missing artifact digest" in str(exc)
    else:
        raise AssertionError("Expected missing required manifest artifact to fail.")


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


def test_contract_verify_reloads_problem_bundle_from_disk(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    bundle = _problem_bundle_from_dir(benchmark_dir, spec)
    contract = BenchmarkContractFactory.create_contract(bundle)

    problem_path = benchmark_dir / "Problem.md"
    problem_path.write_text(
        problem_path.read_text(encoding="utf-8") + "\n\nAdditional stale-contract detail.\n",
        encoding="utf-8",
    )

    try:
        BenchmarkContractFactory.verify_contract(bundle, contract)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("Expected stale problem bundle verification to fail.")


def test_contract_verify_detects_stale_guidelines_source(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    bundle = _problem_bundle_from_dir(benchmark_dir, spec)
    contract = BenchmarkContractFactory.create_contract(bundle)

    guidelines_path = benchmark_dir / "guidelines.md"
    guidelines_path.write_text(
        guidelines_path.read_text(encoding="utf-8") + "\n\nStale guideline detail.\n",
        encoding="utf-8",
    )

    try:
        BenchmarkContractFactory.verify_contract(bundle, contract)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("Expected stale guidelines verification to fail.")


def test_contract_verify_detects_stale_generate_data_source(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    bundle = _problem_bundle_from_dir(benchmark_dir, spec)
    contract = BenchmarkContractFactory.create_contract(bundle)

    generator_path = benchmark_dir / "generate_data.py"
    generator_path.write_text(
        generator_path.read_text(encoding="utf-8") + "\n# stale generator detail\n",
        encoding="utf-8",
    )

    try:
        BenchmarkContractFactory.verify_contract(bundle, contract)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("Expected stale generate_data verification to fail.")


def test_contract_manifest_generation_uses_clean_environment(tmp_path: Path, monkeypatch) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    generator_path = benchmark_dir / "generate_data.py"
    generator_path.write_text(
        generator_path.read_text(encoding="utf-8").replace(
            "import argparse\n",
            "import argparse\nimport os\n\n"
            "if os.environ.get('OPENAI_API_KEY'):\n"
            "    raise RuntimeError('secret leaked into manifest data generation')\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-that-must-not-leak")

    contract = BenchmarkContractFactory.create_contract(_problem_bundle_from_dir(benchmark_dir, spec))

    assert contract.benchmark_source_manifest_digest


def test_contract_verify_detects_stale_existing_data_artifact(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    module = _load_generate_module(benchmark_dir / "generate_data.py")
    module.generate(seed=0, output_dir=benchmark_dir)

    bundle = _problem_bundle_from_dir(benchmark_dir, spec)
    contract = BenchmarkContractFactory.create_contract(bundle)

    (benchmark_dir / "train_data.npz").write_bytes(b"tampered train data")

    try:
        BenchmarkContractFactory.verify_contract(bundle, contract)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("Expected stale train data verification to fail.")


def test_contract_manifest_rejects_partial_data_artifacts(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    module = _load_generate_module(benchmark_dir / "generate_data.py")
    generated_dir = tmp_path / "generated"
    module.generate(seed=0, output_dir=generated_dir)
    shutil.copy2(generated_dir / "train_data.npz", benchmark_dir / "train_data.npz")

    try:
        BenchmarkContractFactory.create_contract(_problem_bundle_from_dir(benchmark_dir, spec))
    except ValueError as exc:
        assert "Partial benchmark data artifacts" in str(exc)
    else:
        raise AssertionError("Expected partial benchmark data artifacts to fail.")


def test_contract_manifest_rejects_partial_validation_artifact(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    module = _load_generate_module(benchmark_dir / "generate_data.py")
    generated_dir = tmp_path / "generated"
    module.generate(seed=0, output_dir=generated_dir)
    shutil.copy2(generated_dir / "val_data.npz", benchmark_dir / "val_data.npz")

    try:
        BenchmarkContractFactory.create_contract(_problem_bundle_from_dir(benchmark_dir, spec))
    except ValueError as exc:
        assert "Partial benchmark data artifacts" in str(exc)
    else:
        raise AssertionError("Expected partial validation artifact to fail.")


def test_contract_manifest_supports_custom_data_config_paths(tmp_path: Path) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / "function_approx"
    shutil.copytree(spec.path, benchmark_dir)
    module = _load_generate_module(benchmark_dir / "generate_data.py")
    generated_dir = tmp_path / "generated"
    module.generate(seed=0, output_dir=generated_dir)
    custom_data_dir = benchmark_dir / "custom_data"
    custom_data_dir.mkdir()
    shutil.copy2(generated_dir / "train_data.npz", custom_data_dir / "custom_train.npz")
    shutil.copy2(generated_dir / "val_data.npz", custom_data_dir / "custom_val.npz")
    (benchmark_dir / "Data_config.json").write_text(
        json.dumps(
            {
                "train_path": "custom_data/custom_train.npz",
                "validation_path": "custom_data/custom_val.npz",
                "description": "custom test paths",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    contract = BenchmarkContractFactory.create_contract(_problem_bundle_from_dir(benchmark_dir, spec))
    restored = EvaluationContract.from_dict(contract.to_dict())

    assert "custom_data/custom_train.npz" in restored.benchmark_source_manifest["artifacts"]
    assert "custom_data/custom_val.npz" in restored.benchmark_source_manifest["artifacts"]


def test_static_catalog_fidelity_requires_the_registered_catalog_path(tmp_path: Path) -> None:
    copied_catalog_name = tmp_path / "function_approx"
    shutil.copytree(BENCHMARKS["function_approx"].path, copied_catalog_name)

    assert benchmark_for_path(copied_catalog_name) is None
    with pytest.raises(ValueError, match="Unknown benchmark"):
        ProblemBundle.load(copied_catalog_name)


@pytest.mark.parametrize("leaked_key", ["x_val", "u_val", "validation_labels"])
def test_contract_rejects_validation_arrays_in_training_npz(
    tmp_path: Path,
    leaked_key: str,
) -> None:
    spec = BENCHMARKS["function_approx"]
    benchmark_dir = tmp_path / f"custom-{leaked_key}"
    shutil.copytree(spec.path, benchmark_dir)
    np.savez(
        benchmark_dir / "custom_train.npz",
        x_train=np.array([[0.0], [1.0]]),
        u_train=np.array([[0.0], [1.0]]),
        **{leaked_key: np.array([[7.0]])},
    )
    np.savez(
        benchmark_dir / "custom_val.npz",
        x_val=np.array([[0.0], [1.0]]),
        u_val=np.array([[0.0], [1.0]]),
    )
    (benchmark_dir / "Data_config.json").write_text(
        json.dumps(
            {
                "train_path": "custom_train.npz",
                "validation_path": "custom_val.npz",
                "description": "validation-leak fixture",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain validation"):
        BenchmarkContractFactory.create_contract(_problem_bundle_from_dir(benchmark_dir, spec))


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
    for item in payload["benchmarks"]:
        boundaries = item["claim_boundaries"]
        assert boundaries["paper_score_reproduction"] == "not_supported"
        assert boundaries["mock"]["scientific_claim"] == "not_supported"
        if item["fidelity_level"] == "proxy":
            assert boundaries["real_llm"]["scientific_claim"] == "proxy_workflow_only"
        else:
            assert boundaries["real_llm"]["scientific_claim"] == "not_validated"


def test_cli_lists_benchmark_fidelity_and_claim(cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agenticsciml.cli", "benchmarks"],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert "\tproxy\t" in result.stdout
    assert "\tfaithful-small\t" in result.stdout
    assert "proxy_workflow_only" in result.stdout
    assert "not_validated" in result.stdout


def _assert_mock_workflow_evidence_is_fail_closed(run_dir: Path) -> None:
    trace_summary = json.loads((run_dir / "trace_summary.json").read_text(encoding="utf-8"))
    readiness = json.loads(
        (run_dir / "reports" / "scientific_discovery_readiness.json").read_text(encoding="utf-8")
    )

    assert trace_summary["quality_gate"]["passed"] is True
    assert readiness["status"] == "blocked"
    assert readiness["scientific_claim_supported"] is False


@pytest.mark.parametrize("name", ORCHESTRATOR_SMOKE_BENCHMARKS)
def test_mock_orchestrator_root_only_runs_all_benchmarks(tmp_path: Path, name: str) -> None:
    spec = BENCHMARKS[name]
    config = ExperimentConfig(
        experiment_id=f"{name}-root",
        benchmark_dir=spec.path,
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, max_debug_retries=0),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    nodes = tree["nodes"]

    assert [node["node_id"] for node in nodes] == ["solution_000"]
    assert nodes[0]["status"] == "evaluated"
    assert nodes[0]["benchmark_name"] == name
    assert nodes[0]["contract_hash"]
    assert checkpoint["completed_iterations"] == 0
    eval_data = json.loads(
        (run_dir / "solutions" / "solution_000" / "eval.json").read_text(encoding="utf-8")
    )
    assert eval_data["metric"] == spec.metric
    _assert_mock_workflow_evidence_is_fail_closed(run_dir)


@pytest.mark.parametrize("name", ORCHESTRATOR_SMOKE_BENCHMARKS)
def test_mock_orchestrator_one_iteration_runs_all_benchmarks(tmp_path: Path, name: str) -> None:
    spec = BENCHMARKS[name]
    config = ExperimentConfig(
        experiment_id=f"{name}-evolution",
        benchmark_dir=spec.path,
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=0),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    nodes = tree["nodes"]

    assert {node["node_id"] for node in nodes} == {
        "solution_000",
        "solution_001",
        "solution_002",
    }
    assert all(node["status"] == "evaluated" for node in nodes)
    assert all(node["benchmark_name"] == name for node in nodes)
    assert all(node["contract_hash"] for node in nodes)
    assert checkpoint["completed_iterations"] == 1
    for node in nodes:
        eval_data = json.loads(
            (run_dir / "solutions" / node["node_id"] / "eval.json").read_text(encoding="utf-8")
        )
        assert eval_data["metric"] == spec.metric
    _assert_mock_workflow_evidence_is_fail_closed(run_dir)
