import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
EVALUATOR_PATHS = tuple(sorted(EXAMPLES_DIR.glob("*/evaluate.py")))
LSHAPE_GENERATORS = (
    EXAMPLES_DIR / "poisson_lshape" / "generate_data.py",
    EXAMPLES_DIR / "poisson_lshape_faithful_small" / "generate_data.py",
)


def _load_module(path: Path):
    module_name = f"scientific_test_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("evaluator_path", EVALUATOR_PATHS, ids=lambda path: path.parent.name)
def test_builtin_evaluators_reject_broadcast_and_reshape_counterexamples(evaluator_path: Path) -> None:
    evaluator = _load_module(evaluator_path)

    scalar_target = np.zeros((5, 1), dtype=float)
    scalar_prediction = np.arange(5, dtype=float)
    np.testing.assert_array_equal(
        evaluator._align_predictions(scalar_prediction, scalar_target),
        scalar_prediction[:, None],
    )

    multi_output_target = np.zeros((5, 3), dtype=float)
    with pytest.raises(ValueError, match="Prediction shape"):
        evaluator._align_predictions(np.zeros((5, 1), dtype=float), multi_output_target)
    with pytest.raises(ValueError, match="Prediction shape"):
        evaluator._align_predictions(np.zeros((3, 5), dtype=float), multi_output_target)


@pytest.mark.parametrize("evaluator_path", EVALUATOR_PATHS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_builtin_evaluators_reject_nonfinite_predictions(
    evaluator_path: Path,
    nonfinite: float,
) -> None:
    evaluator = _load_module(evaluator_path)
    predictions = np.zeros((4, 1), dtype=float)
    predictions[2, 0] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        evaluator._align_predictions(predictions, np.zeros_like(predictions))


@pytest.mark.parametrize("evaluator_path", EVALUATOR_PATHS, ids=lambda path: path.parent.name)
def test_builtin_evaluators_reject_nonfinite_score_and_write_standard_json(
    evaluator_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_module(evaluator_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="score must be finite"):
        evaluator._write_payload({"metric": "counterexample", "score": np.nan})

    payload = {"metric": "finite_control", "score": 0.25, "higher_is_better": False}
    evaluator._write_payload(payload)
    rendered = (tmp_path / "eval.json").read_text(encoding="utf-8")
    assert json.loads(rendered) == payload
    assert all(token not in rendered for token in ("NaN", "Infinity", "-Infinity"))


@pytest.mark.parametrize("generator_path", LSHAPE_GENERATORS, ids=lambda path: path.parent.name)
def test_lshape_singular_solution_is_continuous_on_both_notch_edges(generator_path: Path) -> None:
    generator = _load_module(generator_path)
    epsilon = 1e-10
    points = np.array(
        [
            [0.5, 0.0],
            [0.5, -epsilon],
            [0.0, 0.5],
            [-epsilon, 0.5],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    values = generator.target_solution(points).reshape(-1)
    np.testing.assert_allclose(values, 0.0, atol=1e-8, rtol=0.0)


def test_lshape_analytic_solution_matches_reported_poisson_forcing() -> None:
    generator = _load_module(LSHAPE_GENERATORS[1])
    points = np.array([[-0.55, 0.25], [0.35, -0.45], [-0.3, -0.4]], dtype=float)
    h = 1e-4
    offsets = (
        np.array([h, 0.0]),
        np.array([-h, 0.0]),
        np.array([0.0, h]),
        np.array([0.0, -h]),
    )
    center = generator.target_solution(points)
    laplacian = sum(generator.target_solution(points + offset) for offset in offsets)
    laplacian = (laplacian - 4.0 * center) / (h * h)

    np.testing.assert_allclose(
        -laplacian,
        generator.forcing_term(points),
        rtol=2e-5,
        atol=2e-6,
    )


def test_burgers_reference_satisfies_the_declared_viscous_pde() -> None:
    generator_path = EXAMPLES_DIR / "burgers_pinn_faithful_small" / "generate_data.py"
    generator = _load_module(generator_path)
    points = np.array(
        [[-0.8, 0.15], [-0.35, 0.4], [0.2, 0.7], [0.75, 0.9]],
        dtype=float,
    )
    h = 1e-4
    center = generator.target_solution(points)
    plus_x = generator.target_solution(points + np.array([h, 0.0]))
    minus_x = generator.target_solution(points - np.array([h, 0.0]))
    plus_t = generator.target_solution(points + np.array([0.0, h]))
    minus_t = generator.target_solution(points - np.array([0.0, h]))
    u_t = (plus_t - minus_t) / (2.0 * h)
    u_x = (plus_x - minus_x) / (2.0 * h)
    u_xx = (plus_x - 2.0 * center + minus_x) / (h * h)
    residual = u_t + center * u_x - generator.VISCOSITY * u_xx

    np.testing.assert_allclose(residual, 0.0, atol=2e-6, rtol=0.0)


def _run_evaluator(
    evaluator_path: Path,
    validation_path: Path,
    predictions_path: Path,
    workdir: Path,
) -> dict:
    env = os.environ.copy()
    env["AGENTICSCIML_VALIDATION_DATA"] = str(validation_path)
    result = subprocess.run(
        [sys.executable, str(evaluator_path), "--predictions", str(predictions_path)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((workdir / "eval.json").read_text(encoding="utf-8"))


def test_burgers_prediction_only_evaluator_scores_finite_difference_residual(tmp_path: Path) -> None:
    benchmark_dir = EXAMPLES_DIR / "burgers_pinn_faithful_small"
    generator = _load_module(benchmark_dir / "generate_data.py")
    generator.generate(seed=0, output_dir=tmp_path)
    validation_path = tmp_path / "val_data.npz"
    with np.load(validation_path) as validation:
        x_val = validation["x_val"].copy()
        target = validation["u_val"].copy()
        n_solution = int(validation["n_solution"].item())
        assert int(validation["n_x"].item()) == generator.VAL_X_POINTS
        assert int(validation["n_t"].item()) == generator.VAL_T_POINTS
        assert float(validation["viscosity"].item()) == generator.VISCOSITY

    oracle_path = tmp_path / "oracle_predictions.npz"
    np.savez(oracle_path, predictions=target)
    oracle = _run_evaluator(
        benchmark_dir / "evaluate.py",
        validation_path,
        oracle_path,
        tmp_path,
    )
    assert np.isfinite(oracle["score"])
    assert 0.0 < oracle["components"]["residual_relative_l2"] < 0.02
    assert oracle["components"]["solution_relative_l2"] == 0.0
    assert oracle["components"]["boundary_scaled_rmse"] == 0.0

    perturbed = target.copy()
    perturbation = 0.02 * np.sin(17.0 * np.pi * x_val[:n_solution, 0])
    perturbation *= np.sin(9.0 * np.pi * x_val[:n_solution, 1])
    perturbed[:n_solution, 0] += perturbation
    perturbed_path = tmp_path / "perturbed_predictions.npz"
    np.savez(perturbed_path, predictions=perturbed)
    perturbed_payload = _run_evaluator(
        benchmark_dir / "evaluate.py",
        validation_path,
        perturbed_path,
        tmp_path,
    )
    assert perturbed_payload["components"]["residual_relative_l2"] > (
        10.0 * oracle["components"]["residual_relative_l2"]
    )
