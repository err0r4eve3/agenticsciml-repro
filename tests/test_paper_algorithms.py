import numpy as np

from agenticsciml.algorithm_catalog import ALGORITHMS
from agenticsciml.paper_algorithms import (
    bandlimit_preserving_activation,
    burgers_three_phase_schedule,
    corner_importance_weights,
    derivative_enhanced_loss,
    enforce_spatiotemporal_constraints,
    high_frequency_energy_2d,
    linear_bias_free_deeponet_prediction,
    particular_plus_residual,
    residual_adaptive_refinement_indices,
    sample_by_importance,
    self_adaptive_residual_weights,
    sigmoid_gate,
    sigmoid_moe_prediction,
    spectral_truncate_1d,
)


def test_sigmoid_moe_gate_is_bounded_monotone_and_composes_experts() -> None:
    x = np.linspace(-1.0, 1.0, 9).reshape(-1, 1)
    gate = sigmoid_gate(x, center=0.0, sharpness=6.0)

    assert np.all(gate > 0.0)
    assert np.all(gate < 1.0)
    assert np.all(np.diff(gate.reshape(-1)) > 0.0)

    left = np.full_like(x, -2.0)
    right = np.full_like(x, 3.0)
    prediction = sigmoid_moe_prediction(x, left, right, center=0.0, sharpness=40.0)

    assert prediction[0, 0] < -1.99
    assert prediction[-1, 0] > 2.99
    np.testing.assert_allclose(prediction[4, 0], 0.5)


def test_poisson_decomposition_and_corner_sampling_are_deterministic() -> None:
    particular = np.array([[1.0], [2.0], [3.0]])
    residual = np.array([[0.2], [-0.3], [0.5]])
    np.testing.assert_allclose(particular_plus_residual(particular, residual), particular + residual)

    points = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 1.0], [0.6, 0.6]])
    weights = corner_importance_weights(points, corner=(0.0, 0.0), exponent=1.2)

    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1] > weights[3] > weights[2]
    first = sample_by_importance(points, weights, sample_count=6, seed=11)
    second = sample_by_importance(points, weights, sample_count=6, seed=11)
    np.testing.assert_allclose(first, second)


def test_burgers_schedule_weights_and_rar_indices_are_auditable() -> None:
    schedule = burgers_three_phase_schedule(pretrain_steps=2, residual_steps=3, refine_steps=4)

    assert [phase.name for phase in schedule] == [
        "ic_bc_pretrain",
        "gradient_enhanced_residual",
        "rar_lbfgs_finetune",
    ]
    assert schedule[1].objective_terms == ("pde_residual", "residual_gradient", "self_adaptive_weights")
    assert schedule[2].optimizer == "L-BFGS"

    residuals = np.array([0.1, 0.5, 0.2, 0.5])
    weights = self_adaptive_residual_weights(residuals)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[1] == weights[3]
    assert weights[1] > weights[2] > weights[0]
    np.testing.assert_array_equal(residual_adaptive_refinement_indices(residuals, count=2), np.array([1, 3]))


def test_linear_bias_free_deeponet_preserves_operator_linearity() -> None:
    rng = np.random.default_rng(7)
    branch_weights = rng.normal(size=(4, 3))
    trunk_features = rng.normal(size=(5, 3))
    a = rng.normal(size=(2, 4))
    b = rng.normal(size=(2, 4))

    output_a = linear_bias_free_deeponet_prediction(a, branch_weights, trunk_features)
    output_b = linear_bias_free_deeponet_prediction(b, branch_weights, trunk_features)
    output_sum = linear_bias_free_deeponet_prediction(a + b, branch_weights, trunk_features)
    output_scaled = linear_bias_free_deeponet_prediction(2.5 * a, branch_weights, trunk_features)

    np.testing.assert_allclose(output_sum, output_a + output_b, atol=1e-12)
    np.testing.assert_allclose(output_scaled, 2.5 * output_a, atol=1e-12)


def test_reaction_diffusion_helpers_enforce_constraints_and_derivative_loss() -> None:
    grid = np.linspace(0.0, 1.0, 12)
    times = np.linspace(0.0, 1.0, 6)
    target = np.sin(np.pi * grid).reshape(1, 1, -1) * np.exp(-times).reshape(1, -1, 1)
    shifted = target + 0.1 * grid.reshape(1, 1, -1)

    loss = derivative_enhanced_loss(
        shifted,
        target,
        x_spacing=float(grid[1] - grid[0]),
        t_spacing=float(times[1] - times[0]),
        derivative_weight=0.5,
    )

    assert loss["total"] > loss["value_mse"] > 0.0
    assert loss["x_gradient_mse"] > 0.0
    assert loss["t_gradient_mse"] >= 0.0

    constrained = enforce_spatiotemporal_constraints(
        shifted,
        initial=target[:, 0, :],
        left_boundary=0.0,
        right_boundary=0.0,
    )
    np.testing.assert_allclose(constrained[:, 0, 1:-1], target[:, 0, 1:-1])
    np.testing.assert_allclose(constrained[:, :, 0], 0.0)
    np.testing.assert_allclose(constrained[:, :, -1], 0.0)

    noisy = target + 0.2 * np.sin(10.0 * np.pi * grid).reshape(1, 1, -1)
    smoothed = spectral_truncate_1d(noisy, keep_fraction=0.3)
    assert np.mean((smoothed - target) ** 2) < np.mean((noisy - target) ** 2)


def test_bandlimit_filter_reduces_high_frequency_energy() -> None:
    axis = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    field = np.sin(xx) + 0.4 * np.sin(12.0 * xx + 9.0 * yy)

    before = high_frequency_energy_2d(field)
    filtered = bandlimit_preserving_activation(field, activation="identity", sigma=1.0, kernel_size=5)
    after = high_frequency_energy_2d(filtered)

    assert filtered.shape == field.shape
    assert after < before * 0.5


def test_catalog_exposes_paper_reference_implementations_without_score_claims() -> None:
    algorithms = {item.algorithm_id: item for item in ALGORITHMS}
    expected = {
        "paper_sigmoid_moe_gate",
        "paper_poisson_decomposition_sampler",
        "paper_burgers_staged_pinn_schedule",
        "paper_linear_bias_free_deeponet",
        "paper_reaction_diffusion_fno_helpers",
        "paper_cylinder_bandlimited_filter",
    }

    assert expected <= set(algorithms)
    for algorithm_id in expected:
        spec = algorithms[algorithm_id]
        payload = spec.to_dict()
        assert spec.status == "reference_implementation"
        assert spec.implementation_path is not None
        assert "paper-score" in spec.safety_notes or "paper-scale" in spec.safety_notes
        assert payload["claim_boundary"].startswith("Algorithm catalog entries are planning")
