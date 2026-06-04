from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ArrayLike = np.ndarray | list[float] | list[list[float]]


@dataclass(frozen=True, slots=True)
class TrainingPhase:
    name: str
    objective_terms: tuple[str, ...]
    optimizer: str
    max_steps: int
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "objective_terms": list(self.objective_terms),
            "optimizer": self.optimizer,
            "max_steps": self.max_steps,
            "notes": self.notes,
        }


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def sigmoid_gate(x: ArrayLike, *, center: float = 0.0, sharpness: float = 1.0) -> np.ndarray:
    """Stable sigmoid gate used by the paper's function-approximation MoE pattern."""

    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    shifted = np.clip(sharpness * (_as_float_array(x) - center), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-shifted))


def sigmoid_moe_prediction(
    x: ArrayLike,
    left_expert: ArrayLike,
    right_expert: ArrayLike,
    *,
    center: float = 0.0,
    sharpness: float = 10.0,
) -> np.ndarray:
    left = _as_float_array(left_expert)
    right = _as_float_array(right_expert)
    if left.shape != right.shape:
        raise ValueError("left_expert and right_expert must have the same shape")
    gate = sigmoid_gate(x, center=center, sharpness=sharpness)
    if gate.shape != left.shape:
        gate = np.broadcast_to(gate, left.shape)
    return (1.0 - gate) * left + gate * right


def particular_plus_residual(particular: ArrayLike, residual: ArrayLike) -> np.ndarray:
    particular_array = _as_float_array(particular)
    residual_array = _as_float_array(residual)
    if particular_array.shape != residual_array.shape:
        raise ValueError("particular and residual must have the same shape")
    return particular_array + residual_array


def corner_importance_weights(
    points: ArrayLike,
    *,
    corner: tuple[float, ...] = (0.0, 0.0),
    exponent: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    point_array = _as_float_array(points)
    if point_array.ndim != 2:
        raise ValueError("points must be a 2D array")
    corner_array = np.asarray(corner, dtype=float)
    if corner_array.shape != (point_array.shape[1],):
        raise ValueError("corner dimension must match points")
    distance = np.linalg.norm(point_array - corner_array.reshape(1, -1), axis=1)
    raw = 1.0 / np.maximum(distance, epsilon) ** exponent
    return raw / raw.sum()


def sample_by_importance(
    points: ArrayLike,
    weights: ArrayLike,
    *,
    sample_count: int,
    seed: int,
    replace: bool = True,
) -> np.ndarray:
    point_array = _as_float_array(points)
    weight_array = _as_float_array(weights).reshape(-1)
    if point_array.ndim != 2:
        raise ValueError("points must be a 2D array")
    if len(point_array) != len(weight_array):
        raise ValueError("weights length must match points")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not replace and sample_count > len(point_array):
        raise ValueError("sample_count cannot exceed point count without replacement")
    probabilities = weight_array / weight_array.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(point_array), size=sample_count, replace=replace, p=probabilities)
    return point_array[indices]


def burgers_three_phase_schedule(
    *,
    pretrain_steps: int = 500,
    residual_steps: int = 1000,
    refine_steps: int = 200,
) -> tuple[TrainingPhase, ...]:
    if min(pretrain_steps, residual_steps, refine_steps) <= 0:
        raise ValueError("phase step counts must be positive")
    return (
        TrainingPhase(
            name="ic_bc_pretrain",
            objective_terms=("initial_condition", "periodic_boundary"),
            optimizer="Adam",
            max_steps=pretrain_steps,
            notes="Warm-start the solution on supervised initial and boundary anchors.",
        ),
        TrainingPhase(
            name="gradient_enhanced_residual",
            objective_terms=("pde_residual", "residual_gradient", "self_adaptive_weights"),
            optimizer="Adam",
            max_steps=residual_steps,
            notes="Train with gPINN-style residual-gradient diagnostics and adaptive residual weights.",
        ),
        TrainingPhase(
            name="rar_lbfgs_finetune",
            objective_terms=("residual_adaptive_refinement", "pde_residual"),
            optimizer="L-BFGS",
            max_steps=refine_steps,
            notes="Refine on highest-residual collocation points using a second-order optimizer.",
        ),
    )


def self_adaptive_residual_weights(
    residuals: ArrayLike,
    *,
    temperature: float = 1.0,
    floor: float = 1e-6,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if floor <= 0.0:
        raise ValueError("floor must be positive")
    magnitude = np.abs(_as_float_array(residuals))
    weights = np.maximum(magnitude, floor) ** temperature
    return weights / np.mean(weights)


def residual_adaptive_refinement_indices(residuals: ArrayLike, *, count: int) -> np.ndarray:
    residual_array = np.abs(_as_float_array(residuals)).reshape(-1)
    if count <= 0:
        raise ValueError("count must be positive")
    if count > residual_array.size:
        raise ValueError("count cannot exceed residual count")
    order = np.lexsort((np.arange(residual_array.size), -residual_array))
    return order[:count]


def linear_bias_free_deeponet_prediction(
    branch_inputs: ArrayLike,
    branch_weights: ArrayLike,
    trunk_features: ArrayLike,
) -> np.ndarray:
    inputs = _as_float_array(branch_inputs)
    weights = _as_float_array(branch_weights)
    trunk = _as_float_array(trunk_features)
    if inputs.ndim != 2 or weights.ndim != 2 or trunk.ndim != 2:
        raise ValueError("branch_inputs, branch_weights, and trunk_features must be 2D")
    if inputs.shape[1] != weights.shape[0]:
        raise ValueError("branch input dimension must match branch_weights rows")
    branch_code = inputs @ weights
    if branch_code.shape[1] != trunk.shape[1]:
        raise ValueError("branch code dimension must match trunk feature dimension")
    return branch_code @ trunk.T


def central_difference(values: ArrayLike, *, spacing: float, axis: int = -1) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    array = _as_float_array(values)
    if array.shape[axis] < 2:
        raise ValueError("axis must contain at least two points")
    return np.gradient(array, spacing, axis=axis, edge_order=1)


def derivative_enhanced_loss(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    x_spacing: float,
    t_spacing: float | None = None,
    derivative_weight: float = 1.0,
) -> dict[str, float]:
    if derivative_weight < 0.0:
        raise ValueError("derivative_weight must be non-negative")
    pred = _as_float_array(prediction)
    truth = _as_float_array(target)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same shape")
    value_mse = float(np.mean((pred - truth) ** 2))
    x_gradient_mse = float(
        np.mean((central_difference(pred, spacing=x_spacing, axis=-1) - central_difference(truth, spacing=x_spacing, axis=-1)) ** 2)
    )
    t_gradient_mse = 0.0
    if t_spacing is not None:
        if pred.ndim < 2:
            raise ValueError("time derivative requires at least 2D prediction arrays")
        t_gradient_mse = float(
            np.mean(
                (
                    central_difference(pred, spacing=t_spacing, axis=-2)
                    - central_difference(truth, spacing=t_spacing, axis=-2)
                )
                ** 2
            )
        )
    total = value_mse + derivative_weight * (x_gradient_mse + t_gradient_mse)
    return {
        "total": float(total),
        "value_mse": value_mse,
        "x_gradient_mse": x_gradient_mse,
        "t_gradient_mse": t_gradient_mse,
    }


def enforce_spatiotemporal_constraints(
    field: ArrayLike,
    *,
    initial: ArrayLike | None = None,
    left_boundary: float | ArrayLike = 0.0,
    right_boundary: float | ArrayLike = 0.0,
) -> np.ndarray:
    constrained = _as_float_array(field).copy()
    if constrained.ndim < 2:
        raise ValueError("field must include time and space dimensions")
    if initial is not None:
        initial_array = _as_float_array(initial)
        if initial_array.shape != constrained[..., 0, :].shape:
            raise ValueError("initial shape must match the first time slice")
        constrained[..., 0, :] = initial_array
    constrained[..., :, 0] = left_boundary
    constrained[..., :, -1] = right_boundary
    return constrained


def spectral_truncate_1d(values: ArrayLike, *, keep_fraction: float = 0.25) -> np.ndarray:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    array = _as_float_array(values)
    spectrum = np.fft.rfft(array, axis=-1)
    keep = max(1, int(np.ceil(spectrum.shape[-1] * keep_fraction)))
    truncated = spectrum.copy()
    truncated[..., keep:] = 0.0
    return np.fft.irfft(truncated, n=array.shape[-1], axis=-1)


def gaussian_kernel1d(*, sigma: float, kernel_size: int) -> np.ndarray:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve_axis_reflect(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="reflect")
    moved = np.moveaxis(padded, axis, -1)
    output = np.empty((*moved.shape[:-1], moved.shape[-1] - 2 * radius), dtype=float)
    for idx in np.ndindex(moved.shape[:-1]):
        output[idx] = np.convolve(moved[idx], kernel, mode="valid")
    return np.moveaxis(output, -1, axis)


def gaussian_low_pass_2d(field: ArrayLike, *, sigma: float = 1.0, kernel_size: int = 5) -> np.ndarray:
    array = _as_float_array(field)
    if array.ndim < 2:
        raise ValueError("field must be at least 2D")
    kernel = gaussian_kernel1d(sigma=sigma, kernel_size=kernel_size)
    smoothed = _convolve_axis_reflect(array, kernel, axis=-1)
    return _convolve_axis_reflect(smoothed, kernel, axis=-2)


def bandlimit_preserving_activation(
    field: ArrayLike,
    *,
    negative_slope: float = 0.1,
    sigma: float = 1.0,
    kernel_size: int = 5,
    activation: Literal["leaky_relu", "identity"] = "leaky_relu",
) -> np.ndarray:
    array = _as_float_array(field)
    if activation == "leaky_relu":
        activated = np.where(array >= 0.0, array, negative_slope * array)
    elif activation == "identity":
        activated = array
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return gaussian_low_pass_2d(activated, sigma=sigma, kernel_size=kernel_size)


def high_frequency_energy_2d(field: ArrayLike, *, keep_fraction: float = 0.5) -> float:
    if not 0.0 < keep_fraction < 1.0:
        raise ValueError("keep_fraction must be in (0, 1)")
    array = _as_float_array(field)
    if array.ndim != 2:
        raise ValueError("field must be 2D")
    spectrum = np.fft.rfft2(array)
    row_freq = np.fft.fftfreq(array.shape[0]).reshape(-1, 1)
    col_freq = np.fft.rfftfreq(array.shape[1]).reshape(1, -1)
    radius = np.sqrt(row_freq**2 + col_freq**2)
    mask = radius > keep_fraction * float(radius.max())
    return float(np.sum(np.abs(spectrum[mask]) ** 2))
