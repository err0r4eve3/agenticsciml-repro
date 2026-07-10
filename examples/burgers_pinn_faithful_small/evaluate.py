from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _align_predictions(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    preds = _to_numpy(preds)
    if not np.all(np.isfinite(preds)):
        raise ValueError("Predictions must contain only finite values.")
    if preds.shape == target.shape:
        return preds
    if target.ndim == 2 and target.shape[1] == 1 and preds.shape == (target.shape[0],):
        return preds[:, None]
    raise ValueError(f"Prediction shape {preds.shape} cannot align with target {target.shape}.")


def _load_predictions(path: Path) -> np.ndarray:
    data = np.load(path)
    if "predictions" not in data:
        raise ValueError("predictions.npz must contain a 'predictions' array.")
    return _to_numpy(data["predictions"])


def _write_payload(payload: dict) -> None:
    if not np.isfinite(float(payload["score"])):
        raise ValueError("Evaluation score must be finite.")
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    Path("eval.json").write_text(rendered, encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def _scalar_int(data, key: str, default: int) -> int:
    return int(np.asarray(data[key]).reshape(-1)[0]) if key in data else default


def _scalar_float(data, key: str, default: float) -> float:
    return float(np.asarray(data[key]).reshape(-1)[0]) if key in data else default


def _relative_l2(preds: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(preds - target) / (float(np.linalg.norm(target)) + 1e-12))


def _burgers_terms(
    grid: np.ndarray,
    *,
    dx: float,
    dt: float,
    viscosity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = grid[1:-1, 1:-1]
    u_t = (grid[2:, 1:-1] - grid[:-2, 1:-1]) / (2.0 * dt)
    u_x = (grid[1:-1, 2:] - grid[1:-1, :-2]) / (2.0 * dx)
    u_xx = (grid[1:-1, 2:] - 2.0 * center + grid[1:-1, :-2]) / (dx * dx)
    return u_t, center * u_x, viscosity * u_xx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=os.environ.get("AGENTICSCIML_PREDICTIONS_DATA", "predictions.npz"),
    )
    args = parser.parse_args()

    validation_path = Path(os.environ.get("AGENTICSCIML_VALIDATION_DATA", "val_data.npz"))
    data = np.load(validation_path)
    u_val = data["u_val"]
    preds = _align_predictions(_load_predictions(Path(args.predictions)), u_val)
    n_solution = _scalar_int(data, "n_solution", len(u_val))
    n_initial = _scalar_int(data, "n_initial", 0)
    n_boundary = _scalar_int(data, "n_boundary", 0)
    n_x = _scalar_int(data, "n_x", 0)
    n_t = _scalar_int(data, "n_t", 0)
    dx = _scalar_float(data, "finite_difference_dx", 0.0)
    dt = _scalar_float(data, "finite_difference_dt", 0.0)
    viscosity = _scalar_float(data, "viscosity", 0.0)

    if n_x < 3 or n_t < 3 or n_solution != n_x * n_t:
        raise ValueError("Burgers validation grid metadata is inconsistent with n_solution.")
    if n_solution + n_initial + n_boundary != len(u_val):
        raise ValueError("Burgers validation partition counts do not match u_val.")
    if not all(np.isfinite(value) and value > 0.0 for value in (dx, dt, viscosity)):
        raise ValueError("Burgers dx, dt, and viscosity must be positive finite values.")

    solution_pred = preds[:n_solution]
    solution_target = u_val[:n_solution]
    solution_rel_l2 = _relative_l2(solution_pred, solution_target)

    initial_start = n_solution
    initial_end = initial_start + n_initial
    if n_initial:
        initial_rel_l2 = _relative_l2(preds[initial_start:initial_end], u_val[initial_start:initial_end])
    else:
        initial_rel_l2 = 0.0

    boundary_start = initial_end
    boundary_end = boundary_start + n_boundary
    if n_boundary:
        boundary_error = preds[boundary_start:boundary_end] - u_val[boundary_start:boundary_end]
        solution_rms = float(np.sqrt(np.mean(solution_target * solution_target))) + 1e-12
        boundary_scaled_rmse = float(np.sqrt(np.mean(boundary_error * boundary_error)) / solution_rms)
    else:
        boundary_scaled_rmse = 0.0

    pred_grid = solution_pred.reshape(n_t, n_x, 1)
    target_grid = solution_target.reshape(n_t, n_x, 1)
    pred_terms = _burgers_terms(pred_grid, dx=dx, dt=dt, viscosity=viscosity)
    target_terms = _burgers_terms(target_grid, dx=dx, dt=dt, viscosity=viscosity)
    residual = pred_terms[0] + pred_terms[1] - pred_terms[2]
    residual_scale = sum(float(np.linalg.norm(term)) for term in target_terms) + 1e-12
    residual_rel_l2 = float(np.linalg.norm(residual) / residual_scale)

    score = float(
        solution_rel_l2
        + 0.1 * initial_rel_l2
        + 0.1 * boundary_scaled_rmse
        + 0.01 * residual_rel_l2
    )
    payload = {
        "metric": "burgers_pinn_composite",
        "score": score,
        "higher_is_better": False,
        "components": {
            "solution_relative_l2": solution_rel_l2,
            "initial_relative_l2": initial_rel_l2,
            "boundary_scaled_rmse": boundary_scaled_rmse,
            "residual_relative_l2": residual_rel_l2,
        },
    }
    _write_payload(payload)


if __name__ == "__main__":
    main()
