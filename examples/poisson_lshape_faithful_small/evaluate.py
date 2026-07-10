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
    n_boundary = _scalar_int(data, "n_boundary", 0)
    n_residual = _scalar_int(data, "n_residual", 0)
    h = _scalar_float(data, "finite_difference_h", 1e-3)

    solution_pred = preds[:n_solution]
    solution_target = u_val[:n_solution]
    solution_rel_l2 = float(
        np.linalg.norm(solution_pred - solution_target) / (float(np.linalg.norm(solution_target)) + 1e-12)
    )

    boundary_start = n_solution
    boundary_end = boundary_start + n_boundary
    if n_boundary:
        boundary_pred = preds[boundary_start:boundary_end]
        boundary_target = u_val[boundary_start:boundary_end]
        boundary_rel_l2 = float(
            np.linalg.norm(boundary_pred - boundary_target) / (float(np.linalg.norm(boundary_target)) + 1e-12)
        )
    else:
        boundary_rel_l2 = 0.0

    residual_start = boundary_end
    if n_residual:
        center = preds[residual_start : residual_start + n_residual]
        plus_x = preds[residual_start + n_residual : residual_start + 2 * n_residual]
        minus_x = preds[residual_start + 2 * n_residual : residual_start + 3 * n_residual]
        plus_y = preds[residual_start + 3 * n_residual : residual_start + 4 * n_residual]
        minus_y = preds[residual_start + 4 * n_residual : residual_start + 5 * n_residual]
        laplacian = (plus_x + minus_x + plus_y + minus_y - 4.0 * center) / (h * h)
        forcing = data["f_residual_val"]
        residual = -laplacian - forcing
        residual_rel_l2 = float(
            np.linalg.norm(residual) / (float(np.linalg.norm(forcing)) + 1e-12)
        )
    else:
        residual_rel_l2 = 0.0

    score = float(solution_rel_l2 + 0.1 * boundary_rel_l2 + 0.01 * residual_rel_l2)
    payload = {
        "metric": "poisson_residual_composite",
        "score": score,
        "higher_is_better": False,
        "components": {
            "solution_relative_l2": solution_rel_l2,
            "boundary_relative_l2": boundary_rel_l2,
            "residual_relative_l2": residual_rel_l2,
        },
    }
    _write_payload(payload)


if __name__ == "__main__":
    main()
