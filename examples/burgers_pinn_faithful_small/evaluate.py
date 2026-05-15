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
    if preds.shape == target.shape:
        return preds
    if preds.ndim == 1 and preds.shape[0] == target.shape[0]:
        return preds.reshape(-1, 1)
    if preds.ndim == 2 and preds.shape[0] == target.shape[0] and preds.shape[1] == 1:
        return np.repeat(preds, target.shape[1], axis=1)
    if preds.size == target.size:
        return preds.reshape(target.shape)
    raise ValueError(f"Prediction shape {preds.shape} cannot align with target {target.shape}.")


def _load_predictions(path: Path) -> np.ndarray:
    data = np.load(path)
    if "predictions" not in data:
        raise ValueError("predictions.npz must contain a 'predictions' array.")
    return _to_numpy(data["predictions"])


def _scalar_int(data, key: str, default: int) -> int:
    return int(np.asarray(data[key]).reshape(-1)[0]) if key in data else default


def _relative_l2(preds: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(preds - target) / (float(np.linalg.norm(target)) + 1e-12))


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
        boundary_rel_l2 = _relative_l2(preds[boundary_start:boundary_end], u_val[boundary_start:boundary_end])
    else:
        boundary_rel_l2 = 0.0

    score = float(solution_rel_l2 + 0.1 * initial_rel_l2 + 0.1 * boundary_rel_l2)
    payload = {
        "metric": "burgers_pinn_composite",
        "score": score,
        "higher_is_better": False,
        "components": {
            "solution_relative_l2": solution_rel_l2,
            "initial_relative_l2": initial_rel_l2,
            "boundary_relative_l2": boundary_rel_l2,
        },
    }
    Path("eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
