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


def _mean_relative_l2(preds: np.ndarray, target: np.ndarray) -> float:
    numerator = np.linalg.norm(preds - target, axis=1)
    denominator = np.linalg.norm(target, axis=1) + 1e-12
    return float(np.mean(numerator / denominator))


def _scalar_int(data, key: str, default: int) -> int:
    return int(np.asarray(data[key]).reshape(-1)[0]) if key in data else default


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
    score = _mean_relative_l2(preds, u_val)
    n_t = _scalar_int(data, "n_t", 1)
    n_x = _scalar_int(data, "n_x", u_val.shape[1] // max(n_t, 1))
    pred_grid = preds.reshape(len(preds), n_t, n_x)
    target_grid = u_val.reshape(len(u_val), n_t, n_x)

    time_window = max(1, n_t // 4)
    early_time_relative_l2 = _mean_relative_l2(
        pred_grid[:, :time_window, :].reshape(len(preds), -1),
        target_grid[:, :time_window, :].reshape(len(preds), -1),
    )
    late_time_relative_l2 = _mean_relative_l2(
        pred_grid[:, -time_window:, :].reshape(len(preds), -1),
        target_grid[:, -time_window:, :].reshape(len(preds), -1),
    )
    spatial_gradient_relative_l2 = _mean_relative_l2(
        np.diff(pred_grid, axis=2).reshape(len(preds), -1),
        np.diff(target_grid, axis=2).reshape(len(preds), -1),
    )
    temporal_difference_relative_l2 = _mean_relative_l2(
        np.diff(pred_grid, axis=1).reshape(len(preds), -1),
        np.diff(target_grid, axis=1).reshape(len(preds), -1),
    )
    payload = {
        "metric": "relative_l2",
        "score": score,
        "higher_is_better": False,
        "diagnostics": {
            "early_time_relative_l2": early_time_relative_l2,
            "late_time_relative_l2": late_time_relative_l2,
            "spatial_gradient_relative_l2": spatial_gradient_relative_l2,
            "temporal_difference_relative_l2": temporal_difference_relative_l2,
        },
    }
    Path("eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
