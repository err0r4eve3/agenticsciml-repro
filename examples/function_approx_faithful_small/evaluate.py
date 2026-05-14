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
        return preds
    if preds.size == target.size:
        return preds.reshape(target.shape)
    raise ValueError(f"Prediction shape {preds.shape} cannot align with target {target.shape}.")


def _load_predictions(path: Path) -> np.ndarray:
    data = np.load(path)
    if "predictions" not in data:
        raise ValueError("predictions.npz must contain a 'predictions' array.")
    return _to_numpy(data["predictions"])


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
    score = float(np.mean((preds - u_val) ** 2))
    payload = {
        "metric": "validation_mse",
        "score": score,
        "higher_is_better": False,
    }
    Path("eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
