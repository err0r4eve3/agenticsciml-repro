from __future__ import annotations

import importlib
import os
import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np

MODEL_CHECKPOINT = Path("model.pkl")


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _align_predictions(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    preds = _to_numpy(preds)
    if preds.shape == target.shape:
        return preds
    if preds.ndim == 1 and preds.shape[0] == target.shape[0]:
        preds = preds.reshape(-1, 1)
    if preds.ndim == 2 and preds.shape[0] == target.shape[0] and preds.shape[1] == 1:
        return np.repeat(preds, target.shape[1], axis=1)
    if preds.size == target.size:
        return preds.reshape(target.shape)
    raise ValueError(f"Prediction shape {preds.shape} cannot align with target {target.shape}.")


def load_model():
    solution = importlib.import_module("solution")
    import __main__

    __main__.MODEL = solution.MODEL
    with MODEL_CHECKPOINT.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError("Expected model.pkl from training.")
    data = np.load(os.environ.get("AGENTICSCIML_VALIDATION_DATA", "val_data.npz"))
    x_val = data["x_val"]
    u_val = data["u_val"]
    preds = _align_predictions(load_model().predict(x_val), u_val)
    denom = float(np.linalg.norm(u_val)) + 1e-12
    score = float(np.linalg.norm(preds - u_val) / denom)
    payload = {"metric": "relative_l2", "score": score, "higher_is_better": False}
    Path("eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
