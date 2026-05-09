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


def load_model():
    solution = importlib.import_module("solution")

    # Pickles created while running solution.py as a script may reference
    # __main__.MODEL. Make that name available before loading the checkpoint.
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
    model = load_model()
    preds = _to_numpy(model.predict(x_val)).reshape(u_val.shape)
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
