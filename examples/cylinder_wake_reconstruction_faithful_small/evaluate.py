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


def _mean_per_sample_relative_l2(preds: np.ndarray, target: np.ndarray) -> float:
    numerator = np.linalg.norm(preds - target, axis=1)
    denominator = np.linalg.norm(target, axis=1) + 1e-12
    return float(np.mean(numerator / denominator))


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
    score = _mean_per_sample_relative_l2(preds, u_val)
    payload = {
        "metric": "relative_l2",
        "score": score,
        "higher_is_better": False,
        "diagnostics": {
            "mean_abs_error": float(np.mean(np.abs(preds - u_val))),
            "max_abs_error": float(np.max(np.abs(preds - u_val))),
            "sample_count": int(u_val.shape[0]),
            "output_dimension": int(u_val.shape[1]),
        },
    }
    _write_payload(payload)


if __name__ == "__main__":
    main()
