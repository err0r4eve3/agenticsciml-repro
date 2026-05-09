import subprocess
import sys
from pathlib import Path

import numpy as np


def test_function_approx_data_generation_is_deterministic(tmp_path: Path) -> None:
    script = Path("examples/function_approx/generate_data.py").resolve()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    subprocess.run([sys.executable, str(script), "--seed", "0", "--output-dir", str(out_a)], check=True)
    subprocess.run([sys.executable, str(script), "--seed", "0", "--output-dir", str(out_b)], check=True)

    train_a = np.load(out_a / "train_data.npz")
    train_b = np.load(out_b / "train_data.npz")
    val_a = np.load(out_a / "val_data.npz")

    assert train_a["x_train"].shape == (200, 1)
    assert train_a["u_train"].shape == (200, 1)
    assert val_a["x_val"].shape == (500, 1)
    assert val_a["u_val"].shape == (500, 1)
    np.testing.assert_allclose(train_a["x_train"], train_b["x_train"])
    np.testing.assert_allclose(train_a["u_train"], train_b["u_train"])
