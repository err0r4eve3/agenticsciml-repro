from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def target_function(x: np.ndarray) -> np.ndarray:
    x1 = x[:, 0]
    left = x1**2 - 2.0 * np.exp(-3000.0 * (x1 + 0.5) ** 2)
    right = np.sin(10.0 * np.pi * x1) + 1.0
    return np.where(x1 <= 0.0, left, right).reshape(-1, 1)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = rng.uniform(-1.0, 1.0, size=(200, 1))
    u_train = target_function(x_train)

    x_val = np.linspace(-1.0, 1.0, 500, dtype=float).reshape(-1, 1)
    u_val = target_function(x_val)

    np.savez(output_dir / "train_data.npz", x_train=x_train, u_train=u_train)
    np.savez(output_dir / "val_data.npz", x_val=x_val, u_val=u_val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
