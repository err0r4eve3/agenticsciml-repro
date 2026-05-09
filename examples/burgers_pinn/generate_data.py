from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def target_solution(xt: np.ndarray) -> np.ndarray:
    x = xt[:, 0]
    t = xt[:, 1]
    base = -np.sin(np.pi * x) * np.exp(-0.08 * np.pi**2 * t)
    steepener = 1.0 + 0.35 * t * np.cos(np.pi * x) ** 2
    harmonic = 0.08 * np.sin(3.0 * np.pi * (x - 0.35 * t)) * np.exp(-2.0 * t)
    return (base / steepener + harmonic).reshape(-1, 1)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.column_stack(
        [
            rng.uniform(-1.0, 1.0, size=900),
            rng.uniform(0.0, 1.0, size=900),
        ]
    )
    u_train = target_solution(x_train)

    xs = np.linspace(-1.0, 1.0, 70)
    ts = np.linspace(0.0, 1.0, 45)
    xx, tt = np.meshgrid(xs, ts)
    x_val = np.column_stack([xx.ravel(), tt.ravel()])
    u_val = target_solution(x_val)

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
