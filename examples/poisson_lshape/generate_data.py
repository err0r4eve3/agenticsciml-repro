from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _in_lshape(points: np.ndarray) -> np.ndarray:
    return ~((points[:, 0] > 0.0) & (points[:, 1] > 0.0))


def target_solution(xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    # Measure angle inside the 3*pi/2 wedge from the positive-y notch edge.
    # This puts the other notch edge (positive x approached from below) at
    # theta=3*pi/2, so the singular basis vanishes continuously on both edges.
    theta = np.mod(np.arctan2(y, x) - 0.5 * np.pi, 2.0 * np.pi)
    r = np.sqrt(x * x + y * y)
    singular = np.power(r, 2.0 / 3.0) * np.sin(2.0 * theta / 3.0)
    smooth = 0.15 * np.sin(np.pi * x) * np.sin(np.pi * y)
    return (singular + smooth).reshape(-1, 1)


def _sample_lshape(rng: np.random.Generator, n: int) -> np.ndarray:
    samples: list[np.ndarray] = []
    while sum(len(part) for part in samples) < n:
        candidate = rng.uniform(-1.0, 1.0, size=(n, 2))
        samples.append(candidate[_in_lshape(candidate)])
    return np.concatenate(samples, axis=0)[:n]


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = _sample_lshape(rng, 700)
    u_train = target_solution(x_train)

    axis = np.linspace(-1.0, 1.0, 42)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    x_val = grid[_in_lshape(grid)]
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
