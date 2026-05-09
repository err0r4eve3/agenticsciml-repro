from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GRID_SIZE = 16
SENSOR_POINTS = np.array(
    [
        [0.25, -0.45],
        [0.65, 0.35],
        [1.05, -0.15],
        [1.45, 0.25],
    ],
    dtype=float,
)


def _field_at(points: np.ndarray, t: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    c1 = np.array([0.65 + 0.35 * np.sin(2.0 * np.pi * t), 0.28 * np.cos(2.0 * np.pi * t)])
    c2 = np.array([1.25 + 0.25 * np.cos(2.0 * np.pi * t), -0.25 * np.sin(2.0 * np.pi * t)])
    g1 = np.exp(-((x - c1[0]) ** 2 + (y - c1[1]) ** 2) / 0.08)
    g2 = np.exp(-((x - c2[0]) ** 2 + (y - c2[1]) ** 2) / 0.10)
    shear = 0.15 * np.sin(2.0 * np.pi * (x - 0.35 * t)) * np.exp(-1.5 * y * y)
    return 1.4 * g1 - 1.1 * g2 + shear


def _make_samples(rng: np.random.Generator, n: int, noisy: bool) -> tuple[np.ndarray, np.ndarray]:
    gx = np.linspace(0.0, 2.0, GRID_SIZE)
    gy = np.linspace(-1.0, 1.0, GRID_SIZE)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    xs = []
    ys = []
    for t in np.linspace(0.0, 1.0, n, endpoint=False):
        sensors = _field_at(SENSOR_POINTS, t)
        if noisy:
            sensors = sensors + rng.normal(0.0, 0.015, size=sensors.shape)
        field = _field_at(grid, t)
        xs.append(np.concatenate([sensors, [t]]))
        ys.append(field)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, u_train = _make_samples(rng, 180, noisy=True)
    x_val, u_val = _make_samples(rng, 90, noisy=False)
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
