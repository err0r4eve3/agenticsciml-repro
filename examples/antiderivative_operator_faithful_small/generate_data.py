from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GRID_SIZE = 100
TRAIN_SAMPLES = 320
VAL_SAMPLES_PER_LENGTH_SCALE = 40
VAL_LENGTH_SCALES = (0.05, 0.1, 0.2, 0.5)


def antiderivative_samples(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    dx = np.diff(grid)
    increments = 0.5 * (values[:, 1:] + values[:, :-1]) * dx.reshape(1, -1)
    return np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)


def _grf_samples(
    rng: np.random.Generator,
    grid: np.ndarray,
    n: int,
    length_scale: float,
) -> np.ndarray:
    distance = grid[:, None] - grid[None, :]
    covariance = np.exp(-0.5 * (distance / length_scale) ** 2)
    covariance += 1e-8 * np.eye(len(grid))
    chol = np.linalg.cholesky(covariance)
    samples = rng.normal(size=(n, len(grid))) @ chol.T
    samples -= samples.mean(axis=1, keepdims=True)
    scale = np.std(samples, axis=1, keepdims=True) + 1e-12
    return samples / scale


def _make_functions(
    rng: np.random.Generator,
    grid: np.ndarray,
    length_scales: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(length_scales, dtype=float)
    values = np.empty((len(labels), len(grid)), dtype=float)
    for length_scale in sorted(set(float(item) for item in labels)):
        mask = labels == length_scale
        values[mask] = _grf_samples(rng, grid, int(mask.sum()), length_scale)
    integrals = antiderivative_samples(values, grid)
    return values.astype(float), integrals.astype(float), labels


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, GRID_SIZE)

    train_length_scales = rng.choice(np.asarray(VAL_LENGTH_SCALES), size=TRAIN_SAMPLES, replace=True).tolist()
    x_train, u_train, _ = _make_functions(rng, grid, train_length_scales)

    val_length_scales = [
        length_scale
        for length_scale in VAL_LENGTH_SCALES
        for _ in range(VAL_SAMPLES_PER_LENGTH_SCALE)
    ]
    x_val, u_val, length_scales = _make_functions(rng, grid, val_length_scales)

    np.savez(output_dir / "train_data.npz", x_train=x_train, u_train=u_train, x_grid=grid)
    np.savez(output_dir / "val_data.npz", x_val=x_val, u_val=u_val, x_grid=grid, length_scales=length_scales)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
