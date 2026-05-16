from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GRID_SIZE = 50
TIME_STEPS = 40
TRAIN_SAMPLES = 64
VAL_SAMPLES = 24
INPUT_CHANNEL_COUNT = 3
INNER_STEPS_PER_FRAME = 4


def _random_field(
    rng: np.random.Generator,
    grid: np.ndarray,
    scale: float,
    modes: int = 5,
) -> np.ndarray:
    coeffs = rng.normal(0.0, scale / np.arange(1, modes + 1), size=modes)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=modes)
    field = np.zeros_like(grid)
    for mode in range(1, modes + 1):
        field += coeffs[mode - 1] * np.sin(2.0 * np.pi * mode * grid + phases[mode - 1])
    return field


def unpack_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.shape[1] != INPUT_CHANNEL_COUNT * GRID_SIZE:
        raise ValueError(
            f"Expected {INPUT_CHANNEL_COUNT * GRID_SIZE} feature columns, got {features.shape[1]}."
        )
    diffusion = features[:, :GRID_SIZE]
    source = features[:, GRID_SIZE : 2 * GRID_SIZE]
    initial = features[:, 2 * GRID_SIZE :]
    return diffusion, source, initial


def _sample_inputs(rng: np.random.Generator, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_diffusion = _random_field(rng, grid, scale=0.55, modes=4)
    diffusion = 0.002 + 0.003 * (1.0 + np.tanh(raw_diffusion)) / 2.0
    source = 0.35 * _random_field(rng, grid, scale=0.7, modes=5)
    initial = 0.45 * _random_field(rng, grid, scale=0.8, modes=4)
    initial[0] = 0.0
    initial[-1] = 0.0
    return diffusion, source, initial


def _simulate_reaction_diffusion(
    diffusion: np.ndarray,
    source: np.ndarray,
    initial: np.ndarray,
    grid: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    dx = float(grid[1] - grid[0])
    frame_dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    dt = frame_dt / INNER_STEPS_PER_FRAME
    current = np.asarray(initial, dtype=float).copy()
    frames = [current.copy()]
    for _ in range(1, len(times)):
        for _inner in range(INNER_STEPS_PER_FRAME):
            left_flux = diffusion[1:-1] * (current[1:-1] - current[:-2]) / dx
            right_flux = diffusion[2:] * (current[2:] - current[1:-1]) / dx
            divergence = (right_flux - left_flux) / dx
            reaction = 0.01 * current[1:-1] ** 2
            updated = current.copy()
            updated[1:-1] = current[1:-1] + dt * (divergence + reaction + source[1:-1])
            updated[0] = 0.0
            updated[-1] = 0.0
            current = np.clip(updated, -4.0, 4.0)
        frames.append(current.copy())
    return np.asarray(frames, dtype=float)


def _make_samples(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    times = np.linspace(0.0, 1.0, TIME_STEPS)
    xs = []
    ys = []
    for _ in range(n):
        diffusion, source, initial = _sample_inputs(rng, grid)
        frames = _simulate_reaction_diffusion(diffusion, source, initial, grid, times)
        xs.append(np.concatenate([diffusion, source, initial]))
        ys.append(frames.reshape(-1))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_grid = np.linspace(0.0, 1.0, GRID_SIZE)
    t_grid = np.linspace(0.0, 1.0, TIME_STEPS)
    x_train, u_train = _make_samples(rng, TRAIN_SAMPLES)
    x_val, u_val = _make_samples(rng, VAL_SAMPLES)
    shared = {
        "x_grid": x_grid,
        "t_grid": t_grid,
        "n_x": np.array([GRID_SIZE], dtype=np.int64),
        "n_t": np.array([TIME_STEPS], dtype=np.int64),
        "input_channel_count": np.array([INPUT_CHANNEL_COUNT], dtype=np.int64),
    }
    np.savez(output_dir / "train_data.npz", x_train=x_train, u_train=u_train, **shared)
    np.savez(output_dir / "val_data.npz", x_val=x_val, u_val=u_val, **shared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
