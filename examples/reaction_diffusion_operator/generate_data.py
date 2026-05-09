from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GRID_SIZE = 32


def _random_field(rng: np.random.Generator, s: np.ndarray, scale: float) -> np.ndarray:
    coeffs = rng.normal(0.0, scale, size=4)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    return sum(coeffs[k - 1] * np.sin(2.0 * np.pi * k * s + phases[k - 1]) for k in range(1, 5))


def _make_samples(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, GRID_SIZE)
    xs = []
    ys = []
    for _ in range(n):
        diffusion = 0.08 + 0.04 * (1.0 + np.tanh(_random_field(rng, s, 0.7)))
        source = _random_field(rng, s, 0.5)
        initial = _random_field(rng, s, 0.8)
        smoothing = np.convolve(initial, [0.2, 0.6, 0.2], mode="same")
        reaction = np.tanh(initial + 0.7 * source)
        final = np.exp(-2.5 * diffusion) * smoothing + 0.45 * (1.0 - np.exp(-diffusion)) * source + 0.2 * reaction
        xs.append(np.concatenate([diffusion, source, initial]))
        ys.append(final)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, u_train = _make_samples(rng, 260)
    x_val, u_val = _make_samples(rng, 120)
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
