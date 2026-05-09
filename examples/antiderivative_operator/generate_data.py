from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GRID_SIZE = 64


def _make_functions(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, GRID_SIZE)
    values = []
    integrals = []
    dx = s[1] - s[0]
    for _ in range(n):
        coeffs = rng.normal(0.0, [0.7, 0.35, 0.2, 0.12])
        phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
        a = sum(coeffs[k - 1] * np.sin(2.0 * np.pi * k * s + phases[k - 1]) for k in range(1, 5))
        a += 0.15 * np.cos(9.0 * np.pi * s + phases[0])
        integral = np.cumsum(a) * dx
        integral -= integral[:, None][0] if integral.ndim == 2 else integral[0]
        values.append(a)
        integrals.append(integral)
    return np.asarray(values, dtype=float), np.asarray(integrals, dtype=float)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, u_train = _make_functions(rng, 240)
    x_val, u_val = _make_functions(rng, 120)
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
