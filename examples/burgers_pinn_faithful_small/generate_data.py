from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TRAIN_ANCHORS = 900
TRAIN_INITIAL = 160
TRAIN_BOUNDARY = 240
TRAIN_COLLOCATION = 1200
VAL_X_POINTS = 83
VAL_T_POINTS = 40
VAL_INITIAL = 120
VAL_BOUNDARY = 160


def target_solution(xt: np.ndarray) -> np.ndarray:
    x = xt[:, 0]
    t = xt[:, 1]
    base = -np.sin(np.pi * x) * np.exp(-0.04 * np.pi**2 * t)
    steepener = 1.0 + 0.45 * t * np.cos(np.pi * x) ** 2
    harmonic = 0.06 * np.sin(2.0 * np.pi * (x - 0.2 * t)) * np.exp(-1.5 * t)
    return (base / steepener + harmonic).reshape(-1, 1)


def _random_xt(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.column_stack(
        [
            rng.uniform(-1.0, 1.0, size=n),
            rng.uniform(0.0, 1.0, size=n),
        ]
    )


def _initial_points(n: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n)
    return np.column_stack([x, np.zeros_like(x)])


def _boundary_points(n: int) -> np.ndarray:
    half = n // 2
    t = np.linspace(0.0, 1.0, half)
    left = np.column_stack([np.full_like(t, -1.0), t])
    right = np.column_stack([np.full_like(t, 1.0), t])
    return np.vstack([left, right])


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = _random_xt(rng, TRAIN_ANCHORS)
    u_train = target_solution(x_train)

    x_initial = _initial_points(TRAIN_INITIAL)
    u_initial = target_solution(x_initial)
    x_boundary = _boundary_points(TRAIN_BOUNDARY)
    u_boundary = target_solution(x_boundary)
    x_collocation = _random_xt(rng, TRAIN_COLLOCATION)

    xs = np.linspace(-1.0, 1.0, VAL_X_POINTS)
    ts = np.linspace(0.0, 1.0, VAL_T_POINTS)
    xx, tt = np.meshgrid(xs, ts)
    x_solution = np.column_stack([xx.ravel(), tt.ravel()])
    x_initial_val = _initial_points(VAL_INITIAL)
    x_boundary_val = _boundary_points(VAL_BOUNDARY)
    x_val = np.vstack([x_solution, x_initial_val, x_boundary_val])
    u_val = target_solution(x_val)

    np.savez(
        output_dir / "train_data.npz",
        x_train=x_train,
        u_train=u_train,
        x_initial=x_initial,
        u_initial=u_initial,
        x_boundary=x_boundary,
        u_boundary=u_boundary,
        x_collocation=x_collocation,
    )
    np.savez(
        output_dir / "val_data.npz",
        x_val=x_val,
        u_val=u_val,
        n_solution=np.array([len(x_solution)], dtype=np.int64),
        n_initial=np.array([len(x_initial_val)], dtype=np.int64),
        n_boundary=np.array([len(x_boundary_val)], dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
