from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GRID_SIZE = 24
WINDOW_SIZE = 5
STEPS_PER_TRAJECTORY = 72
TRAIN_SCENARIOS = (
    (0.88, 0.00, 1.00),
    (0.95, 0.17, 0.92),
    (1.03, 0.33, 1.08),
    (1.12, 0.49, 0.97),
)
VAL_SCENARIOS = (
    (0.91, 0.61, 1.03),
    (1.08, 0.78, 0.95),
)
SENSOR_POINTS = np.array(
    [
        [0.35, -0.55],
        [0.35, 0.55],
        [0.75, -0.32],
        [0.75, 0.32],
        [1.15, -0.42],
        [1.15, 0.42],
        [1.65, -0.20],
        [1.65, 0.20],
    ],
    dtype=float,
)

WakeScenario = tuple[float, float, float]


def _field_at(points: np.ndarray, t: float, scenario: WakeScenario) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    shedding_frequency, phase, amplitude = scenario
    theta = 2.0 * np.pi * (shedding_frequency * t + phase)
    convective_phase = 2.0 * np.pi * (x - 0.45 * t)

    c1 = np.array([0.65 + 0.22 * np.sin(theta), 0.34 * np.cos(theta)])
    c2 = np.array([1.18 + 0.25 * np.cos(theta + 0.45), -0.32 * np.sin(theta)])
    c3 = np.array([1.65 + 0.20 * np.sin(2.0 * theta), 0.24 * np.cos(theta + 0.9)])

    g1 = np.exp(-((x - c1[0]) ** 2 + (y - c1[1]) ** 2) / 0.055)
    g2 = np.exp(-((x - c2[0]) ** 2 + (y - c2[1]) ** 2) / 0.075)
    g3 = np.exp(-((x - c3[0]) ** 2 + (y - c3[1]) ** 2) / 0.115)
    shear_layer = 0.16 * np.sin(convective_phase + theta) * np.exp(-1.8 * y**2)
    cross_stream = 0.07 * np.sin(2.0 * np.pi * y + 0.5 * theta) * np.exp(-0.7 * (x - 1.1) ** 2)
    return amplitude * (1.35 * g1 - 1.05 * g2 + 0.55 * g3 + shear_layer + cross_stream)


def build_lagged_sensor_features(
    sensor_series: np.ndarray,
    times: np.ndarray,
    *,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    sensors = np.asarray(sensor_series, dtype=float)
    time_array = np.asarray(times, dtype=float)
    if sensors.ndim != 2:
        raise ValueError("sensor_series must be 2D")
    if time_array.ndim != 1 or len(time_array) != len(sensors):
        raise ValueError("times must be a 1D array matching sensor_series")
    if window_size <= 0 or window_size > len(sensors):
        raise ValueError("window_size must be positive and no larger than the series length")

    features = []
    for end_idx in range(window_size - 1, len(sensors)):
        history = sensors[end_idx - window_size + 1 : end_idx + 1].reshape(-1)
        features.append(np.concatenate([history, [time_array[end_idx]]]))
    return np.asarray(features, dtype=float)


def _trajectory_samples(
    scenario: WakeScenario,
    *,
    noisy_sensors: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.linspace(0.0, 2.2, GRID_SIZE)
    gy = np.linspace(-1.0, 1.0, GRID_SIZE)
    xx, yy = np.meshgrid(gx, gy, indexing="xy")
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    times = np.linspace(0.0, 1.0, STEPS_PER_TRAJECTORY, endpoint=False)

    sensor_rows = []
    field_rows = []
    for t in times:
        sensors = _field_at(SENSOR_POINTS, float(t), scenario)
        if noisy_sensors:
            sensors = sensors + rng.normal(0.0, 0.01, size=sensors.shape)
        sensor_rows.append(sensors)
        field_rows.append(_field_at(grid, float(t), scenario))

    sensor_series = np.asarray(sensor_rows, dtype=float)
    fields = np.asarray(field_rows, dtype=float)
    features = build_lagged_sensor_features(sensor_series, times, window_size=WINDOW_SIZE)
    return features, fields[WINDOW_SIZE - 1 :], times[WINDOW_SIZE - 1 :]


def _make_split(
    scenarios: tuple[tuple[float, float, float], ...],
    *,
    noisy_sensors: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    fields = []
    scenario_ids = []
    for idx, raw in enumerate(scenarios):
        x_part, u_part, _ = _trajectory_samples(raw, noisy_sensors=noisy_sensors, rng=rng)
        features.append(x_part)
        fields.append(u_part)
        scenario_ids.append(np.full(len(x_part), idx, dtype=np.int64))
    return np.vstack(features), np.vstack(fields), np.concatenate(scenario_ids)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_grid = np.linspace(0.0, 2.2, GRID_SIZE)
    y_grid = np.linspace(-1.0, 1.0, GRID_SIZE)
    x_train, u_train, train_scenario_id = _make_split(TRAIN_SCENARIOS, noisy_sensors=True, rng=rng)
    x_val, u_val, val_scenario_id = _make_split(VAL_SCENARIOS, noisy_sensors=False, rng=rng)
    shared = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "sensor_points": SENSOR_POINTS,
        "window_size": np.array([WINDOW_SIZE], dtype=np.int64),
        "sensor_count": np.array([len(SENSOR_POINTS)], dtype=np.int64),
        "n_x": np.array([GRID_SIZE], dtype=np.int64),
        "n_y": np.array([GRID_SIZE], dtype=np.int64),
    }
    np.savez(
        output_dir / "train_data.npz",
        x_train=x_train,
        u_train=u_train,
        scenario_id=train_scenario_id,
        **shared,
    )
    np.savez(
        output_dir / "val_data.npz",
        x_val=x_val,
        u_val=u_val,
        scenario_id=val_scenario_id,
        **shared,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
