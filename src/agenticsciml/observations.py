from __future__ import annotations

import html
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from agenticsciml.benchmarks import benchmark_for_path
from agenticsciml.execution.runner import run_command


def build_data_observation_package(benchmark_dir: Path) -> tuple[dict[str, Any], str]:
    benchmark_dir = benchmark_dir.resolve()
    arrays, source_mode = _load_training_arrays(benchmark_dir)
    spec = benchmark_for_path(benchmark_dir)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_name": spec.name if spec else benchmark_dir.name,
        "source_mode": source_mode,
        "privacy_boundary": "training_data_only_no_validation_labels",
        "arrays": {name: _array_summary(value) for name, value in sorted(arrays.items())},
        "plots": [],
    }
    x_key, y_key = _choose_xy_arrays(arrays, x_prefix="x", y_prefixes=("u", "y", "target"))
    svg = _empty_svg("Training data overview", "No plottable x/y training arrays found.")
    if x_key and y_key:
        svg = _scatter_svg(
            _first_series(arrays[x_key]),
            _first_series(arrays[y_key]),
            title="Training data overview",
            x_label=x_key,
            y_label=y_key,
        )
        manifest["plots"].append(
            {
                "path": "reports/data_overview.svg",
                "kind": "training_target_scatter",
                "x_array": x_key,
                "y_array": y_key,
                "privacy_boundary": "training_data_only_no_validation_labels",
            }
        )
    return manifest, svg


def build_solution_observation_package(
    solution_id: str,
    workspace: Path,
    *,
    run_dir: Path,
) -> tuple[dict[str, Any], str]:
    arrays = _load_solution_arrays(workspace)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "solution_id": solution_id,
        "privacy_boundary": "prediction_only_no_validation_labels",
        "arrays": {name: _array_summary(value) for name, value in sorted(arrays.items())},
        "plots": [],
    }
    x_key, y_key = _choose_xy_arrays(
        arrays,
        x_prefix="predict_input.x",
        y_prefixes=("predictions.predictions", "predictions.y", "predictions.u"),
    )
    svg = _empty_svg("Prediction overview", "No plottable prediction arrays found.")
    if x_key and y_key:
        svg = _scatter_svg(
            _first_series(arrays[x_key]),
            _first_series(arrays[y_key]),
            title="Prediction overview",
            x_label=x_key,
            y_label=y_key,
        )
        plot_path = workspace / "prediction_overview.svg"
        manifest["plots"].append(
            {
                "path": str(plot_path.relative_to(run_dir)),
                "kind": "prediction_scatter",
                "x_array": x_key,
                "y_array": y_key,
                "privacy_boundary": "prediction_only_no_validation_labels",
            }
        )
    return manifest, svg


def prompt_observation_summary(manifest: dict[str, Any]) -> str:
    lines = [
        f"schema_version: {manifest.get('schema_version')}",
        f"privacy_boundary: {manifest.get('privacy_boundary')}",
    ]
    if manifest.get("benchmark_name"):
        lines.append(f"benchmark_name: {manifest['benchmark_name']}")
    if manifest.get("solution_id"):
        lines.append(f"solution_id: {manifest['solution_id']}")
    if manifest.get("source_mode"):
        lines.append(f"source_mode: {manifest['source_mode']}")
    lines.append("arrays:")
    for name, summary in manifest.get("arrays", {}).items():
        lines.append(
            "- "
            f"{name}: shape={summary.get('shape')} "
            f"min={summary.get('min')} max={summary.get('max')} "
            f"mean={summary.get('mean')} std={summary.get('std')}"
        )
    lines.append("plots:")
    for plot in manifest.get("plots", []):
        lines.append(
            "- "
            f"{plot.get('path')} kind={plot.get('kind')} "
            f"x={plot.get('x_array')} y={plot.get('y_array')}"
        )
    return "\n".join(lines)


def _load_training_arrays(benchmark_dir: Path) -> tuple[dict[str, np.ndarray], str]:
    train_path = benchmark_dir / "train_data.npz"
    if train_path.exists():
        return _npz_arrays(train_path), "repo_existing"
    with tempfile.TemporaryDirectory(prefix="agenticsciml-observations-") as tmp:
        tmp_path = Path(tmp)
        result = run_command(
            tmp_path,
            [
                sys.executable,
                str(benchmark_dir / "generate_data.py"),
                "--seed",
                "0",
                "--output-dir",
                str(tmp_path),
            ],
            timeout_s=20,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                "Failed to generate training data observations: "
                f"{result.stderr or result.stdout}"
            )
        return _npz_arrays(tmp_path / "train_data.npz"), "generated_seed0"


def _load_solution_arrays(workspace: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for prefix, filename in (
        ("predict_input", "predict_input.npz"),
        ("predictions", "predictions.npz"),
    ):
        path = workspace / filename
        if not path.exists():
            continue
        for name, value in _npz_arrays(path).items():
            arrays[f"{prefix}.{name}"] = value
    return arrays


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    numeric = np.asarray(array, dtype=float).reshape(-1) if array.size else np.asarray([], dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "size": int(array.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(numeric).sum()) if numeric.size else 0,
        "min": _float_or_none(np.min(finite)) if finite.size else None,
        "max": _float_or_none(np.max(finite)) if finite.size else None,
        "mean": _float_or_none(np.mean(finite)) if finite.size else None,
        "std": _float_or_none(np.std(finite)) if finite.size else None,
    }


def _float_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _choose_xy_arrays(
    arrays: dict[str, np.ndarray],
    *,
    x_prefix: str,
    y_prefixes: tuple[str, ...],
) -> tuple[str | None, str | None]:
    x_key = _first_matching_key(arrays, (x_prefix,))
    y_key = _first_matching_key(arrays, y_prefixes)
    if x_key and y_key:
        return x_key, y_key
    keys = list(arrays)
    if len(keys) >= 2:
        return keys[0], keys[1]
    return None, None


def _first_matching_key(arrays: dict[str, np.ndarray], prefixes: tuple[str, ...]) -> str | None:
    for key in arrays:
        lowered = key.lower()
        if any(lowered.startswith(prefix.lower()) for prefix in prefixes):
            return key
    return None


def _first_series(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    return array.reshape(array.shape[0], -1)[:, 0]


def _scatter_svg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    x_values, y_values = _finite_pairs(x, y)
    if x_values.size == 0:
        return _empty_svg(title, "No finite points found.")
    if x_values.size > 500:
        indices = np.linspace(0, x_values.size - 1, 500, dtype=int)
        x_values = x_values[indices]
        y_values = y_values[indices]

    width = 640
    height = 360
    left = 58
    right = 24
    top = 38
    bottom = 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = _domain(x_values)
    y_min, y_max = _domain(y_values)

    points = []
    for x_item, y_item in zip(x_values, y_values):
        px = left + ((float(x_item) - x_min) / (x_max - x_min)) * plot_w
        py = top + (1.0 - ((float(y_item) - y_min) / (y_max - y_min))) * plot_h
        points.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2" fill="#2563eb" opacity="0.72"/>')

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{left}" y="24" font-family="Arial, sans-serif" font-size="16" fill="#111827">{html.escape(title)}</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
            *points,
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 14}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{html.escape(x_label)}</text>',
            f'<text x="16" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 16 {top + plot_h / 2:.2f})" font-family="Arial, sans-serif" font-size="12" fill="#374151">{html.escape(y_label)}</text>',
            f'<text x="{left}" y="{height - 32}" font-family="Arial, sans-serif" font-size="10" fill="#6b7280">x=[{x_min:.4g}, {x_max:.4g}] y=[{y_min:.4g}, {y_max:.4g}] n={x_values.size}</text>',
            "</svg>",
        ]
    )


def _empty_svg(title: str, message: str) -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="240" viewBox="0 0 640 240">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="32" y="48" font-family="Arial, sans-serif" font-size="16" fill="#111827">{html.escape(title)}</text>',
            f'<text x="32" y="88" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">{html.escape(message)}</text>',
            "</svg>",
        ]
    )


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(x), len(y))
    x_values = np.asarray(x[:count], dtype=float).reshape(-1)
    y_values = np.asarray(y[:count], dtype=float).reshape(-1)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def _domain(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        pad = max(1.0, abs(low) * 0.05)
        return low - pad, high + pad
    pad = (high - low) * 0.05
    return low - pad, high + pad
