from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CUSTOM_BENCHMARK_SPEC = "Benchmark_spec.json"
CUSTOM_BENCHMARK_SCHEMA_VERSION = 1
CUSTOM_EVALUATOR_SYNTHESIS = "evaluator_synthesis.json"
CUSTOM_EVALUATOR_SYNTHESIS_MD = "evaluator_synthesis.md"
CUSTOM_EDA_SCRIPT = "eda/data_eda.py"
CUSTOM_EDA_SEED0 = "eda/data_eda_seed0.json"
CUSTOM_EDA_SVG = "eda/data_overview_seed0.svg"
CUSTOM_SYNTHESIS_LEVEL = "autonomous_eda_evaluator_synthesis"
CUSTOM_EVIDENCE_LEVEL = "workflow_proxy"
CUSTOM_APPROVAL_SCOPE = "workflow_proxy_run_only"

CUSTOM_BENCHMARK_CLAIM_BOUNDARY = (
    "This auto-generated EDA/evaluator synthesis bundle is a deterministic workflow "
    "proxy derived from the user's problem text. It is useful for exercising the "
    "AgenticSciML loop, artifact plumbing, training-only EDA, and private-label "
    "evaluation boundary. It is not a scientific validation of the real problem, "
    "not a paper-like benchmark, and not evidence for paper-score reproduction."
)


@dataclass(frozen=True, slots=True)
class CustomBenchmarkBundle:
    benchmark: str
    benchmark_dir: Path
    files: list[str]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "benchmark_dir": str(self.benchmark_dir),
            "files": list(self.files),
            "claim_boundary": self.claim_boundary,
        }


def create_custom_benchmark_bundle(
    root_dir: Path,
    *,
    problem_statement: str,
    requirements: str = "",
    evaluation_criteria: str = "",
    data_description: str = "",
) -> CustomBenchmarkBundle:
    root_dir = root_dir.resolve(strict=False)
    root_dir.mkdir(parents=True, exist_ok=True)
    source_text = "\n".join(
        item.strip()
        for item in (problem_statement, requirements, evaluation_criteria, data_description)
        if item.strip()
    )
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    benchmark = f"custom_{_slugify(problem_statement)}_{digest[:10]}"
    benchmark_dir = (root_dir / benchmark).resolve(strict=False)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    problem_class = _problem_class_from_text(source_text)
    evaluator_synthesis = _evaluator_synthesis_payload(
        benchmark=benchmark,
        problem_statement=problem_statement,
        requirements=requirements,
        evaluation_criteria=evaluation_criteria,
        data_description=data_description,
        problem_digest=digest,
        problem_class=problem_class,
    )
    seed0_arrays = _proxy_training_arrays(digest, seed=0)
    seed0_eda = _seed_eda_payload(
        benchmark=benchmark,
        problem_digest=digest,
        problem_class=problem_class,
        arrays=seed0_arrays,
    )

    spec = {
        "schema_version": CUSTOM_BENCHMARK_SCHEMA_VERSION,
        "name": benchmark,
        "paper_section": "custom",
        "paper_task_name": "User-defined autonomous proxy evaluator",
        "family": _family_from_problem_class(problem_class),
        "metric": "custom_proxy_relative_l2",
        "description": _one_line(problem_statement),
        "fidelity_level": "proxy",
        "expected_runtime_s": 20,
        "requires_torch": False,
        "requires_gpu": False,
        "paper_gap_notes": CUSTOM_BENCHMARK_CLAIM_BOUNDARY,
        "source_digest": digest,
        "generator": "agenticsciml.custom_benchmarks.v1",
    }
    files = {
        CUSTOM_BENCHMARK_SPEC: json.dumps(spec, indent=2, sort_keys=True) + "\n",
        CUSTOM_EVALUATOR_SYNTHESIS: json.dumps(evaluator_synthesis, indent=2, sort_keys=True) + "\n",
        CUSTOM_EVALUATOR_SYNTHESIS_MD: _evaluator_synthesis_md(evaluator_synthesis),
        CUSTOM_EDA_SCRIPT: _data_eda_py(),
        CUSTOM_EDA_SEED0: json.dumps(seed0_eda, indent=2, sort_keys=True, allow_nan=False) + "\n",
        CUSTOM_EDA_SVG: _eda_svg(seed0_arrays["x_train"], seed0_arrays["u_train"]),
        "Problem.md": _problem_md(problem_statement, data_description),
        "Requirements.md": _requirements_md(requirements),
        "Evaluation.md": _evaluation_md(evaluation_criteria),
        "Data_config.json": json.dumps(
            {
                "train_path": "train_data.npz",
                "validation_path": "val_data.npz",
                "description": (
                    "Auto-generated deterministic proxy dataset. "
                    "Private validation labels are evaluator-only."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "guidelines.md": _guidelines_md(),
        "generate_data.py": _generate_data_py(digest),
        "evaluate.py": _evaluate_py(),
    }
    for relative_path, text in files.items():
        _atomic_write_text(benchmark_dir / relative_path, text)
    return CustomBenchmarkBundle(
        benchmark=benchmark,
        benchmark_dir=benchmark_dir,
        files=sorted(files),
        claim_boundary=CUSTOM_BENCHMARK_CLAIM_BOUNDARY,
    )


def _slugify(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    slug = "_".join(words[:5]) or "problem"
    return slug[:48].strip("_") or "problem"


def _one_line(text: str, *, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _problem_class_from_text(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("inverse", "sensor", "reconstruct", "scattering", "coefficient")):
        return "inverse_reconstruction"
    if any(token in lower for token in ("operator", "deeponet", "function-to-function")):
        return "operator_learning"
    if any(token in lower for token in ("pde", "pinn", "poisson", "burgers", "diffusion")):
        return "pde_residual"
    if any(token in lower for token in ("time", "temporal", "sequence", "forecast")):
        return "time_series_regression"
    return "regression"


def _family_from_problem_class(problem_class: str) -> str:
    return {
        "inverse_reconstruction": "custom inverse reconstruction",
        "operator_learning": "custom operator learning",
        "pde_residual": "custom PINN",
        "time_series_regression": "custom temporal regression",
        "regression": "custom regression",
    }.get(problem_class, "custom regression")


def _evaluator_synthesis_payload(
    *,
    benchmark: str,
    problem_statement: str,
    requirements: str,
    evaluation_criteria: str,
    data_description: str,
    problem_digest: str,
    problem_class: str,
) -> dict[str, Any]:
    complex_requested = "complex" in problem_statement.lower() or "complex" in data_description.lower()
    return {
        "schema_version": 1,
        "synthesis_level": CUSTOM_SYNTHESIS_LEVEL,
        "evidence_level": CUSTOM_EVIDENCE_LEVEL,
        "approval_scope": CUSTOM_APPROVAL_SCOPE,
        "benchmark": benchmark,
        "problem_digest": problem_digest,
        "problem_class": problem_class,
        "input_summary": {
            "problem_statement": _one_line(problem_statement, limit=500),
            "requirements": _one_line(requirements or "No additional requirements provided.", limit=500),
            "evaluation_criteria": _one_line(
                evaluation_criteria or "No domain-specific evaluator criteria provided.",
                limit=500,
            ),
            "data_description": _one_line(data_description or "No structured data description provided.", limit=500),
        },
        "data_schema": {
            "training_features": "x_train",
            "training_targets": "u_train",
            "prediction_input": "x_val",
            "private_target": "u_val",
            "prediction_output": "predictions",
            "array_format": "npz",
            "proxy_projection": "real-valued scalar surrogate",
            "complex_values_requested": complex_requested,
        },
        "eda": {
            "privacy_boundary": "training_data_only_no_private_labels",
            "replay_script": CUSTOM_EDA_SCRIPT,
            "seed0_summary": CUSTOM_EDA_SEED0,
            "seed0_figure": CUSTOM_EDA_SVG,
            "checks": [
                "array_shape_dtype_finite_stats",
                "dynamic_range_and_constant_target_warnings",
                "training_only_first_channel_overview",
            ],
        },
        "metric": {
            "primary": "custom_proxy_relative_l2",
            "secondary": ["custom_proxy_mse", "custom_proxy_mae"],
            "higher_is_better": False,
            "prediction_key": "predictions",
            "target_key": "u_val",
            "implementation": "prediction_only_private_label_relative_l2",
        },
        "quality_gates": {
            "prediction_only": True,
            "private_validation_labels": True,
            "human_domain_review_required": True,
            "paper_score_reproduction_supported": False,
            "scientific_claim_supported": False,
        },
        "review_boundary": {
            "workflow_proxy_run_requires_human_review": False,
            "scientific_claim_requires_human_domain_review": True,
            "paper_level_claim_supported": False,
            "domain_evaluator_replacement_recommended": True,
            "approval_scope": CUSTOM_APPROVAL_SCOPE,
        },
        "synthesis_limits": [
            "The generated data model is a deterministic proxy inferred from text, not the real experiment.",
            "The evaluator does not implement private finite-element, lab, or paper-scale metrics.",
            "Upgrade evidence by replacing evaluate.py/generate_data.py and reviewing the resulting contract hash.",
        ],
        "claim_boundary": CUSTOM_BENCHMARK_CLAIM_BOUNDARY,
    }


def _evaluator_synthesis_md(payload: dict[str, Any]) -> str:
    metric = payload["metric"]
    data_schema = payload["data_schema"]
    quality_gates = payload["quality_gates"]
    return (
        "# Autonomous Evaluator Synthesis\n\n"
        f"- synthesis_level: `{payload['synthesis_level']}`\n"
        f"- evidence_level: `{payload['evidence_level']}`\n"
        f"- approval_scope: `{payload['approval_scope']}`\n"
        f"- benchmark: `{payload['benchmark']}`\n"
        f"- problem_class: `{payload['problem_class']}`\n"
        f"- primary_metric: `{metric['primary']}`\n"
        f"- higher_is_better: `{metric['higher_is_better']}`\n"
        f"- training_features: `{data_schema['training_features']}`\n"
        f"- training_targets: `{data_schema['training_targets']}`\n"
        f"- prediction_input: `{data_schema['prediction_input']}`\n"
        f"- private_target: `{data_schema['private_target']}`\n"
        f"- human_domain_review_required: `{quality_gates['human_domain_review_required']}`\n\n"
        "## Review Boundary\n\n"
        "- workflow proxy runs do not require human domain review.\n"
        "- scientific or paper-level claims require human domain review and a domain evaluator replacement.\n"
        "- paper-level claims are not supported by this generated bundle.\n\n"
        "## Boundary\n\n"
        f"{payload['claim_boundary']}\n"
    )


def _proxy_training_arrays(problem_digest: str, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1.0, 1.0, size=(160, 1))
    u_train = _target_function(problem_digest, x_train) + rng.normal(0.0, 0.02, size=(160, 1))
    return {"x_train": x_train, "u_train": u_train}


def _target_function(problem_digest: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    amplitude, frequency, phase, slope, jump = _coefficients(problem_digest)
    base = amplitude * np.sin(frequency * np.pi * x[:, 0] + phase)
    trend = slope * x[:, 0] + 0.25 * x[:, 0] ** 2
    discontinuity = jump * (x[:, 0] > 0.05).astype(float)
    return (base + trend + discontinuity).reshape(-1, 1)


def _coefficients(problem_digest: str) -> tuple[float, float, float, float, float]:
    values = [int(problem_digest[i:i + 8], 16) / 0xFFFFFFFF for i in range(0, 40, 8)]
    amplitude = 0.7 + 0.8 * values[0]
    frequency = 1.5 + 5.0 * values[1]
    phase = 2.0 * np.pi * values[2]
    slope = -0.8 + 1.6 * values[3]
    jump = -0.35 + 0.7 * values[4]
    return amplitude, frequency, phase, slope, jump


def _seed_eda_payload(
    *,
    benchmark: str,
    problem_digest: str,
    problem_class: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    array_summaries = {name: _array_summary(value) for name, value in sorted(arrays.items())}
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "problem_digest": problem_digest,
        "problem_class": problem_class,
        "privacy_boundary": "training_data_only_no_private_labels",
        "source_mode": "generated_seed0",
        "arrays": array_summaries,
        "array_checks": _array_checks(array_summaries),
        "plots": [
            {
                "path": CUSTOM_EDA_SVG,
                "kind": "training_target_scatter_first_channel",
                "x_array": "x_train",
                "y_array": "u_train",
                "privacy_boundary": "training_data_only_no_private_labels",
            }
        ],
        "claim_boundary": (
            "This EDA artifact is generated from training proxy data only. "
            "It is evaluator-synthesis context, not evaluator truth or paper-score evidence."
        ),
    }


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


def _array_checks(array_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, summary in array_summaries.items():
        notes: list[str] = []
        if summary["nan_count"]:
            notes.append("contains non-finite values")
        if summary["finite_count"] < summary["size"]:
            notes.append("finite count is smaller than total size")
        if summary["std"] == 0:
            notes.append("constant finite values")
        checks.append(
            {
                "array": name,
                "shape": summary["shape"],
                "dtype": summary["dtype"],
                "status": "warning" if notes else "ok",
                "notes": notes or ["no obvious numeric issue from generated seed0 proxy data"],
            }
        )
    return checks


def _float_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _problem_md(problem_statement: str, data_description: str) -> str:
    return (
        "# Custom Problem\n\n"
        "This benchmark was auto-generated from a user problem description.\n\n"
        "## User Problem Statement\n\n"
        f"{problem_statement.strip()}\n\n"
        "## Data Description\n\n"
        f"{data_description.strip() or 'No structured data description was provided.'}\n\n"
        "## Claim Boundary\n\n"
        f"{CUSTOM_BENCHMARK_CLAIM_BOUNDARY}\n"
    )


def _requirements_md(requirements: str) -> str:
    return (
        "# Requirements\n\n"
        f"{requirements.strip() or 'No additional user requirements were provided.'}\n\n"
        "Generated solutions must keep the evaluator contract fixed and must not "
        "read validation labels or evaluator-private files.\n"
    )


def _evaluation_md(evaluation_criteria: str) -> str:
    return (
        "# Evaluation\n\n"
        f"{evaluation_criteria.strip() or 'No domain-specific metric was provided.'}\n\n"
        "The generated proxy evaluator uses private-label relative L2 on a deterministic "
        "one-dimensional surrogate target derived from the problem text hash. This metric "
        "is only workflow evidence.\n"
    )


def _guidelines_md() -> str:
    return (
        "# Custom Proxy Benchmark Guidelines\n\n"
        "- Implement `solution.py` with `--mode=validate`, `--mode=train`, and `--mode=predict`.\n"
        "- `train` may read only `train_data.npz`.\n"
        "- `predict` receives `predict_input.npz` with only `x_val` and must write `predictions.npz`.\n"
        "- `evaluator_synthesis.json` records the autonomous EDA/evaluator synthesis boundary.\n"
        "- `eda/data_eda.py` can replay training-data-only EDA on a public training `.npz`.\n"
        "- Do not read `val_data.npz`, evaluator-private paths, network resources, or host secrets.\n"
        "- Treat this benchmark as workflow proxy evidence only.\n"
    )


def _data_eda_py() -> str:
    return r'''from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Training-data-only EDA for an auto-generated AgenticSciML benchmark.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train_path = Path(args.train_data)
    lowered_parts = [part.lower() for part in train_path.parts]
    if any(part.startswith("val") or "private_eval" in part for part in lowered_parts):
        raise SystemExit("Refusing to inspect private-label data path.")
    with np.load(train_path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    summaries = {name: summarize_array(value) for name, value in sorted(arrays.items())}
    output = {
        "schema_version": 1,
        "privacy_boundary": "training_data_only_no_private_labels",
        "source_mode": "user_supplied_training_npz",
        "arrays": summaries,
        "array_checks": array_checks(summaries),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return 0


def summarize_array(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    numeric = np.asarray(array, dtype=float).reshape(-1) if array.size else np.asarray([], dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "size": int(array.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(numeric).sum()) if numeric.size else 0,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
    }


def array_checks(summaries: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    checks = []
    for name, summary in summaries.items():
        notes = []
        if int(summary.get("nan_count") or 0):
            notes.append("contains non-finite values")
        if int(summary.get("finite_count") or 0) < int(summary.get("size") or 0):
            notes.append("finite count is smaller than total size")
        if summary.get("std") == 0:
            notes.append("constant finite values")
        checks.append(
            {
                "array": name,
                "shape": summary.get("shape"),
                "dtype": summary.get("dtype"),
                "status": "warning" if notes else "ok",
                "notes": notes or ["no obvious numeric issue from training data"],
            }
        )
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _eda_svg(x: np.ndarray, y: np.ndarray) -> str:
    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    count = min(x_values.size, y_values.size)
    x_values = x_values[:count]
    y_values = y_values[:count]
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if x_values.size > 400:
        indices = np.linspace(0, x_values.size - 1, 400, dtype=int)
        x_values = x_values[indices]
        y_values = y_values[indices]
    if x_values.size == 0:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="240" viewBox="0 0 640 240">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<text x="32" y="48" font-family="Arial, sans-serif" font-size="16" fill="#111827">'
            "Autonomous EDA seed0 overview</text>"
            '<text x="32" y="88" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">'
            "No finite training points found.</text></svg>"
        )
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
        points.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2" fill="#0f766e" opacity="0.72"/>')
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{left}" y="24" font-family="Arial, sans-serif" font-size="16" fill="#111827">Autonomous EDA seed0 overview</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1"/>',
            *points,
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 14}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">x_train</text>',
            f'<text x="16" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 16 {top + plot_h / 2:.2f})" font-family="Arial, sans-serif" font-size="12" fill="#374151">u_train</text>',
            f'<text x="{left}" y="{height - 32}" font-family="Arial, sans-serif" font-size="10" fill="#6b7280">training-only proxy seed0 n={x_values.size}</text>',
            "</svg>",
        ]
    )


def _domain(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        pad = max(1.0, abs(low) * 0.05)
        return low - pad, high + pad
    pad = (high - low) * 0.05
    return low - pad, high + pad


def _generate_data_py(problem_digest: str) -> str:
    return f'''from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROBLEM_DIGEST = "{problem_digest}"


def _coefficients() -> tuple[float, float, float, float, float]:
    values = [int(PROBLEM_DIGEST[i:i + 8], 16) / 0xFFFFFFFF for i in range(0, 40, 8)]
    amplitude = 0.7 + 0.8 * values[0]
    frequency = 1.5 + 5.0 * values[1]
    phase = 2.0 * np.pi * values[2]
    slope = -0.8 + 1.6 * values[3]
    jump = -0.35 + 0.7 * values[4]
    return amplitude, frequency, phase, slope, jump


def target_function(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    amplitude, frequency, phase, slope, jump = _coefficients()
    base = amplitude * np.sin(frequency * np.pi * x[:, 0] + phase)
    trend = slope * x[:, 0] + 0.25 * x[:, 0] ** 2
    discontinuity = jump * (x[:, 0] > 0.05).astype(float)
    return (base + trend + discontinuity).reshape(-1, 1)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = rng.uniform(-1.0, 1.0, size=(160, 1))
    u_train = target_function(x_train) + rng.normal(0.0, 0.02, size=(160, 1))
    x_val = np.linspace(-1.0, 1.0, 320, dtype=float).reshape(-1, 1)
    u_val = target_function(x_val)
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
'''


def _evaluate_py() -> str:
    return '''from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _align_predictions(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    preds = _to_numpy(preds)
    if preds.shape == target.shape:
        return preds
    if preds.ndim == 1 and preds.shape[0] == target.shape[0]:
        return preds.reshape(-1, 1)
    if preds.ndim == 2 and preds.shape[0] == target.shape[0] and preds.shape[1] == 1:
        return np.repeat(preds, target.shape[1], axis=1)
    if preds.size == target.size:
        return preds.reshape(target.shape)
    raise ValueError(f"Prediction shape {{preds.shape}} cannot align with target {{target.shape}}.")


def _load_predictions(path: Path) -> np.ndarray:
    data = np.load(path)
    if "predictions" not in data:
        raise ValueError("predictions.npz must contain a 'predictions' array.")
    return _to_numpy(data["predictions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=os.environ.get("AGENTICSCIML_PREDICTIONS_DATA", "predictions.npz"))
    args = parser.parse_args()
    validation_path = Path(os.environ.get("AGENTICSCIML_VALIDATION_DATA", "val_data.npz"))
    data = np.load(validation_path)
    u_val = data["u_val"]
    preds = _align_predictions(_load_predictions(Path(args.predictions)), u_val)
    denom = float(np.linalg.norm(u_val)) + 1e-12
    score = float(np.linalg.norm(preds - u_val) / denom)
    payload = {
        "metric": "custom_proxy_relative_l2",
        "score": score,
        "higher_is_better": False,
    }
    Path("eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
