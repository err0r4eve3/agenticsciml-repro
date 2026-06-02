from __future__ import annotations

import html
import json
import struct
import sys
import tempfile
import zlib
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


def build_data_eda_package(manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    eda_output = {
        "schema_version": 1,
        "benchmark_name": manifest.get("benchmark_name"),
        "source_mode": manifest.get("source_mode"),
        "privacy_boundary": "training_data_only_no_private_labels",
        "input_artifacts": ["reports/data_observations.json"],
        "array_checks": _eda_array_checks(manifest),
        "plot_checks": _eda_plot_checks(manifest),
        "modeling_notes": _eda_modeling_notes(manifest),
        "replay": {
            "script": "reports/data_eda.py",
            "input": "training npz only",
            "command": "python reports/data_eda.py --train-data <train_data.npz> --output reports/data_eda.json",
        },
        "claim_boundary": (
            "This EDA output summarizes training data only. It is workflow context for agents, "
            "not evaluator truth and not paper-score evidence."
        ),
    }
    return eda_output, DATA_EDA_SCRIPT


def build_structured_data_analysis(
    benchmark_dir: Path,
    manifest: dict[str, Any],
    eda_output: dict[str, Any],
    *,
    llm_report: str = "",
) -> dict[str, Any]:
    benchmark_dir = benchmark_dir.resolve()
    spec = benchmark_for_path(benchmark_dir)
    arrays = manifest.get("arrays", {})
    array_map = arrays if isinstance(arrays, dict) else {}
    array_names = sorted(str(name) for name in array_map)
    data_config = _read_json_object(benchmark_dir / "Data_config.json")
    structured = {
        "schema_version": 1,
        "benchmark_name": spec.name if spec else benchmark_dir.name,
        "benchmark_family": spec.family if spec else "unknown",
        "paper_section": spec.paper_section if spec else None,
        "fidelity_level": spec.fidelity_level if spec else "unknown",
        "problem_summary": _doc_excerpt(benchmark_dir / "Problem.md", fallback=spec.description if spec else ""),
        "evaluation_metric": spec.metric if spec else _doc_excerpt(benchmark_dir / "Evaluation.md", fallback="unknown"),
        "training_arrays": {
            name: summary
            for name, summary in sorted(array_map.items())
            if isinstance(summary, dict)
        },
        "training_array_keys": array_names,
        "data_config": data_config,
        "task_specific_observations": _task_specific_observations(spec, array_names, manifest, eda_output),
        "modeling_implications": _structured_modeling_implications(spec, array_names, eda_output),
        "risks": _structured_analysis_risks(spec, manifest, eda_output),
        "private_label_boundary": "training_data_only_no_validation_labels",
        "llm_report_summary": llm_report.strip(),
        "claim_boundary": (
            "Structured data analysis is workflow context from training artifacts only. "
            "It is not evaluator truth, paper-score evidence, or scientific support by itself."
        ),
    }
    return structured


def render_structured_data_analysis(structured: dict[str, Any]) -> str:
    observations = _markdown_list(structured.get("task_specific_observations"))
    implications = _markdown_list(structured.get("modeling_implications"))
    risks = _markdown_list(structured.get("risks"))
    array_keys = ", ".join(str(item) for item in structured.get("training_array_keys", [])) or "none"
    llm_report = str(structured.get("llm_report_summary") or "").strip()
    if not llm_report:
        llm_report = "No LLM summary was recorded."
    return (
        "# Data Analysis\n\n"
        "## Structured Summary\n\n"
        f"- benchmark: {structured.get('benchmark_name')}\n"
        f"- family: {structured.get('benchmark_family')}\n"
        f"- fidelity_level: {structured.get('fidelity_level')}\n"
        f"- metric: {structured.get('evaluation_metric')}\n"
        f"- training_array_keys: {array_keys}\n"
        f"- private_label_boundary: {structured.get('private_label_boundary')}\n\n"
        "## Problem Summary\n\n"
        f"{structured.get('problem_summary') or 'No problem summary available.'}\n\n"
        "## Task-Specific Observations\n\n"
        f"{observations}\n\n"
        "## Modeling Implications\n\n"
        f"{implications}\n\n"
        "## Risks\n\n"
        f"{risks}\n\n"
        "## LLM Summary\n\n"
        f"{llm_report}\n\n"
        "## Claim Boundary\n\n"
        f"{structured.get('claim_boundary')}\n"
    )


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


def build_visual_audit_package(
    solution_id: str,
    workspace: Path,
    *,
    run_dir: Path,
    mode: str,
    provider_capabilities: dict[str, object] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, bytes]]:
    mode = mode if mode in {"off", "mock", "real"} else "off"
    arrays = _load_solution_arrays(workspace)
    capabilities = dict(provider_capabilities or {})
    plots: dict[str, str] = {}
    image_plots: dict[str, bytes] = {}
    artifacts: list[dict[str, object]] = []
    prediction_key = _first_matching_key(
        arrays,
        ("predictions.predictions", "predictions.y", "predictions.u"),
    )
    x_key = _first_matching_key(arrays, ("predict_input.x", "predict_input.t", "predict_input.sensor"))
    if mode == "off":
        pass
    elif prediction_key is None:
        plots["visual_field_diagnostic.svg"] = _empty_svg(
            "Visual field diagnostic",
            "No prediction array was produced.",
        )
        image_plots["visual_field_diagnostic.png"] = _empty_png()
    else:
        y = _first_series(arrays[prediction_key])
        x = _first_series(arrays[x_key]) if x_key else np.arange(len(y), dtype=float)
        plots["visual_field_diagnostic.svg"] = _scatter_svg(
            x,
            y,
            title="Visual field diagnostic",
            x_label=x_key or "sample_index",
            y_label=prediction_key,
        )
        image_plots["visual_field_diagnostic.png"] = _scatter_png(x, y)
        artifacts.append(
            {
                "path": str((workspace / "visual_field_diagnostic.svg").relative_to(run_dir)),
                "kind": "prediction_field_proxy",
                "privacy_boundary": "prediction_only_no_validation_labels",
            }
        )
        artifacts.append(
            {
                "path": str((workspace / "visual_field_diagnostic.png").relative_to(run_dir)),
                "kind": "prediction_field_proxy_png",
                "privacy_boundary": "prediction_only_no_validation_labels",
            }
        )
        residual = _finite_difference_proxy(y)
        plots["visual_residual_diagnostic.svg"] = _scatter_svg(
            np.arange(len(residual), dtype=float),
            residual,
            title="Residual proxy diagnostic",
            x_label="sample_index",
            y_label="absolute_prediction_step",
        )
        image_plots["visual_residual_diagnostic.png"] = _scatter_png(
            np.arange(len(residual), dtype=float),
            residual,
        )
        artifacts.append(
            {
                "path": str((workspace / "visual_residual_diagnostic.svg").relative_to(run_dir)),
                "kind": "prediction_smoothness_proxy",
                "privacy_boundary": "prediction_only_no_validation_labels",
            }
        )
        artifacts.append(
            {
                "path": str((workspace / "visual_residual_diagnostic.png").relative_to(run_dir)),
                "kind": "prediction_smoothness_proxy_png",
                "privacy_boundary": "prediction_only_no_validation_labels",
            }
        )
        boundary = _boundary_proxy(y)
        plots["visual_boundary_diagnostic.svg"] = _scatter_svg(
            np.arange(len(boundary), dtype=float),
            boundary,
            title="Boundary proxy diagnostic",
            x_label="boundary_sample_index",
            y_label=prediction_key,
        )
        image_plots["visual_boundary_diagnostic.png"] = _scatter_png(
            np.arange(len(boundary), dtype=float),
            boundary,
        )
        artifacts.append(
            {
                "path": str((workspace / "visual_boundary_diagnostic.svg").relative_to(run_dir)),
                "kind": "prediction_boundary_proxy",
                "privacy_boundary": "prediction_only_no_validation_labels",
            }
        )
        artifacts.append(
            {
                "path": str((workspace / "visual_boundary_diagnostic.png").relative_to(run_dir)),
                "kind": "prediction_boundary_proxy_png",
                "privacy_boundary": "prediction_only_no_validation_labels",
            }
        )
    actual_image_inputs_used = False
    if mode == "real" and capabilities.get("supports_image_inputs") is True:
        analysis_mode = "real_visual_provider_not_invoked"
        warnings = [
            "visual_audit_mode=real was requested and the provider advertises image input support, "
            "but this local audit did not send image bytes to a real vision provider.",
        ]
    elif mode == "real":
        analysis_mode = "real_requested_provider_text_only"
        warnings = ["visual_audit_mode=real requires a provider with supports_image_inputs=true."]
    elif mode == "mock":
        analysis_mode = "mock_visual_audit_no_real_image_input"
        warnings = ["Mock visual audit generated deterministic artifacts only; no scientific claim is supported."]
    else:
        analysis_mode = "visual_audit_disabled"
        warnings = ["Visual audit is disabled for this run."]
    report = {
        "schema_version": 1,
        "solution_id": solution_id,
        "visual_audit_mode": mode,
        "analysis_mode": analysis_mode,
        "actual_image_inputs_used": actual_image_inputs_used,
        "provider_capabilities": capabilities,
        "privacy_boundary": "prediction_only_no_validation_labels",
        "visual_artifacts": artifacts,
        "physical_consistency_checks": _physical_consistency_checks(arrays, prediction_key),
        "warnings": warnings,
        "summary": (
            "Deterministic prediction-only visual audit artifacts were generated."
            if artifacts
            else "No prediction-only visual artifact could be generated."
        ),
        "claim_boundary": (
            "Visual audit artifacts inspect predictions and public run metadata only. They do not read private "
            "validation labels and do not support scientific claims without real image-provider evidence and "
            "domain review."
        ),
    }
    return report, plots, image_plots


DATA_EDA_SCRIPT = r'''from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Training-data-only EDA for AgenticSciML runs.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train_path = Path(args.train_data)
    if any(part.lower().startswith("val") for part in train_path.parts):
        raise SystemExit("Refusing to inspect private-label data path.")
    with np.load(train_path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    output = {
        "schema_version": 1,
        "privacy_boundary": "training_data_only_no_private_labels",
        "arrays": {name: summarize_array(value) for name, value in sorted(arrays.items())},
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


if __name__ == "__main__":
    raise SystemExit(main())
'''


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


def prompt_eda_summary(eda_output: dict[str, Any]) -> str:
    lines = [
        f"schema_version: {eda_output.get('schema_version')}",
        f"privacy_boundary: {eda_output.get('privacy_boundary')}",
        "array_checks:",
    ]
    for check in eda_output.get("array_checks", []):
        if not isinstance(check, dict):
            continue
        lines.append(
            "- "
            f"{check.get('array')}: status={check.get('status')} "
            f"notes={'; '.join(str(item) for item in check.get('notes', []))}"
        )
    lines.append("plot_checks:")
    for check in eda_output.get("plot_checks", []):
        if not isinstance(check, dict):
            continue
        lines.append(
            "- "
            f"{check.get('path')}: kind={check.get('kind')} "
            f"status={check.get('status')}"
        )
    lines.append("modeling_notes:")
    for note in eda_output.get("modeling_notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines)


def _eda_array_checks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, summary in manifest.get("arrays", {}).items():
        if not isinstance(summary, dict):
            continue
        notes: list[str] = []
        size = _int_or_zero(summary.get("size"))
        finite_count = _int_or_zero(summary.get("finite_count"))
        nan_count = _int_or_zero(summary.get("nan_count"))
        std = summary.get("std")
        min_value = summary.get("min")
        max_value = summary.get("max")
        shape = summary.get("shape")
        if nan_count:
            notes.append("contains non-finite values")
        if size and finite_count < size:
            notes.append("finite count is smaller than total size")
        if isinstance(std, (int, float)) and std == 0:
            notes.append("constant finite values")
        if isinstance(shape, list) and len(shape) > 2:
            notes.append("multi-dimensional tensor; downstream plots use flattened first channel")
        if (
            isinstance(min_value, (int, float))
            and isinstance(max_value, (int, float))
            and min_value != 0
            and abs(max_value / min_value) > 1000
        ):
            notes.append("large dynamic range")
        checks.append(
            {
                "array": name,
                "shape": shape,
                "dtype": summary.get("dtype"),
                "status": "warning" if notes else "ok",
                "notes": notes or ["no obvious numeric issue from summary stats"],
            }
        )
    return checks


def _eda_plot_checks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    plots = manifest.get("plots", [])
    if not isinstance(plots, list) or not plots:
        return [
            {
                "path": None,
                "kind": "none",
                "status": "warning",
                "notes": ["no training-data plot was generated"],
            }
        ]
    checks = []
    for plot in plots:
        if not isinstance(plot, dict):
            continue
        checks.append(
            {
                "path": plot.get("path"),
                "kind": plot.get("kind"),
                "status": "ok",
                "notes": [
                    f"x={plot.get('x_array')}",
                    f"y={plot.get('y_array')}",
                    "privacy boundary preserved",
                ],
            }
        )
    return checks


def _eda_modeling_notes(manifest: dict[str, Any]) -> list[str]:
    notes = ["Use these observations as modeling context only; evaluator contract remains authoritative."]
    arrays = manifest.get("arrays", {})
    if isinstance(arrays, dict):
        high_dimensional = [
            name
            for name, summary in arrays.items()
            if isinstance(summary, dict)
            and isinstance(summary.get("shape"), list)
            and len(summary["shape"]) > 2
        ]
        if high_dimensional:
            notes.append(
                "High-dimensional arrays detected: "
                + ", ".join(str(item) for item in sorted(high_dimensional)[:4])
            )
        target_like = [
            name
            for name in arrays
            if str(name).lower().startswith(("u", "y", "target"))
        ]
        if target_like:
            notes.append("Target-like training arrays available: " + ", ".join(sorted(target_like)[:4]))
    if manifest.get("plots"):
        notes.append("A training-only overview plot is available for qualitative pattern inspection.")
    return notes


def _task_specific_observations(
    spec: Any,
    array_names: list[str],
    manifest: dict[str, Any],
    eda_output: dict[str, Any],
) -> list[str]:
    benchmark_name = spec.name if spec else str(manifest.get("benchmark_name", "unknown"))
    family = spec.family if spec else "unknown"
    observations = [
        f"{benchmark_name} is treated as a {family} benchmark with metric {spec.metric if spec else 'unknown'}.",
        "Training observation uses array keys: " + (", ".join(array_names) if array_names else "none"),
    ]
    lowered_keys = " ".join(array_names).lower()
    lowered_name = benchmark_name.lower()
    if "function" in lowered_name:
        observations.append("Function approximation context: inspect x/u train pairs for discontinuity or oscillation.")
    if "poisson" in lowered_name:
        observations.append("Poisson PINN context: residual and boundary-related coordinates should shape loss design.")
    if "burgers" in lowered_name:
        observations.append("Burgers PINN context: time/space coordinates and shock-like dynamics require stability checks.")
    if "operator" in lowered_name:
        observations.append("Operator-learning context: branch/trunk or input-function structure matters more than scalar fitting.")
    if "reaction" in lowered_name or "diffusion" in lowered_name:
        observations.append("Reaction-diffusion context: multi-input fields and temporal response arrays require shape-aware models.")
    if "cylinder" in lowered_name or "wake" in lowered_name:
        observations.append("Cylinder wake context: sparse sensor history and field reconstruction should preserve spatial layout.")
    if "x_train" in lowered_keys and ("u_train" in lowered_keys or "y_train" in lowered_keys):
        observations.append("Supervised train arrays include feature and target-like keys; validation labels remain private.")
    if isinstance(eda_output.get("array_checks"), list):
        warnings = [
            str(check.get("array"))
            for check in eda_output["array_checks"]
            if isinstance(check, dict) and check.get("status") == "warning"
        ]
        if warnings:
            observations.append("EDA warnings are associated with arrays: " + ", ".join(warnings[:6]))
    return observations


def _structured_modeling_implications(
    spec: Any,
    array_names: list[str],
    eda_output: dict[str, Any],
) -> list[str]:
    notes = list(eda_output.get("modeling_notes", [])) if isinstance(eda_output.get("modeling_notes"), list) else []
    family = (spec.family if spec else "").lower()
    if "pinn" in family:
        notes.append("Favor physics-informed residual or boundary-aware mutations when the local contract exposes those arrays.")
    elif "operator" in family:
        notes.append("Favor architectures that respect function-to-function mapping and array dimensionality.")
    elif "inverse" in family:
        notes.append("Favor reconstruction strategies that preserve sensor-to-field geometry.")
    else:
        notes.append("Favor deterministic baselines that match observed feature/target shapes before adding complexity.")
    if len(array_names) > 4:
        notes.append("Multiple training arrays are present; generated code should name required keys explicitly.")
    return [str(note) for note in notes]


def _structured_analysis_risks(
    spec: Any,
    manifest: dict[str, Any],
    eda_output: dict[str, Any],
) -> list[str]:
    risks = ["Private validation labels are not visible to data analysis or generated solution code."]
    if spec and spec.fidelity_level != "paper-like":
        risks.append(f"{spec.fidelity_level} evidence is not paper-score reproduction evidence.")
    if not manifest.get("plots"):
        risks.append("No training-data plot was generated, so qualitative pattern inspection is limited.")
    if isinstance(eda_output.get("array_checks"), list) and any(
        isinstance(check, dict) and check.get("status") == "warning"
        for check in eda_output["array_checks"]
    ):
        risks.append("At least one training array has EDA warnings that may affect modeling choices.")
    return risks


def _doc_excerpt(path: Path, *, fallback: str = "", limit: int = 420) -> str:
    if not path.exists():
        return fallback
    text = " ".join(path.read_text(encoding="utf-8").split())
    return text[:limit].rstrip() if text else fallback


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _markdown_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "- None recorded."
    return "\n".join(f"- {item}" for item in value)


def _int_or_zero(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


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


def _finite_difference_proxy(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size <= 1:
        return np.asarray([0.0], dtype=float)
    diffs = np.diff(series)
    return np.abs(diffs[np.isfinite(diffs)]) if np.isfinite(diffs).any() else np.asarray([0.0])


def _boundary_proxy(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size <= 8:
        return series
    edge_count = min(16, max(4, series.size // 20))
    return np.concatenate([series[:edge_count], series[-edge_count:]])


def _physical_consistency_checks(
    arrays: dict[str, np.ndarray],
    prediction_key: str | None,
) -> list[str]:
    checks = ["private_validation_labels_not_loaded"]
    if prediction_key is None:
        checks.append("prediction_array_missing")
        return checks
    summary = _array_summary(arrays[prediction_key])
    checks.append(f"prediction_finite_count={summary['finite_count']}")
    checks.append(f"prediction_nan_count={summary['nan_count']}")
    checks.append("field_residual_boundary_proxy_artifacts_prediction_only")
    return checks


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


def _scatter_png(x: np.ndarray, y: np.ndarray) -> bytes:
    x_values, y_values = _finite_pairs(x, y)
    if x_values.size == 0:
        return _empty_png()
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

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_horizontal_line(image, top + plot_h, left, left + plot_w, (17, 24, 39))
    _draw_vertical_line(image, left, top, top + plot_h, (17, 24, 39))
    for x_item, y_item in zip(x_values, y_values):
        px = left + ((float(x_item) - x_min) / (x_max - x_min)) * plot_w
        py = top + (1.0 - ((float(y_item) - y_min) / (y_max - y_min))) * plot_h
        _draw_disc(image, int(round(px)), int(round(py)), radius=2, color=(37, 99, 235))
    return _png_bytes(image)


def _empty_png() -> bytes:
    image = np.full((240, 640, 3), 255, dtype=np.uint8)
    _draw_horizontal_line(image, 120, 64, 576, (209, 213, 219))
    _draw_vertical_line(image, 64, 64, 196, (209, 213, 219))
    _draw_disc(image, 64, 120, radius=3, color=(156, 163, 175))
    return _png_bytes(image)


def _draw_disc(
    image: np.ndarray,
    cx: int,
    cy: int,
    *,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    for y_pos in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x_pos in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x_pos - cx) ** 2 + (y_pos - cy) ** 2 <= radius**2:
                image[y_pos, x_pos] = color


def _draw_horizontal_line(
    image: np.ndarray,
    y_pos: int,
    x_start: int,
    x_end: int,
    color: tuple[int, int, int],
) -> None:
    if y_pos < 0 or y_pos >= image.shape[0]:
        return
    low = max(0, min(x_start, x_end))
    high = min(image.shape[1], max(x_start, x_end) + 1)
    image[y_pos, low:high] = color


def _draw_vertical_line(
    image: np.ndarray,
    x_pos: int,
    y_start: int,
    y_end: int,
    color: tuple[int, int, int],
) -> None:
    if x_pos < 0 or x_pos >= image.shape[1]:
        return
    low = max(0, min(y_start, y_end))
    high = min(image.shape[0], max(y_start, y_end) + 1)
    image[low:high, x_pos] = color


def _png_bytes(rgb: np.ndarray) -> bytes:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG encoder expects an RGB image array")
    height, width = rgb.shape[:2]
    rows = [b"\x00" + np.ascontiguousarray(row, dtype=np.uint8).tobytes() for row in rgb]
    raw = b"".join(rows)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw)),
            _png_chunk(b"IEND", b""),
        ]
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


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
