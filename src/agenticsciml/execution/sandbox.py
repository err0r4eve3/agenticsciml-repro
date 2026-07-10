from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from agenticsciml.config import EvaluationContract
from agenticsciml.execution.runner import RunResult, run_command


BENCHMARK_FILES = [
    "Problem.md",
    "Requirements.md",
    "Evaluation.md",
    "Data_config.json",
    "guidelines.md",
]

GUARDED_FILES = (
    "evaluate.py",
    "guidelines.md",
    "Evaluation.md",
    "Data_config.json",
)

BLOCKED_IMPORT_ROOTS = {
    "generate_data",
    "ftplib",
    "http",
    "paramiko",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}

BLOCKED_CALLS = {
    "os.remove",
    "os.rmdir",
    "os.chmod",
    "os.system",
    "os.symlink",
    "os.unlink",
    "Path.chmod",
    "Path.hardlink_to",
    "Path.symlink_to",
    "Path.unlink",
    "pathlib.Path.chmod",
    "pathlib.Path.hardlink_to",
    "pathlib.Path.symlink_to",
    "pathlib.Path.unlink",
    "shutil.rmtree",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
}


def _normalize_command(command: list[str]) -> list[str]:
    if command and command[0] == "python":
        return [sys.executable, *command[1:]]
    return command


def private_eval_dir_for_workspace(workspace: Path) -> Path:
    if workspace.parent.name == "solutions":
        return workspace.parent.parent / "private_eval" / workspace.name
    return workspace.parent / "private_eval" / workspace.name


def _normalized_data_artifact_path(value: Any, *, default: str, field_name: str) -> str:
    candidate = default if value is None else value
    if not isinstance(candidate, str) or not candidate.strip():
        raise RuntimeError(f"Data_config {field_name} must be a non-empty relative path")
    path = Path(candidate)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise RuntimeError(f"Data_config {field_name} must stay inside the benchmark directory: {candidate}")
    normalized = path.as_posix()
    if Path(normalized).suffix != ".npz":
        raise RuntimeError(f"Data_config {field_name} must reference an .npz artifact: {candidate}")
    return normalized


def _benchmark_data_paths(benchmark_dir: Path) -> tuple[str, str]:
    config_path = benchmark_dir / "Data_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load benchmark Data_config.json: {config_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Benchmark Data_config.json must contain a JSON object: {config_path}")
    train_path = _normalized_data_artifact_path(
        payload.get("train_path"),
        default="train_data.npz",
        field_name="train_path",
    )
    validation_path = _normalized_data_artifact_path(
        payload.get("validation_path"),
        default="val_data.npz",
        field_name="validation_path",
    )
    if train_path == validation_path:
        raise RuntimeError("Data_config train_path and validation_path must be different")
    return train_path, validation_path


def _benchmark_data_artifact_path(benchmark_dir: Path, relative_path: str) -> Path:
    benchmark_root = benchmark_dir.resolve()
    candidate = benchmark_dir / relative_path
    if not candidate.resolve(strict=False).is_relative_to(benchmark_root):
        raise RuntimeError(
            "Data_config artifact resolves outside the benchmark directory: "
            f"{relative_path}"
        )
    return candidate


def _validate_training_npz(path: Path) -> None:
    data: Any = None
    try:
        data = np.load(path, allow_pickle=False)
        files = getattr(data, "files", None)
        if not isinstance(files, list):
            raise ValueError("artifact is not an NPZ archive")
        leaked_keys = sorted(
            key
            for key in files
            if {part for part in key.lower().replace("-", "_").split("_") if part}
            & {"val", "validation"}
        )
    except (EOFError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Training data artifact is not a readable NPZ file: {path}") from exc
    finally:
        close = getattr(data, "close", None)
        if callable(close):
            close()
    if leaked_keys:
        raise RuntimeError(
            "Training data artifact must not contain validation arrays or labels: "
            + ", ".join(leaked_keys)
        )


def prepare_run_inputs(benchmark_dir: Path, run_inputs_dir: Path) -> dict[str, Any]:
    """Create or verify the immutable run-level benchmark input snapshot."""
    return _ensure_run_inputs(benchmark_dir.resolve(), run_inputs_dir)


def prepare_solution_workspace(
    benchmark_dir: Path,
    workspace: Path,
    *,
    run_inputs_dir: Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = benchmark_dir.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    old_evaluator_dir = workspace / ".evaluator"
    if old_evaluator_dir.exists():
        shutil.rmtree(old_evaluator_dir)
    old_private_eval_dir = private_eval_dir_for_workspace(workspace)
    if run_inputs_dir is not None and old_private_eval_dir.exists():
        shutil.rmtree(old_private_eval_dir)
    if run_inputs_dir is not None:
        return _prepare_solution_workspace_from_run_inputs(benchmark_dir, workspace, run_inputs_dir)
    train_path, validation_path = _benchmark_data_paths(benchmark_dir)
    evaluator_dir = private_eval_dir_for_workspace(workspace)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    for filename in BENCHMARK_FILES:
        source = benchmark_dir / filename
        if source.exists():
            shutil.copy2(source, workspace / filename)
    evaluate_source = benchmark_dir / "evaluate.py"
    if evaluate_source.exists():
        shutil.copy2(evaluate_source, evaluator_dir / "evaluate.py")
    train_source = _benchmark_data_artifact_path(benchmark_dir, train_path)
    validation_source = _benchmark_data_artifact_path(benchmark_dir, validation_path)
    if train_source.exists() != validation_source.exists():
        existing = train_source if train_source.exists() else validation_source
        missing = validation_source if train_source.exists() else train_source
        raise RuntimeError(
            f"Partial benchmark data artifacts are not allowed: found {existing}, missing {missing}"
        )
    if train_source.exists() and validation_source.exists():
        (workspace / train_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_source, workspace / train_path)
        (evaluator_dir / validation_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(validation_source, evaluator_dir / validation_path)
    else:
        result = run_command(
            workspace,
            [
                sys.executable,
                str(benchmark_dir / "generate_data.py"),
                "--seed",
                "0",
                "--output-dir",
                ".",
            ],
            timeout_s=20,
        )
        if result.exit_code != 0:
            raise RuntimeError(result.combined_output)
        generated_validation = workspace / validation_path
        if generated_validation.exists():
            (evaluator_dir / validation_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated_validation), evaluator_dir / validation_path)
    if not (workspace / train_path).exists() or not (evaluator_dir / validation_path).exists():
        raise RuntimeError(
            "Benchmark data generator did not provide configured artifacts: "
            f"{train_path}, {validation_path}"
        )
    _validate_training_npz(workspace / train_path)
    if (workspace / validation_path).exists():
        (workspace / validation_path).unlink()
    return {
        "schema_version": 1,
        "layout": "per_solution_legacy",
        "storage_dedup_fallback": "copy",
        "public_dir": None,
        "private_eval_dir": str(evaluator_dir),
        "train_path": train_path,
        "validation_path": validation_path,
    }


def _prepare_solution_workspace_from_run_inputs(
    benchmark_dir: Path,
    workspace: Path,
    run_inputs_dir: Path,
) -> dict[str, Any]:
    manifest = _ensure_run_inputs(benchmark_dir, run_inputs_dir)
    public_dir = run_inputs_dir / "public"
    train_path = str(manifest["train_path"])
    validation_path = str(manifest["validation_path"])
    fallback_modes: list[str] = []
    for filename in [*BENCHMARK_FILES, train_path]:
        source = public_dir / filename
        if not source.exists():
            continue
        target = workspace / filename
        fallback_modes.append(_link_or_copy_public_input(source, target))
    layout = {
        **manifest,
        "solution_workspace": str(workspace),
        "solution_public_input_mode": "copy" if "copy" in fallback_modes else "symlink",
    }
    if layout["solution_public_input_mode"] == "copy":
        layout["storage_dedup_fallback"] = "copy"
    if not (workspace / train_path).exists():
        raise RuntimeError(f"Run-level public inputs did not provide {train_path}")
    if (workspace / validation_path).exists():
        (workspace / validation_path).unlink()
    return layout


def _ensure_run_inputs(benchmark_dir: Path, run_inputs_dir: Path) -> dict[str, Any]:
    manifest_path = run_inputs_dir / "manifest.json"
    public_dir = run_inputs_dir / "public"
    private_dir = run_inputs_dir / "private_eval"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        return _load_and_verify_run_inputs_manifest(run_inputs_dir)
    if any(public_dir.iterdir()) or any(private_dir.iterdir()):
        raise RuntimeError(
            "Run-level inputs are incomplete: files exist without a frozen manifest.json"
        )

    train_path, validation_path = _benchmark_data_paths(benchmark_dir)
    for filename in BENCHMARK_FILES:
        source = benchmark_dir / filename
        if source.exists():
            _copy_frozen_input(source, public_dir / filename)
    evaluate_source = benchmark_dir / "evaluate.py"
    if evaluate_source.exists():
        _copy_frozen_input(evaluate_source, private_dir / "evaluate.py")

    train_source = _benchmark_data_artifact_path(benchmark_dir, train_path)
    validation_source = _benchmark_data_artifact_path(benchmark_dir, validation_path)
    if train_source.exists() and validation_source.exists():
        _validate_training_npz(train_source)
        _copy_frozen_input(train_source, public_dir / train_path)
        _copy_frozen_input(validation_source, private_dir / validation_path)
    elif train_source.exists() != validation_source.exists():
        existing = train_source if train_source.exists() else validation_source
        missing = validation_source if train_source.exists() else train_source
        raise RuntimeError(f"Partial benchmark data artifacts are not allowed: found {existing}, missing {missing}")
    else:
        with tempfile.TemporaryDirectory(prefix="agenticsciml-run-inputs-", dir=run_inputs_dir) as tmp:
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
                raise RuntimeError(result.combined_output)
            generated_train = _benchmark_data_artifact_path(tmp_path, train_path)
            generated_validation = _benchmark_data_artifact_path(tmp_path, validation_path)
            if not generated_train.exists() or not generated_validation.exists():
                raise RuntimeError(
                    "Benchmark data generator did not provide configured artifacts: "
                    f"{train_path}, {validation_path}"
                )
            _validate_training_npz(generated_train)
            _copy_frozen_input(generated_train, public_dir / train_path)
            _copy_frozen_input(generated_validation, private_dir / validation_path)

    _validate_training_npz(public_dir / train_path)
    file_digests = _run_input_file_digests(run_inputs_dir)
    manifest = {
        "schema_version": 1,
        "layout": "run_level_inputs_v1",
        "public_dir": "run_inputs/public",
        "private_eval_dir": "run_inputs/private_eval",
        "train_path": train_path,
        "validation_path": validation_path,
        "public_files": _relative_file_list(public_dir),
        "private_eval_files": _relative_file_list(private_dir),
        "file_digests": file_digests,
        "storage_dedup_fallback": "symlink",
        "private_label_boundary": (
            "Generated solution workspaces receive public inputs only. evaluate.py and the configured "
            "validation artifact remain in run_inputs/private_eval and are used only by the trusted "
            "evaluator phase."
        ),
    }
    manifest["manifest_digest"] = _run_inputs_manifest_digest(manifest)
    _write_run_inputs_manifest(run_inputs_dir, manifest)
    return _load_and_verify_run_inputs_manifest(run_inputs_dir)


def _load_and_verify_run_inputs_manifest(run_inputs_dir: Path) -> dict[str, Any]:
    path = run_inputs_dir / "manifest.json"
    if path.is_symlink():
        raise RuntimeError(f"Frozen run-input manifest must not be a symlink: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load frozen run-input manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Frozen run-input manifest must be a JSON object: {path}")
    if manifest.get("schema_version") != 1 or manifest.get("layout") != "run_level_inputs_v1":
        raise RuntimeError("Frozen run-input manifest has an unsupported schema or layout")
    manifest_digest = manifest.get("manifest_digest")
    if not isinstance(manifest_digest, str) or manifest_digest != _run_inputs_manifest_digest(manifest):
        raise RuntimeError("Frozen run-input manifest digest mismatch")

    train_path = _normalized_data_artifact_path(
        manifest.get("train_path"),
        default="train_data.npz",
        field_name="train_path",
    )
    validation_path = _normalized_data_artifact_path(
        manifest.get("validation_path"),
        default="val_data.npz",
        field_name="validation_path",
    )
    if train_path == validation_path:
        raise RuntimeError("Frozen run-input train_path and validation_path must be different")

    public_files = manifest.get("public_files")
    private_files = manifest.get("private_eval_files")
    file_digests = manifest.get("file_digests")
    if (
        not isinstance(public_files, list)
        or not all(isinstance(item, str) for item in public_files)
        or not isinstance(private_files, list)
        or not all(isinstance(item, str) for item in private_files)
        or not isinstance(file_digests, dict)
        or not all(isinstance(name, str) and isinstance(digest, str) for name, digest in file_digests.items())
    ):
        raise RuntimeError("Frozen run-input manifest has invalid file inventory metadata")

    actual_public_files = _relative_file_list(run_inputs_dir / "public")
    actual_private_files = _relative_file_list(run_inputs_dir / "private_eval")
    if public_files != actual_public_files or private_files != actual_private_files:
        raise RuntimeError("Frozen run-input file inventory mismatch")
    actual_digests = _run_input_file_digests(run_inputs_dir)
    if file_digests != actual_digests:
        changed = sorted(set(file_digests) | set(actual_digests))
        mismatched = [name for name in changed if file_digests.get(name) != actual_digests.get(name)]
        raise RuntimeError("Frozen run-input digest mismatch: " + ", ".join(mismatched))

    public_train = run_inputs_dir / "public" / train_path
    private_validation = run_inputs_dir / "private_eval" / validation_path
    if not public_train.is_file() or not private_validation.is_file():
        raise RuntimeError("Frozen run-input manifest does not provide configured data artifacts")
    if not (run_inputs_dir / "private_eval" / "evaluate.py").is_file():
        raise RuntimeError("Frozen run-input manifest does not provide private evaluate.py")
    _validate_training_npz(public_train)
    return manifest


def _relative_file_list(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Frozen run inputs must not contain symlinks: {path}")
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def _run_input_file_digests(run_inputs_dir: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for dirname in ("public", "private_eval"):
        root = run_inputs_dir / dirname
        for path in root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(run_inputs_dir).as_posix()
                digest = _file_digest(path)
                if digest is None:
                    raise RuntimeError(f"Run input disappeared while hashing: {relative}")
                digests[relative] = digest
    return dict(sorted(digests.items()))


def _run_inputs_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_run_inputs_manifest(run_inputs_dir: Path, manifest: dict[str, Any]) -> None:
    path = run_inputs_dir / "manifest.json"
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
            tmp.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _link_or_copy_public_input(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        relative_source = os.path.relpath(source, start=target.parent)
        os.symlink(relative_source, target)
        return "symlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _copy_frozen_input(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Refusing to overwrite frozen run input: {target}")
    shutil.copy2(source, target)
    _mark_read_only(target)


def _mark_read_only(path: Path) -> None:
    try:
        path.chmod(0o444)
    except OSError:
        pass


def _file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guarded_fingerprints(workspace: Path, evaluator_dir: Path | None = None) -> dict[str, str | None]:
    fingerprints = {filename: _file_digest(workspace / filename) for filename in GUARDED_FILES}
    evaluator_dir = evaluator_dir or private_eval_dir_for_workspace(workspace)
    if evaluator_dir.exists():
        for path in evaluator_dir.rglob("*"):
            if path.is_file():
                relative = path.relative_to(evaluator_dir).as_posix()
                fingerprints[f"private_eval/{relative}"] = _file_digest(path)
    return fingerprints


def _guardrail_violation(
    workspace: Path,
    before: dict[str, str | None],
    evaluator_dir: Path | None = None,
) -> str | None:
    after = _guarded_fingerprints(workspace, evaluator_dir=evaluator_dir)
    changed = [filename for filename, digest in before.items() if after.get(filename) != digest]
    if changed:
        return "Guardrail violation: generated solution modified guarded evaluator file(s): " + ", ".join(changed)
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _static_solution_guardrail_violations(
    solution_path: Path,
    *,
    validation_path: str = "val_data.npz",
) -> list[str]:
    if not solution_path.exists():
        return [f"missing generated solution file: {solution_path.name}"]
    try:
        tree = ast.parse(solution_path.read_text(encoding="utf-8"), filename=str(solution_path))
    except SyntaxError as exc:
        return [f"solution.py syntax error: {exc}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BLOCKED_IMPORT_ROOTS:
                    violations.append(f"blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in BLOCKED_IMPORT_ROOTS:
                violations.append(f"blocked import: {node.module}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in BLOCKED_CALLS:
                violations.append(f"blocked call: {call_name}")
            if call_name in {"open", "Path", "pathlib.Path", "glob", "glob.glob"} and node.args:
                path_value = _constant_string(node.args[0])
                if path_value:
                    if Path(path_value).is_absolute():
                        violations.append(f"blocked absolute path: {path_value}")
                    if path_value == ".." or path_value.startswith("../") or "/../" in path_value:
                        violations.append(f"blocked parent traversal path: {path_value}")
            if call_name in {"Path.home", "Path.expanduser", "os.path.expanduser"}:
                violations.append(f"blocked home-directory path helper: {call_name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if (
                "val_data.npz" in node.value
                or validation_path in node.value
                or ".evaluator" in node.value
                or "private_eval" in node.value
            ):
                violations.append("blocked validation data reference")
            if "AGENTICSCIML_VALIDATION_DATA" in node.value:
                violations.append("blocked validation data environment reference")
            if "generate_data.py" in node.value:
                violations.append("blocked benchmark generator source reference")
    return sorted(set(violations))


def _validation_exposure(workspace: Path, *, validation_path: str = "val_data.npz") -> str | None:
    if (workspace / ".evaluator").exists():
        return "Guardrail violation: evaluator-private directory is exposed during generated solution execution."
    configured_validation = workspace / validation_path
    if configured_validation.exists() or configured_validation.is_symlink():
        return "Guardrail violation: validation data is exposed during generated solution execution."
    exposed = [
        path
        for path in workspace.rglob("*.npz")
        if path.name == "val_data.npz"
    ]
    if exposed:
        return "Guardrail violation: validation data is exposed during generated solution execution."
    return None


def _static_guardrail_result(
    workspace: Path,
    *,
    validation_path: str = "val_data.npz",
) -> RunResult | None:
    violations = _static_solution_guardrail_violations(
        workspace / "solution.py",
        validation_path=validation_path,
    )
    if not violations:
        return None
    message = "Guardrail violation: generated solution failed static sandbox checks: " + "; ".join(violations)
    _write_train_log(workspace, [message])
    return RunResult(
        command=["static_guardrail", "solution.py"],
        exit_code=125,
        stdout="",
        stderr=message,
        duration_s=0.0,
    )


def _prepare_prediction_input(workspace: Path, validation_path: Path) -> Path:
    data = np.load(validation_path)
    if "x_val" not in data:
        raise ValueError("Validation data must contain x_val for prediction-only evaluation.")
    predict_input = workspace / "predict_input.npz"
    np.savez(predict_input, x_val=data["x_val"])
    return predict_input


def _contract_validation_path(contract: EvaluationContract) -> str:
    candidates: list[str] = []
    for entry in contract.evaluator_only_files:
        normalized = entry.replace("\\", "/")
        if not normalized.endswith(".npz"):
            continue
        for marker in ("run_inputs/private_eval/", "private_eval/{solution_id}/"):
            if marker in normalized:
                normalized = normalized.split(marker, 1)[1]
                break
        candidates.append(
            _normalized_data_artifact_path(
                normalized,
                default="val_data.npz",
                field_name="validation_path",
            )
        )
    unique = sorted(set(candidates))
    if not unique:
        return "val_data.npz"
    if len(unique) != 1:
        raise RuntimeError(
            "EvaluationContract must identify exactly one private validation NPZ artifact: "
            + ", ".join(unique)
        )
    return unique[0]


def _prediction_output_path(command: list[str]) -> Path:
    if "--output" in command:
        index = command.index("--output")
        if index + 1 < len(command):
            return Path(command[index + 1])
    return Path("predictions.npz")


def _missing_prediction_output(workspace: Path, command: list[str]) -> str | None:
    output_path = _prediction_output_path(command)
    if output_path.is_absolute():
        return f"Guardrail violation: prediction output path must be relative: {output_path}"
    if not (workspace / output_path).exists():
        return f"Prediction output was not created: {output_path}"
    return None


def _training_data_integrity_violation(phase: str, result: RunResult) -> str | None:
    if phase not in {"validate", "train"} or result.exit_code != 0:
        return None
    text = f"{result.stdout}\n{result.stderr}".lower()
    markers = (
        "error loading data",
        "falling back to synthetic",
        "synthetic training data",
        "cannot infer features/targets",
    )
    if any(marker in text for marker in markers):
        return (
            "Guardrail violation: generated solution reported a training-data "
            f"loading failure or synthetic-data fallback during {phase}."
        )
    return None


def _write_train_log(workspace: Path, stdout_parts: list[str], stderr_parts: list[str] | None = None) -> str:
    parts = [part for part in [*stdout_parts, *(stderr_parts or [])] if part]
    log = "\n".join(parts)
    (workspace / "train.log").write_text(log, encoding="utf-8")
    return log


def train_and_evaluate(
    workspace: Path,
    contract: EvaluationContract,
    timeout_s: int,
    private_eval_dir: Path | None = None,
) -> RunResult:
    try:
        validation_relative_path = _contract_validation_path(contract)
    except RuntimeError as exc:
        message = f"Guardrail violation: {exc}"
        _write_train_log(workspace, [message])
        return RunResult(
            command=["contract_guardrail"],
            exit_code=125,
            stdout="",
            stderr=message,
            duration_s=0.0,
        )
    static_result = _static_guardrail_result(
        workspace,
        validation_path=validation_relative_path,
    )
    if static_result is not None:
        return static_result

    all_stdout: list[str] = []
    all_stderr: list[str] = []
    total_duration = 0.0
    last_command: list[str] = []

    commands = [
        ("validate", contract.validate_command),
        ("train", contract.train_command),
        ("predict", contract.predict_command),
        ("evaluate", contract.evaluate_command),
    ]
    evaluator_dir = private_eval_dir or private_eval_dir_for_workspace(workspace)
    validation_path = evaluator_dir / validation_relative_path

    for phase, command in commands:
        if phase in {"validate", "train"}:
            message = _validation_exposure(
                workspace,
                validation_path=validation_relative_path,
            )
            if message:
                all_stderr.append(message)
                _write_train_log(workspace, all_stdout, all_stderr)
                return RunResult(
                    command=command,
                    exit_code=125,
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                    duration_s=total_duration,
                )
        if phase == "evaluate" and not validation_path.exists():
            message = "Guardrail violation: evaluator-private validation data is missing."
            all_stderr.append(message)
            _write_train_log(workspace, all_stdout, all_stderr)
            return RunResult(
                command=command,
                exit_code=125,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=total_duration,
            )
        if phase == "predict":
            try:
                _prepare_prediction_input(workspace, validation_path)
            except Exception as exc:
                message = f"Guardrail violation: could not prepare prediction input: {exc}"
                all_stderr.append(message)
                _write_train_log(workspace, all_stdout, all_stderr)
                return RunResult(
                    command=command,
                    exit_code=125,
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                    duration_s=total_duration,
                )
        guarded_before = _guarded_fingerprints(workspace, evaluator_dir=evaluator_dir)
        normalized = _phase_command(phase, command, evaluator_dir)
        last_command = normalized
        env = None
        if phase == "evaluate":
            env = {"AGENTICSCIML_VALIDATION_DATA": validation_path}
        result = run_command(workspace, normalized, timeout_s=timeout_s, env=env)
        total_duration += result.duration_s
        all_stdout.append(result.combined_output)
        if result.stderr:
            all_stderr.append(result.stderr)
        if result.exit_code != 0:
            _write_train_log(workspace, all_stdout, all_stderr)
            return RunResult(
                command=normalized,
                exit_code=result.exit_code,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=total_duration,
                timed_out=result.timed_out,
            )
        data_integrity_violation = _training_data_integrity_violation(phase, result)
        if data_integrity_violation:
            all_stderr.append(data_integrity_violation)
            _write_train_log(workspace, all_stdout, all_stderr)
            return RunResult(
                command=normalized,
                exit_code=125,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=total_duration,
            )
        if phase == "predict":
            missing_prediction = _missing_prediction_output(workspace, normalized)
            if missing_prediction:
                all_stderr.append(missing_prediction)
                _write_train_log(workspace, all_stdout, all_stderr)
                return RunResult(
                    command=normalized,
                    exit_code=125,
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                    duration_s=total_duration,
                )
        violation = _guardrail_violation(workspace, guarded_before, evaluator_dir=evaluator_dir)
        if violation:
            all_stderr.append(violation)
            _write_train_log(workspace, all_stdout, all_stderr)
            return RunResult(
                command=normalized,
                exit_code=125,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=total_duration,
            )

    log = "\n".join(all_stdout)
    _write_train_log(workspace, all_stdout, all_stderr)
    return RunResult(
        command=last_command,
        exit_code=0,
        stdout=log,
        stderr="\n".join(all_stderr),
        duration_s=total_duration,
    )


def _phase_command(phase: str, command: list[str], evaluator_dir: Path) -> list[str]:
    normalized = _normalize_command(command)
    if phase != "evaluate" or len(normalized) < 2:
        return normalized
    script = normalized[1]
    if script in {"evaluate.py", ".evaluator/evaluate.py", "<private_eval>/evaluate.py"}:
        return [normalized[0], str(evaluator_dir / "evaluate.py"), *normalized[2:]]
    return normalized
