from __future__ import annotations

import ast
import hashlib
import shutil
import sys
from pathlib import Path

from agenticsciml.config import EvaluationContract
from agenticsciml.execution.runner import RunResult, run_command


BENCHMARK_FILES = [
    "Problem.md",
    "Requirements.md",
    "Evaluation.md",
    "Data_config.json",
    "generate_data.py",
    "guidelines.md",
]

GUARDED_FILES = (
    "evaluate.py",
    ".evaluator/evaluate.py",
    ".evaluator/val_data.npz",
    "guidelines.md",
    "Evaluation.md",
    "Data_config.json",
)

BLOCKED_IMPORT_ROOTS = {
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
    "os.system",
    "os.unlink",
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


def prepare_solution_workspace(benchmark_dir: Path, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    evaluator_dir = workspace / ".evaluator"
    evaluator_dir.mkdir(exist_ok=True)
    for filename in BENCHMARK_FILES:
        source = benchmark_dir / filename
        if source.exists():
            shutil.copy2(source, workspace / filename)
    evaluate_source = benchmark_dir / "evaluate.py"
    if evaluate_source.exists():
        shutil.copy2(evaluate_source, evaluator_dir / "evaluate.py")
    train_source = benchmark_dir / "train_data.npz"
    if train_source.exists():
        shutil.copy2(train_source, workspace / "train_data.npz")
    validation_source = benchmark_dir / "val_data.npz"
    if validation_source.exists():
        shutil.copy2(validation_source, evaluator_dir / "val_data.npz")
    if not (workspace / "train_data.npz").exists() or not (evaluator_dir / "val_data.npz").exists():
        result = run_command(
            workspace,
            [sys.executable, "generate_data.py", "--seed", "0", "--output-dir", "."],
            timeout_s=20,
        )
        if result.exit_code != 0:
            raise RuntimeError(result.combined_output)
        generated_validation = workspace / "val_data.npz"
        if generated_validation.exists():
            shutil.move(str(generated_validation), evaluator_dir / "val_data.npz")
    if (workspace / "val_data.npz").exists():
        (workspace / "val_data.npz").unlink()


def _file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guarded_fingerprints(workspace: Path) -> dict[str, str | None]:
    return {filename: _file_digest(workspace / filename) for filename in GUARDED_FILES}


def _guardrail_violation(
    workspace: Path,
    before: dict[str, str | None],
) -> str | None:
    after = _guarded_fingerprints(workspace)
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


def _static_solution_guardrail_violations(solution_path: Path) -> list[str]:
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
            if call_name in {"open", "Path", "pathlib.Path"} and node.args:
                path_value = _constant_string(node.args[0])
                if path_value and Path(path_value).is_absolute():
                    violations.append(f"blocked absolute path: {path_value}")
            if call_name in {"Path.home", "Path.expanduser", "os.path.expanduser"}:
                violations.append(f"blocked home-directory path helper: {call_name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "val_data.npz" in node.value or ".evaluator" in node.value:
                violations.append("blocked validation data reference")
            if "AGENTICSCIML_VALIDATION_DATA" in node.value:
                violations.append("blocked validation data environment reference")
    return sorted(set(violations))


def _static_guardrail_result(workspace: Path) -> RunResult | None:
    violations = _static_solution_guardrail_violations(workspace / "solution.py")
    if not violations:
        return None
    message = "Guardrail violation: generated solution failed static sandbox checks: " + "; ".join(violations)
    (workspace / "train.log").write_text(message + "\n", encoding="utf-8")
    return RunResult(
        command=["static_guardrail", "solution.py"],
        exit_code=125,
        stdout="",
        stderr=message,
        duration_s=0.0,
    )


def train_and_evaluate(
    workspace: Path,
    contract: EvaluationContract,
    timeout_s: int,
) -> RunResult:
    static_result = _static_guardrail_result(workspace)
    if static_result is not None:
        return static_result

    all_stdout: list[str] = []
    all_stderr: list[str] = []
    total_duration = 0.0
    last_command: list[str] = []

    commands = [
        ("validate", contract.validate_command),
        ("train", contract.train_command),
        ("evaluate", contract.evaluate_command),
    ]
    validation_path = workspace / ".evaluator" / "val_data.npz"

    for phase, command in commands:
        if phase in {"validate", "train"} and (workspace / "val_data.npz").exists():
            message = "Guardrail violation: validation data is exposed during generated solution execution."
            all_stderr.append(message)
            (workspace / "train.log").write_text("\n".join(all_stdout), encoding="utf-8")
            return RunResult(
                command=command,
                exit_code=125,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=total_duration,
            )
        guarded_before = _guarded_fingerprints(workspace)
        normalized = _normalize_command(command)
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
            (workspace / "train.log").write_text("\n".join(all_stdout), encoding="utf-8")
            return RunResult(
                command=normalized,
                exit_code=result.exit_code,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=total_duration,
                timed_out=result.timed_out,
            )
        violation = _guardrail_violation(workspace, guarded_before)
        if violation:
            all_stderr.append(violation)
            (workspace / "train.log").write_text("\n".join(all_stdout), encoding="utf-8")
            return RunResult(
                command=normalized,
                exit_code=125,
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
                duration_s=total_duration,
            )

    log = "\n".join(all_stdout)
    (workspace / "train.log").write_text(log, encoding="utf-8")
    return RunResult(
        command=last_command,
        exit_code=0,
        stdout=log,
        stderr="\n".join(all_stderr),
        duration_s=total_duration,
    )
