from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Mapping


@dataclass(slots=True)
class RunResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        return f"$ {' '.join(self.command)}\n\nSTDOUT:\n{self.stdout}\n\nSTDERR:\n{self.stderr}\n"


def run_command(
    cwd: Path,
    command: list[str],
    timeout_s: int,
    env: Mapping[str, str | PathLike[str]] | None = None,
    clean_env: bool = True,
) -> RunResult:
    started = time.monotonic()
    subprocess_env = _subprocess_env(cwd, env, clean_env=clean_env)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=subprocess_env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return RunResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            command=command,
            exit_code=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"Timed out after {timeout_s}s",
            duration_s=time.monotonic() - started,
            timed_out=True,
        )


def _subprocess_env(
    cwd: Path,
    env: Mapping[str, str | PathLike[str]] | None,
    *,
    clean_env: bool,
) -> dict[str, str] | None:
    if not clean_env:
        return {str(key): str(value) for key, value in env.items()} if env is not None else None
    home = cwd / ".home"
    tmp = cwd / ".tmp"
    home.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    safe = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if os.environ.get("LANG"):
        safe["LANG"] = os.environ["LANG"]
    if env:
        safe.update({str(key): str(value) for key, value in env.items()})
    return safe
