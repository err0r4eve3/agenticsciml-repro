from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


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


def run_command(cwd: Path, command: list[str], timeout_s: int) -> RunResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
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
