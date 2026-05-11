from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from agenticsciml.state import AgentMessage


class ExperimentStorage:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.solutions_dir = run_dir / "solutions"

    @classmethod
    def create(cls, output_dir: Path, experiment_id: str) -> "ExperimentStorage":
        run_dir = output_dir / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "transcripts").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        (run_dir / "champion").mkdir(exist_ok=True)
        (run_dir / "solutions").mkdir(exist_ok=True)
        return cls(run_dir)

    def create_solution_workspace(self, solution_id: str) -> Path:
        workspace = self.solutions_dir / solution_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "transcripts").mkdir(exist_ok=True)
        return workspace

    def solution_workspace(self, solution_id: str) -> Path:
        return self.solutions_dir / solution_id

    def save_json(self, relative_path: str | Path, data: Any) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))
        return path

    def load_json(self, relative_path: str | Path) -> Any:
        return json.loads((self.run_dir / relative_path).read_text(encoding="utf-8"))

    def record_trace(self, event_type: str, name: str, metadata: dict[str, Any] | None = None) -> Path:
        path = self.run_dir / "trace.jsonl"
        event = {
            "event_type": event_type,
            "name": name,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        return path

    def save_text(self, relative_path: str | Path, text: str) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, text)
        return path

    def save_solution_text(self, solution_id: str, filename: str, text: str) -> Path:
        workspace = self.create_solution_workspace(solution_id)
        path = workspace / filename
        _atomic_write_text(path, text)
        return path

    def save_transcript(
        self,
        solution_id: str | None,
        agent_name: str,
        messages: list[AgentMessage],
    ) -> Path:
        if solution_id is None:
            path = self.run_dir / "transcripts" / f"{agent_name}.json"
        else:
            path = self.create_solution_workspace(solution_id) / "transcripts" / f"{agent_name}.json"
        payload = [message.to_dict() for message in messages]
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
        return path


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
        _fsync_directory(path.parent)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        flags = getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, os.O_RDONLY | flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
