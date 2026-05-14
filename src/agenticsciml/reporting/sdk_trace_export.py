from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


SPAN_KIND_BY_EVENT_TYPE = {
    "workflow_span": "workflow",
    "agent_span": "agent",
    "generation_span": "generation",
    "tool_span": "tool",
    "guardrail_span": "guardrail",
}

REDACTED_METADATA_TOKENS = (
    "api_key",
    "cookie",
    "message",
    "password",
    "prompt",
    "raw",
    "request",
    "response",
    "secret",
    "system",
    "token",
)


def write_sdk_trace_export(run_dir: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or run_dir / "openai_sdk_trace.json"
    spans = []
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            spans.append(_span_from_event(event))
    payload = {
        "schema_version": 1,
        "source": "agenticsciml.trace.jsonl",
        "run_dir": str(run_dir),
        "span_count": len(spans),
        "spans": spans,
        "redaction_policy": {
            "metadata_keys_containing": list(REDACTED_METADATA_TOKENS),
            "raw_prompt_or_response_exported": False,
        },
    }
    _atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return output_path


def _span_from_event(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("event_type", "unknown"))
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "event_seq": event.get("event_seq"),
        "span_kind": SPAN_KIND_BY_EVENT_TYPE.get(event_type, "custom"),
        "event_type": event_type,
        "name": str(event.get("name", "")),
        "timestamp": event.get("timestamp"),
        "metadata": _sanitize_metadata(metadata),
    }


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(token in key_text.lower() for token in REDACTED_METADATA_TOKENS):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = _sanitize_metadata(item)
        return result
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
