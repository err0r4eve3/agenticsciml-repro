from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticsciml.storage import atomic_write_text


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
    atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
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
