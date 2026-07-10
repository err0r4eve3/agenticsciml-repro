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
NON_SECRET_TOKEN_COUNT_KEYS = {
    "completion_tokens",
    "cost_per_1k_tokens_usd",
    "input_tokens",
    "max_output_tokens",
    "max_prompt_tokens",
    "max_total_tokens",
    "output_tokens",
    "output_tokens_used",
    "prompt_token_estimate",
    "prompt_tokens",
    "prompt_tokens_used",
    "response_token_estimate",
    "token_budget",
    "token_budget_final",
    "total_tokens",
}
NON_SECRET_TOKEN_COUNT_CONTAINERS = {"token_budget", "token_budget_final"}


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
            "numeric_token_count_fields_preserved": True,
            "non_secret_token_count_keys": sorted(NON_SECRET_TOKEN_COUNT_KEYS),
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
            if (
                key_text.lower() in NON_SECRET_TOKEN_COUNT_CONTAINERS
                and isinstance(item, dict | list)
            ):
                result[key_text] = _sanitize_metadata(item)
            elif _is_non_secret_numeric_count(key_text, item):
                result[key_text] = item
            elif any(token in key_text.lower() for token in REDACTED_METADATA_TOKENS):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = _sanitize_metadata(item)
        return result
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_non_secret_numeric_count(key: str, value: Any) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    normalized = key.lower()
    return (
        normalized in NON_SECRET_TOKEN_COUNT_KEYS
        or normalized.endswith("_token_estimate")
        or normalized.endswith("_token_count")
        or normalized.endswith("_tokens")
    )
