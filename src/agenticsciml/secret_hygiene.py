from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticsciml.storage import _atomic_write_text


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key_pattern", re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{12,}\b")),
    ("openai_api_key_assignment", re.compile(r"\bOPENAI_API_KEY\b\s*[:=]")),
    ("github_token_pattern", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("generic_bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b")),
)
SENSITIVE_ENV_NAMES = {
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
}
SENSITIVE_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".mmd",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_SCAN_BYTES = 2_000_000


@dataclass(frozen=True, slots=True)
class SecretHygieneResult:
    report_json: Path
    passed: bool


def write_secret_hygiene_report(
    root_dir: Path,
    *,
    output_json: Path | None = None,
) -> SecretHygieneResult:
    root = root_dir.resolve()
    report = scan_secret_hygiene(root)
    path = output_json.resolve() if output_json else root / "secret_hygiene_report.json"
    _atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return SecretHygieneResult(report_json=path, passed=bool(report["passed"]))


def scan_secret_hygiene(root_dir: Path) -> dict[str, Any]:
    root = root_dir.resolve()
    findings: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    env_values = _sensitive_env_values()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not _is_text_artifact(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > MAX_SCAN_BYTES:
            skipped.append(
                {
                    "path": _relative_path(path, root),
                    "reason": "file_too_large",
                    "size_bytes": stat.st_size,
                }
            )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped.append({"path": _relative_path(path, root), "reason": "non_utf8"})
            continue
        findings.extend(_pattern_findings(path, root, text))
        findings.extend(_env_value_findings(path, root, text, env_values))
    return {
        "schema_version": 1,
        "root_dir": str(root),
        "passed": not findings,
        "finding_count": len(findings),
        "findings": findings,
        "skipped": skipped,
        "claim_boundary": (
            "This scanner checks common secret patterns and configured sensitive env values in "
            "text run artifacts. It never prints matched secret values."
        ),
    }


def _pattern_findings(path: Path, root: Path, text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for pattern_id, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                {
                    "path": _relative_path(path, root),
                    "pattern_id": pattern_id,
                    "line": _line_for_offset(text, match.start()),
                    "match_redacted": True,
                }
            )
    return findings


def _env_value_findings(
    path: Path,
    root: Path,
    text: str,
    env_values: dict[str, str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for name, value in env_values.items():
        if value and value in text:
            findings.append(
                {
                    "path": _relative_path(path, root),
                    "pattern_id": "sensitive_env_value",
                    "env_name": name,
                    "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    "match_redacted": True,
                }
            )
    return findings


def _sensitive_env_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        if name in SENSITIVE_ENV_NAMES or name.endswith(SENSITIVE_ENV_SUFFIXES):
            values[name] = value
    return values


def _is_text_artifact(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name in {"leaderboard.csv", "trace.jsonl"}


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())
