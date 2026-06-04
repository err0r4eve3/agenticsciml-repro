from __future__ import annotations

import hashlib
import json
import math
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
    ("jwt_token_pattern", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |)?PRIVATE KEY-----",
        ),
    ),
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    (?P<key>\b(?:api[_-]?key|token|secret|password|authorization|credential)s?\b)
    \s*[:=]\s*
    (?P<quote>["'])?
    (?P<value>[A-Za-z0-9._~+/=-]{16,})
    (?P=quote)?
    """
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
        findings.extend(_sensitive_assignment_findings(path, root, text))
    return {
        "schema_version": 1,
        "root_dir": str(root),
        "passed": not findings,
        "finding_count": len(findings),
        "findings": findings,
        "skipped": skipped,
        "claim_boundary": (
            "This scanner checks common secret patterns, high-entropy sensitive assignments, "
            "private-key/JWT shapes, and configured sensitive env values in text run artifacts. "
            "It never prints matched secret values and does not prove the absence of every possible secret."
        ),
    }


def _pattern_findings(path: Path, root: Path, text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for pattern_id, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line = _line_for_offset(text, match.start())
            column = _column_for_offset(text, match.start())
            path_text = _relative_path(path, root)
            findings.append(
                {
                    "finding_id": _finding_id(
                        path=path_text,
                        pattern_id=pattern_id,
                        line=line,
                        column=column,
                    ),
                    "path": path_text,
                    "pattern_id": pattern_id,
                    "line": line,
                    "column": column,
                    "match_redacted": True,
                    "value_redacted": True,
                    "secret_derived_fingerprint": False,
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
        if not value:
            continue
        start = 0
        while True:
            offset = text.find(value, start)
            if offset < 0:
                break
            line = _line_for_offset(text, offset)
            column = _column_for_offset(text, offset)
            path_text = _relative_path(path, root)
            findings.append(
                {
                    "finding_id": _finding_id(
                        path=path_text,
                        pattern_id="sensitive_env_value",
                        line=line,
                        column=column,
                        env_name=name,
                    ),
                    "path": path_text,
                    "pattern_id": "sensitive_env_value",
                    "env_name": name,
                    "line": line,
                    "column": column,
                    "match_redacted": True,
                    "value_redacted": True,
                    "secret_derived_fingerprint": False,
                }
            )
            start = offset + max(1, len(value))
    return findings


def _sensitive_assignment_findings(path: Path, root: Path, text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for match in SENSITIVE_ASSIGNMENT_PATTERN.finditer(text):
        key = match.group("key")
        value = match.group("value")
        if _looks_low_risk_literal(value):
            continue
        if len(value) < 20 and _shannon_entropy(value) < 3.5:
            continue
        line = _line_for_offset(text, match.start("value"))
        column = _column_for_offset(text, match.start("value"))
        path_text = _relative_path(path, root)
        findings.append(
            {
                "finding_id": _finding_id(
                    path=path_text,
                    pattern_id="sensitive_assignment_value",
                    line=line,
                    column=column,
                    key_name=key,
                ),
                "path": path_text,
                "pattern_id": "sensitive_assignment_value",
                "line": line,
                "column": column,
                "key_name": key,
                "match_redacted": True,
                "value_redacted": True,
                "secret_derived_fingerprint": False,
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


def _column_for_offset(text: str, offset: int) -> int:
    return offset - text.rfind("\n", 0, offset)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _finding_id(
    *,
    path: str,
    pattern_id: str,
    line: int,
    column: int,
    env_name: str | None = None,
    key_name: str | None = None,
) -> str:
    identity = {
        "path": path,
        "pattern_id": pattern_id,
        "line": line,
        "column": column,
        "env_name": env_name,
        "key_name": key_name,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "finding_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _looks_low_risk_literal(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"true", "false", "none", "null", "present", "redacted", "placeholder"}


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())
