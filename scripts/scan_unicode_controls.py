#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCAN_SUFFIXES = {
    ".css",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
ALLOWED_CONTROL_CHARS = {"\n", "\r", "\t"}
DEFAULT_ALLOWLIST = Path(".unicode-control-allowlist.json")
EXCLUDED_DIR_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "runs",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    codepoint: str
    name: str
    category: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "codepoint": self.codepoint,
            "name": self.name,
            "category": self.category,
        }


@dataclass(frozen=True)
class AllowlistEntry:
    path: str
    codepoint: str
    reason: str
    line: int | None = None
    column: int | None = None

    def matches(self, finding: Finding) -> bool:
        if self.path != finding.path or self.codepoint != finding.codepoint:
            return False
        if self.line is not None and self.line != finding.line:
            return False
        if self.column is not None and self.column != finding.column:
            return False
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan source files for hidden Unicode control characters.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root or directory to scan")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="JSON allowlist file; defaults to <root>/.unicode-control-allowlist.json when present",
    )
    parser.add_argument("--json", action="store_true", help="write machine-readable scan result")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    allowlist_path = args.allowlist
    if allowlist_path is None:
        default = root / DEFAULT_ALLOWLIST
        allowlist_path = default if default.exists() else None
    try:
        allowlist = load_allowlist(allowlist_path, root=root)
        files = discover_files(root)
        findings = scan_files(files, root=root)
    except ValueError as exc:
        print(f"unicode-control scan error: {exc}", file=sys.stderr)
        return 2
    unapproved = [finding for finding in findings if not any(entry.matches(finding) for entry in allowlist)]
    result = {
        "checked_file_count": len(files),
        "finding_count": len(findings),
        "unapproved_finding_count": len(unapproved),
        "findings": [finding.to_dict() for finding in unapproved],
        "allowlist_path": str(allowlist_path) if allowlist_path is not None else None,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif unapproved:
        print("Unicode control-character scan failed:")
        for finding in unapproved:
            print(
                f"{finding.path}:{finding.line}:{finding.column}: "
                f"{finding.codepoint} {finding.name} ({finding.category})"
            )
        print("Add a justified entry to .unicode-control-allowlist.json only for intentional cases.")
    else:
        print(f"Unicode control-character scan passed: {len(files)} files checked.")
    return 1 if unapproved else 0


def discover_files(root: Path) -> list[Path]:
    git_files = _git_tracked_files(root)
    if git_files is not None:
        candidates = [root / path for path in git_files]
    else:
        candidates = [path for path in root.rglob("*") if path.is_file()]
    return sorted(
        path
        for path in candidates
        if path.suffix in SCAN_SUFFIXES and not _is_excluded(path, root=root)
    )


def scan_files(files: list[Path], *, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            relative = _relative_path(path, root=root)
            raise ValueError(f"{relative} is not valid UTF-8: {exc}") from exc
        line = 1
        column = 0
        for character in text:
            if character == "\n":
                line += 1
                column = 0
                continue
            column += 1
            if _is_disallowed_control(character):
                findings.append(
                    Finding(
                        path=_relative_path(path, root=root),
                        line=line,
                        column=column,
                        codepoint=_codepoint(character),
                        name=unicodedata.name(character, "UNKNOWN"),
                        category=unicodedata.category(character),
                    )
                )
    return findings


def load_allowlist(path: Path | None, *, root: Path) -> list[AllowlistEntry]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"allowlist file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"allowlist file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("allowlist must be an object with schema_version=1")
    raw_entries = payload.get("allowlist")
    if not isinstance(raw_entries, list):
        raise ValueError("allowlist must contain an allowlist array")
    entries: list[AllowlistEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"allowlist entry {index} must be an object")
        reason = raw_entry.get("reason")
        entry_path = raw_entry.get("path")
        codepoint = raw_entry.get("codepoint")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"allowlist entry {index} must include a non-empty reason")
        if not isinstance(entry_path, str) or not entry_path:
            raise ValueError(f"allowlist entry {index} must include path")
        if not isinstance(codepoint, str) or not codepoint.startswith("U+"):
            raise ValueError(f"allowlist entry {index} must include codepoint like U+202E")
        line = _optional_positive_int(raw_entry.get("line"), f"allowlist entry {index} line")
        column = _optional_positive_int(raw_entry.get("column"), f"allowlist entry {index} column")
        normalized_path = _normalize_allowlist_path(entry_path, root=root)
        entries.append(
            AllowlistEntry(
                path=normalized_path,
                codepoint=codepoint.upper(),
                reason=reason,
                line=line,
                column=column,
            )
        )
    return entries


def _git_tracked_files(root: Path) -> list[Path] | None:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [Path(item) for item in completed.stdout.decode("utf-8").split("\0") if item]


def _is_disallowed_control(character: str) -> bool:
    if character in ALLOWED_CONTROL_CHARS:
        return False
    return unicodedata.category(character) in {"Cc", "Cf"}


def _is_excluded(path: Path, *, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    return any(part in EXCLUDED_DIR_PARTS for part in relative.parts)


def _relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_allowlist_path(path: str, *, root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"allowlist path escapes root: {path}") from exc
    if ".." in candidate.parts:
        raise ValueError(f"allowlist path must not contain '..': {path}")
    return candidate.as_posix()


def _codepoint(character: str) -> str:
    return f"U+{ord(character):04X}"


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
