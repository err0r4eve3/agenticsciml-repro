from __future__ import annotations

import re
from pathlib import Path


SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_path_segment(value: str, *, label: str) -> str:
    if value in {".", ".."} or not SAFE_PATH_SEGMENT_RE.fullmatch(value):
        raise ValueError(
            f"{label} must be a safe single path segment containing only letters, "
            "numbers, _, . or -"
        )
    return value


def resolve_contained_child(
    root: Path,
    child_name: str,
    *,
    label: str,
    strict: bool = False,
) -> Path:
    validate_path_segment(child_name, label=label)
    resolved_root = root.resolve(strict=False)
    resolved_child = (resolved_root / child_name).resolve(strict=strict)
    if resolved_child.parent != resolved_root:
        raise ValueError(f"{label} escapes its configured root")
    return resolved_child
