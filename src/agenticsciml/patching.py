from __future__ import annotations

import difflib
import hashlib
import re


class PatchApplicationError(RuntimeError):
    pass


def solution_digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def make_unified_patch(old: str, new: str, filename: str = "solution.py") -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=filename,
            tofile=filename,
        )
    )


def _parse_range(value: str) -> tuple[int, int]:
    if "," in value:
        start, count = value.split(",", 1)
        return int(start), int(count)
    return int(value), 1


def apply_unified_patch(original: str, patch: str) -> str:
    patch_lines = patch.splitlines(keepends=True)
    if len(patch_lines) < 3 or not patch_lines[0].startswith("--- ") or not patch_lines[1].startswith("+++ "):
        raise PatchApplicationError("Malformed unified patch header.")

    original_lines = original.splitlines(keepends=True)
    output: list[str] = []
    source_index = 0
    patch_index = 2
    saw_hunk = False

    while patch_index < len(patch_lines):
        header = patch_lines[patch_index]
        match = re.match(r"@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@", header)
        if not match:
            raise PatchApplicationError(f"Malformed patch hunk header: {header.strip()}")
        saw_hunk = True
        old_start, _ = _parse_range(match.group(1))
        patch_index += 1

        target_index = old_start - 1
        if target_index < source_index:
            raise PatchApplicationError("Patch hunks overlap or move backwards.")
        output.extend(original_lines[source_index:target_index])
        source_index = target_index

        while patch_index < len(patch_lines) and not patch_lines[patch_index].startswith("@@ "):
            line = patch_lines[patch_index]
            patch_index += 1
            if line.startswith("\\"):
                continue
            if not line:
                raise PatchApplicationError("Malformed empty patch line.")
            marker = line[0]
            text = line[1:]
            if marker == " ":
                if source_index >= len(original_lines) or original_lines[source_index] != text:
                    raise PatchApplicationError("Patch context does not match parent code.")
                output.append(original_lines[source_index])
                source_index += 1
            elif marker == "-":
                if source_index >= len(original_lines) or original_lines[source_index] != text:
                    raise PatchApplicationError("Patch deletion does not match parent code.")
                source_index += 1
            elif marker == "+":
                output.append(text)
            else:
                raise PatchApplicationError(f"Malformed patch line: {line.strip()}")

    if not saw_hunk:
        raise PatchApplicationError("Unified patch has no hunks.")
    output.extend(original_lines[source_index:])
    return "".join(output)
