import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/scan_unicode_controls.py")


def _run_scan(root: Path, *, allowlist: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), "--root", str(root)]
    if allowlist is not None:
        command.extend(["--allowlist", str(allowlist)])
    return subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_unicode_control_scan_passes_clean_files(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text("print('safe')\n", encoding="utf-8")

    result = _run_scan(tmp_path)

    assert result.returncode == 0
    assert "passed" in result.stdout


def test_repository_unicode_control_scan_passes() -> None:
    result = _run_scan(Path.cwd())

    assert result.returncode == 0, result.stdout + result.stderr


def test_unicode_control_scan_fails_on_bidi_control(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("safe" + chr(0x202E) + "text\n", encoding="utf-8")

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "bad.md:1:5" in result.stdout
    assert "U+202E" in result.stdout


def test_unicode_control_scan_allows_documented_exception(tmp_path: Path) -> None:
    (tmp_path / "fixture.md").write_text("safe" + chr(0x202E) + "text\n", encoding="utf-8")
    allowlist = tmp_path / "unicode-allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "allowlist": [
                    {
                        "path": "fixture.md",
                        "codepoint": "U+202E",
                        "line": 1,
                        "column": 5,
                        "reason": "test fixture documents an intentional bidi control character",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_scan(tmp_path, allowlist=allowlist)

    assert result.returncode == 0
    assert "passed" in result.stdout


def test_unicode_control_scan_rejects_allowlist_without_reason(tmp_path: Path) -> None:
    (tmp_path / "fixture.md").write_text("safe\n", encoding="utf-8")
    allowlist = tmp_path / "unicode-allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "allowlist": [
                    {
                        "path": "fixture.md",
                        "codepoint": "U+202E",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_scan(tmp_path, allowlist=allowlist)

    assert result.returncode == 2
    assert "reason" in result.stderr
