from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.secret_hygiene import scan_secret_hygiene, write_secret_hygiene_report


def test_secret_hygiene_passes_clean_artifacts(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"provider": "openai", "api_key_present": True}),
        encoding="utf-8",
    )

    report = scan_secret_hygiene(tmp_path)

    assert report["passed"] is True
    assert report["findings"] == []


def test_secret_hygiene_detects_patterns_without_printing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret_value = "test-secret-value-12345"
    monkeypatch.setenv("OPENAI_API_KEY", secret_value)
    artifact = tmp_path / "trace.jsonl"
    artifact.write_text(
        '{"message": "token sk-testsecretvalue1234567890"}\n'
        f'{{"message": "{secret_value}"}}\n',
        encoding="utf-8",
    )

    result = write_secret_hygiene_report(tmp_path)
    payload = json.loads(result.report_json.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True)

    assert result.passed is False
    assert payload["finding_count"] == 2
    assert {finding["pattern_id"] for finding in payload["findings"]} == {
        "openai_key_pattern",
        "sensitive_env_value",
    }
    assert "OPENAI_API_KEY" in encoded
    assert secret_value not in encoded
    assert "sk-testsecretvalue1234567890" not in encoded


def test_cli_secret_hygiene_fail_on_findings(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    (tmp_path / "transcript.md").write_text("leaked sk-testsecretvalue1234567890\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "secret-hygiene",
            str(tmp_path),
            "--fail-on-findings",
        ],
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert result.returncode == 1
    report_path = Path(result.stdout.strip().splitlines()[-1])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["findings"][0]["match_redacted"] is True
