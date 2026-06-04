from __future__ import annotations

import hashlib
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
    assert hashlib.sha256(secret_value.encode("utf-8")).hexdigest() not in encoded
    assert "value_sha256" not in encoded
    assert "value_hash" not in encoded
    assert all(finding["secret_derived_fingerprint"] is False for finding in payload["findings"])
    assert all(finding["value_redacted"] is True for finding in payload["findings"])
    assert all(str(finding["finding_id"]).startswith("finding_") for finding in payload["findings"])


def test_secret_hygiene_detects_jwt_private_key_and_sensitive_assignment(tmp_path: Path) -> None:
    api_token = "AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    jwt_value = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturepart"
    artifact = tmp_path / "transcript.md"
    artifact.write_text(
        "\n".join(
            [
                f"token = {api_token}",
                f"authorization: {jwt_value}",
                "-----BEGIN PRIVATE KEY-----",
                "not-the-key-body",
            ]
        ),
        encoding="utf-8",
    )

    report = scan_secret_hygiene(tmp_path)
    encoded = json.dumps(report, sort_keys=True)
    pattern_ids = {finding["pattern_id"] for finding in report["findings"]}

    assert report["passed"] is False
    assert "sensitive_assignment_value" in pattern_ids
    assert "jwt_token_pattern" in pattern_ids
    assert "private_key_block" in pattern_ids
    assert api_token not in encoded
    assert jwt_value not in encoded
    assert "value_sha256" not in encoded
    assert "key_name" in encoded


def test_secret_hygiene_finding_id_is_not_secret_derived(tmp_path: Path) -> None:
    artifact = tmp_path / "transcript.md"
    first_secret = "AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    second_secret = "ZyXwVuTsRqPoNmLkJiHgFeDcBa654321"

    artifact.write_text(f"token = {first_secret}\n", encoding="utf-8")
    first = scan_secret_hygiene(tmp_path)
    first_finding = next(
        finding for finding in first["findings"] if finding["pattern_id"] == "sensitive_assignment_value"
    )
    first_encoded = json.dumps(first, sort_keys=True)

    artifact.write_text(f"token = {second_secret}\n", encoding="utf-8")
    second = scan_secret_hygiene(tmp_path)
    second_finding = next(
        finding for finding in second["findings"] if finding["pattern_id"] == "sensitive_assignment_value"
    )
    second_encoded = json.dumps(second, sort_keys=True)

    assert first_finding["finding_id"] == second_finding["finding_id"]
    assert first_finding["secret_derived_fingerprint"] is False
    assert second_finding["secret_derived_fingerprint"] is False
    assert hashlib.sha256(first_secret.encode("utf-8")).hexdigest() not in first_encoded
    assert hashlib.sha256(second_secret.encode("utf-8")).hexdigest() not in second_encoded

    moved_artifact = tmp_path / "nested" / "transcript.md"
    moved_artifact.parent.mkdir()
    moved_artifact.write_text(f"token = {second_secret}\n", encoding="utf-8")
    moved = scan_secret_hygiene(tmp_path)
    moved_finding = next(
        finding
        for finding in moved["findings"]
        if finding["pattern_id"] == "sensitive_assignment_value" and finding["path"] == "nested/transcript.md"
    )

    assert moved_finding["finding_id"] != first_finding["finding_id"]


def test_secret_hygiene_ignores_low_risk_boolean_assignment(tmp_path: Path) -> None:
    (tmp_path / "run_metadata.json").write_text(
        json.dumps({"api_key_present": True, "token_budget": {"max_calls": 80}}),
        encoding="utf-8",
    )

    report = scan_secret_hygiene(tmp_path)

    assert report["passed"] is True


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
