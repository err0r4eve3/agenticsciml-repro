from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agenticsciml.iteration_campaign import (
    build_iteration_campaign,
    record_iteration_round_evidence,
    verify_iteration_campaign,
    write_iteration_campaign,
)


def test_iteration_campaign_builds_sixty_rounds_in_batches() -> None:
    campaign = build_iteration_campaign(
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        rounds=60,
        batch_size=10,
        env={},
    )

    assert campaign["schema_version"] == 1
    assert campaign["rounds_requested"] == 60
    assert len(campaign["rounds"]) == 60
    assert len(campaign["batches"]) == 6
    assert campaign["batches"][0]["round_start"] == 1
    assert campaign["batches"][-1]["round_end"] == 60
    assert campaign["rounds"][0]["target_id"] == "real_provider_budget"
    assert campaign["rounds"][9]["target_id"] == "documentation_and_repro_packet"
    assert campaign["rounds"][10]["target_id"] == "real_provider_budget"
    assert campaign["rounds"][0]["status"] == "blocked_by_readiness"
    assert campaign["rounds"][5]["status"] == "planned"
    assert campaign["claim_boundary"]


def test_iteration_campaign_rejects_invalid_rounds() -> None:
    with pytest.raises(ValueError, match="rounds must be positive"):
        build_iteration_campaign(
            benchmark_dir=Path("examples/function_approx").resolve(),
            rounds=0,
            env={},
        )


def test_write_iteration_campaign_outputs_json_and_markdown(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )

    campaign = json.loads(Path(result["paths"]["campaign_json"]).read_text(encoding="utf-8"))

    assert campaign["rounds_requested"] == 12
    assert len(campaign["batches"]) == 3
    assert Path(result["paths"]["campaign_md"]).exists()


def test_cli_plan_iteration_campaign_writes_sixty_rounds(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    output_dir = tmp_path / "campaign"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-iteration-campaign",
            "examples/cylinder_wake_reconstruction_faithful_small",
            "--output-dir",
            str(output_dir),
            "--rounds",
            "60",
            "--batch-size",
            "10",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    campaign_path = Path(result.stdout.strip())
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))

    assert campaign_path == output_dir / "iteration_campaign.json"
    assert campaign["rounds_requested"] == 60
    assert len(campaign["rounds"]) == 60
    assert (output_dir / "iteration_campaign.md").exists()


def test_record_iteration_round_evidence_updates_campaign(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    evidence_path = tmp_path / "round_006_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")

    record = record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        notes="multi-seed ablation verifier implementation landed",
    )
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    round_six = next(item for item in campaign["rounds"] if item["round_index"] == 6)

    assert record["round_index"] == 6
    assert record["target_id"] == "multi_seed_ablation"
    assert record["evidence"]["sha256"]
    assert record["validation_command"] == "pytest tests/test_iteration_campaign.py -q"
    assert round_six["status"] == "completed"
    assert round_six["evidence_record_path"] == "iteration_round_006_record.json"
    assert campaign["completed_rounds"] == 1
    assert campaign["remaining_rounds"] == 11
    assert (tmp_path / "iteration_round_006_record.json").exists()


def test_record_iteration_round_evidence_rejects_blocked_round(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    evidence_path = tmp_path / "round_001_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="round 1 is blocked_by_readiness"):
        record_iteration_round_evidence(
            Path(result["paths"]["campaign_json"]),
            round_index=1,
            evidence_path=evidence_path,
            validation_command="pytest tests/test_iteration_campaign.py -q",
        )


def test_cli_record_iteration_round_updates_campaign(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    campaign_dir = tmp_path / "campaign"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-iteration-campaign",
            "examples/function_approx",
            "--output-dir",
            str(campaign_dir),
            "--rounds",
            "12",
            "--batch-size",
            "5",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    evidence_path = campaign_dir / "round_006_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "record-iteration-round",
            str(campaign_dir / "iteration_campaign.json"),
            "--round",
            "6",
            "--evidence-path",
            str(evidence_path),
            "--validation-command",
            "pytest tests/test_iteration_campaign.py -q",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    record_path = Path(result.stdout.strip())
    campaign = json.loads((campaign_dir / "iteration_campaign.json").read_text(encoding="utf-8"))

    assert record_path == campaign_dir / "iteration_round_006_record.json"
    assert json.loads(record_path.read_text(encoding="utf-8"))["round_index"] == 6
    assert campaign["completed_rounds"] == 1


def test_verify_iteration_campaign_passes_recorded_round(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    evidence_path = tmp_path / "round_006_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
    )

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is True
    assert verification["completed_round_count"] == 1
    assert verification["integrity_issue_count"] == 0
    assert verification["records"][0]["round_index"] == 6
    assert verification["records"][0]["evidence_digest_match"] is True


def test_verify_iteration_campaign_detects_tampered_evidence(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    evidence_path = tmp_path / "round_006_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
    )
    evidence_path.write_text('{"validated": false}\n', encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["integrity_issue_count"] == 1
    assert "evidence digest mismatch for round 6" in verification["issues"]


def test_cli_verify_iteration_campaign_writes_report(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    campaign_dir = tmp_path / "campaign"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-iteration-campaign",
            "examples/function_approx",
            "--output-dir",
            str(campaign_dir),
            "--rounds",
            "12",
            "--batch-size",
            "5",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    evidence_path = campaign_dir / "round_006_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "record-iteration-round",
            str(campaign_dir / "iteration_campaign.json"),
            "--round",
            "6",
            "--evidence-path",
            str(evidence_path),
            "--validation-command",
            "pytest tests/test_iteration_campaign.py -q",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-iteration-campaign",
            str(campaign_dir / "iteration_campaign.json"),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    report_path = Path(result.stdout.strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path == campaign_dir / "iteration_campaign_verification.json"
    assert report["passed"] is True
    assert report["completed_round_count"] == 1
