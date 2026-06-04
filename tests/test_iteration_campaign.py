from __future__ import annotations

import hashlib
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
    assert campaign["llm_problem_context_pack"]["llm_execution_required"] is True
    assert campaign["llm_problem_context_pack"]["scientific_claim_supported"] is False
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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")

    record = record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
        notes="multi-seed ablation verifier implementation landed",
    )
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    round_six = next(item for item in campaign["rounds"] if item["round_index"] == 6)

    assert record["round_index"] == 6
    assert record["target_id"] == "multi_seed_ablation"
    assert record["evidence"]["sha256"]
    assert record["validation_command"] == "pytest tests/test_iteration_campaign.py -q"
    assert record["validation_result"]["exit_code"] == 0
    assert record["validation_result"]["output"]["sha256"]
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
            validation_exit_code=0,
            validation_output_path=evidence_path,
        )


def test_record_iteration_round_evidence_rejects_tampered_external_blocker_removal(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["rounds"][0]["status"] = "planned"
    campaign["rounds"][0]["blocked_by"] = None
    campaign["batches"][0]["blocked_round_count"] = 4
    campaign["batches"][0]["planned_round_count"] = 1
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    evidence_path = tmp_path / "round_001_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="round 1 still has readiness blocker real_llm_credentials"):
        record_iteration_round_evidence(
            campaign_path,
            round_index=1,
            evidence_path=evidence_path,
            validation_command="pytest tests/test_iteration_campaign.py -q",
            validation_exit_code=0,
            validation_output_path=evidence_path,
        )


def test_record_iteration_round_evidence_rejects_removed_top_level_readiness_blockers(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["readiness_blockers"] = []
    campaign["paper_workflow_readiness_status"] = "ready"
    campaign["status"] = "ready"
    campaign["rounds"][0]["status"] = "planned"
    campaign["rounds"][0]["blocked_by"] = None
    campaign["batches"][0]["blocked_round_count"] = 4
    campaign["batches"][0]["planned_round_count"] = 1
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    evidence_path = tmp_path / "round_001_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="round 1 still has readiness blocker real_llm_credentials"):
        record_iteration_round_evidence(
            campaign_path,
            round_index=1,
            evidence_path=evidence_path,
            validation_command="pytest tests/test_iteration_campaign.py -q",
            validation_exit_code=0,
            validation_output_path=evidence_path,
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
    validation_output = campaign_dir / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")

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
            "--validation-exit-code",
            "0",
            "--validation-output-path",
            str(validation_output),
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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is True
    assert verification["completed_round_count"] == 1
    assert verification["integrity_issue_count"] == 0
    assert verification["records"][0]["round_index"] == 6
    assert verification["records"][0]["evidence_digest_match"] is True
    assert verification["records"][0]["validation_output_digest_match"] is True


def test_verify_iteration_campaign_require_complete_blocks_incomplete_campaign(tmp_path: Path) -> None:
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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )

    verification = verify_iteration_campaign(campaign_path, require_complete=True)

    assert verification["passed"] is False
    assert verification["require_complete"] is True
    assert verification["fully_completed"] is False
    assert "campaign is not complete: completed=1, expected=12" in verification["issues"]


def test_verify_iteration_campaign_detects_round_schema_tampering(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["rounds"][1]["round_index"] = 1
    campaign["rounds"][2]["status"] = "done"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["round_integrity"]["duplicate_round_indices"] == [1]
    assert verification["round_integrity"]["missing_round_indices"] == [2]
    assert "duplicate round_index values: [1]" in verification["issues"]
    assert "missing round_index values: [2]" in verification["issues"]
    assert "round 3 has unsupported status: done" in verification["issues"]


def test_verify_iteration_campaign_detects_round_target_tampering(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["target_sequence"][5]["target_id"] = "paper_like_benchmark"
    campaign["rounds"][5]["target_id"] = "paper_like_benchmark"
    campaign["rounds"][5]["batch_index"] = 99
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["round_integrity"]["target_sequence_valid"] is False
    assert verification["round_integrity"]["round_target_mismatches"] == [
        {
            "round_index": 6,
            "field": "target_id",
            "expected": "multi_seed_ablation",
            "actual": "paper_like_benchmark",
        }
    ]
    assert verification["round_integrity"]["batch_index_mismatches"] == [
        {"round_index": 6, "expected": 2, "actual": 99}
    ]
    assert "target_sequence does not match campaign schema targets" in verification["issues"]
    assert (
        "round 6 target_id mismatch: expected multi_seed_ablation, got paper_like_benchmark"
        in verification["issues"]
    )
    assert "round 6 batch_index mismatch: expected 2, got 99" in verification["issues"]


def test_verify_iteration_campaign_detects_batch_summary_tampering(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["batches"][0]["round_end"] = 4
    campaign["batches"][0]["target_ids"] = ["documentation_and_repro_packet"]
    campaign["batches"][0]["planned_round_count"] = 99
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["batch_integrity"]["batch_schema_valid"] is False
    assert {
        "batch_index": 1,
        "field": "round_end",
        "expected": 5,
        "actual": 4,
    } in verification["batch_integrity"]["batch_mismatches"]
    assert any(issue.startswith("batch 1 target_ids mismatch") for issue in verification["issues"])
    assert "batch 1 planned_round_count mismatch: expected 0, got 99" in verification["issues"]


def test_verify_iteration_campaign_detects_campaign_metadata_tampering(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["campaign_version"] = "iteration_campaign.v999"
    campaign["claim_boundary"] = "scientific discovery ready"
    campaign["status"] = "ready"
    campaign["paper_workflow_readiness_status"] = "ready"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["campaign_metadata_integrity"]["campaign_metadata_valid"] is False
    assert {
        "field": "campaign_version",
        "expected": "iteration_campaign.v1",
        "actual": "iteration_campaign.v999",
    } in verification["campaign_metadata_integrity"]["metadata_mismatches"]
    assert "campaign_version mismatch: expected iteration_campaign.v1, got iteration_campaign.v999" in verification[
        "issues"
    ]
    assert "campaign status mismatch: expected blocked, got ready" in verification["issues"]
    assert "paper_workflow_readiness_status mismatch: expected blocked, got ready" in verification["issues"]
    assert "claim_boundary mismatch for iteration campaign" in verification["issues"]


def test_verify_iteration_campaign_detects_record_metadata_tampering(tmp_path: Path) -> None:
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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )
    record_path = tmp_path / "iteration_round_006_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["target_id"] = "paper_like_benchmark"
    record["previous_status"] = "blocked_by_readiness"
    record["validation_result"]["command"] = "pytest tests/test_other.py -q"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["rounds"][5]["evidence_sha256"] = "tampered"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["records"][0]["record_metadata_match"] is False
    assert {
        "field": "target_id",
        "expected": "multi_seed_ablation",
        "actual": "paper_like_benchmark",
    } in verification["records"][0]["record_metadata_mismatches"]
    assert (
        "record target_id mismatch for round 6: expected multi_seed_ablation, got paper_like_benchmark"
        in verification["issues"]
    )
    assert "record previous_status mismatch for round 6: expected planned, got blocked_by_readiness" in verification[
        "issues"
    ]
    assert "record validation command mismatch for round 6" in verification["issues"]
    assert "campaign evidence_sha256 mismatch for round 6" in verification["issues"]


def test_verify_iteration_campaign_detects_record_claim_boundary_tampering(tmp_path: Path) -> None:
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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )
    record_path = tmp_path / "iteration_round_006_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["claim_boundary"] = "This record proves a real scientific discovery."
    record_path.write_text(json.dumps(record), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["records"][0]["record_metadata_match"] is False
    assert {
        "field": "claim_boundary",
        "expected": "This record proves an engineering iteration artifact exists; it is not scientific discovery evidence.",
        "actual": "This record proves a real scientific discovery.",
    } in verification["records"][0]["record_metadata_mismatches"]
    assert "record claim_boundary mismatch for round 6" in verification["issues"]


def test_verify_iteration_campaign_rejects_completed_round_with_readiness_blocker(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    evidence_path = tmp_path / "round_001_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")
    validation_output = tmp_path / "round_001_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    evidence_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    validation_digest = hashlib.sha256(validation_output.read_bytes()).hexdigest()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    round_one = campaign["rounds"][0]
    round_one["status"] = "completed"
    round_one["evidence_record_path"] = "iteration_round_001_record.json"
    round_one["evidence_sha256"] = evidence_digest
    campaign["completed_rounds"] = 1
    campaign["remaining_rounds"] = 11
    campaign["batches"][0]["blocked_round_count"] = 4
    campaign["batches"][0]["completed_round_count"] = 1
    campaign["batches"][0]["remaining_round_count"] = 4
    record = {
        "schema_version": 1,
        "record_version": "iteration_round_record.v1",
        "round_index": 1,
        "batch_index": 1,
        "target_id": "real_provider_budget",
        "readiness_check_id": "real_llm_credentials",
        "previous_status": "planned",
        "validation_command": "pytest tests/test_iteration_campaign.py -q",
        "validation_result": {
            "command": "pytest tests/test_iteration_campaign.py -q",
            "exit_code": 0,
            "output": {
                "path": "round_001_validation.log",
                "sha256": validation_digest,
                "size_bytes": validation_output.stat().st_size,
            },
        },
        "notes": "",
        "evidence": {
            "path": "round_001_evidence.json",
            "sha256": evidence_digest,
            "size_bytes": evidence_path.stat().st_size,
        },
        "claim_boundary": "This record proves an engineering iteration artifact exists; it is not scientific discovery evidence.",
    }
    (tmp_path / "iteration_round_001_record.json").write_text(json.dumps(record), encoding="utf-8")
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["round_integrity"]["completed_rounds_with_blockers"] == [1]
    assert "round 1 is completed but still has readiness blocker real_llm_credentials" in verification["issues"]


def test_verify_iteration_campaign_detects_removed_external_round_blocker(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["rounds"][0]["status"] = "planned"
    campaign["rounds"][0]["blocked_by"] = None
    campaign["batches"][0]["blocked_round_count"] = 4
    campaign["batches"][0]["planned_round_count"] = 1
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert {
        "round_index": 1,
        "field": "blocked_by",
        "expected": {
            "check_id": "real_llm_credentials",
            "category": "provider",
            "next_action": "Set OPENAI_API_KEY without writing it into artifacts.",
        },
        "actual": None,
    } in verification["round_integrity"]["round_blocker_mismatches"]
    assert "round 1 blocked_by mismatch for readiness blocker real_llm_credentials" in verification["issues"]
    assert "round 1 status mismatch: expected blocked_by_readiness while blocker remains, got planned" in verification[
        "issues"
    ]


def test_verify_iteration_campaign_detects_removed_top_level_readiness_blockers(tmp_path: Path) -> None:
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["readiness_blockers"] = []
    campaign["paper_workflow_readiness_status"] = "ready"
    campaign["status"] = "ready"
    for round_item in campaign["rounds"]:
        if round_item["requires_external_asset"]:
            round_item["status"] = "planned"
            round_item["blocked_by"] = None
    for batch in campaign["batches"]:
        items = [
            item
            for item in campaign["rounds"]
            if batch["round_start"] <= item["round_index"] <= batch["round_end"]
        ]
        batch["blocked_round_count"] = sum(1 for item in items if item["status"] == "blocked_by_readiness")
        batch["planned_round_count"] = sum(1 for item in items if item["status"] == "planned")
        batch["completed_round_count"] = sum(1 for item in items if item["status"] == "completed")
        batch["remaining_round_count"] = sum(1 for item in items if item["status"] != "completed")
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["campaign_metadata_integrity"]["readiness_blockers_match_failed_checks"] is False
    assert "readiness_blockers do not match failed paper_workflow_readiness_checks" in verification["issues"]
    assert "paper_workflow_readiness_status mismatch: expected blocked, got ready" in verification["issues"]
    assert "round 1 blocked_by mismatch for readiness blocker real_llm_credentials" in verification["issues"]


def test_verify_iteration_campaign_requires_canonical_record_path(tmp_path: Path) -> None:
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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )
    canonical_record = tmp_path / "iteration_round_006_record.json"
    alternate_record = tmp_path / "alternate_round_006_record.json"
    alternate_record.write_text(canonical_record.read_text(encoding="utf-8"), encoding="utf-8")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["rounds"][5]["evidence_record_path"] = alternate_record.name
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert (
        "record path mismatch for round 6: expected iteration_round_006_record.json, got "
        "alternate_round_006_record.json"
    ) in verification["issues"]


def test_record_iteration_round_rejects_external_evidence_paths(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=campaign_dir,
        rounds=12,
        batch_size=5,
        env={},
    )
    evidence_path = external_dir / "round_006_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")
    validation_output = campaign_dir / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="evidence_path must resolve inside campaign directory"):
        record_iteration_round_evidence(
            Path(result["paths"]["campaign_json"]),
            round_index=6,
            evidence_path=evidence_path,
            validation_command="pytest tests/test_iteration_campaign.py -q",
            validation_exit_code=0,
            validation_output_path=validation_output,
        )


def test_verify_iteration_campaign_detects_artifact_path_escapes(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    result = write_iteration_campaign(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=campaign_dir,
        rounds=12,
        batch_size=5,
        env={},
    )
    campaign_path = Path(result["paths"]["campaign_json"])
    evidence_path = campaign_dir / "round_006_evidence.json"
    evidence_path.write_text('{"validated": true}\n', encoding="utf-8")
    validation_output = campaign_dir / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )
    external_evidence = external_dir / "external_evidence.json"
    external_evidence.write_text('{"validated": true}\n', encoding="utf-8")
    external_validation = external_dir / "external_validation.log"
    external_validation.write_text("1 passed\n", encoding="utf-8")
    record_path = campaign_dir / "iteration_round_006_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["evidence"]["path"] = str(external_evidence)
    record["validation_result"]["output"]["path"] = "../external/external_validation.log"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert "evidence path for round 6 must be relative to campaign directory" in verification["issues"]
    assert "validation output path for round 6 escapes campaign directory" in verification["issues"]


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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )
    evidence_path.write_text('{"validated": false}\n', encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["integrity_issue_count"] == 1
    assert "evidence digest mismatch for round 6" in verification["issues"]


def test_verify_iteration_campaign_detects_tampered_validation_output(tmp_path: Path) -> None:
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
    validation_output = tmp_path / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
    record_iteration_round_evidence(
        campaign_path,
        round_index=6,
        evidence_path=evidence_path,
        validation_command="pytest tests/test_iteration_campaign.py -q",
        validation_exit_code=0,
        validation_output_path=validation_output,
    )
    validation_output.write_text("failed after edit\n", encoding="utf-8")

    verification = verify_iteration_campaign(campaign_path)

    assert verification["passed"] is False
    assert verification["integrity_issue_count"] == 1
    assert "validation output digest mismatch for round 6" in verification["issues"]


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
    validation_output = campaign_dir / "round_006_validation.log"
    validation_output.write_text("1 passed\n", encoding="utf-8")
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
            "--validation-exit-code",
            "0",
            "--validation-output-path",
            str(validation_output),
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


def test_cli_verify_iteration_campaign_require_complete_exits_nonzero(
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

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-iteration-campaign",
            str(campaign_dir / "iteration_campaign.json"),
            "--require-complete",
            "--fail-on-issues",
        ],
        text=True,
        capture_output=True,
        env=cli_env,
    )
    report = json.loads((campaign_dir / "iteration_campaign_verification.json").read_text(encoding="utf-8"))

    assert result.returncode == 1
    assert report["require_complete"] is True
    assert report["fully_completed"] is False
    assert "campaign is not complete: completed=0, expected=12" in report["issues"]
