from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agenticsciml.paper_workflow_readiness import build_paper_workflow_readiness_bundle
from agenticsciml.storage import _atomic_write_text


ITERATION_CAMPAIGN_SCHEMA_VERSION = 1
ITERATION_CAMPAIGN_VERSION = "iteration_campaign.v1"
ITERATION_CAMPAIGN_CLAIM_BOUNDARY = (
    "This campaign is an execution plan for iterative engineering evidence. Planned or blocked rounds "
    "do not count as scientific discovery evidence until validated run artifacts exist."
)
ITERATION_ROUND_RECORD_CLAIM_BOUNDARY = (
    "This record proves an engineering iteration artifact exists; it is not scientific discovery evidence."
)
DEFAULT_CAMPAIGN_ROUNDS = 60
DEFAULT_BATCH_SIZE = 10
SUPPORTED_CAMPAIGN_STATUSES = frozenset({"blocked", "ready"})
SUPPORTED_READINESS_STATUSES = frozenset({"blocked", "ready"})
SUPPORTED_ROUND_STATUSES = frozenset({"planned", "blocked_by_readiness", "completed"})

CAMPAIGN_TARGETS = (
    {
        "target_id": "real_provider_budget",
        "readiness_check_id": "real_llm_credentials",
        "objective": "Configure a real provider and explicit LLM budget without storing secrets.",
        "artifact": "paper_workflow_readiness.json provider_readiness",
        "validation": "agenticsciml plan-paper-workflow --fail-on-blockers after credentials are configured",
        "requires_external_asset": True,
    },
    {
        "target_id": "real_multimodal_input",
        "readiness_check_id": "real_multimodal_provider",
        "objective": "Run visual audit through an image-capable provider and require actual image input evidence.",
        "artifact": "reports/visual_audit_manifest.json",
        "validation": "trace_summary.json must show actual_image_inputs_used=true only after provider image call",
        "requires_external_asset": True,
    },
    {
        "target_id": "heterogeneous_selector",
        "readiness_check_id": "heterogeneous_real_selector",
        "objective": "Exercise at least two non-mock selector provider/model paths at runtime.",
        "artifact": "selector_votes.json and reports/selector_heterogeneity.json",
        "validation": "selector_heterogeneity.heterogeneous_selector_evidence=true",
        "requires_external_asset": True,
    },
    {
        "target_id": "paper_like_benchmark",
        "readiness_check_id": "paper_like_benchmark",
        "objective": "Replace faithful-small pilot with paper-equivalent data, evaluator, and manifest.",
        "artifact": "paper_benchmark_manifest.json and reports/paper_like_benchmark_dossier.json",
        "validation": "benchmark fidelity is paper-like and paper_like_ready=true",
        "requires_external_asset": True,
    },
    {
        "target_id": "domain_review_packet",
        "readiness_check_id": "domain_approval_packet",
        "objective": "Collect domain reviewer approval with required checklist and failure sample review.",
        "artifact": "domain_approval_template.json and reports/domain_approval.json",
        "validation": "domain_approval_readiness.approved=true",
        "requires_external_asset": True,
    },
    {
        "target_id": "multi_seed_ablation",
        "readiness_check_id": "multi_seed_ablation_output",
        "objective": "Run and verify multi-seed ablations for baseline and non-baseline variants.",
        "artifact": "ablation_runs.csv, ablation_summary.csv, multi_seed_ablation_verified_manifest.json",
        "validation": "verify-ablation-evidence exits zero and readiness gate passes",
        "requires_external_asset": False,
    },
    {
        "target_id": "paper_equivalent_kb",
        "readiness_check_id": "paper_equivalent_kb",
        "objective": "Attach paper/code/data provenance with digests and paper-equivalent KB flag.",
        "artifact": "benchmark kb manifest with paper_kb_equivalent=true",
        "validation": "paper_equivalent_kb readiness check passes",
        "requires_external_asset": True,
    },
    {
        "target_id": "method_experience_failure_attribution",
        "readiness_check_id": "failure_attribution",
        "objective": "Accumulate success, failure, plateau, and policy mismatch experience records.",
        "artifact": "reports/method_experience_cache.json",
        "validation": "scientific_discovery_readiness failure_attribution check passes",
        "requires_external_asset": False,
    },
    {
        "target_id": "trace_and_claim_audit",
        "readiness_check_id": "scientific_claim_supported",
        "objective": "Keep trace summary and run metadata fail-closed against any claim overstatement.",
        "artifact": "trace_summary.json and champion/claim_gate.json",
        "validation": "trace-summary quality_gate passes without overclaim issues",
        "requires_external_asset": False,
    },
    {
        "target_id": "documentation_and_repro_packet",
        "readiness_check_id": "documentation",
        "objective": "Update docs, command recipes, and validation notes for each evidence increment.",
        "artifact": "docs/version_notes.md and docs/paper_workflow_readiness.md",
        "validation": "targeted tests plus docs linked from docs/index.md",
        "requires_external_asset": False,
    },
)


def build_iteration_campaign(
    *,
    benchmark_dir: Path,
    rounds: int = DEFAULT_CAMPAIGN_ROUNDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    selector_panel: list[dict[str, object]] | None = None,
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    readiness = build_paper_workflow_readiness_bundle(
        benchmark_dir=benchmark_dir,
        selector_panel=selector_panel,
        resource_constraints=resource_constraints,
        expert_blueprint_id=expert_blueprint_id,
        domain_approval_path=domain_approval_path,
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds,
        expected_variants=expected_variants,
        env=env,
    )
    blockers_by_check = {
        str(blocker.get("check_id")): dict(blocker)
        for blocker in readiness.get("blockers", [])
        if isinstance(blocker, dict)
    }
    rounds_payload = [
        _round_payload(index, batch_size, blockers_by_check)
        for index in range(1, rounds + 1)
    ]
    batches = _batches(rounds_payload, batch_size)
    return {
        "schema_version": ITERATION_CAMPAIGN_SCHEMA_VERSION,
        "campaign_version": ITERATION_CAMPAIGN_VERSION,
        "rounds_requested": rounds,
        "batch_size": batch_size,
        "status": "blocked" if readiness.get("status") == "blocked" else "ready",
        "completed_rounds": 0,
        "remaining_rounds": rounds,
        "benchmark": readiness.get("benchmark", {}),
        "paper_workflow_readiness_status": readiness.get("status"),
        "readiness_blockers": readiness.get("blockers", []),
        "target_sequence": [dict(target) for target in CAMPAIGN_TARGETS],
        "batches": batches,
        "rounds": rounds_payload,
        "claim_boundary": ITERATION_CAMPAIGN_CLAIM_BOUNDARY,
    }


def write_iteration_campaign(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    rounds: int = DEFAULT_CAMPAIGN_ROUNDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    selector_panel: list[dict[str, object]] | None = None,
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign = build_iteration_campaign(
        benchmark_dir=benchmark_dir,
        rounds=rounds,
        batch_size=batch_size,
        selector_panel=selector_panel,
        resource_constraints=resource_constraints,
        expert_blueprint_id=expert_blueprint_id,
        domain_approval_path=domain_approval_path,
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds,
        expected_variants=expected_variants,
        env=env,
    )
    campaign_json = output_dir / "iteration_campaign.json"
    campaign_md = output_dir / "iteration_campaign.md"
    _atomic_write_text(campaign_json, json.dumps(campaign, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(campaign_md, render_iteration_campaign_markdown(campaign))
    return {"campaign": campaign, "paths": {"campaign_json": str(campaign_json), "campaign_md": str(campaign_md)}}


def record_iteration_round_evidence(
    campaign_path: Path,
    *,
    round_index: int,
    evidence_path: Path,
    validation_command: str,
    validation_exit_code: int,
    validation_output_path: Path,
    notes: str = "",
) -> dict[str, Any]:
    if round_index <= 0:
        raise ValueError("round_index must be positive")
    if not validation_command.strip():
        raise ValueError("validation_command is required")
    campaign = _load_campaign(campaign_path)
    campaign_dir = campaign_path.parent
    evidence_file = _input_file_inside_campaign(
        evidence_path,
        campaign_dir,
        field_name="evidence_path",
    )
    validation_output_file = _input_file_inside_campaign(
        validation_output_path,
        campaign_dir,
        field_name="validation_output_path",
    )
    rounds = campaign.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("campaign rounds must be a list")
    round_item = _round_by_index(rounds, round_index)
    status = str(round_item.get("status", ""))
    if status == "blocked_by_readiness":
        raise ValueError(f"round {round_index} is blocked_by_readiness")
    if status == "completed":
        raise ValueError(f"round {round_index} is already completed")
    record_path = campaign_dir / f"iteration_round_{round_index:03d}_record.json"
    record = {
        "schema_version": 1,
        "record_version": "iteration_round_record.v1",
        "round_index": round_index,
        "batch_index": round_item.get("batch_index"),
        "target_id": round_item.get("target_id"),
        "readiness_check_id": round_item.get("readiness_check_id"),
        "previous_status": status,
        "validation_command": validation_command.strip(),
        "validation_result": {
            "command": validation_command.strip(),
            "exit_code": int(validation_exit_code),
            "output": _evidence_descriptor(validation_output_file, campaign_dir),
        },
        "notes": notes.strip(),
        "evidence": _evidence_descriptor(evidence_file, campaign_dir),
        "claim_boundary": ITERATION_ROUND_RECORD_CLAIM_BOUNDARY,
    }
    _atomic_write_text(record_path, json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    round_item["status"] = "completed"
    round_item["evidence_record_path"] = record_path.name
    round_item["evidence_sha256"] = record["evidence"]["sha256"]
    _refresh_campaign_progress(campaign)
    _atomic_write_text(campaign_path, json.dumps(campaign, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(campaign_path.with_suffix(".md"), render_iteration_campaign_markdown(campaign))
    return record


def verify_iteration_campaign(campaign_path: Path, *, require_complete: bool = False) -> dict[str, Any]:
    campaign = _load_campaign(campaign_path)
    campaign_dir = campaign_path.parent
    rounds = campaign.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("campaign rounds must be a list")
    issues: list[str] = []
    campaign_metadata_integrity = _validate_campaign_metadata(campaign, issues)
    valid_rounds, round_integrity = _validate_round_integrity(campaign, rounds, issues)
    batch_integrity = _validate_batch_integrity(campaign, valid_rounds, issues)
    completed_rounds = [
        item
        for item in valid_rounds
        if item.get("status") == "completed" and isinstance(item.get("round_index"), int)
    ]
    record_summaries: list[dict[str, Any]] = []
    for round_item in completed_rounds:
        round_index = int(round_item["round_index"])
        record_path_value = str(round_item.get("evidence_record_path") or "").strip()
        if not record_path_value:
            issues.append(f"completed round {round_index} is missing evidence_record_path")
            continue
        expected_record_path = f"iteration_round_{round_index:03d}_record.json"
        if record_path_value != expected_record_path:
            issues.append(
                f"record path mismatch for round {round_index}: "
                f"expected {expected_record_path}, got {record_path_value}"
            )
        record_path = _campaign_reference_file(
            record_path_value,
            campaign_dir,
            kind="record",
            round_index=round_index,
            issues=issues,
        )
        if record_path is None:
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"record file invalid JSON for round {round_index}: {record_path_value}")
            continue
        if not isinstance(record, dict):
            issues.append(f"record file must be an object for round {round_index}: {record_path_value}")
            continue
        record_metadata_mismatches = _record_metadata_mismatches(round_item, record, round_index, issues)
        record_round_index = record.get("round_index")
        if record_round_index != round_index:
            issues.append(f"record round mismatch for round {round_index}: got {record_round_index}")
        evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
        evidence_path_value = str(evidence.get("path") or "").strip()
        expected_digest = str(evidence.get("sha256") or "").strip()
        digest_match = False
        if not evidence_path_value:
            issues.append(f"record evidence path missing for round {round_index}")
        elif not expected_digest:
            issues.append(f"record evidence sha256 missing for round {round_index}")
        else:
            campaign_evidence_digest = str(round_item.get("evidence_sha256") or "").strip()
            if campaign_evidence_digest != expected_digest:
                record_metadata_mismatches.append(
                    {
                        "field": "evidence_sha256",
                        "expected": expected_digest,
                        "actual": campaign_evidence_digest,
                    }
                )
                issues.append(f"campaign evidence_sha256 mismatch for round {round_index}")
            evidence_file = _campaign_reference_file(
                evidence_path_value,
                campaign_dir,
                kind="evidence",
                round_index=round_index,
                issues=issues,
            )
            if evidence_file is not None:
                actual_digest = _sha256(evidence_file)
                digest_match = actual_digest == expected_digest
                if not digest_match:
                    issues.append(f"evidence digest mismatch for round {round_index}")
        validation = record.get("validation_result") if isinstance(record.get("validation_result"), dict) else {}
        validation_output = validation.get("output") if isinstance(validation.get("output"), dict) else {}
        validation_output_path_value = str(validation_output.get("path") or "").strip()
        expected_validation_digest = str(validation_output.get("sha256") or "").strip()
        validation_digest_match = False
        exit_code = validation.get("exit_code")
        if exit_code != 0:
            issues.append(f"validation exit code nonzero for round {round_index}: {exit_code}")
        if not validation_output_path_value:
            issues.append(f"validation output path missing for round {round_index}")
        elif not expected_validation_digest:
            issues.append(f"validation output sha256 missing for round {round_index}")
        else:
            validation_output_file = _campaign_reference_file(
                validation_output_path_value,
                campaign_dir,
                kind="validation output",
                round_index=round_index,
                issues=issues,
            )
            if validation_output_file is not None:
                actual_validation_digest = _sha256(validation_output_file)
                validation_digest_match = actual_validation_digest == expected_validation_digest
                if not validation_digest_match:
                    issues.append(f"validation output digest mismatch for round {round_index}")
        record_summaries.append(
            {
                "round_index": round_index,
                "target_id": round_item.get("target_id"),
                "record_path": record_path_value,
                "evidence_path": evidence_path_value or None,
                "evidence_digest_match": digest_match,
                "validation_exit_code": exit_code,
                "validation_output_path": validation_output_path_value or None,
                "validation_output_digest_match": validation_digest_match,
                "record_metadata_match": not record_metadata_mismatches,
                "record_metadata_mismatches": record_metadata_mismatches,
            }
        )
    declared_completed = campaign.get("completed_rounds")
    if declared_completed != len(completed_rounds):
        issues.append(
            f"completed_rounds mismatch: declared={declared_completed}, actual={len(completed_rounds)}"
        )
    declared_remaining = campaign.get("remaining_rounds")
    expected_remaining = len(valid_rounds) - len(completed_rounds)
    if declared_remaining != expected_remaining:
        issues.append(f"remaining_rounds mismatch: declared={declared_remaining}, actual={expected_remaining}")
    fully_completed = len(valid_rounds) > 0 and len(completed_rounds) == len(valid_rounds)
    if require_complete and not fully_completed:
        issues.append(f"campaign is not complete: completed={len(completed_rounds)}, expected={len(valid_rounds)}")
    return {
        "schema_version": 1,
        "verification_version": "iteration_campaign_verification.v1",
        "passed": not issues,
        "campaign_path": str(campaign_path),
        "require_complete": require_complete,
        "fully_completed": fully_completed,
        "round_count": len(valid_rounds),
        "completed_round_count": len(completed_rounds),
        "remaining_round_count": expected_remaining,
        "integrity_issue_count": len(issues),
        "campaign_metadata_integrity": campaign_metadata_integrity,
        "round_integrity": round_integrity,
        "batch_integrity": batch_integrity,
        "issues": issues,
        "records": record_summaries,
        "claim_boundary": (
            "This verification checks campaign accounting and evidence-record integrity only. "
            "It does not validate scientific claims or blocked external assets."
        ),
    }


def write_iteration_campaign_verification(campaign_path: Path, *, require_complete: bool = False) -> dict[str, Any]:
    verification = verify_iteration_campaign(campaign_path, require_complete=require_complete)
    output_path = campaign_path.parent / "iteration_campaign_verification.json"
    _atomic_write_text(output_path, json.dumps(verification, indent=2, sort_keys=True, allow_nan=False))
    return {"verification": verification, "path": str(output_path)}


def _validate_campaign_metadata(campaign: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    metadata_mismatches: list[dict[str, Any]] = []

    def record_mismatch(field: str, expected: object, actual: object, message: str) -> None:
        metadata_mismatches.append({"field": field, "expected": expected, "actual": actual})
        issues.append(message)

    campaign_version = campaign.get("campaign_version")
    if campaign_version != ITERATION_CAMPAIGN_VERSION:
        record_mismatch(
            "campaign_version",
            ITERATION_CAMPAIGN_VERSION,
            campaign_version,
            f"campaign_version mismatch: expected {ITERATION_CAMPAIGN_VERSION}, got {campaign_version}",
        )

    claim_boundary = campaign.get("claim_boundary")
    if claim_boundary != ITERATION_CAMPAIGN_CLAIM_BOUNDARY:
        record_mismatch(
            "claim_boundary",
            ITERATION_CAMPAIGN_CLAIM_BOUNDARY,
            claim_boundary,
            "claim_boundary mismatch for iteration campaign",
        )

    readiness_blockers = campaign.get("readiness_blockers")
    readiness_blockers_valid = isinstance(readiness_blockers, list)
    if not readiness_blockers_valid:
        issues.append("readiness_blockers must be a list")
    blocker_count = len(readiness_blockers) if readiness_blockers_valid else None
    expected_readiness_status = None
    if blocker_count is not None:
        expected_readiness_status = "blocked" if blocker_count > 0 else "ready"

    readiness_status = campaign.get("paper_workflow_readiness_status")
    if readiness_status not in SUPPORTED_READINESS_STATUSES:
        issues.append(f"paper_workflow_readiness_status has unsupported value: {readiness_status}")
    if expected_readiness_status is not None and readiness_status != expected_readiness_status:
        record_mismatch(
            "paper_workflow_readiness_status",
            expected_readiness_status,
            readiness_status,
            (
                "paper_workflow_readiness_status mismatch: "
                f"expected {expected_readiness_status}, got {readiness_status}"
            ),
        )

    expected_campaign_status = expected_readiness_status
    campaign_status = campaign.get("status")
    if campaign_status not in SUPPORTED_CAMPAIGN_STATUSES:
        issues.append(f"campaign status has unsupported value: {campaign_status}")
    if expected_campaign_status is not None and campaign_status != expected_campaign_status:
        record_mismatch(
            "status",
            expected_campaign_status,
            campaign_status,
            f"campaign status mismatch: expected {expected_campaign_status}, got {campaign_status}",
        )

    return {
        "campaign_metadata_valid": not metadata_mismatches and readiness_blockers_valid,
        "expected_campaign_version": ITERATION_CAMPAIGN_VERSION,
        "expected_claim_boundary": ITERATION_CAMPAIGN_CLAIM_BOUNDARY,
        "expected_status": expected_campaign_status,
        "expected_paper_workflow_readiness_status": expected_readiness_status,
        "readiness_blocker_count": blocker_count,
        "metadata_mismatches": metadata_mismatches,
    }


def _validate_batch_integrity(
    campaign: dict[str, Any],
    valid_rounds: list[dict[str, Any]],
    issues: list[str],
) -> dict[str, Any]:
    batches = campaign.get("batches")
    batch_mismatches: list[dict[str, Any]] = []
    if not isinstance(batches, list):
        issues.append("campaign batches must be a list")
        return {
            "batch_schema_valid": False,
            "expected_batch_count": 0,
            "actual_batch_count": 0,
            "batch_mismatches": batch_mismatches,
        }
    batch_size = campaign.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        return {
            "batch_schema_valid": False,
            "expected_batch_count": 0,
            "actual_batch_count": len(batches),
            "batch_mismatches": batch_mismatches,
        }
    expected_batches = _recompute_batch_summaries(valid_rounds, batch_size)
    if len(batches) != len(expected_batches):
        issues.append(f"batch count mismatch: expected {len(expected_batches)}, got {len(batches)}")
    for index, expected in enumerate(expected_batches):
        if index >= len(batches):
            batch_mismatches.append(
                {
                    "batch_index": expected["batch_index"],
                    "field": "batch",
                    "expected": expected,
                    "actual": None,
                }
            )
            continue
        actual = batches[index]
        if not isinstance(actual, dict):
            batch_mismatches.append(
                {
                    "batch_index": expected["batch_index"],
                    "field": "batch",
                    "expected": expected,
                    "actual": actual,
                }
            )
            issues.append(f"batch entry {index + 1} is not an object")
            continue
        for field, expected_value in expected.items():
            actual_value = actual.get(field)
            if actual_value != expected_value:
                batch_mismatches.append(
                    {
                        "batch_index": expected["batch_index"],
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
                issues.append(
                    f"batch {expected['batch_index']} {field} mismatch: "
                    f"expected {expected_value}, got {actual_value}"
                )
    return {
        "batch_schema_valid": not batch_mismatches and len(batches) == len(expected_batches),
        "expected_batch_count": len(expected_batches),
        "actual_batch_count": len(batches),
        "batch_mismatches": batch_mismatches,
    }


def _record_metadata_mismatches(
    round_item: dict[str, Any],
    record: dict[str, Any],
    round_index: int,
    issues: list[str],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    if record.get("schema_version") != 1:
        mismatches.append({"field": "schema_version", "expected": 1, "actual": record.get("schema_version")})
        issues.append(f"record schema_version mismatch for round {round_index}: got {record.get('schema_version')}")
    if record.get("record_version") != "iteration_round_record.v1":
        mismatches.append(
            {
                "field": "record_version",
                "expected": "iteration_round_record.v1",
                "actual": record.get("record_version"),
            }
        )
        issues.append(f"record_version mismatch for round {round_index}: got {record.get('record_version')}")
    if record.get("claim_boundary") != ITERATION_ROUND_RECORD_CLAIM_BOUNDARY:
        mismatches.append(
            {
                "field": "claim_boundary",
                "expected": ITERATION_ROUND_RECORD_CLAIM_BOUNDARY,
                "actual": record.get("claim_boundary"),
            }
        )
        issues.append(f"record claim_boundary mismatch for round {round_index}")
    for field in ("batch_index", "target_id", "readiness_check_id"):
        expected_value = round_item.get(field)
        actual_value = record.get(field)
        if actual_value != expected_value:
            mismatches.append({"field": field, "expected": expected_value, "actual": actual_value})
            issues.append(
                f"record {field} mismatch for round {round_index}: "
                f"expected {expected_value}, got {actual_value}"
            )
    previous_status = record.get("previous_status")
    if previous_status != "planned":
        mismatches.append({"field": "previous_status", "expected": "planned", "actual": previous_status})
        issues.append(
            f"record previous_status mismatch for round {round_index}: "
            f"expected planned, got {previous_status}"
        )
    validation = record.get("validation_result") if isinstance(record.get("validation_result"), dict) else {}
    validation_command = str(record.get("validation_command") or "").strip()
    validation_result_command = str(validation.get("command") or "").strip()
    if not validation_command or validation_command != validation_result_command:
        mismatches.append(
            {
                "field": "validation_command",
                "expected": validation_command,
                "actual": validation_result_command,
            }
        )
        issues.append(f"record validation command mismatch for round {round_index}")
    return mismatches


def _input_file_inside_campaign(path: Path, campaign_dir: Path, *, field_name: str) -> Path:
    candidate = path if path.is_absolute() else campaign_dir / path
    if not candidate.is_file():
        raise ValueError(f"{field_name} does not exist: {path}")
    resolved = candidate.resolve(strict=True)
    root = campaign_dir.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must resolve inside campaign directory: {path}") from exc
    return resolved


def _campaign_reference_file(
    path_value: str,
    campaign_dir: Path,
    *,
    kind: str,
    round_index: int,
    issues: list[str],
) -> Path | None:
    path = Path(path_value)
    if path.is_absolute():
        issues.append(f"{kind} path for round {round_index} must be relative to campaign directory")
        return None
    candidate = campaign_dir / path
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        issues.append(f"{kind} file missing for round {round_index}: {path_value}")
        return None
    root = campaign_dir.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        issues.append(f"{kind} path for round {round_index} escapes campaign directory")
        return None
    return resolved


def _validate_round_integrity(
    campaign: dict[str, Any],
    rounds: list[object],
    issues: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_rounds: list[dict[str, Any]] = []
    round_indices: list[int] = []
    duplicate_indices: list[int] = []
    seen_indices: set[int] = set()
    invalid_round_entries = 0
    unsupported_statuses: list[dict[str, Any]] = []
    round_target_mismatches: list[dict[str, Any]] = []
    batch_index_mismatches: list[dict[str, Any]] = []
    completed_rounds_with_blockers: list[int] = []
    schema_target_sequence = [dict(target) for target in CAMPAIGN_TARGETS]
    target_sequence_valid = campaign.get("target_sequence") == schema_target_sequence
    if not target_sequence_valid:
        issues.append("target_sequence does not match campaign schema targets")
    batch_size = campaign.get("batch_size")
    batch_size_valid = isinstance(batch_size, int) and batch_size > 0
    if not batch_size_valid:
        issues.append(f"batch_size must be a positive integer, got {batch_size}")
    for position, item in enumerate(rounds, start=1):
        if not isinstance(item, dict):
            invalid_round_entries += 1
            issues.append(f"round entry {position} is not an object")
            continue
        valid_rounds.append(item)
        round_index = item.get("round_index")
        if not isinstance(round_index, int) or round_index <= 0:
            invalid_round_entries += 1
            issues.append(f"round entry {position} has invalid round_index: {round_index}")
        else:
            round_indices.append(round_index)
            if round_index in seen_indices and round_index not in duplicate_indices:
                duplicate_indices.append(round_index)
            seen_indices.add(round_index)
            expected_target = schema_target_sequence[(round_index - 1) % len(schema_target_sequence)]
            for round_field, target_field in (
                ("target_id", "target_id"),
                ("readiness_check_id", "readiness_check_id"),
                ("objective", "objective"),
                ("evidence_artifact", "artifact"),
                ("validation", "validation"),
                ("requires_external_asset", "requires_external_asset"),
            ):
                expected_value = expected_target[target_field]
                actual_value = item.get(round_field)
                if actual_value != expected_value:
                    round_target_mismatches.append(
                        {
                            "round_index": round_index,
                            "field": round_field,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    )
                    issues.append(
                        f"round {round_index} {round_field} mismatch: "
                        f"expected {expected_value}, got {actual_value}"
                    )
            if batch_size_valid:
                expected_batch_index = ((round_index - 1) // batch_size) + 1
                actual_batch_index = item.get("batch_index")
                if actual_batch_index != expected_batch_index:
                    batch_index_mismatches.append(
                        {
                            "round_index": round_index,
                            "expected": expected_batch_index,
                            "actual": actual_batch_index,
                        }
                    )
                    issues.append(
                        f"round {round_index} batch_index mismatch: "
                        f"expected {expected_batch_index}, got {actual_batch_index}"
                    )
        status = item.get("status")
        if status not in SUPPORTED_ROUND_STATUSES:
            unsupported_statuses.append({"round_index": round_index, "status": status})
            issues.append(f"round {round_index} has unsupported status: {status}")
        if status == "completed" and item.get("requires_external_asset") is True and item.get("blocked_by"):
            completed_rounds_with_blockers.append(round_index)
            issues.append(
                "round {round_index} is completed but still has readiness blocker {check_id}".format(
                    round_index=round_index,
                    check_id=item.get("readiness_check_id"),
                )
            )
    rounds_requested = campaign.get("rounds_requested")
    rounds_requested_valid = isinstance(rounds_requested, int) and rounds_requested > 0
    expected_indices = set(range(1, rounds_requested + 1)) if rounds_requested_valid else set()
    actual_indices = set(round_indices)
    missing_indices = sorted(expected_indices - actual_indices)
    extra_indices = sorted(actual_indices - expected_indices)
    duplicate_indices.sort()
    if duplicate_indices:
        issues.append(f"duplicate round_index values: {duplicate_indices}")
    if missing_indices:
        issues.append(f"missing round_index values: {missing_indices}")
    if extra_indices:
        issues.append(f"unexpected round_index values: {extra_indices}")
    if not rounds_requested_valid:
        issues.append(f"rounds_requested must be a positive integer, got {rounds_requested}")
    return valid_rounds, {
        "round_schema_valid": not (
            invalid_round_entries
            or unsupported_statuses
            or duplicate_indices
            or missing_indices
            or extra_indices
            or not rounds_requested_valid
            or not batch_size_valid
            or not target_sequence_valid
            or round_target_mismatches
            or batch_index_mismatches
            or completed_rounds_with_blockers
        ),
        "invalid_round_entry_count": invalid_round_entries,
        "duplicate_round_indices": duplicate_indices,
        "missing_round_indices": missing_indices,
        "unexpected_round_indices": extra_indices,
        "unsupported_statuses": unsupported_statuses,
        "target_sequence_valid": target_sequence_valid,
        "round_target_mismatches": round_target_mismatches,
        "batch_index_mismatches": batch_index_mismatches,
        "completed_rounds_with_blockers": completed_rounds_with_blockers,
    }


def render_iteration_campaign_markdown(campaign: dict[str, Any]) -> str:
    lines = [
        "# Iteration Campaign",
        "",
        f"- status: {campaign.get('status')}",
        f"- rounds_requested: {campaign.get('rounds_requested')}",
        f"- batch_size: {campaign.get('batch_size')}",
        f"- completed_rounds: {campaign.get('completed_rounds')}",
        f"- remaining_rounds: {campaign.get('remaining_rounds')}",
        f"- paper_workflow_readiness_status: {campaign.get('paper_workflow_readiness_status')}",
        "",
        "## Batches",
        "",
    ]
    for batch in campaign.get("batches", []):
        if not isinstance(batch, dict):
            continue
        lines.append(
            "- batch {batch_index}: rounds {round_start}-{round_end}; targets: {targets}".format(
                batch_index=batch.get("batch_index"),
                round_start=batch.get("round_start"),
                round_end=batch.get("round_end"),
                targets=", ".join(str(item) for item in batch.get("target_ids", [])),
            )
        )
    lines.extend(["", "## First 10 Rounds", ""])
    for round_item in campaign.get("rounds", [])[:10]:
        if not isinstance(round_item, dict):
            continue
        lines.append(
            "- round {round_index}: {target_id} [{status}]".format(
                round_index=round_item.get("round_index"),
                target_id=round_item.get("target_id"),
                status=round_item.get("status"),
            )
        )
    lines.extend(["", "## Claim Boundary", "", str(campaign.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _load_campaign(campaign_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"campaign_path does not exist: {campaign_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"campaign_path is invalid JSON: {campaign_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("campaign payload must be a JSON object")
    if payload.get("schema_version") != ITERATION_CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("unsupported iteration campaign schema_version")
    return payload


def _round_by_index(rounds: list[object], round_index: int) -> dict[str, Any]:
    for item in rounds:
        if isinstance(item, dict) and item.get("round_index") == round_index:
            return item
    raise ValueError(f"round {round_index} is not present in campaign")


def _evidence_descriptor(evidence_file: Path, campaign_dir: Path) -> dict[str, Any]:
    resolved = evidence_file.resolve(strict=True)
    root = campaign_dir.resolve(strict=False)
    try:
        display_path = str(resolved.relative_to(root))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_campaign_progress(campaign: dict[str, Any]) -> None:
    rounds = [item for item in campaign.get("rounds", []) if isinstance(item, dict)]
    completed = sum(1 for item in rounds if item.get("status") == "completed")
    campaign["completed_rounds"] = completed
    campaign["remaining_rounds"] = max(0, len(rounds) - completed)
    batch_size = campaign.get("batch_size")
    if isinstance(batch_size, int) and batch_size > 0:
        campaign["batches"] = _recompute_batch_summaries(rounds, batch_size)


def _round_payload(
    round_index: int,
    batch_size: int,
    blockers_by_check: dict[str, dict[str, object]],
) -> dict[str, Any]:
    target = CAMPAIGN_TARGETS[(round_index - 1) % len(CAMPAIGN_TARGETS)]
    readiness_check_id = str(target["readiness_check_id"])
    blocker = blockers_by_check.get(readiness_check_id)
    status = "blocked_by_readiness" if blocker and target["requires_external_asset"] else "planned"
    return {
        "round_index": round_index,
        "batch_index": ((round_index - 1) // batch_size) + 1,
        "target_id": target["target_id"],
        "readiness_check_id": readiness_check_id,
        "status": status,
        "objective": target["objective"],
        "evidence_artifact": target["artifact"],
        "validation": target["validation"],
        "requires_external_asset": target["requires_external_asset"],
        "blocked_by": blocker or None,
        "claim_boundary": "Round output must be validated by artifacts before it can affect readiness.",
    }


def _batches(rounds_payload: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    return _recompute_batch_summaries(rounds_payload, batch_size)


def _recompute_batch_summaries(rounds_payload: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    valid_items = [
        item
        for item in rounds_payload
        if isinstance(item.get("round_index"), int) and int(item["round_index"]) > 0
    ]
    valid_items.sort(key=lambda item: int(item["round_index"]))
    for start in range(0, len(valid_items), batch_size):
        items = valid_items[start : start + batch_size]
        batches.append(
            {
                "batch_index": (start // batch_size) + 1,
                "round_start": items[0]["round_index"],
                "round_end": items[-1]["round_index"],
                "target_ids": sorted({str(item["target_id"]) for item in items}),
                "blocked_round_count": sum(1 for item in items if item["status"] == "blocked_by_readiness"),
                "planned_round_count": sum(1 for item in items if item["status"] == "planned"),
                "completed_round_count": sum(1 for item in items if item["status"] == "completed"),
                "remaining_round_count": sum(1 for item in items if item["status"] != "completed"),
            }
        )
    return batches
