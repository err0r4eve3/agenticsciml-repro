from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agenticsciml.paper_workflow_readiness import build_paper_workflow_readiness_bundle
from agenticsciml.storage import _atomic_write_text


ITERATION_CAMPAIGN_SCHEMA_VERSION = 1
DEFAULT_CAMPAIGN_ROUNDS = 60
DEFAULT_BATCH_SIZE = 10

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
        "campaign_version": "iteration_campaign.v1",
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
        "claim_boundary": (
            "This campaign is an execution plan for iterative engineering evidence. Planned or blocked rounds "
            "do not count as scientific discovery evidence until validated run artifacts exist."
        ),
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
    evidence_file = evidence_path if evidence_path.is_absolute() else campaign_dir / evidence_path
    if not evidence_file.is_file():
        raise ValueError(f"evidence_path does not exist: {evidence_path}")
    validation_output_file = (
        validation_output_path
        if validation_output_path.is_absolute()
        else campaign_dir / validation_output_path
    )
    if not validation_output_file.is_file():
        raise ValueError(f"validation_output_path does not exist: {validation_output_path}")
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
        "claim_boundary": "This record proves an engineering iteration artifact exists; it is not scientific discovery evidence.",
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
    valid_rounds = [item for item in rounds if isinstance(item, dict)]
    completed_rounds = [item for item in valid_rounds if item.get("status") == "completed"]
    issues: list[str] = []
    record_summaries: list[dict[str, Any]] = []
    for round_item in completed_rounds:
        round_index = int(round_item["round_index"])
        record_path_value = str(round_item.get("evidence_record_path") or "").strip()
        if not record_path_value:
            issues.append(f"completed round {round_index} is missing evidence_record_path")
            continue
        record_path = campaign_dir / record_path_value
        if not record_path.is_file():
            issues.append(f"record file missing for round {round_index}: {record_path_value}")
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"record file invalid JSON for round {round_index}: {record_path_value}")
            continue
        if not isinstance(record, dict):
            issues.append(f"record file must be an object for round {round_index}: {record_path_value}")
            continue
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
            evidence_path = Path(evidence_path_value)
            evidence_file = evidence_path if evidence_path.is_absolute() else campaign_dir / evidence_path
            if not evidence_file.is_file():
                issues.append(f"evidence file missing for round {round_index}: {evidence_path_value}")
            else:
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
            validation_output_path = Path(validation_output_path_value)
            validation_output_file = (
                validation_output_path
                if validation_output_path.is_absolute()
                else campaign_dir / validation_output_path
            )
            if not validation_output_file.is_file():
                issues.append(
                    f"validation output file missing for round {round_index}: {validation_output_path_value}"
                )
            else:
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
    for batch in campaign.get("batches", []):
        if not isinstance(batch, dict):
            continue
        start = int(batch.get("round_start", 0))
        end = int(batch.get("round_end", 0))
        batch_rounds = [
            item
            for item in rounds
            if isinstance(item.get("round_index"), int) and start <= int(item["round_index"]) <= end
        ]
        batch["completed_round_count"] = sum(1 for item in batch_rounds if item.get("status") == "completed")
        batch["remaining_round_count"] = len(batch_rounds) - int(batch["completed_round_count"])


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
    batches: list[dict[str, Any]] = []
    for start in range(0, len(rounds_payload), batch_size):
        items = rounds_payload[start : start + batch_size]
        batches.append(
            {
                "batch_index": items[0]["batch_index"],
                "round_start": items[0]["round_index"],
                "round_end": items[-1]["round_index"],
                "target_ids": sorted({str(item["target_id"]) for item in items}),
                "blocked_round_count": sum(1 for item in items if item["status"] != "planned"),
                "planned_round_count": sum(1 for item in items if item["status"] == "planned"),
            }
        )
    return batches
