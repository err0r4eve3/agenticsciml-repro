from __future__ import annotations

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


def render_iteration_campaign_markdown(campaign: dict[str, Any]) -> str:
    lines = [
        "# Iteration Campaign",
        "",
        f"- status: {campaign.get('status')}",
        f"- rounds_requested: {campaign.get('rounds_requested')}",
        f"- batch_size: {campaign.get('batch_size')}",
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
