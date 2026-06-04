from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agenticsciml.storage import _atomic_write_text


SELECTOR_EVIDENCE_PACKET_SCHEMA_VERSION = 1
SELECTOR_VOTES_PATH = Path("reports") / "selector_votes.json"
SELECTOR_HETEROGENEITY_PATH = Path("reports") / "selector_heterogeneity.json"
SELECTOR_EVIDENCE_PACKET_JSON = Path("reports") / "selector_evidence_packet.json"
SELECTOR_EVIDENCE_PACKET_MD = Path("reports") / "selector_evidence_packet.md"


def build_selector_evidence_packet(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    selector_votes_path = run_dir / SELECTOR_VOTES_PATH
    selector_report_path = run_dir / SELECTOR_HETEROGENEITY_PATH
    votes_payload = _read_json(selector_votes_path)
    selector_report = _read_json(selector_report_path)
    votes = votes_payload.get("votes") if isinstance(votes_payload.get("votes"), list) else []
    members = (
        votes_payload.get("selector_panel_members")
        if isinstance(votes_payload.get("selector_panel_members"), list)
        else []
    )
    summary = _runtime_vote_summary(votes)
    ensemble_mode = str(votes_payload.get("ensemble_mode", ""))
    selector_report_ready = selector_report.get("heterogeneous_selector_evidence") is True
    paper_workflow_selector_ready = (
        selector_votes_path.exists()
        and ensemble_mode == "configured_selector_panel"
        and summary["real_vote_count"] >= 2
        and summary["distinct_member_vote_count"] >= 2
        and summary["mock_vote_count"] == 0
        and summary["repeated_member_votes"] is False
        and (summary["provider_diversity"] is True or summary["actual_model_diversity"] is True)
    )
    blockers = _selector_blockers(
        selector_votes_exists=selector_votes_path.exists(),
        ensemble_mode=ensemble_mode,
        summary=summary,
    )
    return {
        "schema_version": SELECTOR_EVIDENCE_PACKET_SCHEMA_VERSION,
        "status": "ready" if paper_workflow_selector_ready else "blocked",
        "paper_workflow_selector_ready": paper_workflow_selector_ready,
        "scientific_claim_supported": False,
        "run_dir": str(run_dir),
        "selector_votes_schema_version": votes_payload.get("schema_version"),
        "ensemble_mode": ensemble_mode,
        "runtime_vote_summary": summary,
        "selector_panel_member_count": len(members),
        "selector_heterogeneity_report_ready": selector_report_ready,
        "source_artifacts": {
            "selector_votes": _source_artifact(selector_votes_path, run_dir=run_dir),
            "selector_heterogeneity": _source_artifact(selector_report_path, run_dir=run_dir),
        },
        "blockers": blockers,
        "claim_boundary": (
            "This packet proves only runtime selector evidence. It does not support a scientific claim, "
            "does not replace evaluator scores, and is paper_workflow-ready only with at least two "
            "real heterogeneous selector votes from distinct runtime paths."
        ),
    }


def write_selector_evidence_packet(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_root = Path(output_dir) if output_dir is not None else run_dir
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    packet = build_selector_evidence_packet(run_dir)
    packet_json = reports_dir / SELECTOR_EVIDENCE_PACKET_JSON.name
    packet_md = reports_dir / SELECTOR_EVIDENCE_PACKET_MD.name
    _atomic_write_text(packet_json, json.dumps(packet, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(packet_md, render_selector_evidence_packet_markdown(packet))
    return {
        "packet": packet,
        "paths": {
            "packet_json": str(packet_json),
            "packet_md": str(packet_md),
        },
    }


def render_selector_evidence_packet_markdown(packet: dict[str, Any]) -> str:
    summary = packet.get("runtime_vote_summary") if isinstance(packet.get("runtime_vote_summary"), dict) else {}
    blockers = packet.get("blockers") if isinstance(packet.get("blockers"), list) else []
    lines = [
        "# Selector Evidence Packet",
        "",
        f"- status: {packet.get('status')}",
        f"- paper_workflow_selector_ready: {packet.get('paper_workflow_selector_ready')}",
        f"- scientific_claim_supported: {packet.get('scientific_claim_supported')}",
        f"- real_vote_count: {summary.get('real_vote_count')}",
        f"- unique_providers: {', '.join(summary.get('unique_providers', [])) if isinstance(summary.get('unique_providers'), list) else ''}",
        f"- unique_actual_models: {', '.join(summary.get('unique_actual_models', [])) if isinstance(summary.get('unique_actual_models'), list) else ''}",
        "",
        "## Blockers",
    ]
    if blockers:
        lines.extend(f"- {item.get('check_id')}: {item.get('message')}" for item in blockers if isinstance(item, dict))
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", "", str(packet.get("claim_boundary", ""))])
    return "\n".join(lines) + "\n"


def _runtime_vote_summary(votes: list[object]) -> dict[str, Any]:
    normalized_votes = [vote for vote in votes if isinstance(vote, dict)]
    real_votes = [vote for vote in normalized_votes if not _selector_vote_is_mock(vote)]
    member_vote_counts: dict[str, int] = {}
    for vote in normalized_votes:
        member_id = str(vote.get("member_id", "unknown"))
        member_vote_counts[member_id] = member_vote_counts.get(member_id, 0) + 1
    unique_models = sorted({str(vote.get("actual_model")) for vote in real_votes if vote.get("actual_model")})
    unique_providers = sorted({str(vote.get("provider")) for vote in real_votes if vote.get("provider")})
    unique_adapter_types = sorted(
        {str(vote.get("adapter_type")) for vote in real_votes if vote.get("adapter_type")}
    )
    return {
        "actual_vote_count": len(normalized_votes),
        "real_vote_count": len(real_votes),
        "mock_vote_count": len(normalized_votes) - len(real_votes),
        "distinct_member_vote_count": len(member_vote_counts),
        "member_vote_counts": dict(sorted(member_vote_counts.items())),
        "repeated_member_votes": any(count > 1 for count in member_vote_counts.values()),
        "unique_actual_models": unique_models,
        "unique_providers": unique_providers,
        "unique_adapter_types": unique_adapter_types,
        "actual_model_diversity": len(unique_models) > 1,
        "provider_diversity": len(unique_providers) > 1,
    }


def _selector_blockers(
    *,
    selector_votes_exists: bool,
    ensemble_mode: str,
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not selector_votes_exists:
        blockers.append(
            {
                "check_id": "selector_votes_artifact",
                "message": "reports/selector_votes.json is missing",
            }
        )
    if ensemble_mode != "configured_selector_panel":
        blockers.append(
            {
                "check_id": "configured_selector_panel",
                "message": "selector evidence requires ensemble_mode=configured_selector_panel",
            }
        )
    if summary["real_vote_count"] < 2 or summary["mock_vote_count"] > 0:
        blockers.append(
            {
                "check_id": "real_selector_votes",
                "message": "at least two non-mock runtime selector votes are required",
            }
        )
    if summary["distinct_member_vote_count"] < 2 or summary["repeated_member_votes"] is True:
        blockers.append(
            {
                "check_id": "distinct_runtime_members",
                "message": "selector votes must come from distinct selector members without repeated-member fanout",
            }
        )
    if summary["provider_diversity"] is not True and summary["actual_model_diversity"] is not True:
        blockers.append(
            {
                "check_id": "heterogeneous_runtime_path",
                "message": "runtime selector votes are not heterogeneous by provider or actual model",
            }
        )
    return blockers


def _selector_vote_is_mock(vote: dict[str, object]) -> bool:
    return (
        str(vote.get("actual_model", "")).lower() == "mock"
        or str(vote.get("provider", "")).lower() == "mockllmclient"
        or str(vote.get("adapter_type", "")).lower() == "mock_local"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_artifact(path: Path, *, run_dir: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path.relative_to(run_dir)) if exists and path.is_relative_to(run_dir) else str(path),
        "exists": exists,
        "sha256": _sha256(path) if exists else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
