from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.selector_evidence import (
    build_selector_evidence_packet,
    write_selector_evidence_packet,
)


def test_selector_evidence_packet_accepts_runtime_heterogeneous_real_votes(tmp_path: Path) -> None:
    run_dir = _write_run_with_selector_votes(
        tmp_path,
        votes=[
            _vote("selector_openai", model="gpt-5-mini", provider="openai"),
            _vote("selector_compatible", model="deepseek-v4-pro", provider="api.deepseek.com"),
        ],
    )

    packet = build_selector_evidence_packet(run_dir)

    assert packet["status"] == "ready"
    assert packet["paper_workflow_selector_ready"] is True
    assert packet["scientific_claim_supported"] is False
    assert packet["runtime_vote_summary"]["real_vote_count"] == 2
    assert packet["runtime_vote_summary"]["unique_providers"] == ["api.deepseek.com", "openai"]
    assert packet["runtime_vote_summary"]["unique_actual_models"] == ["deepseek-v4-pro", "gpt-5-mini"]
    assert packet["source_artifacts"]["selector_votes"]["sha256"]
    assert packet["blockers"] == []


def test_selector_evidence_packet_blocks_mock_or_non_heterogeneous_votes(tmp_path: Path) -> None:
    run_dir = _write_run_with_selector_votes(
        tmp_path,
        votes=[
            _vote("selector_a", model="mock", provider="MockLLMClient", adapter="mock_local"),
            _vote("selector_b", model="mock", provider="MockLLMClient", adapter="mock_local"),
        ],
    )

    packet = build_selector_evidence_packet(run_dir)

    assert packet["status"] == "blocked"
    assert packet["paper_workflow_selector_ready"] is False
    blocker_ids = {item["check_id"] for item in packet["blockers"]}
    assert "real_selector_votes" in blocker_ids
    assert "heterogeneous_runtime_path" in blocker_ids
    assert packet["scientific_claim_supported"] is False


def test_write_selector_evidence_packet_outputs_json_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_run_with_selector_votes(
        tmp_path,
        votes=[
            _vote("selector_openai", model="gpt-5-mini", provider="openai"),
            _vote("selector_compatible", model="deepseek-v4-pro", provider="api.deepseek.com"),
        ],
    )

    result = write_selector_evidence_packet(run_dir)

    packet_path = Path(result["paths"]["packet_json"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet_path == run_dir / "reports" / "selector_evidence_packet.json"
    assert packet["paper_workflow_selector_ready"] is True
    assert (run_dir / "reports" / "selector_evidence_packet.md").exists()


def test_cli_generate_selector_evidence_writes_packet(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    run_dir = _write_run_with_selector_votes(
        tmp_path,
        votes=[
            _vote("selector_openai", model="gpt-5-mini", provider="openai"),
            _vote("selector_compatible", model="deepseek-v4-pro", provider="api.deepseek.com"),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "generate-selector-evidence",
            str(run_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    packet_path = Path(result.stdout.strip())
    assert packet_path == run_dir / "reports" / "selector_evidence_packet.json"
    assert json.loads(packet_path.read_text(encoding="utf-8"))["status"] == "ready"


def _write_run_with_selector_votes(tmp_path: Path, *, votes: list[dict[str, object]]) -> Path:
    run_dir = tmp_path / "selector-run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    members = [
        {
            "member_id": vote["member_id"],
            "role": "selector",
            "configured_model": vote["configured_model"],
            "configured_base_url": vote["configured_base_url"],
            "actual_model": vote["actual_model"],
            "provider": vote["provider"],
            "adapter_type": vote["adapter_type"],
            "provider_capabilities": vote["provider_capabilities"],
            "source": "selector_panel",
        }
        for vote in votes
    ]
    (reports / "selector_votes.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "ensemble_mode": "configured_selector_panel",
                "selector_panel_members": members,
                "votes": votes,
                "vote_counts": {"solution_001": 2},
                "selected_parent_ids": ["solution_000", "solution_001"],
                "claim_boundary": "runtime selector votes only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def _vote(
    member_id: str,
    *,
    model: str,
    provider: str,
    adapter: str = "openai_compatible_chat",
) -> dict[str, object]:
    return {
        "vote_index": 1,
        "member_id": member_id,
        "member_role": "selector",
        "configured_model": model,
        "configured_base_url": None if provider == "openai" else f"https://{provider}",
        "actual_model": model,
        "provider": provider,
        "adapter_type": adapter,
        "provider_capabilities": {
            "supports_responses": provider == "openai",
            "supports_structured_outputs": True,
            "supports_image_inputs": provider == "openai",
        },
        "source": "selector_panel",
        "selected_parent_ids": ["solution_001"],
        "rationale": "Test vote.",
    }
