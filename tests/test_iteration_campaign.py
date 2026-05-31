from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agenticsciml.iteration_campaign import build_iteration_campaign, write_iteration_campaign


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
