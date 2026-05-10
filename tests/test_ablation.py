from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.ablation import run_ablation


def test_ablation_runner_writes_summary_and_report(tmp_path: Path) -> None:
    result = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        seeds=[0],
        variants=["root_only", "no_kb", "kb", "random_kb", "no_critic", "no_debugger"],
    )

    rows = list(csv.DictReader(result.summary_csv.open(encoding="utf-8")))
    variants = {row["variant"] for row in rows}
    report = result.report_md.read_text(encoding="utf-8")

    assert variants == {"root_only", "no_kb", "kb", "random_kb", "no_critic", "no_debugger"}
    assert result.summary_csv.exists()
    assert result.report_md.exists()
    assert "champion/root improvement" in rows[0]
    assert "valid_runs" in rows[0]
    assert "Champion/root improvement" in report
    assert "Valid runs" in report
    for row in rows:
        assert Path(row["example_run_dir"]).exists()

    no_critic_run = Path(next(row["example_run_dir"] for row in rows if row["variant"] == "no_critic"))
    no_critic_events = [
        json.loads(line)
        for line in (no_critic_run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not any(
        event.get("event_type") == "generation_span"
        and event.get("metadata", {}).get("spec_role") == "critic"
        for event in no_critic_events
    )

    no_debugger_run = Path(next(row["example_run_dir"] for row in rows if row["variant"] == "no_debugger"))
    no_debugger_config = json.loads((no_debugger_run / "config.json").read_text(encoding="utf-8"))
    assert no_debugger_config["evolution"]["use_debugger"] is False


def test_cli_ablate_command_runs_mock_pipeline(tmp_path: Path, cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "ablate",
            "examples/function_approx",
            "--seeds",
            "0",
            "--variants",
            "root_only,kb",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    summary_path = Path(result.stdout.strip().splitlines()[-1])
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))

    assert summary_path.name == "ablation_summary.csv"
    assert {row["variant"] for row in rows} == {"root_only", "kb"}
    assert (summary_path.parent / "ablation_report.md").exists()
