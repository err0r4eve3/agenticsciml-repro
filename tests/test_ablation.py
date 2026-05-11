from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.ablation import _aggregate, run_ablation
from agenticsciml.evidence import EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE, SCIENTIFIC_CLAIM_NOT_SUPPORTED


def test_ablation_runner_writes_summary_and_report(tmp_path: Path) -> None:
    result = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        seeds=[0],
        variants=[
            "root_only",
            "no_kb",
            "kb",
            "random_kb",
            "no_critic",
            "no_debugger",
            "branch_context",
            "no_branch_context",
        ],
    )

    rows = list(csv.DictReader(result.summary_csv.open(encoding="utf-8")))
    variants = {row["variant"] for row in rows}
    report = result.report_md.read_text(encoding="utf-8")

    assert variants == {
        "root_only",
        "no_kb",
        "kb",
        "random_kb",
        "no_critic",
        "no_debugger",
        "branch_context",
        "no_branch_context",
    }
    assert result.summary_csv.exists()
    assert result.report_md.exists()
    assert "champion/root improvement" in rows[0]
    assert "valid_runs" in rows[0]
    assert rows[0]["evidence_mode"] == EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE
    assert rows[0]["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED
    assert "Champion/root improvement" in report
    assert "Valid runs" in report
    assert EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE in report
    assert "cannot prove emergent discovery" in report
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

    branch_context_row = next(row for row in rows if row["variant"] == "branch_context")
    no_branch_context_row = next(row for row in rows if row["variant"] == "no_branch_context")
    assert int(branch_context_row["branch_context_count_total"]) > 0
    assert branch_context_row["branch_intents"]
    assert int(no_branch_context_row["branch_context_count_total"]) == 0

    no_branch_context_run = Path(no_branch_context_row["example_run_dir"])
    no_branch_context_config = json.loads(
        (no_branch_context_run / "config.json").read_text(encoding="utf-8")
    )
    assert no_branch_context_config["evolution"]["use_branch_context"] is False


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


def test_ablation_aggregate_respects_score_direction() -> None:
    base_row = {
        "variant": "accuracy_metric",
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "valid_solution_rate": 1.0,
        "timeout_count": 0,
        "debug_success_count": 0,
        "llm_calls": 0,
        "wall_time_s": 0.0,
        "run_dir": "runs/demo",
        "champion/root improvement": 0.0,
    }
    rows = [
        {**base_row, "seed": 0, "higher_is_better": True, "champion_score": 0.8},
        {**base_row, "seed": 1, "higher_is_better": True, "champion_score": 0.9},
    ]

    summary = _aggregate(rows)[0]

    assert summary["higher_is_better"] is True
    assert summary["champion_score_best"] == 0.9
    assert summary["champion_score_worst"] == 0.8
