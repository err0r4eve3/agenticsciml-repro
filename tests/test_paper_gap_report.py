from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.paper_gap_report import build_paper_gap_report


def test_paper_gap_report_blocks_proxy_catalog_without_run_evidence() -> None:
    report = build_paper_gap_report(benchmark_dirs=[Path("examples/function_approx")])

    assert report["status"] == "blocked"
    assert report["summary"]["benchmark_count"] == 1
    assert report["summary"]["run_evidence_count"] == 0
    benchmark = report["benchmarks"][0]
    assert benchmark["name"] == "function_approx"
    gap_items = {item["check_id"]: item for item in benchmark["gap_items"]}
    assert gap_items["benchmark_fidelity"]["status"] == "gap"
    assert gap_items["completed_run_artifacts"]["status"] == "gap"
    assert "not paper-like" in gap_items["benchmark_fidelity"]["blocker"]
    assert benchmark["summary"]["open_gap_count"] == len(benchmark["gap_items"])


def test_paper_gap_report_attaches_run_evidence_without_overclaiming(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "faithful-small-real"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True)
    claim_gate = {
        "status": "allowed",
        "paper_level_claim_supported": False,
        "scientific_claim_supported": False,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_state": "exported",
                "benchmark_name": "function_approx_faithful_small",
                "champion": "solution_000",
                "solution_count": 1,
                "llm_mode": "real",
                "evidence_mode": "real_llm_faithful-small_benchmark",
                "scientific_claim": "not_validated",
                "claim_gate": claim_gate,
                "multi_seed_ablation": {"verified_multi_seed_ablation": True},
                "scientific_discovery_readiness": {
                    "status": "blocked",
                    "scientific_claim_supported": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "trace_summary.json").write_text(
        json.dumps({"quality_gate": {"passed": True}, "claim_gate": claim_gate}, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "scientific_result_card.json").write_text(
        json.dumps(
            {
                "benchmark_name": "function_approx_faithful_small",
                "evidence_grade": "workflow_evidence_only",
                "claim_support": {"scientific_claim_supported": False},
                "uncertainty_flags": ["needs replication"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (reports_dir / "scientific_discovery_readiness.json").write_text(
        json.dumps({"status": "blocked", "scientific_claim_supported": False}, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "multi_seed_ablation_evidence.json").write_text(
        json.dumps({"verified_multi_seed_ablation": True}, sort_keys=True),
        encoding="utf-8",
    )

    report = build_paper_gap_report(
        benchmark_dirs=[Path("examples/function_approx_faithful_small")],
        run_dirs=[run_dir],
    )

    run = report["run_evidence"][0]
    assert run["issues"] == []
    assert run["completed_run_artifacts"] is True
    assert run["trace_quality_gate_passed"] is True
    assert run["multi_seed_ablation_verified"] is True
    assert run["paper_level_claim_supported"] is False

    gap_items = {item["check_id"]: item for item in report["benchmarks"][0]["gap_items"]}
    assert gap_items["completed_run_artifacts"]["status"] == "satisfied"
    assert gap_items["trace_quality_gate"]["status"] == "satisfied"
    assert gap_items["real_llm_execution"]["status"] == "satisfied"
    assert gap_items["multi_seed_ablation"]["status"] == "satisfied"
    assert gap_items["benchmark_fidelity"]["status"] == "gap"
    assert gap_items["scientific_readiness"]["status"] == "gap"
    assert gap_items["claim_gate_support"]["status"] == "gap"
    assert report["status"] == "blocked"


def test_cli_paper_gap_report_writes_json_and_markdown(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    output_dir = tmp_path / "paper gap output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "paper-gap-report",
            "--benchmark-dir",
            "examples/function_approx",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    report_path = Path(result.stdout.strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == output_dir / "paper_gap_report.json"
    assert report["status"] == "blocked"
    assert (output_dir / "paper_gap_report.md").exists()
