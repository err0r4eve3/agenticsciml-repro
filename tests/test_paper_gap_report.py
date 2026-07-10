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


def test_paper_gap_report_keeps_complete_ablation_bundle_fail_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "complete-stage-a-collection"
    bundle.mkdir()
    (bundle / "multi_seed_ablation_verified_manifest.json").write_text(
        json.dumps(
            {
                "verified": True,
                "run_count": 20,
                "seed_count": 5,
                "variants": ["root_only", "no_kb", "kb", "random_kb"],
                "scientific_claims": ["not_supported"],
                "claim_boundary": (
                    "This manifest verifies local ablation output shape, seed coverage, "
                    "and variant coverage. It does not prove paper-score improvement or "
                    "scientific discovery."
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_paper_gap_report(
        benchmark_dirs=[Path("examples/function_approx_faithful_small")],
        run_dirs=[bundle],
    )

    assert report["status"] == "blocked"
    assert report["summary"]["run_evidence_count"] == 1
    assert report["summary"]["unmatched_run_count"] == 1
    run = report["run_evidence"][0]
    assert run["benchmark_name"] is None
    assert run["completed_run_artifacts"] is False
    assert run["trace_quality_gate_passed"] is False
    assert run["multi_seed_ablation_verified"] is False
    assert any("run_metadata.json" in issue for issue in run["issues"])
    benchmark = report["benchmarks"][0]
    gap_items = {item["check_id"]: item for item in benchmark["gap_items"]}
    assert gap_items["benchmark_fidelity"]["status"] == "gap"
    assert gap_items["completed_run_artifacts"]["status"] == "gap"
    assert gap_items["multi_seed_ablation"]["status"] == "gap"


def test_paper_gap_report_does_not_stitch_claim_readiness_across_runs(
    tmp_path: Path,
) -> None:
    benchmark_name = "function_approx_faithful_small"
    runs = [
        _write_run_evidence(tmp_path / "completed", benchmark_name, completed=True),
        _write_run_evidence(tmp_path / "trace", benchmark_name, trace_passed=True),
        _write_run_evidence(tmp_path / "real", benchmark_name, real_llm=True),
        _write_run_evidence(tmp_path / "readiness", benchmark_name, readiness_supported=True),
        _write_run_evidence(tmp_path / "claim", benchmark_name, claim_supported=True),
    ]

    report = build_paper_gap_report(
        benchmark_dirs=[Path("examples/function_approx_faithful_small")],
        run_dirs=runs,
    )

    gap_items = {item["check_id"]: item for item in report["benchmarks"][0]["gap_items"]}
    assert gap_items["completed_run_artifacts"]["status"] == "satisfied"
    assert gap_items["trace_quality_gate"]["status"] == "satisfied"
    assert gap_items["real_llm_execution"]["status"] == "satisfied"
    assert gap_items["scientific_readiness"]["status"] == "satisfied"
    assert gap_items["claim_gate_support"]["status"] == "satisfied"
    assert gap_items["same_run_paper_claim_evidence"]["status"] == "gap"
    assert all(run["same_run_paper_claim_evidence"] is False for run in report["run_evidence"])
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


def _write_run_evidence(
    run_dir: Path,
    benchmark_name: str,
    *,
    completed: bool = False,
    trace_passed: bool = False,
    real_llm: bool = False,
    readiness_supported: bool = False,
    claim_supported: bool = False,
) -> Path:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True)
    claim_gate = {
        "status": "allowed",
        "paper_level_claim_supported": claim_supported,
        "scientific_claim_supported": claim_supported,
    }
    readiness = {
        "status": "ready" if readiness_supported else "blocked",
        "scientific_claim_supported": readiness_supported,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_state": "exported" if completed else "partial",
                "benchmark_name": benchmark_name,
                "llm_mode": "real" if real_llm else "mock",
                "claim_gate": claim_gate,
                "scientific_discovery_readiness": readiness,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trace_summary.json").write_text(
        json.dumps(
            {
                "quality_gate": {"passed": trace_passed},
                "claim_gate": claim_gate,
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "scientific_discovery_readiness.json").write_text(
        json.dumps(readiness),
        encoding="utf-8",
    )
    return run_dir
