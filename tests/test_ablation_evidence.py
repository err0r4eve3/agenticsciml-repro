from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.ablation_evidence import (
    build_multi_seed_ablation_verified_manifest,
    has_ablation_output_source,
)
from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.evidence import EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE, SCIENTIFIC_CLAIM_NOT_SUPPORTED
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


def test_ablation_evidence_builder_verifies_local_output(tmp_path: Path) -> None:
    _write_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])

    manifest = build_multi_seed_ablation_verified_manifest(
        {
            "ablation_output_dir": str(tmp_path),
            "verified_by": "ablation-reviewer",
            "expected_seeds": [0, 1],
            "expected_variants": ["root_only", "kb"],
        }
    )

    assert manifest["verified"] is True
    assert manifest["seed_count"] == 2
    assert manifest["ablation_count"] == 1
    assert manifest["baseline_variants"] == ["root_only"]
    assert manifest["ablation_variants"] == ["kb"]
    assert manifest["seed_coverage_by_variant"]["kb"] == [0, 1]
    assert manifest["source_artifacts"]["runs_csv"]["sha256"]
    assert manifest["scientific_claims"] == [SCIENTIFIC_CLAIM_NOT_SUPPORTED]


def test_ablation_evidence_builder_blocks_single_seed_output(tmp_path: Path) -> None:
    _write_ablation_output(tmp_path, seeds=[0], variants=["root_only", "kb"])

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )

    assert manifest["verified"] is False
    assert "at least two seeds are required in ablation_runs.csv" in manifest["blockers"]
    assert "each non-baseline ablation variant must include at least two seeds" in manifest["blockers"]


def test_legacy_attached_summary_path_does_not_trigger_output_parser() -> None:
    assert (
        has_ablation_output_source(
            {
                "seed_count": 2,
                "ablation_count": 1,
                "verified": True,
                "verified_by": "external-reviewer",
                "summary_path": "reports/external_ablation_summary.json",
            }
        )
        is False
    )


def test_cli_verify_ablation_evidence_writes_manifest(tmp_path: Path, cli_env: dict[str, str]) -> None:
    _write_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-ablation-evidence",
            str(tmp_path),
            "--verified-by",
            "ablation-reviewer",
            "--expected-seeds",
            "0",
            "1",
            "--expected-variants",
            "root_only,kb",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    manifest_path = Path(result.stdout.strip())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "multi_seed_ablation_verified_manifest.json"
    assert manifest["verified"] is True
    assert manifest["seed_count"] == 2
    assert manifest["ablation_count"] == 1


def test_orchestrator_attaches_verified_ablation_output_manifest(tmp_path: Path) -> None:
    ablation_dir = tmp_path / "ablation"
    _write_ablation_output(ablation_dir, seeds=[0, 1], variants=["root_only", "branch_context"])
    output_dir = tmp_path / "runs"
    config = ExperimentConfig(
        experiment_id="verified-ablation-output-run",
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        output_dir=output_dir,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        multi_seed_ablation={
            "ablation_output_dir": str(ablation_dir),
            "verified_by": "ablation-reviewer",
            "expected_seeds": [0, 1],
            "expected_variants": ["root_only", "branch_context"],
        },
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    report = json.loads(
        (run_dir / "reports" / "multi_seed_ablation_evidence.json").read_text(encoding="utf-8")
    )
    verified_manifest = json.loads(
        (run_dir / "reports" / "multi_seed_ablation_verified_manifest.json").read_text(encoding="utf-8")
    )
    readiness = json.loads(
        (run_dir / "reports" / "scientific_discovery_readiness.json").read_text(encoding="utf-8")
    )

    assert report["verified_multi_seed_ablation"] is True
    assert report["seed_count"] == 2
    assert report["ablation_count"] == 1
    assert report["verifier"] == "ablation-reviewer"
    assert "reports/multi_seed_ablation_verified_manifest.json" in report["attached_artifact_paths"]
    assert verified_manifest["source_type"] == "ablation_output"
    assert verified_manifest["source_artifacts"]["runs_csv"]["sha256"]
    checks = {check["check_id"]: check for check in readiness["checks"]}
    assert checks["multi_seed_ablation"]["passed"] is True
    assert readiness["scientific_claim_supported"] is False


def _write_ablation_output(output_dir: Path, *, seeds: list[int], variants: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            run_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
                    "llm_mode": "mock",
                    "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
                    "run_dir": str(output_dir / "runs" / f"{variant}-seed-{seed}"),
                    "valid_solution_rate": 1.0,
                }
            )
    _write_csv(output_dir / "ablation_runs.csv", run_rows)
    _write_csv(
        output_dir / "ablation_summary.csv",
        [
            {
                "variant": variant,
                "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
                "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
                "runs": len(seeds),
                "valid_runs": len(seeds),
                "example_run_dir": str(output_dir / "runs" / f"{variant}-seed-{seeds[0]}"),
            }
            for variant in variants
        ],
    )
    (output_dir / "ablation_report.md").write_text("# Ablation Report\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
