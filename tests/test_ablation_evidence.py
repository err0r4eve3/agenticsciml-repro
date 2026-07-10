from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from agenticsciml.ablation_evidence import (
    build_multi_seed_ablation_verified_manifest,
    has_ablation_output_source,
)
from agenticsciml.ablation import (
    _aggregate,
    _benchmark_content_hash,
    _hash_payload,
    _render_report,
    _write_ablation_evidence_bundle,
    _write_csv,
)
from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.benchmarks import benchmark_for_path
from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    EVIDENCE_MODE_REAL_LLM_ABLATION,
    LLM_MODE_MOCK,
    LLM_MODE_REAL,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
)
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.scientific_readiness import build_scientific_discovery_readiness_report


def test_ablation_evidence_builder_verifies_local_output(tmp_path: Path) -> None:
    _write_bound_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])

    manifest = build_multi_seed_ablation_verified_manifest(
        {
            "ablation_output_dir": str(tmp_path),
            "verified_by": "ablation-reviewer",
            "expected_seeds": [0, 1],
            "expected_variants": ["root_only", "kb"],
        }
    )

    assert manifest["verified"] is True
    assert manifest["scientific_multi_seed_verified"] is True
    assert manifest["workflow_shape_verified"] is True
    assert manifest["seed_count"] == 2
    assert manifest["ablation_count"] == 1
    assert manifest["baseline_variants"] == ["root_only"]
    assert manifest["ablation_variants"] == ["kb"]
    assert manifest["seed_coverage_by_variant"]["kb"] == [0, 1]
    assert manifest["source_artifacts"]["runs_csv"]["sha256"]
    assert manifest["scientific_claims"] == [SCIENTIFIC_CLAIM_NOT_SUPPORTED]
    assert manifest["seed_provenance"]["seed_provenance_complete"] is True
    assert manifest["seed_provenance"]["scientific_random_dimension_varied"] is True


def test_ablation_evidence_builder_blocks_single_seed_output(tmp_path: Path) -> None:
    _write_bound_ablation_output(tmp_path, seeds=[0], variants=["root_only", "kb"])

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )

    assert manifest["verified"] is False
    assert "at least two search seeds are required in ablation_runs.csv" in manifest["blockers"]
    assert (
        "each non-baseline ablation variant must include at least two search seeds"
        in manifest["blockers"]
    )


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
    _write_bound_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])

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
    _write_bound_ablation_output(
        ablation_dir,
        seeds=[0, 1],
        variants=["root_only", "branch_context"],
    )
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
    assert checks["multi_seed_ablation"]["passed"] is False
    assert readiness["scientific_claim_supported"] is False


def test_mock_ablation_remains_workflow_shape_only(tmp_path: Path) -> None:
    _write_bound_ablation_output(
        tmp_path,
        seeds=[0, 1],
        variants=["root_only", "kb"],
        execution_mode="mock",
        complete_seed_provenance=False,
        vary_scientific_seeds=False,
    )

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "workflow-reviewer"}
    )

    assert manifest["artifact_integrity_verified"] is True
    assert manifest["workflow_shape_verified"] is True
    assert manifest["verified"] is False
    assert manifest["scientific_multi_seed_verified"] is False
    assert any("execution_mode=real" in blocker for blocker in manifest["blockers"])


def test_real_ablation_blocks_missing_model_and_provider_seed_provenance(tmp_path: Path) -> None:
    _write_bound_ablation_output(
        tmp_path,
        seeds=[0, 1],
        variants=["root_only", "kb"],
        complete_seed_provenance=False,
    )

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )

    assert manifest["workflow_shape_verified"] is True
    assert manifest["verified"] is False
    assert any("model_seed is not backed by run artifacts" in item for item in manifest["blockers"])
    assert any("provider_seed is not backed by run artifacts" in item for item in manifest["blockers"])


def test_real_ablation_requires_a_scientific_seed_dimension_to_vary(tmp_path: Path) -> None:
    _write_bound_ablation_output(
        tmp_path,
        seeds=[0, 1],
        variants=["root_only", "kb"],
        vary_scientific_seeds=False,
    )

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )

    assert manifest["verified"] is False
    assert any("must actually vary" in item for item in manifest["blockers"])


def test_ablation_evidence_rejects_stale_summary_after_bundle_creation(tmp_path: Path) -> None:
    _write_bound_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])
    with (tmp_path / "ablation_summary.csv").open("a", encoding="utf-8") as f:
        f.write("stale,row\n")

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )

    assert manifest["verified"] is False
    assert "evidence bundle sha256 does not match current summary_csv" in manifest["blockers"]


def test_ablation_evidence_rejects_stale_plan_and_manifest(tmp_path: Path) -> None:
    _write_bound_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])
    plan_path = tmp_path / "real_llm_ablation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["variants"] = ["root_only"]
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")

    stale_plan = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )
    assert stale_plan["verified"] is False
    assert any("current plan" in item for item in stale_plan["blockers"])

    _write_bound_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])
    manifest_path = tmp_path / "real_llm_ablation_manifest.json"
    execution_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution_manifest["execution_mode"] = "mock"
    manifest_path.write_text(json.dumps(execution_manifest, sort_keys=True), encoding="utf-8")

    stale_manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )
    assert stale_manifest["verified"] is False
    assert any("current manifest" in item for item in stale_manifest["blockers"])


def test_ablation_evidence_rejects_benchmark_drift(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    shutil.copytree(Path("examples/function_approx").resolve(), benchmark_dir)
    output_dir = tmp_path / "ablation"
    _write_bound_ablation_output(
        output_dir,
        seeds=[0, 1],
        variants=["root_only", "kb"],
        benchmark_dir=benchmark_dir,
    )
    with (benchmark_dir / "Problem.md").open("a", encoding="utf-8") as f:
        f.write("\nchanged after execution\n")

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(output_dir), "verified_by": "ablation-reviewer"}
    )

    assert manifest["verified"] is False
    assert "current benchmark content no longer matches the recorded plan" in manifest["blockers"]


def test_legacy_csv_seed_labels_are_not_verified(tmp_path: Path) -> None:
    _write_legacy_ablation_output(tmp_path, seeds=[0, 1], variants=["root_only", "kb"])

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "ablation-reviewer"}
    )

    assert manifest["verified"] is False
    assert manifest["artifact_integrity_verified"] is False
    assert any("legacy or incomplete" in item for item in manifest["blockers"])


def test_scientific_readiness_rejects_declared_or_mock_multiseed_evidence() -> None:
    declared = {
        "seed_count": 2,
        "ablation_count": 1,
        "verified": True,
        "verified_by": "reviewer",
    }
    strict_manifest = {
        "schema_version": 2,
        "source_type": "ablation_output",
        "verified": True,
        "scientific_multi_seed_verified": True,
        "workflow_shape_verified": True,
        "artifact_integrity_verified": True,
        "execution_mode": "real",
        "seed_count": 2,
        "ablation_count": 1,
        "ablation_variants": ["kb"],
        "source_artifacts": {
            name: {"exists": True, "sha256": "fixture"}
            for name in (
                "runs_csv",
                "summary_csv",
                "plan_json",
                "manifest_json",
                "evidence_bundle",
            )
        },
        "seed_provenance": {
            "seed_provenance_complete": True,
            "scientific_random_dimension_varied": True,
        },
    }
    real_report = _readiness_report(
        use_mock=False,
        configured_multi_seed=declared,
        evidence_manifest={
            "verified_multi_seed_ablation": True,
            "verified_ablation_output_manifest": strict_manifest,
        },
    )
    mock_report = _readiness_report(
        use_mock=True,
        configured_multi_seed={},
        evidence_manifest={
            "verified_multi_seed_ablation": True,
            "verified_ablation_output_manifest": strict_manifest,
        },
    )
    declared_report = _readiness_report(
        use_mock=False,
        configured_multi_seed=declared,
        evidence_manifest={},
    )
    planner_report = _readiness_report(
        use_mock=False,
        configured_multi_seed={},
        evidence_manifest={},
        planner_snapshot={"ablation_manifest": strict_manifest},
    )

    assert _check_by_id(real_report, "multi_seed_ablation")["passed"] is True
    assert _check_by_id(mock_report, "multi_seed_ablation")["passed"] is False
    assert _check_by_id(declared_report, "multi_seed_ablation")["passed"] is False
    assert _check_by_id(planner_report, "multi_seed_ablation")["passed"] is False


def _write_bound_ablation_output(
    output_dir: Path,
    *,
    seeds: list[int],
    variants: list[str],
    execution_mode: str = "real",
    complete_seed_provenance: bool = True,
    vary_scientific_seeds: bool = True,
    benchmark_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = benchmark_dir or Path("examples/function_approx").resolve()
    evidence_mode = (
        EVIDENCE_MODE_REAL_LLM_ABLATION
        if execution_mode == "real"
        else EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE
    )
    llm_mode = LLM_MODE_REAL if execution_mode == "real" else LLM_MODE_MOCK
    planned_runs = [
        {
            "variant": variant,
            "seed": seed,
            "experiment_id": f"{variant}-seed-{seed}",
            "evolution": {"random_seed": seed},
        }
        for variant in variants
        for seed in seeds
    ]
    plan = {
        "schema_version": 1,
        "benchmark_dir": str(benchmark_dir),
        "benchmark_content_hash": _benchmark_content_hash(benchmark_dir),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "variants": variants,
        "execution_mode": execution_mode,
        "real_mode_explicit": execution_mode == "real",
        "provider_calls_enabled": execution_mode == "real",
        "evidence_mode": evidence_mode,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "runs": planned_runs,
        "full_stage_plan_hash": "fixture-stage-plan",
    }
    plan_path = output_dir / (
        "real_llm_ablation_plan.json" if execution_mode == "real" else "ablation_plan.json"
    )
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "execution_mode": execution_mode,
        "real_mode_explicit": execution_mode == "real",
        "provider_calls_enabled": execution_mode == "real",
        "benchmark_dir": str(benchmark_dir),
        "benchmark_content_hash": plan["benchmark_content_hash"],
        "plan_hash": _hash_payload(plan),
        "run_count": len(planned_runs),
        "full_stage_run_count": len(planned_runs),
        "full_stage_plan_hash": plan["full_stage_plan_hash"],
        "evidence_mode": evidence_mode,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
    }
    manifest_path = output_dir / (
        "real_llm_ablation_manifest.json"
        if execution_mode == "real"
        else "ablation_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    plan_sha256 = _file_sha256(plan_path)
    manifest_sha256 = _file_sha256(manifest_path)

    run_rows: list[dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            experiment_id = f"{variant}-seed-{seed}"
            run_dir = output_dir / "runs" / experiment_id
            run_dir.mkdir(parents=True, exist_ok=True)
            scientific_seed = seed if vary_scientific_seeds else 0
            model_seed = scientific_seed if complete_seed_provenance else None
            provider_seed = scientific_seed if complete_seed_provenance else None
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "experiment_id": experiment_id,
                        "evolution": {"random_seed": seed},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "llm_mode": llm_mode,
                        "run_state": "exported",
                        "llm_calls": {"total": 1 if execution_mode == "real" else 0},
                        "seed_provenance": {
                            "model_seed": model_seed,
                            "provider_seed": provider_seed,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "trace_summary.json").write_text(
                json.dumps({"quality_gate": {"passed": execution_mode == "real"}}),
                encoding="utf-8",
            )
            if execution_mode == "real":
                (run_dir / "llm_call_ledger.jsonl").write_text(
                    json.dumps({"call_index": 1}) + "\n",
                    encoding="utf-8",
                )
            (run_dir / "evaluation_contract.json").write_text(
                json.dumps(
                    {
                        "benchmark_name": "function_approx",
                        "contract_hash": f"contract-{scientific_seed}",
                        "benchmark_source_manifest": {"data_seed": scientific_seed},
                        "benchmark_source_manifest_digest": f"manifest-{scientific_seed}",
                    }
                ),
                encoding="utf-8",
            )
            run_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "experiment_id": experiment_id,
                    "search_seed": seed,
                    "data_seed": scientific_seed,
                    "model_seed": model_seed,
                    "provider_seed": provider_seed,
                    "search_seed_source": "config.json:evolution.random_seed",
                    "data_seed_source": (
                        "evaluation_contract.json:benchmark_source_manifest.data_seed"
                    ),
                    "model_seed_source": (
                        "run_metadata.json:seed_provenance.model_seed"
                        if complete_seed_provenance
                        else "unavailable"
                    ),
                    "provider_seed_source": (
                        "run_metadata.json:seed_provenance.provider_seed"
                        if complete_seed_provenance
                        else "unavailable"
                    ),
                    "seed_provenance_complete": complete_seed_provenance,
                    "execution_mode": execution_mode,
                    "evidence_mode": evidence_mode,
                    "llm_mode": llm_mode,
                    "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
                    "run_evidence_mode": evidence_mode,
                    "run_scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
                    "run_dir": str(run_dir),
                    "benchmark_name": "function_approx",
                    "benchmark_contract_hash": f"contract-{scientific_seed}",
                    "benchmark_source_manifest_digest": f"manifest-{scientific_seed}",
                    "benchmark_content_hash": plan["benchmark_content_hash"],
                    "plan_hash": _hash_payload(plan),
                    "plan_sha256": plan_sha256,
                    "manifest_sha256": manifest_sha256,
                    "full_stage_plan_hash": plan["full_stage_plan_hash"],
                    "branch_context_enabled": False,
                    "champion_node_id": "solution_000",
                    "metric_name": "mse",
                    "higher_is_better": False,
                    "champion_score": float(seed + 1),
                    "root_score": float(seed + 1),
                    "champion/root improvement": 0.0,
                    "valid_solution_rate": 1.0,
                    "timeout_count": 0,
                    "debug_success_count": 0,
                    "branch_context_count": 0,
                    "branch_intents": "",
                    "llm_calls": 1 if execution_mode == "real" else 0,
                    "wall_time_s": 1.0,
                }
            )
    runs_csv = output_dir / "ablation_runs.csv"
    summary_csv = output_dir / "ablation_summary.csv"
    report_md = output_dir / "ablation_report.md"
    _write_csv(runs_csv, run_rows)
    summary_rows = _aggregate(run_rows)
    _write_csv(summary_csv, summary_rows)
    report_md.write_text(_render_report(summary_rows), encoding="utf-8")
    _write_ablation_evidence_bundle(
        output_dir=output_dir,
        plan=plan,
        plan_path=plan_path,
        manifest=manifest,
        manifest_path=manifest_path,
        runs_csv=runs_csv,
        summary_csv=summary_csv,
        report_md=report_md,
        run_rows=run_rows,
    )


def _write_legacy_ablation_output(
    output_dir: Path,
    *,
    seeds: list[int],
    variants: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "variant": variant,
            "seed": seed,
            "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
            "llm_mode": LLM_MODE_MOCK,
            "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
            "run_dir": str(output_dir / "runs" / f"{variant}-seed-{seed}"),
        }
        for variant in variants
        for seed in seeds
    ]
    _write_simple_csv(output_dir / "ablation_runs.csv", rows)
    _write_simple_csv(
        output_dir / "ablation_summary.csv",
        [{"variant": variant, "runs": len(seeds)} for variant in variants],
    )
    (output_dir / "ablation_report.md").write_text("# Legacy\n", encoding="utf-8")


def _write_simple_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readiness_report(
    *,
    use_mock: bool,
    configured_multi_seed: dict[str, object],
    evidence_manifest: dict[str, object],
    planner_snapshot: dict[str, object] | None = None,
) -> dict[str, object]:
    benchmark = benchmark_for_path(
        Path("examples/cylinder_wake_reconstruction_faithful_small").resolve()
    )
    assert benchmark is not None
    return build_scientific_discovery_readiness_report(
        benchmark=benchmark,
        use_mock=use_mock,
        claim_level="paper_workflow",
        claim_gate={"status": "blocked", "scientific_claim_supported": False},
        selector_diversity={},
        kb_manifest={},
        visual_audit_manifest={},
        method_experience_manifest={},
        domain_evaluator_approved=False,
        domain_reviewer=None,
        domain_review_notes=None,
        paper_benchmark_approved=False,
        resource_constraints={},
        expert_blueprint_id=None,
        problem_intake={},
        planner_snapshot=planner_snapshot or {},
        multi_seed_ablation=configured_multi_seed,
        multi_seed_ablation_evidence=evidence_manifest,
    )


def _check_by_id(report: dict[str, object], check_id: str) -> dict[str, object]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return next(
        check
        for check in checks
        if isinstance(check, dict) and check.get("check_id") == check_id
    )
