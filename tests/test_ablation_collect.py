from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agenticsciml.ablation import run_ablation
from agenticsciml.ablation_collect import AblationBatchCollectionError, collect_ablation_batches
from agenticsciml.ablation_evidence import build_multi_seed_ablation_verified_manifest
from agenticsciml.evidence import EVIDENCE_MODE_REAL_LLM_ABLATION, LLM_MODE_REAL, SCIENTIFIC_CLAIM_NOT_SUPPORTED


def test_collect_ablation_batches_writes_complete_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
    )

    result = collect_ablation_batches(
        stage_plan_dir=tmp_path / "stage",
        batch_dirs=[batch_dir],
        output_dir=tmp_path / "collected",
    )

    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(result.runs_csv.open(encoding="utf-8")))
    summary = list(csv.DictReader(result.summary_csv.open(encoding="utf-8")))
    report_md = result.manifest_json.parent / "ablation_report.md"

    assert result.passed is True
    assert manifest["collection_status"] == "complete"
    assert manifest["full_stage_plan_hash"] == stage_manifest["full_stage_plan_hash"]
    assert manifest["expected_run_count"] == 1
    assert manifest["collected_run_count"] == 1
    assert manifest["created_outputs"]["report_md"] == str(report_md)
    assert rows[0]["variant"] == "root_only"
    assert summary[0]["variant"] == "root_only"
    assert report_md.is_file()
    assert "Ablation Report" in report_md.read_text(encoding="utf-8")


def test_collect_ablation_batches_requires_allow_partial(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "25")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only", "kb"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
    )

    with pytest.raises(AblationBatchCollectionError, match="planned experiment_id"):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[batch_dir],
            output_dir=tmp_path / "blocked",
        )

    result = collect_ablation_batches(
        stage_plan_dir=tmp_path / "stage",
        batch_dirs=[batch_dir],
        output_dir=tmp_path / "partial",
        allow_partial=True,
    )
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))

    assert result.passed is False
    assert manifest["collection_status"] == "partial"
    assert manifest["missing_run_count"] == 1
    assert manifest["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED


def test_complete_collection_writes_bound_bundle_and_verifies_seed_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0, 1],
        variants=["kb"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        scientific_seed_provenance=True,
    )
    output_dir = tmp_path / "collected"

    collect_ablation_batches(
        stage_plan_dir=tmp_path / "stage",
        batch_dirs=[batch_dir],
        output_dir=output_dir,
    )
    verified = build_multi_seed_ablation_verified_manifest(
        {
            "ablation_output_dir": str(output_dir),
            "verified_by": "collection-reviewer",
            "expected_seeds": [0, 1],
            "expected_variants": ["kb"],
        }
    )

    assert (output_dir / "real_llm_ablation_plan.json").is_file()
    assert (output_dir / "real_llm_ablation_manifest.json").is_file()
    assert (output_dir / "ablation_evidence_bundle.json").is_file()
    assert verified["artifact_integrity_verified"] is True
    assert verified["scientific_multi_seed_verified"] is True
    assert verified["verified"] is True


def test_partial_collection_bundle_remains_scientifically_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "25")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0, 1],
        variants=["kb"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        scientific_seed_provenance=True,
    )
    output_dir = tmp_path / "partial"

    collect_ablation_batches(
        stage_plan_dir=tmp_path / "stage",
        batch_dirs=[batch_dir],
        output_dir=output_dir,
        allow_partial=True,
    )
    verified = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(output_dir), "verified_by": "collection-reviewer"}
    )

    assert verified["verified"] is False
    assert any("collection must be complete" in item for item in verified["blockers"])


def test_collector_rejects_tampered_stage_plan_before_rebinding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0, 1],
        variants=["kb"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        scientific_seed_provenance=True,
    )
    stage_plan["timeout_s"] = int(stage_plan["timeout_s"]) + 1
    stage.plan_json.write_text(json.dumps(stage_plan), encoding="utf-8")

    with pytest.raises(AblationBatchCollectionError, match="plan_hash"):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[batch_dir],
            output_dir=tmp_path / "collected",
        )


def test_collection_verifier_rejects_failed_trace_and_zero_real_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0, 1],
        variants=["kb"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        scientific_seed_provenance=True,
    )
    failed_run = batch_dir / "runs" / "kb-seed-0"
    (failed_run / "trace_summary.json").write_text(
        json.dumps({"quality_gate": {"passed": False}}),
        encoding="utf-8",
    )
    metadata = json.loads((failed_run / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["llm_calls"] = {"total": 0}
    (failed_run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    output_dir = tmp_path / "collected"

    collect_ablation_batches(
        stage_plan_dir=tmp_path / "stage",
        batch_dirs=[batch_dir],
        output_dir=output_dir,
    )
    verified = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(output_dir), "verified_by": "collection-reviewer"}
    )

    assert verified["verified"] is False
    assert any("trace quality_gate did not pass" in item for item in verified["blockers"])
    assert any("positive real llm_calls" in item for item in verified["blockers"])


def test_collect_ablation_batches_rejects_duplicate_and_extra_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        extra_rows=[_row("kb", 99, tmp_path / "batch-1")],
    )

    with pytest.raises(AblationBatchCollectionError, match="canonical budget batch"):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[batch_dir],
            output_dir=tmp_path / "collected",
        )


def test_collect_ablation_batches_accepts_legacy_missing_hash_only_with_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "25")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only", "kb"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "legacy-batch",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        legacy_missing_hash=True,
    )

    with pytest.raises(AblationBatchCollectionError, match="missing full_stage_plan_hash"):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[batch_dir],
            output_dir=tmp_path / "blocked",
            allow_partial=True,
        )

    result = collect_ablation_batches(
        stage_plan_dir=tmp_path / "stage",
        batch_dirs=[batch_dir],
        output_dir=tmp_path / "partial",
        allow_partial=True,
        allow_legacy_missing_full_stage_hash=True,
    )
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))

    assert manifest["collection_status"] == "partial"
    assert manifest["legacy_batch_manifest_count"] == 1


def test_collect_ablation_batches_rejects_failed_secret_hygiene(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        secret_passed=False,
    )

    with pytest.raises(AblationBatchCollectionError, match="secret hygiene"):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[batch_dir],
            output_dir=tmp_path / "collected",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"full_stage_plan_hash": "different"}, "full_stage_plan_hash"),
        ({"benchmark_content_hash": "different"}, "benchmark_content_hash"),
        ({"provider": "different-provider"}, "provider"),
        ({"model": "different-model"}, "model"),
        ({"adapter_type": "different-adapter"}, "adapter_type"),
        ({"scientific_claim": "paper_score_supported"}, "scientific_claim"),
    ],
)
def test_collect_ablation_batches_rejects_manifest_identity_mismatch(
    tmp_path: Path,
    monkeypatch,
    overrides: dict[str, str],
    message: str,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        manifest_overrides=overrides,
    )

    with pytest.raises(AblationBatchCollectionError, match=message):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[batch_dir],
            output_dir=tmp_path / "collected",
        )


def test_collect_ablation_batches_rejects_missing_trace_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    batch_dir = _write_batch_dir(
        tmp_path / "batch-1",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
        trace_summary_present=False,
    )

    with pytest.raises(AblationBatchCollectionError, match="trace summaries"):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[batch_dir],
            output_dir=tmp_path / "collected",
        )


def test_collect_ablation_batches_rejects_duplicate_batch_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")
    stage = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "stage",
        seeds=[0],
        variants=["root_only"],
        mock=False,
        dry_run=True,
    )
    stage_plan = json.loads(stage.plan_json.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage.manifest_json.read_text(encoding="utf-8"))
    first_batch = _write_batch_dir(
        tmp_path / "batch-1a",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
    )
    second_batch = _write_batch_dir(
        tmp_path / "batch-1b",
        stage_plan=stage_plan,
        stage_manifest=stage_manifest,
        batch_index=1,
    )

    with pytest.raises(AblationBatchCollectionError, match="duplicate selected_budget_batch_index"):
        collect_ablation_batches(
            stage_plan_dir=tmp_path / "stage",
            batch_dirs=[first_batch, second_batch],
            output_dir=tmp_path / "collected",
        )


def _write_batch_dir(
    batch_dir: Path,
    *,
    stage_plan: dict,
    stage_manifest: dict,
    batch_index: int,
    legacy_missing_hash: bool = False,
    secret_passed: bool = True,
    extra_rows: list[dict[str, str]] | None = None,
    manifest_overrides: dict[str, str] | None = None,
    trace_summary_present: bool = True,
    scientific_seed_provenance: bool = False,
) -> Path:
    batch_dir.mkdir(parents=True)
    batch = next(item for item in stage_plan["budget_batch_plan"]["batches"] if item["batch_index"] == batch_index)
    rows = [
        _row_from_experiment_id(
            experiment_id,
            batch_dir,
            scientific_seed_provenance=scientific_seed_provenance,
        )
        for experiment_id in batch["experiment_ids"]
    ]
    rows.extend(extra_rows or [])
    _write_csv(batch_dir / "ablation_runs.csv", rows)
    _write_csv(batch_dir / "ablation_summary.csv", rows)
    for row in rows:
        run_dir = Path(row["run_dir"])
        run_dir.mkdir(parents=True)
        if trace_summary_present:
            (run_dir / "trace_summary.json").write_text(json.dumps({"quality_gate": {"status": "pass"}}), encoding="utf-8")
        if scientific_seed_provenance:
            seed = int(row["search_seed"])
            experiment_id = str(row["experiment_id"])
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
                        "llm_mode": LLM_MODE_REAL,
                        "run_state": "exported",
                        "llm_calls": {"total": 1},
                        "seed_provenance": {
                            "model_seed": seed,
                            "provider_seed": seed,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "trace_summary.json").write_text(
                json.dumps({"quality_gate": {"passed": True}}),
                encoding="utf-8",
            )
            (run_dir / "llm_call_ledger.jsonl").write_text(
                json.dumps({"call_index": 1}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "evaluation_contract.json").write_text(
                json.dumps(
                    {
                        "benchmark_name": "function_approx",
                        "contract_hash": f"contract-{seed}",
                        "benchmark_source_manifest": {"data_seed": seed},
                        "benchmark_source_manifest_digest": f"manifest-{seed}",
                    }
                ),
                encoding="utf-8",
            )
    manifest = {
        "schema_version": 1,
        "benchmark_dir": stage_plan["benchmark_dir"],
        "benchmark_content_hash": stage_plan["benchmark_content_hash"],
        "selected_budget_batch_index": batch_index,
        "budget_batch_selection": batch,
        "full_stage_run_count": len(stage_plan["runs"]),
        "full_stage_expected_llm_call_range": stage_plan["expected_llm_call_range"],
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "provider": stage_manifest["provider"],
        "model": stage_manifest["model"],
        "adapter_type": stage_manifest["adapter_type"],
    }
    if not legacy_missing_hash:
        manifest["full_stage_plan_hash"] = stage_manifest["full_stage_plan_hash"]
    manifest.update(manifest_overrides or {})
    (batch_dir / "real_llm_ablation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (batch_dir / "secret_hygiene_report.json").write_text(json.dumps({"passed": secret_passed}), encoding="utf-8")
    return batch_dir


def _row_from_experiment_id(
    experiment_id: str,
    batch_dir: Path,
    *,
    scientific_seed_provenance: bool = False,
) -> dict[str, str]:
    variant, seed_text = experiment_id.rsplit("-seed-", 1)
    row = _row(variant, int(seed_text), batch_dir)
    if scientific_seed_provenance:
        seed = int(seed_text)
        row.update(
            {
                "experiment_id": experiment_id,
                "execution_mode": "real",
                "search_seed": str(seed),
                "data_seed": str(seed),
                "model_seed": str(seed),
                "provider_seed": str(seed),
                "search_seed_source": "config.json:evolution.random_seed",
                "data_seed_source": (
                    "evaluation_contract.json:benchmark_source_manifest.data_seed"
                ),
                "model_seed_source": "run_metadata.json:seed_provenance.model_seed",
                "provider_seed_source": "run_metadata.json:seed_provenance.provider_seed",
                "seed_provenance_complete": "True",
                "benchmark_name": "function_approx",
                "benchmark_contract_hash": f"contract-{seed}",
                "benchmark_source_manifest_digest": f"manifest-{seed}",
            }
        )
    return row


def _row(variant: str, seed: int, batch_dir: Path) -> dict[str, str]:
    run_dir = batch_dir / "runs" / f"{variant}-seed-{seed}"
    return {
        "variant": variant,
        "seed": str(seed),
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_ABLATION,
        "llm_mode": LLM_MODE_REAL,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "run_evidence_mode": "real_llm_proxy_benchmark",
        "run_scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "run_dir": str(run_dir),
        "branch_context_enabled": "False",
        "champion_node_id": "solution_000",
        "metric_name": "mse",
        "higher_is_better": "False",
        "champion_score": "1.0",
        "root_score": "1.0",
        "champion/root improvement": "0.0",
        "valid_solution_rate": "1.0",
        "timeout_count": "0",
        "debug_success_count": "0",
        "branch_context_count": "0",
        "branch_intents": "",
        "llm_calls": "1",
        "wall_time_s": "1.0",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
