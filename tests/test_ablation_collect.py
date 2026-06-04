from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agenticsciml.ablation import run_ablation
from agenticsciml.ablation_collect import AblationBatchCollectionError, collect_ablation_batches
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

    assert result.passed is True
    assert manifest["collection_status"] == "complete"
    assert manifest["full_stage_plan_hash"] == stage_manifest["full_stage_plan_hash"]
    assert manifest["expected_run_count"] == 1
    assert manifest["collected_run_count"] == 1
    assert rows[0]["variant"] == "root_only"
    assert summary[0]["variant"] == "root_only"


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
) -> Path:
    batch_dir.mkdir(parents=True)
    batch = next(item for item in stage_plan["budget_batch_plan"]["batches"] if item["batch_index"] == batch_index)
    rows = [
        _row_from_experiment_id(experiment_id, batch_dir)
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


def _row_from_experiment_id(experiment_id: str, batch_dir: Path) -> dict[str, str]:
    variant, seed_text = experiment_id.rsplit("-seed-", 1)
    return _row(variant, int(seed_text), batch_dir)


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
