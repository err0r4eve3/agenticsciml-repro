from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticsciml.ablation import (
    _aggregate,
    _bind_run_rows_to_execution_artifacts,
    _full_stage_plan_hash,
    _hash_payload,
    _render_report,
    _write_ablation_evidence_bundle,
    _write_csv,
)
from agenticsciml.evidence import EVIDENCE_MODE_REAL_LLM_ABLATION, SCIENTIFIC_CLAIM_NOT_SUPPORTED
from agenticsciml.storage import _atomic_write_text


class AblationBatchCollectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AblationBatchCollectionResult:
    manifest_json: Path
    runs_csv: Path
    summary_csv: Path
    passed: bool


def collect_ablation_batches(
    *,
    stage_plan_dir: Path,
    batch_dirs: list[Path],
    output_dir: Path,
    allow_partial: bool = False,
    allow_legacy_missing_full_stage_hash: bool = False,
) -> AblationBatchCollectionResult:
    stage = _load_stage(stage_plan_dir)
    planned_ids = _planned_experiment_ids(stage["plan"])
    if not planned_ids:
        raise AblationBatchCollectionError("stage plan has no planned runs")

    rows_by_experiment_id: dict[str, dict[str, Any]] = {}
    batch_entries: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    extra_ids: list[str] = []
    legacy_count = 0
    seen_batch_indexes: set[str] = set()

    for batch_dir in batch_dirs:
        batch = _load_batch(batch_dir)
        batch_ids = _batch_experiment_ids(batch)
        _validate_batch_identity(
            stage,
            batch,
            batch_ids,
            allow_legacy_missing_full_stage_hash=allow_legacy_missing_full_stage_hash,
        )
        if not batch["secret_hygiene_passed"]:
            raise AblationBatchCollectionError(f"secret hygiene did not pass for batch: {batch_dir}")
        if not batch["trace_summary_all_present"]:
            raise AblationBatchCollectionError(f"trace summaries are incomplete for batch: {batch_dir}")
        if batch["legacy_missing_full_stage_hash"]:
            legacy_count += 1

        selected_index_key = str(batch["selected_budget_batch_index"])
        if selected_index_key in seen_batch_indexes:
            raise AblationBatchCollectionError(f"duplicate selected_budget_batch_index: {selected_index_key}")
        seen_batch_indexes.add(selected_index_key)

        for experiment_id, row in zip(batch_ids, batch["run_rows"], strict=True):
            if experiment_id not in planned_ids:
                extra_ids.append(experiment_id)
                continue
            if experiment_id in rows_by_experiment_id:
                duplicate_ids.append(experiment_id)
                continue
            rows_by_experiment_id[experiment_id] = row
        batch_entries.append(
            {
                "path": str(batch_dir.resolve()),
                "selected_budget_batch_index": batch["selected_budget_batch_index"],
                "experiment_ids": batch_ids,
                "legacy_missing_full_stage_hash": batch["legacy_missing_full_stage_hash"],
            }
        )

    if extra_ids:
        raise AblationBatchCollectionError("batch contains plan-external experiment_id(s): " + ", ".join(sorted(extra_ids)))
    if duplicate_ids:
        raise AblationBatchCollectionError("duplicate experiment_id(s): " + ", ".join(sorted(duplicate_ids)))

    missing_ids = [experiment_id for experiment_id in planned_ids if experiment_id not in rows_by_experiment_id]
    if missing_ids and not allow_partial:
        raise AblationBatchCollectionError("planned experiment_id(s) missing: " + ", ".join(missing_ids))

    ordered_rows = [rows_by_experiment_id[experiment_id] for experiment_id in planned_ids if experiment_id in rows_by_experiment_id]
    collection_status = "partial" if missing_ids else "complete"

    output_dir.mkdir(parents=True, exist_ok=True)
    source_stage_plan_path = output_dir / "source_stage_plan.json"
    source_stage_manifest_path = output_dir / "source_stage_manifest.json"
    _atomic_write_text(
        source_stage_plan_path,
        json.dumps(stage["plan"], indent=2, sort_keys=True, allow_nan=False),
    )
    _atomic_write_text(
        source_stage_manifest_path,
        json.dumps(stage["manifest"], indent=2, sort_keys=True, allow_nan=False),
    )
    plan = _collection_execution_plan(
        stage,
        ordered_rows=ordered_rows,
        output_dir=output_dir,
    )
    plan_path = output_dir / "real_llm_ablation_plan.json"
    _atomic_write_text(
        plan_path,
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False),
    )
    execution_manifest = _collection_execution_manifest(
        stage,
        plan=plan,
        collected_run_count=len(ordered_rows),
        collection_status=collection_status,
        output_dir=output_dir,
    )
    execution_manifest_path = output_dir / "real_llm_ablation_manifest.json"
    _atomic_write_text(
        execution_manifest_path,
        json.dumps(execution_manifest, indent=2, sort_keys=True, allow_nan=False),
    )
    _bind_run_rows_to_execution_artifacts(
        ordered_rows,
        plan=plan,
        plan_path=plan_path,
        manifest_path=execution_manifest_path,
    )
    summary_rows = _aggregate(ordered_rows)
    runs_csv = output_dir / "ablation_runs.csv"
    summary_csv = output_dir / "ablation_summary.csv"
    report_md = output_dir / "ablation_report.md"
    _write_csv(runs_csv, ordered_rows)
    _write_csv(summary_csv, summary_rows)
    _atomic_write_text(report_md, _render_report(summary_rows))
    manifest = {
        "schema_version": 1,
        "source_type": "ablation_batch_collection",
        "collection_status": collection_status,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_ABLATION,
        "full_stage_plan_hash": stage["full_stage_plan_hash"],
        "benchmark_dir": stage["plan"].get("benchmark_dir"),
        "benchmark_content_hash": stage["plan"].get("benchmark_content_hash"),
        "expected_run_count": len(planned_ids),
        "collected_run_count": len(ordered_rows),
        "missing_run_count": len(missing_ids),
        "missing_experiment_ids": missing_ids,
        "duplicate_experiment_ids": sorted(set(duplicate_ids)),
        "extra_experiment_ids": sorted(set(extra_ids)),
        "batch_dirs": batch_entries,
        "legacy_batch_manifest_count": legacy_count,
        "secret_hygiene_all_passed": True,
        "trace_summary_all_present": True,
        "created_outputs": {
            "runs_csv": str(runs_csv),
            "summary_csv": str(summary_csv),
            "report_md": str(report_md),
            "manifest_json": str(output_dir / "batch_collection_manifest.json"),
            "plan_json": str(plan_path),
            "execution_manifest_json": str(execution_manifest_path),
            "evidence_bundle_json": str(output_dir / "ablation_evidence_bundle.json"),
            "source_stage_plan_json": str(source_stage_plan_path),
            "source_stage_manifest_json": str(source_stage_manifest_path),
        },
        "claim_boundary": (
            "This collector verifies batch identity and output shape only. It does not prove "
            "paper-score reproduction or scientific discovery."
        ),
    }
    manifest_json = output_dir / "batch_collection_manifest.json"
    _atomic_write_text(manifest_json, json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    _write_ablation_evidence_bundle(
        output_dir=output_dir,
        plan=plan,
        plan_path=plan_path,
        manifest=execution_manifest,
        manifest_path=execution_manifest_path,
        runs_csv=runs_csv,
        summary_csv=summary_csv,
        report_md=report_md,
        run_rows=ordered_rows,
        source_type="ablation_batch_collection",
        additional_artifacts={
            "collection_manifest": manifest_json,
            "source_stage_plan_json": source_stage_plan_path,
            "source_stage_manifest_json": source_stage_manifest_path,
        },
    )
    return AblationBatchCollectionResult(
        manifest_json=manifest_json,
        runs_csv=runs_csv,
        summary_csv=summary_csv,
        passed=collection_status == "complete",
    )


def _collection_execution_plan(
    stage: dict[str, Any],
    *,
    ordered_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    canonical_plan = dict(stage["plan"])
    entries_by_id = {
        str(entry.get("experiment_id")): dict(entry)
        for entry in canonical_plan.get("runs", [])
        if isinstance(entry, dict)
    }
    experiment_ids = [_row_experiment_id(row) for row in ordered_rows]
    canonical_plan.update(
        {
            "output_dir": str(output_dir.resolve()),
            "execution_mode": "real",
            "real_mode_explicit": True,
            "provider_calls_enabled": True,
            "runs": [entries_by_id[experiment_id] for experiment_id in experiment_ids],
            "seeds": sorted({int(row["seed"]) for row in ordered_rows}),
            "variants": sorted({str(row["variant"]) for row in ordered_rows}),
            "selected_budget_batch_index": None,
            "budget_batch_selection": None,
        }
    )
    return canonical_plan


def _collection_execution_manifest(
    stage: dict[str, Any],
    *,
    plan: dict[str, Any],
    collected_run_count: int,
    collection_status: str,
    output_dir: Path,
) -> dict[str, Any]:
    canonical_manifest = dict(stage["manifest"])
    canonical_manifest.update(
        {
            "source_type": "ablation_batch_collection",
            "collection_status": collection_status,
            "execution_mode": "real",
            "real_mode_explicit": True,
            "provider_calls_enabled": True,
            "output_dir": str(output_dir.resolve()),
            "run_count": collected_run_count,
            "full_stage_run_count": len(stage["plan"].get("runs", [])),
            "full_stage_plan_hash": stage["full_stage_plan_hash"],
            "plan_hash": _hash_payload(plan),
            "selected_budget_batch_index": None,
            "budget_batch_selection": None,
            "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
            "evidence_mode": EVIDENCE_MODE_REAL_LLM_ABLATION,
        }
    )
    return canonical_manifest


def _load_stage(stage_plan_dir: Path) -> dict[str, Any]:
    root = stage_plan_dir.resolve()
    plan_path = root / "real_llm_ablation_plan.json"
    manifest_path = root / "real_llm_ablation_manifest.json"
    if not plan_path.is_file():
        raise AblationBatchCollectionError(f"stage plan is missing: {plan_path}")
    if not manifest_path.is_file():
        raise AblationBatchCollectionError(f"stage manifest is missing: {manifest_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_hash = _hash_payload(plan)
    if manifest.get("plan_hash") != plan_hash:
        raise AblationBatchCollectionError("stage manifest plan_hash does not match stage plan")
    plan_full_stage_hash = str(plan.get("full_stage_plan_hash") or "")
    manifest_full_stage_hash = str(manifest.get("full_stage_plan_hash") or "")
    if not plan_full_stage_hash or not manifest_full_stage_hash:
        raise AblationBatchCollectionError("stage plan missing full_stage_plan_hash")
    if plan_full_stage_hash != manifest_full_stage_hash:
        raise AblationBatchCollectionError("stage plan and manifest full_stage_plan_hash mismatch")
    if _full_stage_plan_hash(plan) != plan_full_stage_hash:
        raise AblationBatchCollectionError("stage full_stage_plan_hash does not match canonical plan")
    return {
        "root": root,
        "plan": plan,
        "manifest": manifest,
        "plan_path": plan_path,
        "manifest_path": manifest_path,
        "full_stage_plan_hash": plan_full_stage_hash,
    }


def _load_batch(batch_dir: Path) -> dict[str, Any]:
    root = batch_dir.resolve()
    manifest_path = root / "real_llm_ablation_manifest.json"
    runs_path = root / "ablation_runs.csv"
    summary_path = root / "ablation_summary.csv"
    secret_path = root / "secret_hygiene_report.json"
    if not manifest_path.is_file():
        raise AblationBatchCollectionError(f"batch manifest is missing: {manifest_path}")
    if not runs_path.is_file():
        raise AblationBatchCollectionError(f"batch runs CSV is missing: {runs_path}")
    if not summary_path.is_file():
        raise AblationBatchCollectionError(f"batch summary CSV is missing: {summary_path}")
    if not secret_path.is_file():
        raise AblationBatchCollectionError(f"batch secret hygiene report is missing: {secret_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    secret_report = json.loads(secret_path.read_text(encoding="utf-8"))
    run_rows = _read_csv(runs_path)
    _validate_run_rows(run_rows, root)
    return {
        "root": root,
        "manifest": manifest,
        "run_rows": run_rows,
        "selected_budget_batch_index": manifest.get("selected_budget_batch_index"),
        "secret_hygiene_passed": secret_report.get("passed") is True,
        "trace_summary_all_present": _trace_summary_all_present(run_rows, root),
        "legacy_missing_full_stage_hash": not bool(manifest.get("full_stage_plan_hash")),
    }


def _validate_batch_identity(
    stage: dict[str, Any],
    batch: dict[str, Any],
    batch_ids: list[str],
    *,
    allow_legacy_missing_full_stage_hash: bool,
) -> None:
    manifest = batch["manifest"]
    full_stage_hash = str(manifest.get("full_stage_plan_hash") or "")
    if not full_stage_hash:
        if not allow_legacy_missing_full_stage_hash:
            raise AblationBatchCollectionError("batch missing full_stage_plan_hash")
    elif full_stage_hash != stage["full_stage_plan_hash"]:
        raise AblationBatchCollectionError("batch full_stage_plan_hash does not match stage plan")

    if manifest.get("benchmark_content_hash") and manifest.get("benchmark_content_hash") != stage["plan"].get("benchmark_content_hash"):
        raise AblationBatchCollectionError("batch benchmark_content_hash does not match stage plan")
    if manifest.get("benchmark_dir") and manifest.get("benchmark_dir") != stage["plan"].get("benchmark_dir"):
        raise AblationBatchCollectionError("batch benchmark_dir does not match stage plan")

    for field in ("provider", "model", "adapter_type"):
        expected = stage["manifest"].get(field)
        actual = manifest.get(field)
        if expected not in (None, "") and actual in (None, ""):
            raise AblationBatchCollectionError(f"batch {field} is required")
        if expected not in (None, "") and actual != expected:
            raise AblationBatchCollectionError(f"batch {field} does not match stage manifest")

    if manifest.get("scientific_claim") not in (None, SCIENTIFIC_CLAIM_NOT_SUPPORTED):
        raise AblationBatchCollectionError("batch manifest scientific_claim must be not_supported")
    for row in batch["run_rows"]:
        if str(row.get("scientific_claim", "")) != SCIENTIFIC_CLAIM_NOT_SUPPORTED:
            raise AblationBatchCollectionError("batch run row scientific_claim must be not_supported")

    selected_index = manifest.get("selected_budget_batch_index")
    if selected_index in (None, ""):
        raise AblationBatchCollectionError("batch selected_budget_batch_index is required")
    expected_batch_ids = _expected_batch_ids(stage["plan"], int(selected_index))
    if expected_batch_ids and batch_ids != expected_batch_ids:
        raise AblationBatchCollectionError("batch experiment_ids do not match canonical budget batch")

    selection = manifest.get("budget_batch_selection")
    if isinstance(selection, dict):
        selection_ids = [str(item) for item in selection.get("experiment_ids", [])]
        if selection_ids and selection_ids != batch_ids:
            raise AblationBatchCollectionError("batch budget_batch_selection experiment_ids disagree with run rows")


def _planned_experiment_ids(plan: dict[str, Any]) -> list[str]:
    return [str(entry["experiment_id"]) for entry in plan.get("runs", []) if isinstance(entry, dict)]


def _expected_batch_ids(plan: dict[str, Any], batch_index: int) -> list[str]:
    batch_plan = plan.get("budget_batch_plan")
    if not isinstance(batch_plan, dict):
        return []
    batches = batch_plan.get("batches")
    if not isinstance(batches, list):
        return []
    for batch in batches:
        if isinstance(batch, dict) and int(batch.get("batch_index", -1)) == batch_index:
            return [str(item) for item in batch.get("experiment_ids", [])]
    return []


def _batch_experiment_ids(batch: dict[str, Any]) -> list[str]:
    return [_row_experiment_id(row) for row in batch["run_rows"]]


def _row_experiment_id(row: dict[str, Any]) -> str:
    return f"{row.get('variant')}-seed-{int(row.get('seed', 0))}"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _validate_run_rows(rows: list[dict[str, Any]], batch_dir: Path) -> None:
    if not rows:
        raise AblationBatchCollectionError("batch ablation_runs.csv has no rows")
    for row in rows:
        if not row.get("variant") or row.get("seed") in (None, ""):
            raise AblationBatchCollectionError("batch run row missing variant or seed")
        run_dir = Path(str(row.get("run_dir", "")))
        if not run_dir.is_absolute():
            run_dir = batch_dir / run_dir
        if not run_dir.exists():
            raise AblationBatchCollectionError(f"batch run_dir is missing: {run_dir}")


def _trace_summary_all_present(rows: list[dict[str, Any]], batch_dir: Path) -> bool:
    for row in rows:
        run_dir = Path(str(row.get("run_dir", "")))
        if not run_dir.is_absolute():
            run_dir = batch_dir / run_dir
        if not (run_dir / "trace_summary.json").is_file():
            return False
    return True
