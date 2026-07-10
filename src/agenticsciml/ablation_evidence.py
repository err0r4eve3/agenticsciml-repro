from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    EVIDENCE_MODE_REAL_LLM_ABLATION,
    LLM_MODE_REAL,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
)


ABLATION_EVIDENCE_SCHEMA_VERSION = 2
BASELINE_VARIANTS = {"root_only", "baseline"}
SEED_PROVENANCE_FIELDS = (
    "search_seed",
    "data_seed",
    "model_seed",
    "provider_seed",
)
SCIENTIFIC_RANDOM_SEED_FIELDS = ("data_seed", "model_seed", "provider_seed")
BINDING_RUN_COLUMNS = {
    "variant",
    "seed",
    "experiment_id",
    "execution_mode",
    "evidence_mode",
    "llm_mode",
    "scientific_claim",
    "run_dir",
    "benchmark_name",
    "benchmark_content_hash",
    "plan_hash",
    "plan_sha256",
    "manifest_sha256",
    "full_stage_plan_hash",
}


def has_ablation_output_source(source: Mapping[str, object]) -> bool:
    return any(
        isinstance(source.get(key), str) and str(source.get(key)).strip()
        for key in (
            "ablation_output_dir",
            "output_dir",
            "runs_csv_path",
            "runs_path",
            "summary_csv_path",
        )
    )


def build_multi_seed_ablation_verified_manifest(
    source: Mapping[str, object],
) -> dict[str, object]:
    """Verify a completed ablation bundle without promoting workflow-shape evidence.

    ``verified`` is intentionally the scientific multi-seed result. Mock runs can
    still set ``workflow_shape_verified`` when their bound artifacts are intact.
    """
    paths = _source_paths(source)
    verifier = _verifier(source)
    integrity_blockers: list[str] = []
    coverage_blockers: list[str] = []
    scientific_blockers: list[str] = []
    warnings: list[str] = []

    run_rows, run_blockers = _read_csv(paths["runs_csv"])
    summary_rows, summary_blockers = _read_csv(paths["summary_csv"])
    integrity_blockers.extend(f"ablation_runs.csv: {item}" for item in run_blockers)
    integrity_blockers.extend(f"ablation_summary.csv: {item}" for item in summary_blockers)
    plan, plan_blockers = _read_json(paths["plan_json"])
    manifest, manifest_blockers = _read_json(paths["manifest_json"])
    evidence_bundle, bundle_blockers = _read_json(paths["evidence_bundle"])
    integrity_blockers.extend(f"plan: {item}" for item in plan_blockers)
    integrity_blockers.extend(f"manifest: {item}" for item in manifest_blockers)
    integrity_blockers.extend(f"evidence bundle: {item}" for item in bundle_blockers)
    collection_manifest: dict[str, Any] = {}
    is_collection = evidence_bundle.get("source_type") == "ablation_batch_collection"
    if is_collection:
        collection_manifest, collection_blockers = _read_json(paths["collection_manifest"])
        integrity_blockers.extend(
            f"batch collection manifest: {item}" for item in collection_blockers
        )

    if not paths["report_md"].is_file():
        integrity_blockers.append(f"ablation_report.md is missing: {paths['report_md']}")

    run_columns = set(run_rows[0]) if run_rows else set()
    summary_columns = set(summary_rows[0]) if summary_rows else set()
    missing_binding_columns = sorted(BINDING_RUN_COLUMNS - run_columns)
    if missing_binding_columns:
        integrity_blockers.append(
            "ablation_runs.csv is legacy or incomplete; missing binding columns: "
            + ", ".join(missing_binding_columns)
        )
    if summary_rows and "variant" not in summary_columns:
        integrity_blockers.append("ablation_summary.csv must include variant column")

    identity = _validate_execution_identity(
        source=source,
        paths=paths,
        plan=plan,
        manifest=manifest,
        bundle=evidence_bundle,
        run_rows=run_rows,
        integrity_blockers=integrity_blockers,
    )
    _validate_plan_run_matrix(plan, run_rows, integrity_blockers)
    _validate_summary_binding(run_rows, summary_rows, integrity_blockers)

    seeds = _sorted_values(_collect_seeds(run_rows))
    variants = _sorted_values(_collect_variants(run_rows, summary_rows))
    baseline_variants = [variant for variant in variants if str(variant) in BASELINE_VARIANTS]
    ablation_variants = [variant for variant in variants if str(variant) not in BASELINE_VARIANTS]
    seed_coverage_by_variant = _seed_coverage_by_variant(run_rows)

    if len(seeds) < 2:
        coverage_blockers.append("at least two search seeds are required in ablation_runs.csv")
    if not ablation_variants:
        coverage_blockers.append("at least one non-baseline ablation variant is required")
    under_seeded = {
        variant: coverage
        for variant, coverage in seed_coverage_by_variant.items()
        if variant in ablation_variants and len(coverage) < 2
    }
    if under_seeded:
        coverage_blockers.append(
            "each non-baseline ablation variant must include at least two search seeds"
        )

    expected_seeds = _optional_sequence(source, "expected_seeds")
    expected_variants = _optional_sequence(source, "expected_variants")
    if expected_seeds:
        missing = [seed for seed in expected_seeds if seed not in seeds]
        unexpected = [seed for seed in seeds if seed not in expected_seeds]
        if missing or unexpected:
            coverage_blockers.append(
                "ablation search seeds do not exactly match expected_seeds"
                f" (missing={_join_values(missing) or 'none'}, "
                f"unexpected={_join_values(unexpected) or 'none'})"
            )
    if expected_variants:
        missing_variants = [variant for variant in expected_variants if variant not in variants]
        unexpected_variants = [variant for variant in variants if variant not in expected_variants]
        if missing_variants or unexpected_variants:
            coverage_blockers.append(
                "ablation variants do not exactly match expected_variants"
                f" (missing={_join_values(missing_variants) or 'none'}, "
                f"unexpected={_join_values(unexpected_variants) or 'none'})"
            )

    seed_provenance = _audit_seed_provenance(
        run_rows,
        output_dir=paths["output_dir"],
        ablation_variants=[str(value) for value in ablation_variants],
        bound_run_provenance=evidence_bundle.get("run_provenance_artifacts"),
        allow_external_run_dirs=is_collection,
    )
    scientific_blockers.extend(str(item) for item in seed_provenance["blockers"])
    repeated_result_variants = _repeated_result_variants(run_rows)
    if repeated_result_variants:
        warnings.append(
            "identical recorded outcomes across search seeds for variant(s): "
            + ", ".join(repeated_result_variants)
        )

    evidence_modes = _sorted_values(_collect_values(run_rows, summary_rows, "evidence_mode"))
    llm_modes = _sorted_values(_collect_values(run_rows, summary_rows, "llm_mode"))
    execution_modes = _sorted_values(_collect_values(run_rows, summary_rows, "execution_mode"))
    scientific_claims = _sorted_values(_collect_values(run_rows, summary_rows, "scientific_claim"))

    execution_mode = str(identity.get("execution_mode") or "")
    if execution_mode != "real":
        scientific_blockers.append(
            f"scientific multi-seed evidence requires execution_mode=real, got {execution_mode or 'missing'}"
        )
    if manifest.get("real_mode_explicit") is not True:
        scientific_blockers.append("real ablation manifest must record real_mode_explicit=true")
    if manifest.get("provider_calls_enabled") is not True:
        scientific_blockers.append("real ablation manifest must record provider_calls_enabled=true")
    selected_batch = manifest.get("selected_budget_batch_index")
    run_count = manifest.get("run_count")
    full_stage_run_count = manifest.get("full_stage_run_count", run_count)
    if selected_batch not in (None, "") or run_count != full_stage_run_count:
        scientific_blockers.append(
            "a selected or partial budget batch cannot satisfy full multi-seed evidence"
        )
    if evidence_modes != [EVIDENCE_MODE_REAL_LLM_ABLATION]:
        scientific_blockers.append(
            "scientific multi-seed evidence requires only real_llm_ablation evidence rows"
        )
    if llm_modes != [LLM_MODE_REAL]:
        scientific_blockers.append("scientific multi-seed evidence requires only real LLM rows")
    if scientific_claims and scientific_claims != [SCIENTIFIC_CLAIM_NOT_SUPPORTED]:
        scientific_blockers.append("ablation outputs must keep scientific_claim=not_supported")
    if not verifier:
        scientific_blockers.append("verified_by or reviewer is required")
    if is_collection:
        if collection_manifest.get("collection_status") != "complete":
            scientific_blockers.append(
                "batch collection must be complete before it can satisfy scientific multi-seed evidence"
            )
        if collection_manifest.get("secret_hygiene_all_passed") is not True:
            scientific_blockers.append("batch collection secret hygiene evidence is incomplete")
        if collection_manifest.get("trace_summary_all_present") is not True:
            scientific_blockers.append("batch collection trace summary evidence is incomplete")

    source_artifacts = _artifact_descriptors(paths)
    artifact_integrity_verified = not integrity_blockers
    workflow_shape_verified = artifact_integrity_verified and not coverage_blockers
    all_scientific_blockers = _unique_strings(
        [*integrity_blockers, *coverage_blockers, *scientific_blockers]
    )
    scientific_multi_seed_verified = workflow_shape_verified and not scientific_blockers

    return {
        "schema_version": ABLATION_EVIDENCE_SCHEMA_VERSION,
        "source_type": "ablation_output",
        "verified": scientific_multi_seed_verified,
        "scientific_multi_seed_verified": scientific_multi_seed_verified,
        "workflow_shape_verified": workflow_shape_verified,
        "artifact_integrity_verified": artifact_integrity_verified,
        "verified_by": verifier or None,
        "execution_mode": execution_mode or None,
        "seed_count": len(seeds),
        "ablation_count": len(ablation_variants),
        "run_count": len(run_rows),
        "summary_variant_count": len(summary_rows),
        "seeds": seeds,
        "variants": variants,
        "baseline_variants": baseline_variants,
        "ablation_variants": ablation_variants,
        "seed_coverage_by_variant": seed_coverage_by_variant,
        "seed_provenance": seed_provenance,
        "expected_seeds": expected_seeds,
        "expected_variants": expected_variants,
        "evidence_modes": evidence_modes,
        "llm_modes": llm_modes,
        "execution_modes": execution_modes,
        "scientific_claims": scientific_claims,
        "execution_identity": identity,
        "source_artifacts": source_artifacts,
        "workflow_shape_blockers": _unique_strings([*integrity_blockers, *coverage_blockers]),
        "scientific_blockers": all_scientific_blockers,
        "blockers": all_scientific_blockers,
        "warnings": _unique_strings(warnings),
        "claim_boundary": (
            "workflow_shape_verified checks a bound local workflow artifact bundle. verified and "
            "scientific_multi_seed_verified additionally require real execution, complete artifact-backed "
            "search/data/model/provider seed provenance, and variation in at least one data/model/provider "
            "seed dimension. No field in this manifest proves paper-score improvement or discovery."
        ),
    }


def _source_paths(source: Mapping[str, object]) -> dict[str, Path]:
    output_dir_value = source.get("ablation_output_dir") or source.get("output_dir") or ""
    output_dir = (
        Path(str(output_dir_value)).expanduser()
        if str(output_dir_value).strip()
        else Path.cwd()
    )
    runs_csv = _path_value(source, "runs_csv_path") or _path_value(source, "runs_path")
    summary_csv = _path_value(source, "summary_csv_path") or _path_value(source, "summary_path")
    report_md = _path_value(source, "report_md_path")
    evidence_bundle = _path_value(source, "evidence_bundle_path")
    plan_json = _path_value(source, "plan_json_path")
    manifest_json = _path_value(source, "manifest_json_path")
    if plan_json is None:
        real_plan = output_dir / "real_llm_ablation_plan.json"
        plan_json = real_plan if real_plan.is_file() else output_dir / "ablation_plan.json"
    if manifest_json is None:
        real_manifest = output_dir / "real_llm_ablation_manifest.json"
        manifest_json = (
            real_manifest if real_manifest.is_file() else output_dir / "ablation_manifest.json"
        )
    return {
        "output_dir": output_dir,
        "runs_csv": runs_csv or output_dir / "ablation_runs.csv",
        "summary_csv": summary_csv or output_dir / "ablation_summary.csv",
        "report_md": report_md or output_dir / "ablation_report.md",
        "plan_json": plan_json,
        "manifest_json": manifest_json,
        "evidence_bundle": evidence_bundle or output_dir / "ablation_evidence_bundle.json",
        "collection_manifest": output_dir / "batch_collection_manifest.json",
    }


def _path_value(source: Mapping[str, object], key: str) -> Path | None:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], [f"missing file: {path}"]
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return [], [f"could not read {path}: {exc}"]
    if not rows:
        return [], [f"no rows in {path}"]
    return rows, []


def _read_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"missing file: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, [f"could not read {path}: {exc}"]
    if not isinstance(payload, dict):
        return {}, [f"JSON object required: {path}"]
    return payload, []


def _validate_execution_identity(
    *,
    source: Mapping[str, object],
    paths: dict[str, Path],
    plan: dict[str, Any],
    manifest: dict[str, Any],
    bundle: dict[str, Any],
    run_rows: list[dict[str, str]],
    integrity_blockers: list[str],
) -> dict[str, Any]:
    plan_hash = _hash_payload(plan) if plan else ""
    plan_sha256 = _sha256(paths["plan_json"]) if paths["plan_json"].is_file() else ""
    manifest_sha256 = (
        _sha256(paths["manifest_json"]) if paths["manifest_json"].is_file() else ""
    )
    execution_values = {
        str(value)
        for value in (
            plan.get("execution_mode"),
            manifest.get("execution_mode"),
            bundle.get("execution_mode"),
            *[row.get("execution_mode") for row in run_rows],
        )
        if value not in (None, "")
    }
    execution_mode = next(iter(execution_values)) if len(execution_values) == 1 else ""
    if len(execution_values) != 1:
        integrity_blockers.append(
            "execution_mode mismatch across plan, manifest, evidence bundle, and run CSV"
        )

    if plan and manifest.get("plan_hash") != plan_hash:
        integrity_blockers.append("manifest plan_hash does not match the current plan JSON")
    if bundle and bundle.get("plan_hash") != plan_hash:
        integrity_blockers.append("evidence bundle plan_hash does not match the current plan JSON")
    if bundle and bundle.get("manifest_payload_hash") != _hash_payload(manifest):
        integrity_blockers.append(
            "evidence bundle manifest_payload_hash does not match the current manifest JSON"
        )

    artifact_map = bundle.get("artifacts") if isinstance(bundle.get("artifacts"), dict) else {}
    if not artifact_map:
        integrity_blockers.append("evidence bundle artifacts map is missing")
    required_bundle_artifacts = [
        "plan_json",
        "manifest_json",
        "runs_csv",
        "summary_csv",
        "report_md",
    ]
    if bundle.get("source_type") == "ablation_batch_collection":
        required_bundle_artifacts.append("collection_manifest")
    for name in required_bundle_artifacts:
        descriptor = artifact_map.get(name) if isinstance(artifact_map, dict) else None
        if not isinstance(descriptor, dict):
            integrity_blockers.append(f"evidence bundle artifact descriptor missing: {name}")
            continue
        actual_path = paths[name]
        if not actual_path.is_file():
            continue
        expected_path = _resolve_bundle_path(paths["output_dir"], descriptor.get("path"))
        if expected_path is None or expected_path.resolve() != actual_path.resolve():
            integrity_blockers.append(f"evidence bundle path does not match current {name}")
        actual_sha = _sha256(actual_path)
        if descriptor.get("sha256") != actual_sha:
            integrity_blockers.append(f"evidence bundle sha256 does not match current {name}")
        if name in {"plan_json", "manifest_json"}:
            payload = plan if name == "plan_json" else manifest
            if descriptor.get("payload_hash") != _hash_payload(payload):
                integrity_blockers.append(
                    f"evidence bundle payload_hash does not match current {name}"
                )
    if bundle.get("source_type") == "ablation_batch_collection":
        for name in ("source_stage_plan_json", "source_stage_manifest_json"):
            descriptor = artifact_map.get(name) if isinstance(artifact_map, dict) else None
            if not isinstance(descriptor, dict):
                integrity_blockers.append(f"evidence bundle artifact descriptor missing: {name}")
                continue
            bound_path = _resolve_bundle_path(paths["output_dir"], descriptor.get("path"))
            if bound_path is None or not bound_path.is_file():
                integrity_blockers.append(f"evidence bundle source artifact is missing: {name}")
                continue
            if descriptor.get("sha256") != _sha256(bound_path):
                integrity_blockers.append(f"evidence bundle sha256 does not match current {name}")
            payload, issues = _read_json(bound_path)
            if issues or descriptor.get("payload_hash") != _hash_payload(payload):
                integrity_blockers.append(
                    f"evidence bundle payload_hash does not match current {name}"
                )

    benchmark_dir_value = str(plan.get("benchmark_dir") or "").strip()
    benchmark_dir = Path(benchmark_dir_value).expanduser() if benchmark_dir_value else None
    current_benchmark_hash = ""
    if benchmark_dir is None or not benchmark_dir.is_dir():
        integrity_blockers.append("plan benchmark_dir is missing or no longer exists")
    else:
        current_benchmark_hash = _benchmark_content_hash(benchmark_dir)
    benchmark_hash_values = {
        str(value)
        for value in (
            plan.get("benchmark_content_hash"),
            manifest.get("benchmark_content_hash"),
            bundle.get("benchmark_content_hash"),
            *[row.get("benchmark_content_hash") for row in run_rows],
        )
        if value not in (None, "")
    }
    if len(benchmark_hash_values) != 1:
        integrity_blockers.append(
            "benchmark_content_hash mismatch across plan, manifest, bundle, and run CSV"
        )
    recorded_benchmark_hash = next(iter(benchmark_hash_values), "")
    if current_benchmark_hash and recorded_benchmark_hash != current_benchmark_hash:
        integrity_blockers.append("current benchmark content no longer matches the recorded plan")

    expected_identity = {
        "expected_execution_mode": execution_mode,
        "expected_plan_hash": plan_hash,
        "expected_manifest_sha256": manifest_sha256,
        "expected_benchmark_content_hash": current_benchmark_hash,
    }
    for key, actual in expected_identity.items():
        expected = source.get(key)
        if expected not in (None, "") and str(expected) != str(actual):
            integrity_blockers.append(f"{key} does not match the current ablation bundle")

    full_stage_hashes = {
        str(value)
        for value in (
            plan.get("full_stage_plan_hash"),
            manifest.get("full_stage_plan_hash"),
            bundle.get("full_stage_plan_hash"),
            *[row.get("full_stage_plan_hash") for row in run_rows],
        )
        if value not in (None, "")
    }
    if len(full_stage_hashes) != 1:
        integrity_blockers.append("full_stage_plan_hash is missing or inconsistent")

    for row in run_rows:
        experiment_id = row.get("experiment_id") or "unknown-run"
        if row.get("plan_hash") != plan_hash:
            integrity_blockers.append(f"{experiment_id}: plan_hash does not match current plan")
        if row.get("plan_sha256") != plan_sha256:
            integrity_blockers.append(f"{experiment_id}: plan_sha256 does not match current plan")
        if row.get("manifest_sha256") != manifest_sha256:
            integrity_blockers.append(
                f"{experiment_id}: manifest_sha256 does not match current manifest"
            )

    if bundle.get("status") != "completed":
        integrity_blockers.append("evidence bundle status must be completed")
    actual_run_count = len(run_rows)
    planned_runs = plan.get("runs")
    planned_run_count = len(planned_runs) if isinstance(planned_runs, list) else None
    recorded_counts = (
        planned_run_count,
        _strict_count(manifest.get("run_count")),
        _strict_count(bundle.get("run_count")),
    )
    if any(count != actual_run_count for count in recorded_counts):
        integrity_blockers.append(
            "run_count mismatch across plan, manifest, evidence bundle, and run CSV"
        )
    return {
        "execution_mode": execution_mode or None,
        "plan_path": str(paths["plan_json"]),
        "plan_hash": plan_hash or None,
        "plan_sha256": plan_sha256 or None,
        "manifest_path": str(paths["manifest_json"]),
        "manifest_sha256": manifest_sha256 or None,
        "evidence_bundle_path": str(paths["evidence_bundle"]),
        "full_stage_plan_hash": next(iter(full_stage_hashes), None),
        "benchmark_dir": str(benchmark_dir) if benchmark_dir else None,
        "recorded_benchmark_content_hash": recorded_benchmark_hash or None,
        "current_benchmark_content_hash": current_benchmark_hash or None,
    }


def _resolve_bundle_path(output_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else output_dir / candidate


def _validate_plan_run_matrix(
    plan: dict[str, Any],
    rows: list[dict[str, str]],
    blockers: list[str],
) -> None:
    planned_runs = plan.get("runs")
    if not isinstance(planned_runs, list):
        blockers.append("plan runs must be a list")
        return
    planned_matrix: list[tuple[str, object, str]] = []
    for entry in planned_runs:
        if not isinstance(entry, dict):
            blockers.append("plan run entries must be objects")
            continue
        planned_matrix.append(
            (
                str(entry.get("variant") or ""),
                _normalize_seed(entry.get("seed")),
                str(entry.get("experiment_id") or ""),
            )
        )
    actual_matrix = [
        (
            str(row.get("variant") or ""),
            _normalize_seed(row.get("seed")),
            str(row.get("experiment_id") or ""),
        )
        for row in rows
    ]
    if actual_matrix != planned_matrix:
        blockers.append("ablation_runs.csv matrix does not exactly match the current plan")


def _validate_summary_binding(
    run_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    blockers: list[str],
) -> None:
    run_counts: dict[str, int] = {}
    for row in run_rows:
        variant = str(row.get("variant") or "")
        if variant:
            run_counts[variant] = run_counts.get(variant, 0) + 1
    summary_counts: dict[str, int] = {}
    for row in summary_rows:
        variant = str(row.get("variant") or "")
        if not variant or variant in summary_counts:
            blockers.append("ablation_summary.csv must contain exactly one row per variant")
            continue
        try:
            summary_counts[variant] = int(str(row.get("runs") or ""))
        except ValueError:
            blockers.append(f"ablation_summary.csv has invalid runs count for {variant}")
    if summary_counts != run_counts:
        blockers.append("ablation_summary.csv variant/run counts do not match ablation_runs.csv")


def _audit_seed_provenance(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
    ablation_variants: list[str],
    bound_run_provenance: object,
    allow_external_run_dirs: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    values: dict[str, set[int]] = {field: set() for field in SEED_PROVENANCE_FIELDS}
    tuples_by_variant: dict[str, set[tuple[int, int, int]]] = {}
    checked_runs = 0
    for row in rows:
        experiment_id = str(row.get("experiment_id") or "unknown-run")
        row_values = {field: _integer_seed(row.get(field)) for field in SEED_PROVENANCE_FIELDS}
        legacy_seed = _integer_seed(row.get("seed"))
        if legacy_seed != row_values["search_seed"]:
            blockers.append(f"{experiment_id}: seed must equal search_seed")
        for field, value in row_values.items():
            if value is None:
                blockers.append(f"{experiment_id}: {field} is missing or not an integer")
            source = str(row.get(f"{field}_source") or "").strip()
            if not source or source.startswith("unavailable"):
                blockers.append(f"{experiment_id}: {field} has no auditable source")
        if not _truthy(row.get("seed_provenance_complete")):
            blockers.append(f"{experiment_id}: seed_provenance_complete is not true")

        run_dir = _resolve_run_dir(output_dir, row.get("run_dir"))
        if run_dir is None or not run_dir.is_dir():
            blockers.append(f"{experiment_id}: run_dir is missing")
            continue
        if not _run_provenance_binding_matches(
            experiment_id,
            run_dir,
            output_dir=output_dir,
            bindings=bound_run_provenance,
        ):
            blockers.append(f"{experiment_id}: run provenance artifact binding is missing or stale")
            continue
        if (
            not allow_external_run_dirs
            and not _is_relative_to(run_dir.resolve(), output_dir.resolve())
        ):
            blockers.append(f"{experiment_id}: run_dir is outside the ablation output directory")
            continue
        run_config, config_issues = _read_json(run_dir / "config.json")
        run_metadata, metadata_issues = _read_json(run_dir / "run_metadata.json")
        contract, contract_issues = _read_json(run_dir / "evaluation_contract.json")
        trace_summary, trace_issues = _read_json(run_dir / "trace_summary.json")
        ledger_path = run_dir / "llm_call_ledger.jsonl"
        if config_issues or metadata_issues or contract_issues or trace_issues:
            blockers.append(f"{experiment_id}: seed provenance run artifacts are incomplete")
            continue
        if run_metadata.get("run_state") not in {"completed", "exported", "finalized"}:
            blockers.append(f"{experiment_id}: run_state is not exported/completed/finalized")
        quality_gate = trace_summary.get("quality_gate")
        if not isinstance(quality_gate, dict) or quality_gate.get("passed") is not True:
            blockers.append(f"{experiment_id}: trace quality_gate did not pass")
        llm_calls = run_metadata.get("llm_calls")
        total_llm_calls = llm_calls.get("total") if isinstance(llm_calls, dict) else None
        if (
            not isinstance(total_llm_calls, int)
            or isinstance(total_llm_calls, bool)
            or total_llm_calls <= 0
        ):
            blockers.append(f"{experiment_id}: positive real llm_calls total is required")
        if not ledger_path.is_file() or not ledger_path.read_text(encoding="utf-8").strip():
            blockers.append(f"{experiment_id}: non-empty llm_call_ledger.jsonl is required")
        artifact_values = _artifact_seed_values(run_config, run_metadata, contract)
        for field in SEED_PROVENANCE_FIELDS:
            if artifact_values[field] is None:
                blockers.append(f"{experiment_id}: {field} is not backed by run artifacts")
            elif row_values[field] != artifact_values[field]:
                blockers.append(f"{experiment_id}: {field} disagrees with run artifacts")
            else:
                values[field].add(artifact_values[field])
        if str(run_config.get("experiment_id") or "") != experiment_id:
            blockers.append(f"{experiment_id}: config experiment_id mismatch")
        if str(run_metadata.get("llm_mode") or "") != str(row.get("llm_mode") or ""):
            blockers.append(f"{experiment_id}: llm_mode disagrees with run_metadata.json")
        benchmark_name = str(contract.get("benchmark_name") or "")
        if benchmark_name != str(row.get("benchmark_name") or ""):
            blockers.append(f"{experiment_id}: benchmark_name disagrees with evaluation_contract.json")
        contract_hash = str(contract.get("contract_hash") or "")
        if contract_hash != str(row.get("benchmark_contract_hash") or ""):
            blockers.append(
                f"{experiment_id}: benchmark_contract_hash disagrees with evaluation_contract.json"
            )
        source_manifest_digest = str(contract.get("benchmark_source_manifest_digest") or "")
        if (
            not source_manifest_digest
            or source_manifest_digest
            != str(row.get("benchmark_source_manifest_digest") or "")
        ):
            blockers.append(
                f"{experiment_id}: benchmark_source_manifest_digest is missing or disagrees "
                "with evaluation_contract.json"
            )
        scientific_tuple = tuple(
            artifact_values[field] for field in SCIENTIFIC_RANDOM_SEED_FIELDS
        )
        if all(isinstance(value, int) and not isinstance(value, bool) for value in scientific_tuple):
            tuples_by_variant.setdefault(str(row.get("variant") or ""), set()).add(
                scientific_tuple  # type: ignore[arg-type]
            )
        checked_runs += 1

    sorted_values = {field: sorted(field_values) for field, field_values in values.items()}
    varied_dimensions = [
        field for field in SCIENTIFIC_RANDOM_SEED_FIELDS if len(sorted_values[field]) >= 2
    ]
    if not varied_dimensions:
        blockers.append(
            "at least one scientific random dimension (data_seed, model_seed, provider_seed) "
            "must actually vary"
        )
    nonvarying_variants = [
        variant for variant in ablation_variants if len(tuples_by_variant.get(variant, set())) < 2
    ]
    if nonvarying_variants:
        blockers.append(
            "each non-baseline variant must vary an artifact-backed scientific seed tuple: "
            + ", ".join(sorted(nonvarying_variants))
        )
    return {
        "required_fields": list(SEED_PROVENANCE_FIELDS),
        "scientific_random_fields": list(SCIENTIFIC_RANDOM_SEED_FIELDS),
        "values": sorted_values,
        "seed_provenance_complete": bool(rows) and not any(
            "seed" in blocker and (
                "missing" in blocker
                or "no auditable source" in blocker
                or "not backed" in blocker
                or "disagrees" in blocker
                or "not true" in blocker
            )
            for blocker in blockers
        ),
        "scientific_random_dimension_varied": bool(varied_dimensions),
        "varied_scientific_dimensions": varied_dimensions,
        "scientific_seed_tuple_count_by_variant": {
            variant: len(tuples) for variant, tuples in sorted(tuples_by_variant.items())
        },
        "checked_run_count": checked_runs,
        "blockers": _unique_strings(blockers),
    }


def _artifact_seed_values(
    run_config: dict[str, Any],
    run_metadata: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, int | None]:
    evolution = run_config.get("evolution")
    source_manifest = contract.get("benchmark_source_manifest")
    metadata_provenance = run_metadata.get("seed_provenance")
    return {
        "search_seed": (
            _integer_seed(evolution.get("random_seed")) if isinstance(evolution, dict) else None
        ),
        "data_seed": (
            _integer_seed(source_manifest.get("data_seed"))
            if isinstance(source_manifest, dict)
            else None
        ),
        "model_seed": _metadata_seed(run_metadata, metadata_provenance, "model_seed"),
        "provider_seed": _metadata_seed(run_metadata, metadata_provenance, "provider_seed"),
    }


def _run_provenance_binding_matches(
    experiment_id: str,
    run_dir: Path,
    *,
    output_dir: Path,
    bindings: object,
) -> bool:
    if not isinstance(bindings, dict):
        return False
    binding = bindings.get(experiment_id)
    if not isinstance(binding, dict):
        return False
    if Path(str(binding.get("run_dir") or "")).resolve() != run_dir.resolve():
        return False
    for name in (
        "config.json",
        "run_metadata.json",
        "evaluation_contract.json",
        "trace_summary.json",
        "llm_call_ledger.jsonl",
    ):
        descriptor = binding.get(name)
        path = run_dir / name
        if not isinstance(descriptor, dict) or not path.is_file():
            return False
        bound_path = _resolve_bundle_path(output_dir, descriptor.get("path"))
        if bound_path is None or bound_path.resolve() != path.resolve():
            return False
        if descriptor.get("sha256") != _sha256(path):
            return False
        if name.endswith(".json"):
            payload, issues = _read_json(path)
            if issues or descriptor.get("payload_hash") != _hash_payload(payload):
                return False
    return True


def _metadata_seed(
    run_metadata: dict[str, Any],
    provenance: object,
    key: str,
) -> int | None:
    if isinstance(provenance, dict):
        nested = _integer_seed(provenance.get(key))
        if nested is not None:
            return nested
    return _integer_seed(run_metadata.get(key))


def _resolve_run_dir(output_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else output_dir / path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _repeated_result_variants(rows: list[dict[str, str]]) -> list[str]:
    fields = (
        "champion_score",
        "root_score",
        "champion/root improvement",
        "valid_solution_rate",
        "timeout_count",
        "debug_success_count",
    )
    signatures: dict[str, set[tuple[str, ...]]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        variant = str(row.get("variant") or "")
        signatures.setdefault(variant, set()).add(tuple(str(row.get(field) or "") for field in fields))
        counts[variant] = counts.get(variant, 0) + 1
    return sorted(
        variant
        for variant, values in signatures.items()
        if counts.get(variant, 0) >= 2 and len(values) == 1
    )


def _collect_seeds(rows: list[dict[str, str]]) -> list[object]:
    return [
        normalized
        for row in rows
        if (normalized := _normalize_seed(row.get("search_seed") or row.get("seed"))) is not None
    ]


def _collect_variants(
    run_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> list[str]:
    variants: list[str] = []
    for row in [*run_rows, *summary_rows]:
        value = (row.get("variant") or "").strip()
        if value:
            variants.append(value)
    return variants


def _seed_coverage_by_variant(rows: list[dict[str, str]]) -> dict[str, list[object]]:
    coverage: dict[str, list[object]] = {}
    for row in rows:
        variant = (row.get("variant") or "").strip()
        seed = _normalize_seed(row.get("search_seed") or row.get("seed"))
        if not variant or seed is None:
            continue
        coverage.setdefault(variant, []).append(seed)
    return {variant: _sorted_values(seeds) for variant, seeds in sorted(coverage.items())}


def _collect_values(
    run_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    key: str,
) -> list[str]:
    values: list[str] = []
    for row in [*run_rows, *summary_rows]:
        value = (row.get(key) or "").strip()
        if value:
            values.append(value)
    return values


def _optional_sequence(source: Mapping[str, object], key: str) -> list[object]:
    value = source.get(key)
    if not isinstance(value, list):
        return []
    normalized = [
        _normalize_seed(item) if key == "expected_seeds" else str(item)
        for item in value
    ]
    return _sorted_values([item for item in normalized if item is not None])


def _normalize_seed(value: object) -> object | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _integer_seed(value: object) -> int | None:
    normalized = _normalize_seed(value)
    if isinstance(normalized, bool) or not isinstance(normalized, int):
        return None
    return normalized


def _strict_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _truthy(value: object) -> bool:
    return value in (True, "True", "true", "1", 1)


def _sorted_values(values: list[object]) -> list[object]:
    unique = {value for value in values if value not in (None, "")}
    return sorted(unique, key=lambda value: (str(type(value)), str(value)))


def _verifier(source: Mapping[str, object]) -> str:
    return str(
        source.get("verified_by")
        or source.get("reviewer")
        or source.get("verification_source")
        or ""
    ).strip()


def _artifact_descriptors(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    descriptors: dict[str, dict[str, object]] = {}
    for name in (
        "runs_csv",
        "summary_csv",
        "report_md",
        "plan_json",
        "manifest_json",
        "evidence_bundle",
        "collection_manifest",
    ):
        path = paths[name]
        exists = path.is_file()
        descriptors[name] = {
            "path": str(path),
            "exists": exists,
            "sha256": _sha256(path) if exists else None,
        }
    return descriptors


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _benchmark_content_hash(benchmark_dir: Path) -> str:
    root = benchmark_dir.resolve()
    items: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        items.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    return _hash_payload({"schema_version": 1, "artifacts": items})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_values(values: list[object]) -> str:
    return ", ".join(str(value) for value in values)


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
