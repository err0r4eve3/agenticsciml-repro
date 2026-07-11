from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    EVIDENCE_MODE_REAL_LLM_ABLATION,
    LLM_MODE_MOCK,
    LLM_MODE_REAL,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
)
from agenticsciml.llm.base import LLMClient
from agenticsciml.llm.budget import (
    LLMBudget,
    combine_llm_call_ranges,
    estimate_orchestrator_llm_call_range,
    llm_call_budget_preflight,
    require_llm_call_budget_preflight,
)
from agenticsciml.llm.capabilities import capabilities_for_openai_compatible
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.llm_smoke import _RecordingLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.storage import atomic_write_text


DEFAULT_VARIANTS = (
    "root_only",
    "no_kb",
    "kb",
    "random_kb",
    "no_critic",
    "no_debugger",
    "branch_context",
    "no_branch_context",
)

SEED_PROVENANCE_FIELDS = (
    "search_seed",
    "data_seed",
    "model_seed",
    "provider_seed",
)
SCIENTIFIC_RANDOM_SEED_FIELDS = ("data_seed", "model_seed", "provider_seed")


@dataclass(frozen=True, slots=True)
class AblationResult:
    summary_csv: Path | None
    report_md: Path
    runs_csv: Path | None = None
    plan_json: Path | None = None
    manifest_json: Path | None = None


def run_ablation(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    seeds: list[int],
    variants: list[str] | None = None,
    mock: bool = True,
    dry_run: bool = False,
    timeout_s: int = 60,
    llm_timeout_s: float | None = None,
    llm_max_retries: int | None = None,
    llm_fast_mode: bool = False,
    budget_batch_index: int | None = None,
    llm_client: LLMClient | None = None,
) -> AblationResult:
    if mock and dry_run:
        raise ValueError("ablation dry-run is only supported with real LLM mode.")
    if mock and budget_batch_index is not None:
        raise ValueError("budget batch selection is only supported with real LLM mode.")
    selected_variants = variants or list(DEFAULT_VARIANTS)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mock:
        return _run_real_ablation(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            seeds=seeds,
            variants=selected_variants,
            dry_run=dry_run,
            timeout_s=timeout_s,
            llm_timeout_s=llm_timeout_s,
            llm_max_retries=llm_max_retries,
            llm_fast_mode=llm_fast_mode,
            budget_batch_index=budget_batch_index,
            llm_client=llm_client,
        )

    plan = _build_mock_ablation_plan(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        seeds=seeds,
        variants=selected_variants,
        timeout_s=timeout_s,
    )
    plan_path = output_dir / "ablation_plan.json"
    atomic_write_text(
        plan_path,
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False),
    )
    manifest = _build_mock_ablation_manifest(plan)
    manifest_path = output_dir / "ablation_manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
    )

    run_rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        for seed in seeds:
            run_rows.append(
                _run_variant(
                    benchmark_dir,
                    output_dir,
                    variant,
                    seed,
                    mock=True,
                    timeout_s=timeout_s,
                )
            )

    _bind_run_rows_to_execution_artifacts(
        run_rows,
        plan=plan,
        plan_path=plan_path,
        manifest_path=manifest_path,
    )
    runs_csv = output_dir / "ablation_runs.csv"
    _write_csv(runs_csv, run_rows)
    summary_rows = _aggregate(run_rows)
    summary_csv = output_dir / "ablation_summary.csv"
    _write_csv(summary_csv, summary_rows)
    report_md = output_dir / "ablation_report.md"
    atomic_write_text(report_md, _render_report(summary_rows))
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
    return AblationResult(summary_csv=summary_csv, report_md=report_md, runs_csv=runs_csv)


def _run_real_ablation(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    seeds: list[int],
    variants: list[str],
    dry_run: bool,
    timeout_s: int,
    llm_timeout_s: float | None,
    llm_max_retries: int | None,
    llm_fast_mode: bool,
    budget_batch_index: int | None,
    llm_client: LLMClient | None,
) -> AblationResult:
    budget = LLMBudget.from_env()
    plan = _build_real_ablation_plan(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        seeds=seeds,
        variants=variants,
        timeout_s=timeout_s,
        dry_run=dry_run,
    )
    plan = _with_budget_batch_plan(plan, budget)
    if budget_batch_index is not None:
        plan = _select_budget_batch(plan, budget_batch_index)
    plan_path = output_dir / "real_llm_ablation_plan.json"
    atomic_write_text(plan_path, json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    manifest = _build_real_ablation_manifest(plan, llm_client=llm_client, budget=budget)
    manifest_path = output_dir / "real_llm_ablation_manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
    )
    report_md = output_dir / "ablation_report.md"
    if dry_run:
        atomic_write_text(report_md, _render_real_dry_run_report(plan))
        return AblationResult(
            summary_csv=None,
            report_md=report_md,
            plan_json=plan_path,
            manifest_json=manifest_path,
        )
    preflight = manifest.get("budget_preflight", {})
    if isinstance(preflight, dict) and preflight.get("passed") is not True:
        atomic_write_text(report_md, _render_real_budget_blocked_report(plan, preflight))
        require_llm_call_budget_preflight(preflight)

    try:
        inner_llm = llm_client or OpenAIAdapter(timeout_s=llm_timeout_s, max_retries=llm_max_retries)
    except Exception as exc:
        atomic_write_text(report_md, _render_real_failure_report(plan, "adapter_init_error", exc))
        raise

    run_rows: list[dict[str, Any]] = []
    try:
        for entry in plan["runs"]:
            variant = str(entry["variant"])
            seed = int(entry["seed"])
            expected_run_dir = output_dir / "runs" / str(entry["experiment_id"])
            ledger_path = expected_run_dir / "llm_call_ledger.jsonl"
            if ledger_path.exists():
                ledger_path.unlink()
            recording_llm = _RecordingLLMClient(inner_llm, ledger_path, budget)
            run_rows.append(
                _run_variant(
                    benchmark_dir,
                    output_dir,
                    variant,
                    seed,
                    mock=False,
                    timeout_s=timeout_s,
                    llm_fast_mode=llm_fast_mode,
                    llm_client=recording_llm,
                )
            )
    except Exception as exc:
        atomic_write_text(report_md, _render_real_failure_report(plan, "orchestrator_error", exc))
        raise

    _bind_run_rows_to_execution_artifacts(
        run_rows,
        plan=plan,
        plan_path=plan_path,
        manifest_path=manifest_path,
    )
    runs_csv = output_dir / "ablation_runs.csv"
    _write_csv(runs_csv, run_rows)
    summary_rows = _aggregate(run_rows)
    summary_csv = output_dir / "ablation_summary.csv"
    _write_csv(summary_csv, summary_rows)
    atomic_write_text(report_md, _render_report(summary_rows))
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
    return AblationResult(
        summary_csv=summary_csv,
        report_md=report_md,
        runs_csv=runs_csv,
        plan_json=plan_path,
        manifest_json=manifest_path,
    )


def _run_variant(
    benchmark_dir: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    *,
    mock: bool,
    timeout_s: int,
    llm_fast_mode: bool = False,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    config = _variant_config(variant, seed, timeout_s=timeout_s)
    experiment_id = f"{variant}-seed-{seed}"
    run_config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=benchmark_dir,
        output_dir=output_dir / "runs",
        evolution=config,
        use_mock=mock,
        llm_fast_mode=llm_fast_mode,
    )
    llm = MockLLMClient() if mock else llm_client
    if llm is None:
        raise RuntimeError("real ablation requires an LLM client")
    run_dir = AgenticSciMLOrchestrator(run_config, llm).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    run_config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    evaluation_contract = json.loads(
        (run_dir / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    seed_provenance = _seed_provenance_from_run_artifacts(
        search_seed=seed,
        run_config=run_config_payload,
        run_metadata=metadata,
        evaluation_contract=evaluation_contract,
    )
    nodes = tree["nodes"]
    root = next(node for node in nodes if node["node_id"] == "solution_000")
    champion = _champion(nodes)
    root_score = _score_value(root)
    champion_score = _score_value(champion)
    improvement = _improvement(root, champion)
    evaluated_count = sum(1 for node in nodes if node.get("status") == "evaluated")
    timeout_count = sum(1 for node in nodes if node.get("failure_kind") == "timeout")
    debug_success_count = sum(
        1
        for node in nodes
        if int(node.get("num_debug_attempts", 0)) > 0 and node.get("status") == "evaluated"
    )
    branch_tags = sorted(
        {
            tag
            for node in nodes
            for tag in node.get("method_tags", [])
            if isinstance(tag, str) and tag.startswith("branch:")
        }
    )
    branch_context_count = sum(
        1
        for node in nodes
        if (run_dir / "solutions" / str(node["node_id"]) / "branch_context.json").exists()
        and json.loads(
            (run_dir / "solutions" / str(node["node_id"]) / "branch_context.json").read_text(
                encoding="utf-8"
            )
        )
    )
    run_llm_mode = str(metadata.get("llm_mode") or (LLM_MODE_MOCK if mock else LLM_MODE_REAL))
    run_evidence_mode = str(metadata.get("evidence_mode") or "")
    run_scientific_claim = str(metadata.get("scientific_claim") or "")
    return {
        "variant": variant,
        "seed": seed,
        "experiment_id": experiment_id,
        "search_seed": seed_provenance["search_seed"],
        "data_seed": seed_provenance["data_seed"],
        "model_seed": seed_provenance["model_seed"],
        "provider_seed": seed_provenance["provider_seed"],
        "search_seed_source": seed_provenance["search_seed_source"],
        "data_seed_source": seed_provenance["data_seed_source"],
        "model_seed_source": seed_provenance["model_seed_source"],
        "provider_seed_source": seed_provenance["provider_seed_source"],
        "seed_provenance_complete": seed_provenance["seed_provenance_complete"],
        "execution_mode": "mock" if mock else "real",
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE if mock else EVIDENCE_MODE_REAL_LLM_ABLATION,
        "llm_mode": run_llm_mode,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "run_evidence_mode": run_evidence_mode,
        "run_scientific_claim": run_scientific_claim,
        "run_dir": str(run_dir),
        "benchmark_name": str(evaluation_contract.get("benchmark_name") or ""),
        "benchmark_contract_hash": str(evaluation_contract.get("contract_hash") or ""),
        "benchmark_source_manifest_digest": str(
            evaluation_contract.get("benchmark_source_manifest_digest") or ""
        ),
        "branch_context_enabled": bool(metadata.get("branch_context_enabled", False)),
        "champion_node_id": champion["node_id"],
        "metric_name": champion.get("score", {}).get("metric", ""),
        "higher_is_better": _higher_is_better(champion),
        "champion_score": champion_score,
        "root_score": root_score,
        "champion/root improvement": improvement,
        "valid_solution_rate": evaluated_count / len(nodes) if nodes else 0.0,
        "timeout_count": timeout_count,
        "debug_success_count": debug_success_count,
        "branch_context_count": branch_context_count,
        "branch_intents": ",".join(tag.replace("branch:", "", 1) for tag in branch_tags),
        "llm_calls": metadata.get("llm_calls", {}).get("total", 0),
        "wall_time_s": metadata.get("wall_time_s", 0.0),
    }


def _variant_config(variant: str, seed: int, *, timeout_s: int = 60) -> EvolutionConfig:
    common = {
        "parallel_mutations": 1,
        "max_debug_retries": 1,
        "random_seed": seed,
        "timeout_s": timeout_s,
    }
    if variant == "root_only":
        return EvolutionConfig(max_iterations=0, use_kb=True, **common)
    if variant == "no_kb":
        return EvolutionConfig(max_iterations=1, use_kb=False, **common)
    if variant == "kb":
        return EvolutionConfig(max_iterations=1, use_kb=True, **common)
    if variant == "random_kb":
        return EvolutionConfig(max_iterations=1, use_kb=True, random_kb=True, **common)
    if variant == "no_critic":
        return EvolutionConfig(max_iterations=1, use_kb=True, use_critic=False, **common)
    if variant == "no_debugger":
        return EvolutionConfig(
            max_iterations=1,
            use_kb=True,
            use_debugger=False,
            max_debug_retries=0,
            parallel_mutations=1,
            random_seed=seed,
        )
    if variant == "branch_context":
        branch_common = {**common, "parallel_mutations": 2}
        return EvolutionConfig(max_iterations=1, use_kb=True, **branch_common)
    if variant == "no_branch_context":
        return EvolutionConfig(
            max_iterations=1,
            use_kb=True,
            use_branch_context=False,
            parallel_mutations=2,
            max_debug_retries=1,
            timeout_s=timeout_s,
            random_seed=seed,
        )
    raise ValueError(f"Unknown ablation variant: {variant}")


def _seed_provenance_from_run_artifacts(
    *,
    search_seed: int,
    run_config: dict[str, Any],
    run_metadata: dict[str, Any],
    evaluation_contract: dict[str, Any],
) -> dict[str, Any]:
    evolution = run_config.get("evolution")
    configured_search_seed = (
        _integer_seed(evolution.get("random_seed")) if isinstance(evolution, dict) else None
    )
    source_manifest = evaluation_contract.get("benchmark_source_manifest")
    data_seed = (
        _integer_seed(source_manifest.get("data_seed"))
        if isinstance(source_manifest, dict)
        else None
    )
    metadata_provenance = run_metadata.get("seed_provenance")
    model_seed = _seed_from_metadata(run_metadata, metadata_provenance, "model_seed")
    provider_seed = _seed_from_metadata(run_metadata, metadata_provenance, "provider_seed")
    values = {
        "search_seed": configured_search_seed,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "provider_seed": provider_seed,
    }
    return {
        **values,
        "search_seed_source": (
            "config.json:evolution.random_seed"
            if configured_search_seed is not None and configured_search_seed == search_seed
            else "unavailable_or_mismatched"
        ),
        "data_seed_source": (
            "evaluation_contract.json:benchmark_source_manifest.data_seed"
            if data_seed is not None
            else "unavailable"
        ),
        "model_seed_source": (
            "run_metadata.json:seed_provenance.model_seed"
            if isinstance(metadata_provenance, dict)
            and _integer_seed(metadata_provenance.get("model_seed")) is not None
            else "run_metadata.json:model_seed"
            if _integer_seed(run_metadata.get("model_seed")) is not None
            else "unavailable"
        ),
        "provider_seed_source": (
            "run_metadata.json:seed_provenance.provider_seed"
            if isinstance(metadata_provenance, dict)
            and _integer_seed(metadata_provenance.get("provider_seed")) is not None
            else "run_metadata.json:provider_seed"
            if _integer_seed(run_metadata.get("provider_seed")) is not None
            else "unavailable"
        ),
        "seed_provenance_complete": all(values[field] is not None for field in SEED_PROVENANCE_FIELDS),
    }


def _seed_from_metadata(
    run_metadata: dict[str, Any],
    seed_provenance: object,
    key: str,
) -> int | None:
    if isinstance(seed_provenance, dict):
        nested = _integer_seed(seed_provenance.get(key))
        if nested is not None:
            return nested
    return _integer_seed(run_metadata.get(key))


def _integer_seed(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _build_mock_ablation_plan(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    seeds: list[int],
    variants: list[str],
    timeout_s: int,
) -> dict[str, Any]:
    runs = [
        {
            "variant": variant,
            "seed": seed,
            "experiment_id": f"{variant}-seed-{seed}",
            "evolution": _variant_config(variant, seed, timeout_s=timeout_s).to_dict(),
        }
        for variant in variants
        for seed in seeds
    ]
    plan = {
        "schema_version": 1,
        "benchmark_dir": str(benchmark_dir),
        "benchmark_content_hash": _benchmark_content_hash(benchmark_dir),
        "output_dir": str(output_dir),
        "seeds": list(seeds),
        "variants": list(variants),
        "timeout_s": timeout_s,
        "execution_mode": "mock",
        "real_mode_explicit": False,
        "provider_calls_enabled": False,
        "seed_dimensions": list(SEED_PROVENANCE_FIELDS),
        "scientific_random_seed_dimensions": list(SCIENTIFIC_RANDOM_SEED_FIELDS),
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "runs": runs,
        "claim_boundary": {
            "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
            "paper_score_reproduction": False,
            "emergent_discovery_supported": False,
            "notes": "Mock ablation validates workflow shape only.",
        },
    }
    plan["full_stage_plan_hash"] = _hash_payload(
        {
            key: plan[key]
            for key in (
                "schema_version",
                "benchmark_dir",
                "benchmark_content_hash",
                "seeds",
                "variants",
                "timeout_s",
                "execution_mode",
                "evidence_mode",
                "scientific_claim",
                "runs",
                "claim_boundary",
            )
        }
    )
    return plan


def _build_mock_ablation_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_mode": "mock",
        "real_mode_explicit": False,
        "provider_calls_enabled": False,
        "seed_dimensions": list(SEED_PROVENANCE_FIELDS),
        "scientific_random_seed_dimensions": list(SCIENTIFIC_RANDOM_SEED_FIELDS),
        "benchmark_dir": plan["benchmark_dir"],
        "benchmark_content_hash": plan["benchmark_content_hash"],
        "seeds": list(plan["seeds"]),
        "variants": list(plan["variants"]),
        "run_count": len(plan["runs"]),
        "full_stage_plan_hash": plan["full_stage_plan_hash"],
        "plan_hash": _hash_payload(plan),
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "claim_boundary": plan["claim_boundary"],
    }


def _build_real_ablation_plan(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    seeds: list[int],
    variants: list[str],
    timeout_s: int,
    dry_run: bool,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            config = _variant_config(variant, seed, timeout_s=timeout_s)
            expected_call_range = estimate_orchestrator_llm_call_range(
                max_iterations=config.max_iterations,
                parallel_mutations=config.parallel_mutations,
            )
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "seed_provenance": {
                        "search_seed": seed,
                        "data_seed": None,
                        "model_seed": None,
                        "provider_seed": None,
                        "status": "resolved_from_completed_run_artifacts",
                    },
                    "experiment_id": f"{variant}-seed-{seed}",
                    "evolution": config.to_dict(),
                    "expected_llm_call_range": expected_call_range,
                }
            )
    expected_llm_call_range = combine_llm_call_ranges(
        [
            dict(entry["expected_llm_call_range"])
            for entry in runs
            if isinstance(entry.get("expected_llm_call_range"), dict)
        ]
    )
    plan = {
        "schema_version": 1,
        "benchmark_dir": str(benchmark_dir),
        "benchmark_content_hash": _benchmark_content_hash(benchmark_dir),
        "output_dir": str(output_dir),
        "seeds": list(seeds),
        "variants": list(variants),
        "timeout_s": timeout_s,
        "timeout_scope": "generated solution subprocesses; provider HTTP request timeout is adapter-specific",
        "execution_mode": "dry_run" if dry_run else "real",
        "real_mode_explicit": not dry_run,
        "provider_calls_enabled": not dry_run,
        "seed_dimensions": list(SEED_PROVENANCE_FIELDS),
        "scientific_random_seed_dimensions": list(SCIENTIFIC_RANDOM_SEED_FIELDS),
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_ABLATION,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "runs": runs,
        "full_stage_run_count": len(runs),
        "expected_llm_call_range": expected_llm_call_range,
        "full_stage_expected_llm_call_range": expected_llm_call_range,
        "expected_artifacts": _real_ablation_expected_artifacts(runs),
        "claim_boundary": {
            "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
            "paper_score_reproduction": False,
            "emergent_discovery_supported": False,
            "notes": (
                "Real LLM ablation is workflow contrast evidence only until completed runs, "
                "multi-seed coverage, failure review, and paper/workflow readiness gates are satisfied."
            ),
        },
    }
    plan["full_stage_plan_hash"] = _full_stage_plan_hash(plan)
    return plan


def _with_budget_batch_plan(plan: dict[str, Any], budget: LLMBudget) -> dict[str, Any]:
    updated = dict(plan)
    updated["budget_batch_plan"] = _build_budget_batch_plan(
        list(plan.get("runs", [])),
        max_calls=budget.max_calls,
    )
    return updated


def _select_budget_batch(plan: dict[str, Any], budget_batch_index: int) -> dict[str, Any]:
    if budget_batch_index < 1:
        raise ValueError("--budget-batch-index must be >= 1")
    batch_plan = plan.get("budget_batch_plan")
    if not isinstance(batch_plan, dict):
        raise ValueError("budget batch plan is missing")
    batches = batch_plan.get("batches")
    if not isinstance(batches, list):
        raise ValueError("budget batch plan has no batches")
    selected_batch = next(
        (
            batch
            for batch in batches
            if isinstance(batch, dict) and int(batch.get("batch_index", -1)) == budget_batch_index
        ),
        None,
    )
    if selected_batch is None:
        raise ValueError(f"Unknown budget batch index: {budget_batch_index}")
    if selected_batch.get("status") != "ready":
        raise ValueError(f"Budget batch {budget_batch_index} is not ready")
    experiment_ids = {
        str(experiment_id)
        for experiment_id in selected_batch.get("experiment_ids", [])
        if isinstance(experiment_id, str)
    }
    selected_runs = [
        dict(entry)
        for entry in plan.get("runs", [])
        if str(entry.get("experiment_id")) in experiment_ids
    ]
    if len(selected_runs) != int(selected_batch.get("run_count", 0)):
        raise ValueError(f"Budget batch {budget_batch_index} does not match planned runs")
    selected = dict(plan)
    selected["full_stage_run_count"] = len(plan.get("runs", []))
    selected["full_stage_expected_llm_call_range"] = dict(plan.get("expected_llm_call_range") or {})
    selected["full_stage_plan_hash"] = str(plan.get("full_stage_plan_hash", ""))
    selected["selected_budget_batch_index"] = budget_batch_index
    selected["budget_batch_selection"] = dict(selected_batch)
    selected["runs"] = selected_runs
    selected["expected_llm_call_range"] = combine_llm_call_ranges(
        [
            dict(entry["expected_llm_call_range"])
            for entry in selected_runs
            if isinstance(entry.get("expected_llm_call_range"), dict)
        ]
    )
    selected["expected_artifacts"] = _real_ablation_expected_artifacts(selected_runs)
    return selected


def _build_budget_batch_plan(runs: list[dict[str, Any]], *, max_calls: int | None) -> dict[str, Any]:
    total_call_range = combine_llm_call_ranges(
        [
            dict(entry["expected_llm_call_range"])
            for entry in runs
            if isinstance(entry.get("expected_llm_call_range"), dict)
        ]
    )
    if max_calls is None:
        return {
            "schema_version": 1,
            "status": "not_configured",
            "full_stage_budget_status": "not_configured",
            "batch_plan_status": "not_configured",
            "batching_required": False,
            "configured_max_llm_calls": None,
            "required_for_execution": False,
            "planned_run_count": len(runs),
            "coverage_run_count": len(runs),
            "total_expected_llm_call_range": total_call_range,
            "batch_count": 1 if runs else 0,
            "batches": [_budget_batch(1, runs)] if runs else [],
            "blocked_runs": [],
            "strategy": "single_batch_without_call_budget",
        }
    batches: list[dict[str, Any]] = []
    blocked_runs: list[dict[str, Any]] = []
    current_runs: list[dict[str, Any]] = []
    current_max = 0
    for entry in runs:
        call_range = dict(entry.get("expected_llm_call_range") or {})
        entry_max = int(call_range.get("max", 0))
        if entry_max > max_calls:
            blocked_runs.append(
                {
                    "experiment_id": str(entry.get("experiment_id")),
                    "variant": str(entry.get("variant")),
                    "seed": int(entry.get("seed", 0)),
                    "expected_llm_call_range": call_range,
                    "status": "single_run_exceeds_budget",
                }
            )
            continue
        if current_runs and current_max + entry_max > max_calls:
            batches.append(_budget_batch(len(batches) + 1, current_runs))
            current_runs = []
            current_max = 0
        current_runs.append(entry)
        current_max += entry_max
    if current_runs:
        batches.append(_budget_batch(len(batches) + 1, current_runs))
    coverage_run_count = sum(int(batch["run_count"]) for batch in batches)
    total_max = int(total_call_range.get("max", 0))
    batching_required = total_max > max_calls
    full_stage_budget_status = (
        "blocked_by_single_run"
        if blocked_runs
        else "blocked_by_budget"
        if batching_required
        else "ready"
    )
    batch_plan_status = (
        "blocked_by_single_run"
        if blocked_runs
        else "ready"
        if coverage_run_count == len(runs)
        and all(int(dict(batch.get("expected_llm_call_range") or {}).get("max", 0)) <= max_calls for batch in batches)
        else "invalid_plan"
    )
    return {
        "schema_version": 1,
        "status": full_stage_budget_status,
        "full_stage_budget_status": full_stage_budget_status,
        "batch_plan_status": batch_plan_status,
        "batching_required": batching_required,
        "configured_max_llm_calls": max_calls,
        "required_for_execution": batching_required,
        "planned_run_count": len(runs),
        "coverage_run_count": coverage_run_count,
        "total_expected_llm_call_range": total_call_range,
        "batch_count": len(batches),
        "batches": batches,
        "blocked_runs": blocked_runs,
        "strategy": "greedy_by_expected_max_calls",
    }


def _budget_batch(batch_index: int, runs: list[dict[str, Any]]) -> dict[str, Any]:
    call_range = combine_llm_call_ranges(
        [
            dict(entry["expected_llm_call_range"])
            for entry in runs
            if isinstance(entry.get("expected_llm_call_range"), dict)
        ]
    )
    return {
        "batch_index": batch_index,
        "status": "ready",
        "run_count": len(runs),
        "experiment_ids": [str(entry["experiment_id"]) for entry in runs],
        "variants": sorted({str(entry["variant"]) for entry in runs}),
        "seeds": sorted({int(entry["seed"]) for entry in runs}),
        "expected_llm_call_range": call_range,
    }


def _real_ablation_expected_artifacts(runs: list[dict[str, Any]]) -> list[str]:
    expected_run_artifacts = [
        artifact
        for entry in runs
        for artifact in [
            f"runs/{entry['experiment_id']}/run_metadata.json",
            f"runs/{entry['experiment_id']}/tree.json",
            f"runs/{entry['experiment_id']}/checkpoint.json",
            f"runs/{entry['experiment_id']}/trace_summary.json",
            f"runs/{entry['experiment_id']}/reports/scientific_result_card.json",
            f"runs/{entry['experiment_id']}/llm_call_ledger.jsonl",
        ]
    ]
    return [
        "real_llm_ablation_plan.json",
        "real_llm_ablation_manifest.json",
        "ablation_report.md",
        "ablation_runs.csv",
        "ablation_summary.csv",
        *expected_run_artifacts,
    ]


def _build_real_ablation_manifest(
    plan: dict[str, Any],
    *,
    llm_client: LLMClient | None,
    budget: LLMBudget,
) -> dict[str, Any]:
    provider_capabilities = _provider_capabilities(llm_client)
    run_count = len(plan.get("runs", []))
    expected_llm_call_range = dict(plan.get("expected_llm_call_range") or {})
    budget_preflight = llm_call_budget_preflight(
        budget=budget,
        expected_llm_call_range=expected_llm_call_range,
    )
    return {
        "schema_version": 1,
        "execution_mode": plan["execution_mode"],
        "real_mode_explicit": plan["real_mode_explicit"],
        "provider_calls_enabled": plan.get("provider_calls_enabled", False),
        "provider": provider_capabilities["provider"],
        "model": getattr(llm_client, "model", None) or os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "adapter_type": getattr(llm_client, "adapter_type", None) or provider_capabilities["adapter_type"],
        "provider_capabilities": provider_capabilities,
        "benchmark_content_hash": plan.get("benchmark_content_hash"),
        "benchmark_dir": plan.get("benchmark_dir"),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "seeds": plan["seeds"],
        "variants": plan["variants"],
        "run_count": run_count,
        "full_stage_run_count": plan.get("full_stage_run_count", run_count),
        "full_stage_plan_hash": plan.get("full_stage_plan_hash", ""),
        "selected_budget_batch_index": plan.get("selected_budget_batch_index"),
        "timeout_s": plan["timeout_s"],
        "timeout_scope": plan["timeout_scope"],
        "output_dir": plan["output_dir"],
        "plan_hash": _hash_payload(plan),
        "config_hash": _hash_payload({"runs": plan["runs"], "benchmark_dir": plan["benchmark_dir"]}),
        "token_budget": budget.to_dict(),
        "expected_llm_call_range": expected_llm_call_range,
        "full_stage_expected_llm_call_range": plan.get(
            "full_stage_expected_llm_call_range",
            expected_llm_call_range,
        ),
        "budget_preflight": budget_preflight,
        "budget_batch_plan": plan.get("budget_batch_plan"),
        "budget_batch_selection": plan.get("budget_batch_selection"),
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_ABLATION,
        "seed_dimensions": list(SEED_PROVENANCE_FIELDS),
        "scientific_random_seed_dimensions": list(SCIENTIFIC_RANDOM_SEED_FIELDS),
        "claim_boundary": plan["claim_boundary"],
    }


def _provider_capabilities(llm_client: LLMClient | None) -> dict[str, Any]:
    capabilities = getattr(llm_client, "provider_capabilities", None)
    if isinstance(capabilities, dict):
        return dict(capabilities)
    if capabilities is not None and hasattr(capabilities, "to_dict"):
        return dict(capabilities.to_dict())
    return capabilities_for_openai_compatible(os.environ.get("OPENAI_BASE_URL")).to_dict()


def _package_versions() -> dict[str, str | None]:
    packages = ("agenticsciml", "openai", "numpy")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _full_stage_plan_hash(plan: dict[str, Any]) -> str:
    return _hash_payload(
        {
            "schema_version": plan.get("schema_version"),
            "benchmark_dir": plan.get("benchmark_dir"),
            "benchmark_content_hash": plan.get("benchmark_content_hash"),
            "seeds": plan.get("seeds"),
            "variants": plan.get("variants"),
            "timeout_s": plan.get("timeout_s"),
            "timeout_scope": plan.get("timeout_scope"),
            "evidence_mode": plan.get("evidence_mode"),
            "scientific_claim": plan.get("scientific_claim"),
            "runs": plan.get("runs"),
            "expected_llm_call_range": plan.get("expected_llm_call_range"),
            "claim_boundary": plan.get("claim_boundary"),
        }
    )


def _benchmark_content_hash(benchmark_dir: Path) -> str:
    root = benchmark_dir.resolve()
    items: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        items.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return _hash_payload({"schema_version": 1, "artifacts": items})


def _bind_run_rows_to_execution_artifacts(
    run_rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    plan_path: Path,
    manifest_path: Path,
) -> None:
    plan_hash = _hash_payload(plan)
    plan_sha256 = _sha256_file(plan_path)
    manifest_sha256 = _sha256_file(manifest_path)
    benchmark_content_hash = str(plan.get("benchmark_content_hash") or "")
    full_stage_plan_hash = str(plan.get("full_stage_plan_hash") or "")
    execution_mode = str(plan.get("execution_mode") or "")
    for row in run_rows:
        row.update(
            {
                "benchmark_content_hash": benchmark_content_hash,
                "plan_hash": plan_hash,
                "plan_sha256": plan_sha256,
                "manifest_sha256": manifest_sha256,
                "full_stage_plan_hash": full_stage_plan_hash,
                "execution_mode": execution_mode,
            }
        )


def _write_ablation_evidence_bundle(
    *,
    output_dir: Path,
    plan: dict[str, Any],
    plan_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    runs_csv: Path,
    summary_csv: Path,
    report_md: Path,
    run_rows: list[dict[str, Any]],
    source_type: str = "direct_ablation",
    additional_artifacts: dict[str, Path] | None = None,
) -> Path:
    seed_provenance = _seed_provenance_summary(run_rows)
    artifacts = {
        "plan_json": _bundle_artifact_descriptor(plan_path, output_dir, include_payload_hash=True),
        "manifest_json": _bundle_artifact_descriptor(
            manifest_path,
            output_dir,
            include_payload_hash=True,
        ),
        "runs_csv": _bundle_artifact_descriptor(runs_csv, output_dir),
        "summary_csv": _bundle_artifact_descriptor(summary_csv, output_dir),
        "report_md": _bundle_artifact_descriptor(report_md, output_dir),
    }
    for name, path in (additional_artifacts or {}).items():
        artifacts[name] = _bundle_artifact_descriptor(path, output_dir, include_payload_hash=True)
    bundle = {
        "schema_version": 1,
        "source_type": source_type,
        "status": "completed",
        "execution_mode": str(plan.get("execution_mode") or ""),
        "evidence_mode": str(plan.get("evidence_mode") or ""),
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "benchmark_dir": str(plan.get("benchmark_dir") or ""),
        "benchmark_content_hash": str(plan.get("benchmark_content_hash") or ""),
        "full_stage_plan_hash": str(plan.get("full_stage_plan_hash") or ""),
        "plan_hash": _hash_payload(plan),
        "manifest_payload_hash": _hash_payload(manifest),
        "run_count": len(run_rows),
        "summary_variant_count": len({str(row.get("variant") or "") for row in run_rows}),
        "seed_provenance": seed_provenance,
        "artifacts": artifacts,
        "run_provenance_artifacts": _run_provenance_artifacts(run_rows, output_dir),
        "claim_boundary": (
            "This bundle binds the ablation plan, execution manifest, current benchmark digest, "
            "run CSV, summary CSV, report, and seed provenance. Mock or incomplete seed provenance "
            "remains workflow-shape evidence only."
        ),
    }
    bundle_path = output_dir / "ablation_evidence_bundle.json"
    atomic_write_text(
        bundle_path,
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False),
    )
    return bundle_path


def _run_provenance_artifacts(
    run_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}
    for row in run_rows:
        experiment_id = str(row.get("experiment_id") or "").strip()
        raw_run_dir = str(row.get("run_dir") or "").strip()
        if not experiment_id or not raw_run_dir:
            continue
        run_dir = Path(raw_run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = output_dir / run_dir
        artifacts: dict[str, Any] = {"run_dir": str(run_dir.resolve())}
        for name in (
            "config.json",
            "run_metadata.json",
            "evaluation_contract.json",
            "trace.jsonl",
            "trace_summary.json",
            "tree.json",
            "checkpoint.json",
            "llm_call_ledger.jsonl",
            "invocation_history.json",
        ):
            path = run_dir / name
            if path.is_file():
                artifacts[name] = _bundle_artifact_descriptor(
                    path,
                    output_dir,
                    include_payload_hash=name.endswith(".json"),
                )
        descriptors[experiment_id] = artifacts
    return descriptors


def _bundle_artifact_descriptor(
    path: Path,
    output_dir: Path,
    *,
    include_payload_hash: bool = False,
) -> dict[str, Any]:
    try:
        relative_path = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative_path = str(path.resolve())
    descriptor: dict[str, Any] = {
        "path": relative_path,
        "sha256": _sha256_file(path),
    }
    if include_payload_hash:
        payload = json.loads(path.read_text(encoding="utf-8"))
        descriptor["payload_hash"] = _hash_payload(payload)
    return descriptor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_provenance_summary(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        field: sorted(
            {
                seed
                for row in run_rows
                if (seed := _csv_seed(row.get(field))) is not None
            }
        )
        for field in SEED_PROVENANCE_FIELDS
    }
    varied_scientific_dimensions = [
        field for field in SCIENTIFIC_RANDOM_SEED_FIELDS if len(values[field]) >= 2
    ]
    return {
        "required_fields": list(SEED_PROVENANCE_FIELDS),
        "scientific_random_fields": list(SCIENTIFIC_RANDOM_SEED_FIELDS),
        "values": values,
        "seed_provenance_complete": bool(run_rows)
        and all(_truthy(row.get("seed_provenance_complete")) for row in run_rows),
        "varied_scientific_dimensions": varied_scientific_dimensions,
        "scientific_random_dimension_varied": bool(varied_scientific_dimensions),
    }


def _csv_seed(value: object) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _champion(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [node for node in nodes if node.get("score")]
    if not scored:
        return nodes[0]
    higher_is_better = bool(scored[0]["score"].get("higher_is_better", False))
    return sorted(
        scored,
        key=lambda node: float(node["score"]["value"]),
        reverse=higher_is_better,
    )[0]


def _score_value(node: dict[str, Any]) -> float | None:
    score = node.get("score")
    return float(score["value"]) if score else None


def _improvement(root: dict[str, Any], champion: dict[str, Any]) -> float | None:
    root_score = _score_value(root)
    champion_score = _score_value(champion)
    if root_score is None or champion_score is None:
        return None
    higher_is_better = bool(root["score"].get("higher_is_better", False))
    return champion_score - root_score if higher_is_better else root_score - champion_score


def _aggregate(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({str(row["variant"]) for row in run_rows})
    summary: list[dict[str, Any]] = []
    for variant in variants:
        rows = [row for row in run_rows if row["variant"] == variant]
        champion_scores = _numbers(rows, "champion_score")
        improvements = _numbers(rows, "champion/root improvement")
        valid_rates = _numbers(rows, "valid_solution_rate")
        higher_is_better = _variant_higher_is_better(rows)
        seed_summary = _seed_provenance_summary(rows)
        summary.append(
            {
                "variant": variant,
                "execution_mode": _joined_unique(row.get("execution_mode", "") for row in rows),
                "evidence_mode": _joined_unique(row.get("evidence_mode", "") for row in rows)
                or EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
                "llm_mode": _joined_unique(row.get("llm_mode", "") for row in rows),
                "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
                "run_evidence_mode": _joined_unique(row.get("run_evidence_mode", "") for row in rows),
                "run_scientific_claim": _joined_unique(row.get("run_scientific_claim", "") for row in rows),
                "benchmark_name": _joined_unique(row.get("benchmark_name", "") for row in rows),
                "benchmark_content_hash": _joined_unique(
                    row.get("benchmark_content_hash", "") for row in rows
                ),
                "plan_hash": _joined_unique(row.get("plan_hash", "") for row in rows),
                "manifest_sha256": _joined_unique(
                    row.get("manifest_sha256", "") for row in rows
                ),
                "search_seed_count": len(seed_summary["values"]["search_seed"]),
                "data_seed_count": len(seed_summary["values"]["data_seed"]),
                "model_seed_count": len(seed_summary["values"]["model_seed"]),
                "provider_seed_count": len(seed_summary["values"]["provider_seed"]),
                "seed_provenance_complete": seed_summary["seed_provenance_complete"],
                "scientific_random_dimensions_varied": ",".join(
                    seed_summary["varied_scientific_dimensions"]
                ),
                "branch_context_enabled": any(_truthy(row.get("branch_context_enabled")) for row in rows),
                "higher_is_better": higher_is_better,
                "runs": len(rows),
                "valid_runs": sum(1 for row in rows if float(row["valid_solution_rate"]) > 0),
                "champion_score_median": _median(champion_scores),
                "champion_score_mean": _mean(champion_scores),
                "champion_score_std": _std(champion_scores),
                "champion_score_best": _best_score(champion_scores, higher_is_better),
                "champion_score_worst": _worst_score(champion_scores, higher_is_better),
                "champion_score_iqr": _iqr(champion_scores),
                "champion/root improvement": _median(improvements),
                "champion/root improvement_median": _median(improvements),
                "champion/root improvement_mean": _mean(improvements),
                "champion/root improvement_std": _std(improvements),
                "champion/root improvement_best": max(improvements) if improvements else "",
                "champion/root improvement_worst": min(improvements) if improvements else "",
                "champion/root improvement_iqr": _iqr(improvements),
                "valid_solution_rate_mean": _mean(valid_rates),
                "timeout_count_total": sum(int(row["timeout_count"]) for row in rows),
                "debug_success_count_total": sum(int(row["debug_success_count"]) for row in rows),
                "branch_context_count_total": sum(int(row.get("branch_context_count", 0)) for row in rows),
                "branch_intents": _joined_unique(row.get("branch_intents", "") for row in rows),
                "llm_calls_total": sum(int(row["llm_calls"]) for row in rows),
                "wall_time_s_total": sum(float(row["wall_time_s"]) for row in rows),
                "example_run_dir": str(rows[0]["run_dir"]),
            }
        )
    return summary


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def _joined_unique(values: Any) -> str:
    items: set[str] = set()
    for value in values:
        if not value:
            continue
        for item in str(value).split(","):
            item = item.strip()
            if item:
                items.add(item)
    return ",".join(sorted(items))


def _higher_is_better(node: dict[str, Any]) -> bool:
    score = node.get("score")
    if not score:
        return False
    return bool(score.get("higher_is_better", False))


def _variant_higher_is_better(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        value = row.get("higher_is_better")
        if _truthy(value):
            return True
        if value in (False, "False", "false", "0", 0):
            return False
    return False


def _truthy(value: Any) -> bool:
    return value in (True, "True", "true", "1", 1)


def _best_score(values: list[float], higher_is_better: bool) -> float | str:
    if not values:
        return ""
    return max(values) if higher_is_better else min(values)


def _worst_score(values: list[float], higher_is_better: bool) -> float | str:
    if not values:
        return ""
    return min(values) if higher_is_better else max(values)


def _median(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def _mean(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _iqr(values: list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    if len(ordered) < 4:
        return 0.0
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _render_report(summary_rows: list[dict[str, Any]]) -> str:
    evidence_mode = _joined_unique(row.get("evidence_mode", "") for row in summary_rows)
    if not evidence_mode:
        evidence_mode = EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE
    boundary = (
        "Mock-mode ablation checks workflow behavior, variant switches, and artifact production. "
        "It cannot prove emergent discovery or paper-score reproduction."
        if evidence_mode == EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE
        else (
            "Real LLM ablation checks controlled workflow contrasts with real provider calls. "
            "It still cannot prove emergent discovery or paper-score reproduction without "
            "paper/readiness gates, failure review, and domain validation."
        )
    )
    lines = [
        "# Ablation Report",
        "",
        f"Evidence mode: `{evidence_mode}`.",
        "",
        boundary,
        "",
        "| Variant | Runs | Valid runs | Champion score median | Champion/root improvement median | Valid solution rate mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {variant} | {runs} | {valid_runs} | {score} | {improvement} | {valid_rate} |".format(
                variant=row["variant"],
                runs=row["runs"],
                valid_runs=row["valid_runs"],
                score=row["champion_score_median"],
                improvement=row["champion/root improvement_median"],
                valid_rate=row["valid_solution_rate_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "- Champion/root improvement is positive when the champion improves over the root under the benchmark metric direction.",
            "- Valid runs count runs with at least one evaluated solution.",
            "- Exact paper improvement factors are out of scope for this MVP.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_real_dry_run_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Real LLM Ablation Dry Run",
        "",
        f"Evidence mode: `{EVIDENCE_MODE_REAL_LLM_ABLATION}`.",
        "",
        "No provider calls were made and no `ablation_runs.csv` was written. This dry-run bundle must not pass `verify-ablation-evidence`.",
        "",
        f"Benchmark: `{plan['benchmark_dir']}`",
        f"Run count: `{len(plan.get('runs', []))}`",
        f"Variants: `{','.join(str(item) for item in plan.get('variants', []))}`",
        f"Seeds: `{','.join(str(item) for item in plan.get('seeds', []))}`",
        "",
        "## Claim Boundary",
        "",
        "- Scientific claim support: `false`.",
        "- Paper score reproduction: `false`.",
        "- Emergent discovery support: `false`.",
        "",
        "## Planned Runs",
        "",
        "| Variant | Seed | Experiment ID |",
        "| --- | ---: | --- |",
    ]
    for entry in plan.get("runs", []):
        lines.append(f"| {entry['variant']} | {entry['seed']} | `{entry['experiment_id']}` |")
    lines.extend(_budget_batch_markdown(plan))
    lines.append("")
    return "\n".join(lines)


def _render_real_failure_report(plan: dict[str, Any], failure_kind: str, exc: Exception) -> str:
    return "\n".join(
        [
            "# Real LLM Ablation Failed",
            "",
            f"Evidence mode: `{EVIDENCE_MODE_REAL_LLM_ABLATION}`.",
            "",
            f"Failure kind: `{failure_kind}`.",
            f"Error type: `{type(exc).__name__}`.",
            "",
            "The run remains blocked. Do not treat this output as multi-seed ablation evidence.",
            "",
            f"Planned run count: `{len(plan.get('runs', []))}`.",
            "",
        ]
    )


def _render_real_budget_blocked_report(plan: dict[str, Any], preflight: dict[str, Any]) -> str:
    blockers = preflight.get("blockers")
    blocker_lines = [
        f"- {item}" for item in blockers if isinstance(item, str)
    ] if isinstance(blockers, list) else []
    if not blocker_lines:
        blocker_lines = ["- budget preflight failed"]
    return "\n".join(
        [
            "# Real LLM Ablation Blocked",
            "",
            f"Evidence mode: `{EVIDENCE_MODE_REAL_LLM_ABLATION}`.",
            "",
            "Failure kind: `blocked_by_budget`.",
            "",
            "No provider calls were made. Increase the explicit call budget, reduce variants/seeds, or run a smaller matrix.",
            "",
            "## Budget Preflight",
            "",
            f"- expected min calls: `{preflight.get('expected_min_llm_calls')}`",
            f"- expected max calls: `{preflight.get('expected_max_llm_calls')}`",
            f"- configured max calls: `{preflight.get('configured_max_llm_calls')}`",
            "",
            "## Blockers",
            "",
            *blocker_lines,
            "",
            *_budget_batch_markdown(plan),
            f"Planned run count: `{len(plan.get('runs', []))}`.",
            "",
        ]
    )


def _budget_batch_markdown(plan: dict[str, Any]) -> list[str]:
    batch_plan = plan.get("budget_batch_plan")
    if not isinstance(batch_plan, dict):
        return []
    batches = batch_plan.get("batches")
    if not isinstance(batches, list) or not batches:
        return []
    lines = [
        "",
        "## Budget Batches",
        "",
        f"- status: `{batch_plan.get('status')}`",
        f"- configured max calls: `{batch_plan.get('configured_max_llm_calls')}`",
        f"- full expected max calls: `{dict(batch_plan.get('total_expected_llm_call_range') or {}).get('max')}`",
        f"- required for execution: `{batch_plan.get('required_for_execution')}`",
        "",
        "| Batch | Runs | Expected max calls | Experiment IDs |",
        "| ---: | ---: | ---: | --- |",
    ]
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        call_range = dict(batch.get("expected_llm_call_range") or {})
        experiment_ids = ",".join(str(item) for item in batch.get("experiment_ids", []))
        lines.append(
            f"| {batch.get('batch_index')} | {batch.get('run_count')} | "
            f"{call_range.get('max')} | `{experiment_ids}` |"
        )
    selected_index = plan.get("selected_budget_batch_index")
    if selected_index is not None:
        lines.extend(["", f"Selected budget batch: `{selected_index}`."])
    lines.append("")
    return lines
