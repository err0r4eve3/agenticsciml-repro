from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.evidence import EVIDENCE_MODE_REAL_LLM_SMOKE, SCIENTIFIC_CLAIM_NOT_SUPPORTED
from agenticsciml.llm.base import LLMClient
from agenticsciml.llm.budget import (
    LLMBudget,
    LLMBudgetPreflightError,
    RecordingLLMClient,
    combine_llm_call_ranges,
    estimate_orchestrator_llm_call_range,
    llm_call_budget_preflight,
    require_llm_call_budget_preflight,
)
from agenticsciml.llm.capabilities import capabilities_for_openai_compatible
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.reporting.trace_summary import summarize_trace


DEFAULT_SMOKE_VARIANTS = ("branch_context", "no_branch_context")


@dataclass(frozen=True, slots=True)
class LLMSmokeResult:
    plan_json: Path
    report_md: Path
    manifest_json: Path
    runs_csv: Path | None = None


@dataclass(frozen=True, slots=True)
class LLMSmokeVerification:
    verification_json: Path
    passed: bool


_RecordingLLMClient = RecordingLLMClient


@contextmanager
def _exclusive_bundle_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir.parent / f".{output_dir.name}.real-llm-smoke.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise RuntimeError(f"Unsafe real LLM smoke bundle lock path: {lock_path}")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Real LLM smoke bundle has an active writer or verifier: {output_dir}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def run_llm_smoke(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    variants: list[str] | None = None,
    seed: int = 0,
    dry_run: bool = True,
    timeout_s: int = 60,
    max_iterations: int = 1,
    parallel_mutations: int = 2,
    llm_fast_mode: bool = False,
    llm_client: LLMClient | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LLMSmokeResult:
    output_dir = Path(output_dir).resolve()
    with _exclusive_bundle_lock(output_dir):
        return _run_llm_smoke_locked(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            variants=variants,
            seed=seed,
            dry_run=dry_run,
            timeout_s=timeout_s,
            max_iterations=max_iterations,
            parallel_mutations=parallel_mutations,
            llm_fast_mode=llm_fast_mode,
            llm_client=llm_client,
            progress_callback=progress_callback,
        )


def _run_llm_smoke_locked(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    variants: list[str] | None = None,
    seed: int = 0,
    dry_run: bool = True,
    timeout_s: int = 60,
    max_iterations: int = 1,
    parallel_mutations: int = 2,
    llm_fast_mode: bool = False,
    llm_client: LLMClient | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LLMSmokeResult:
    benchmark_dir = Path(benchmark_dir).resolve()
    selected_variants = variants or list(DEFAULT_SMOKE_VARIANTS)
    _validate_smoke_variants(selected_variants)
    if not dry_run:
        _require_paired_contrast(selected_variants)
    _validate_smoke_arguments(
        benchmark_dir=benchmark_dir,
        timeout_s=timeout_s,
        max_iterations=max_iterations,
        parallel_mutations=parallel_mutations,
    )
    benchmark_snapshot = _validated_benchmark_snapshot(benchmark_dir)
    backup_dir: Path | None = None
    completed_bundle_exists = _is_completed_smoke_bundle(output_dir)
    if dry_run and completed_bundle_exists:
        raise ValueError(
            "Refusing to overwrite a completed real LLM smoke bundle with a dry run; "
            "choose a different --output-dir"
        )
    if not dry_run and completed_bundle_exists:
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.previous-{uuid.uuid4().hex}"
        )
        output_dir.rename(backup_dir)
    try:
        result = _run_llm_smoke_once(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            selected_variants=selected_variants,
            seed=seed,
            dry_run=dry_run,
            timeout_s=timeout_s,
            max_iterations=max_iterations,
            parallel_mutations=parallel_mutations,
            llm_fast_mode=llm_fast_mode,
            llm_client=llm_client,
            progress_callback=progress_callback,
            benchmark_snapshot=benchmark_snapshot,
        )
    except Exception:
        if backup_dir is not None:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            backup_dir.rename(output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)
    return result


def _run_llm_smoke_once(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    selected_variants: list[str],
    seed: int,
    dry_run: bool,
    timeout_s: int,
    max_iterations: int,
    parallel_mutations: int,
    llm_fast_mode: bool,
    llm_client: LLMClient | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    benchmark_snapshot: dict[str, object],
) -> LLMSmokeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    budget = LLMBudget.from_env()
    plan = _build_plan(
        benchmark_dir,
        output_dir,
        selected_variants,
        seed,
        timeout_s,
        max_iterations,
        parallel_mutations,
        dry_run=dry_run,
        benchmark_snapshot=benchmark_snapshot,
    )
    plan_path = output_dir / "real_llm_smoke_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    manifest = _build_manifest(plan, llm_client=llm_client, budget=budget)
    manifest_path = output_dir / "real_llm_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    if dry_run:
        report_path = output_dir / "real_llm_smoke_report.md"
        report_path.write_text(_render_dry_run_report(plan), encoding="utf-8")
        return LLMSmokeResult(plan_json=plan_path, report_md=report_path, manifest_json=manifest_path)
    preflight = manifest.get("budget_preflight", {})
    if isinstance(preflight, dict) and preflight.get("passed") is not True:
        report_path = output_dir / "real_llm_smoke_report.md"
        report_path.write_text(_render_budget_blocked_report(plan, preflight), encoding="utf-8")
        _finalize_failed_manifest(
            manifest_path,
            manifest,
            report_path,
            budget,
            failure_kind="blocked_by_budget",
            error_type=LLMBudgetPreflightError.__name__,
        )
        require_llm_call_budget_preflight(preflight)

    try:
        llm = llm_client or OpenAIAdapter()
    except Exception as exc:
        report_path = output_dir / "real_llm_smoke_report.md"
        report_path.write_text(_render_failure_report(plan, "adapter_init_error", exc), encoding="utf-8")
        _finalize_failed_manifest(
            manifest_path,
            manifest,
            report_path,
            budget,
            failure_kind="adapter_init_error",
            error_type=type(exc).__name__,
        )
        raise

    rows: list[dict[str, Any]] = []
    gate_issues: list[str] = []
    for entry in plan["runs"]:
        variant = str(entry["variant"])
        expected_run_dir = output_dir / "runs" / str(entry["experiment_id"])
        ledger_path = expected_run_dir / "llm_call_ledger.jsonl"
        scoped_progress_callback = (
            (
                lambda payload, current_variant=variant: progress_callback(
                    {**payload, "variant": current_variant}
                )
            )
            if progress_callback is not None
            else None
        )
        if ledger_path.exists():
            ledger_path.unlink()
        config = ExperimentConfig(
            experiment_id=str(entry["experiment_id"]),
            benchmark_dir=benchmark_dir,
            output_dir=output_dir / "runs",
            evolution=_smoke_variant_config(
                variant,
                seed=seed,
                timeout_s=timeout_s,
                max_iterations=max_iterations,
                parallel_mutations=parallel_mutations,
            ),
            use_mock=False,
            llm_fast_mode=llm_fast_mode,
        )
        try:
            recording_llm = _RecordingLLMClient(
                llm,
                ledger_path,
                budget,
                progress_callback=scoped_progress_callback,
            )
            run_dir = AgenticSciMLOrchestrator(config, recording_llm).run()
        except Exception as exc:
            report_path = output_dir / "real_llm_smoke_report.md"
            report_path.write_text(_render_failure_report(plan, "orchestrator_error", exc), encoding="utf-8")
            _finalize_failed_manifest(
                manifest_path,
                manifest,
                report_path,
                budget,
                failure_kind="orchestrator_error",
                error_type=type(exc).__name__,
            )
            raise
        try:
            row = _smoke_row(run_dir, variant, seed)
        except Exception as exc:
            report_path = output_dir / "real_llm_smoke_report.md"
            report_path.write_text(_render_failure_report(plan, "run_artifact_error", exc), encoding="utf-8")
            _finalize_failed_manifest(
                manifest_path,
                manifest,
                report_path,
                budget,
                failure_kind="run_artifact_error",
                error_type=type(exc).__name__,
            )
            raise
        rows.append(row)
        if not _truthy(row["smoke_gate_passed"]):
            gate_issues.append(f"{variant}: {row['smoke_gate_issues']}")
    paired_gate = _paired_contrast_gate(rows)
    if not paired_gate["passed"]:
        gate_issues.append(f"paired_contrast: {'; '.join(paired_gate['issues'])}")

    runs_csv = output_dir / "real_llm_smoke_runs.csv"
    _write_csv(runs_csv, rows)
    report_path = output_dir / "real_llm_smoke_report.md"
    report_path.write_text(_render_real_report(plan, rows, paired_gate), encoding="utf-8")
    if gate_issues:
        error = RuntimeError(f"Real LLM smoke gate failed; see {report_path}: {'; '.join(gate_issues)}")
        _finalize_failed_manifest(
            manifest_path,
            manifest,
            report_path,
            budget,
            failure_kind="smoke_gate_failed",
            error_type=type(error).__name__,
        )
        raise error
    manifest["status"] = "completed"
    manifest["report_status"] = "passed"
    manifest["report_sha256"] = _file_sha256(report_path)
    manifest["runs_csv_sha256"] = _file_sha256(runs_csv)
    manifest["token_budget_final"] = budget.to_dict()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return LLMSmokeResult(plan_json=plan_path, report_md=report_path, manifest_json=manifest_path, runs_csv=runs_csv)


def verify_llm_smoke_output(output_dir: Path) -> LLMSmokeVerification:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_bundle_lock(output_dir):
        payload = _verify_llm_smoke_output(output_dir)
        path = output_dir / "real_llm_smoke_verification.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    return LLMSmokeVerification(verification_json=path, passed=bool(payload["passed"]))


def _build_plan(
    benchmark_dir: Path,
    output_dir: Path,
    variants: list[str],
    seed: int,
    timeout_s: int,
    max_iterations: int,
    parallel_mutations: int,
    *,
    dry_run: bool,
    benchmark_snapshot: dict[str, object],
) -> dict[str, Any]:
    runs = [
        {
            "variant": variant,
            "experiment_id": f"smoke-{variant}-seed-{seed}",
            "max_iterations": max_iterations,
            "parallel_mutations": parallel_mutations,
            "use_branch_context": variant != "no_branch_context",
            "expected_llm_call_range": estimate_orchestrator_llm_call_range(
                max_iterations=max_iterations,
                parallel_mutations=parallel_mutations,
            ),
        }
        for variant in variants
    ]
    expected_llm_call_range = combine_llm_call_ranges(
        [
            dict(entry["expected_llm_call_range"])
            for entry in runs
            if isinstance(entry.get("expected_llm_call_range"), dict)
        ]
    )
    expected_artifacts = [
        artifact
        for entry in runs
        for artifact in [
            f"runs/{entry['experiment_id']}/tree.json",
            f"runs/{entry['experiment_id']}/trace_summary.json",
            f"runs/{entry['experiment_id']}/run_metadata.json",
            f"runs/{entry['experiment_id']}/solutions/<solution_id>/proposal.md",
            f"runs/{entry['experiment_id']}/solutions/<solution_id>/engineering_summary.md",
        ]
    ]
    return {
        "schema_version": 1,
        "bundle_id": uuid.uuid4().hex,
        "benchmark_dir": str(benchmark_dir),
        "benchmark": benchmark_snapshot,
        "output_dir": str(output_dir),
        "seed": seed,
        "timeout_s": timeout_s,
        "timeout_scope": "generated solution subprocesses; provider HTTP request timeout is adapter-specific",
        "execution_mode": "dry_run" if dry_run else "real",
        "real_mode_explicit": not dry_run,
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_SMOKE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "agent_call_sequence": [
            "data_analyst",
            "evaluator",
            "root_engineer",
            "selector",
            "retriever",
            "proposer",
            "critic",
            "engineer",
            "debugger",
            "result_analyst",
        ],
        "expected_artifacts": [
            "real_llm_smoke_plan.json",
            "real_llm_smoke_report.md",
            *expected_artifacts,
        ],
        "runs": runs,
        "expected_llm_call_range": expected_llm_call_range,
        "claim_boundary": (
            "This smoke can provide prompt-delivery and behavioral-difference evidence only. "
            "It is not a paper-scale scientific reproduction."
        ),
    }


def _smoke_variant_config(
    variant: str,
    *,
    seed: int,
    timeout_s: int,
    max_iterations: int,
    parallel_mutations: int,
) -> EvolutionConfig:
    _validate_smoke_variants([variant])
    return EvolutionConfig(
        max_iterations=max_iterations,
        parallel_mutations=parallel_mutations,
        max_debug_retries=1,
        timeout_s=timeout_s,
        random_seed=seed,
        use_kb=True,
        use_branch_context=variant != "no_branch_context",
    )


def _smoke_row(run_dir: Path, variant: str, seed: int) -> dict[str, Any]:
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    stored_trace_summary = json.loads(
        (run_dir / "trace_summary.json").read_text(encoding="utf-8")
    )
    trace_summary = summarize_trace(run_dir)
    trace_summary_issues = []
    if stored_trace_summary != trace_summary:
        trace_summary_issues.append(
            f"{variant}: stored trace_summary.json is stale or does not match run evidence"
        )
    nodes = tree["nodes"]
    score_diagnostics = _score_diagnostics(nodes)
    branch_tags = sorted(
        {
            tag
            for node in nodes
            for tag in node.get("method_tags", [])
            if isinstance(tag, str) and tag.startswith("branch:")
        }
    )
    proposal_titles = _proposal_titles(run_dir, nodes)
    llm_calls_total = _llm_call_count(metadata, variant, [])
    ledger_entries, ledger_issues = _read_llm_call_ledger(run_dir, variant)
    ledger_issues.extend(trace_summary_issues)
    ledger_call_count = len(ledger_entries)
    generation_spans = _generation_span_metadata(run_dir)
    ledger_issues.extend(
        _llm_token_accounting_issues(
            ledger_entries,
            generation_spans,
            variant,
        )
    )
    generation_span_count = len(generation_spans)
    gate = _smoke_gate(
        run_dir,
        variant,
        metadata,
        trace_summary,
        nodes,
        branch_tags,
        ledger_issues=ledger_issues,
        ledger_call_count=ledger_call_count,
        generation_span_count=generation_span_count,
    )
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_SMOKE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "branch_context_enabled": bool(metadata.get("branch_context_enabled", False)),
        "solution_count": len(nodes),
        **score_diagnostics,
        "branch_intents": ",".join(tag.replace("branch:", "", 1) for tag in branch_tags),
        "proposal_titles": " | ".join(proposal_titles),
        "llm_calls": llm_calls_total if llm_calls_total is not None else 0,
        "llm_ledger_calls": ledger_call_count,
        "llm_ledger_call_ids": _join_sequence(entry.get("call_id") for entry in ledger_entries),
        "llm_ledger_providers": ",".join(sorted({str(entry.get("provider", "")) for entry in ledger_entries})),
        "llm_ledger_models": ",".join(sorted({str(entry.get("model", "")) for entry in ledger_entries})),
        "llm_ledger_methods": _join_sequence(entry.get("method") for entry in ledger_entries),
        "llm_ledger_schema_names": _join_sequence(entry.get("schema_name") for entry in ledger_entries),
        "llm_trace_call_ids": _join_sequence(event.get("llm_call_id") for event in generation_spans),
        "llm_trace_providers": ",".join(sorted({str(event.get("provider", "")) for event in generation_spans})),
        "llm_trace_models": ",".join(sorted({str(event.get("model", "")) for event in generation_spans})),
        "llm_trace_methods": _join_sequence(event.get("method") for event in generation_spans),
        "llm_trace_schema_names": _join_sequence(event.get("schema_name") for event in generation_spans),
        "llm_ledger_call_fingerprints": _join_sequence(sorted(_call_fingerprints(ledger_entries, id_key="call_id"))),
        "llm_trace_call_fingerprints": _join_sequence(sorted(_call_fingerprints(generation_spans, id_key="llm_call_id"))),
        "generation_span_count": generation_span_count,
        "trace_quality_gate_passed": bool(trace_summary.get("quality_gate", {}).get("passed", False)),
        "smoke_gate_passed": gate["passed"],
        "smoke_gate_issues": "; ".join(gate["issues"]),
    }


def _score_diagnostics(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_nodes = [node for node in nodes if node.get("status") == "evaluated"]
    failed_nodes = [node for node in nodes if node.get("status") == "failed"]
    failed_solution_kinds = ",".join(
        sorted({str(node.get("failure_kind") or "unknown") for node in failed_nodes})
    )
    debug_attempted_nodes = [
        node for node in nodes if int(node.get("num_debug_attempts", 0)) > 0
    ]
    debug_attempt_count = sum(int(node.get("num_debug_attempts", 0)) for node in nodes)
    debug_recovered_node_count = sum(
        1 for node in debug_attempted_nodes if node.get("status") == "evaluated"
    )
    debug_failed_node_count = sum(
        1 for node in debug_attempted_nodes if node.get("status") == "failed"
    )
    debug_recovery_rate: float | str = (
        debug_recovered_node_count / len(debug_attempted_nodes)
        if debug_attempted_nodes
        else ""
    )
    scored_nodes = [node for node in evaluated_nodes if isinstance(node.get("score"), dict)]
    root = next((node for node in nodes if node.get("parent_id") is None), None)
    root_score_data = root.get("score") if isinstance(root, dict) else None
    score_reference = root_score_data if isinstance(root_score_data, dict) else (
        scored_nodes[0]["score"] if scored_nodes else None
    )
    metric = score_reference.get("metric", "") if isinstance(score_reference, dict) else ""
    higher_is_better = (
        bool(score_reference.get("higher_is_better", False))
        if isinstance(score_reference, dict)
        else ""
    )
    root_score = (
        float(root_score_data["value"])
        if isinstance(root_score_data, dict) and "value" in root_score_data
        else ""
    )
    scored_children = [node for node in scored_nodes if node.get("parent_id") is not None]
    best_child = (
        sorted(
            scored_children,
            key=lambda node: float(node["score"]["value"]),
            reverse=higher_is_better is True,
        )[0]
        if scored_children
        else None
    )
    best_child_score = float(best_child["score"]["value"]) if best_child is not None else ""
    if isinstance(root_score, float) and isinstance(best_child_score, float):
        best_child_improvement = (
            best_child_score - root_score if higher_is_better is True else root_score - best_child_score
        )
        mutation_improved: bool | str = best_child_improvement > 0
    else:
        best_child_improvement = ""
        mutation_improved = ""
    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "evaluated_solution_count": len(evaluated_nodes),
        "failed_solution_count": len(failed_nodes),
        "failed_solution_kinds": failed_solution_kinds,
        "debug_attempted_node_count": len(debug_attempted_nodes),
        "debug_attempt_count": debug_attempt_count,
        "debug_recovered_node_count": debug_recovered_node_count,
        "debug_failed_node_count": debug_failed_node_count,
        "debug_recovery_rate": debug_recovery_rate,
        "root_score": root_score,
        "best_child_score": best_child_score,
        "best_child_improvement_vs_root": best_child_improvement,
        "mutation_improved": mutation_improved,
    }


def _proposal_titles(run_dir: Path, nodes: list[dict[str, Any]]) -> list[str]:
    titles: list[str] = []
    for node in nodes:
        if node.get("parent_id") is None:
            continue
        proposal_path = run_dir / "solutions" / str(node["node_id"]) / "proposal.md"
        if not proposal_path.exists():
            continue
        for line in proposal_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                titles.append(line.removeprefix("# ").strip())
                break
    return titles


def _render_dry_run_report(plan: dict[str, Any]) -> str:
    run_lines = "\n".join(
        f"- `{entry['variant']}`: branch_context={entry['use_branch_context']}, "
        f"parallel_mutations={entry['parallel_mutations']}"
        for entry in plan["runs"]
    )
    return (
        "# Real LLM Smoke Dry Run\n\n"
        f"- Evidence mode: `{plan['evidence_mode']}`\n"
        f"- Scientific claim: `{plan['scientific_claim']}`\n"
        f"- Execution mode: `{plan['execution_mode']}`\n"
        "- API calls: none; dry run only.\n\n"
        "## Manifest\n\n"
        "- `real_llm_smoke_manifest.json`\n\n"
        "## Planned Runs\n\n"
        f"{run_lines}\n\n"
        "## Boundary\n\n"
        f"{plan['claim_boundary']}\n"
    )


def _finalize_failed_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    report_path: Path,
    budget: LLMBudget,
    *,
    failure_kind: str,
    error_type: str,
) -> None:
    manifest.update(
        {
            "status": "failed",
            "report_status": "failed",
            "failure_kind": failure_kind,
            "error_type": error_type,
            "report_sha256": _file_sha256(report_path),
            "token_budget_final": budget.to_dict(),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _render_real_report(plan: dict[str, Any], rows: list[dict[str, Any]], paired_gate: dict[str, Any]) -> str:
    row_lines = "\n".join(
        f"- `{row['variant']}`: branch_context={row['branch_context_enabled']}, "
        f"solutions={row['solution_count']}, intents={row['branch_intents'] or 'none'}, "
        f"score={row['metric'] or 'unknown'}, "
        f"higher_is_better={row['higher_is_better'] if row['higher_is_better'] != '' else 'unknown'}, "
        f"root={row['root_score'] if row['root_score'] != '' else 'unknown'}, "
        f"best_child={row['best_child_score'] if row['best_child_score'] != '' else 'unknown'}, "
        "improvement_vs_root="
        f"{row['best_child_improvement_vs_root'] if row['best_child_improvement_vs_root'] != '' else 'unknown'}, "
        f"mutation_improved={row['mutation_improved'] if row['mutation_improved'] != '' else 'unknown'}, "
        f"evaluated={row['evaluated_solution_count']}, failed={row['failed_solution_count']}, "
        f"failure_kinds={row['failed_solution_kinds'] or 'none'}, "
        f"debug_nodes={row['debug_attempted_node_count']}, "
        f"debug_attempts={row['debug_attempt_count']}, "
        f"debug_recovered={row['debug_recovered_node_count']}, "
        f"debug_failed={row['debug_failed_node_count']}, "
        f"debug_recovery_rate={row['debug_recovery_rate'] if row['debug_recovery_rate'] != '' else 'not_applicable'}, "
        f"trace_gate={row['trace_quality_gate_passed']}, smoke_gate={row['smoke_gate_passed']}, "
        f"gate_issues={row['smoke_gate_issues'] or 'none'}"
        for row in rows
    )
    return (
        "# Real LLM Smoke Report\n\n"
        f"- Evidence mode: `{plan['evidence_mode']}`\n"
        f"- Scientific claim: `{plan['scientific_claim']}`\n\n"
        "## Manifest\n\n"
        "- `real_llm_smoke_manifest.json`\n\n"
        "## Runs\n\n"
        f"{row_lines}\n\n"
        "## Paired Contrast Gate\n\n"
        f"- paired_contrast_passed: `{paired_gate['passed']}`\n"
        f"- issues: `{'; '.join(paired_gate['issues']) or 'none'}`\n\n"
        "## Performance Boundary\n\n"
        "- performance_comparison_supported: `false`\n"
        "- Root solutions are independently generated for each variant; cross-variant scores "
        "do not isolate the effect of branch context.\n"
        "- Mutation score diagnostics are within-run evidence only and do not change the smoke gate.\n\n"
        "## Boundary\n\n"
        f"{plan['claim_boundary']}\n"
    )


def _render_failure_report(plan: dict[str, Any], failure_kind: str, exc: Exception) -> str:
    return (
        "# Real LLM Smoke Failed\n\n"
        f"- Evidence mode: `{plan['evidence_mode']}`\n"
        f"- Scientific claim: `{plan['scientific_claim']}`\n"
        f"- failure_kind: `{failure_kind}`\n"
        f"- error_type: `{type(exc).__name__}`\n"
        f"- error: `{exc}`\n\n"
        "## Manifest\n\n"
        "- `real_llm_smoke_manifest.json`\n\n"
        "## Boundary\n\n"
        f"{plan['claim_boundary']}\n"
    )


def _render_budget_blocked_report(plan: dict[str, Any], preflight: dict[str, Any]) -> str:
    blockers = preflight.get("blockers")
    blocker_lines = [
        f"- {item}" for item in blockers if isinstance(item, str)
    ] if isinstance(blockers, list) else []
    if not blocker_lines:
        blocker_lines = ["- budget preflight failed"]
    return "\n".join(
        [
            "# Real LLM Smoke Blocked",
            "",
            f"- Evidence mode: `{plan['evidence_mode']}`",
            f"- Scientific claim: `{plan['scientific_claim']}`",
            "- failure_kind: `blocked_by_budget`",
            "",
            "No provider calls were made. Reduce the smoke matrix or increase the explicit call budget.",
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
            "## Manifest",
            "",
            "- `real_llm_smoke_manifest.json`",
            "",
            "## Boundary",
            "",
            str(plan["claim_boundary"]),
            "",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _verify_llm_smoke_output(output_dir: Path) -> dict[str, Any]:
    runs_root = output_dir / "runs"
    if runs_root.is_symlink():
        return _failed_smoke_verification_payload(
            output_dir,
            ["smoke bundle runs directory must not be a symlink"],
        )
    run_dirs = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: str(path.resolve()),
    ) if runs_root.is_dir() else []
    lock_handles: list[Any] = []
    lock_issues: list[str] = []
    try:
        for run_dir in run_dirs:
            lock_path = run_dir / ".invocation.lock"
            if run_dir.is_symlink() or lock_path.is_symlink():
                lock_issues.append(f"{run_dir.name}: unsafe invocation lock path")
                break
            try:
                handle = lock_path.open("a+", encoding="utf-8")
            except OSError as exc:
                lock_issues.append(
                    f"{run_dir.name}: could not open invocation lock: {type(exc).__name__}"
                )
                break
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                lock_issues.append(f"{run_dir.name}: run has an active invocation")
                break
            lock_handles.append(handle)
        if lock_issues:
            return _failed_smoke_verification_payload(output_dir, lock_issues)
        bundle_snapshot_before = _bundle_evidence_snapshot(output_dir)
        snapshotted_run_dirs = {
            output_dir / key
            for key, value in bundle_snapshot_before.items()
            if key.startswith("runs/") and value == ("directory", "")
        }
        if set(run_dirs) != snapshotted_run_dirs:
            return _failed_smoke_verification_payload(
                output_dir,
                ["smoke bundle run directory set changed before verification"],
            )
        snapshots_before = {
            run_dir: _run_evidence_snapshot(run_dir)
            for run_dir in run_dirs
        }
        payload = _verify_llm_smoke_output_locked(output_dir)
        verified_benchmark_snapshot = payload.get("verified_benchmark_snapshot")
        if isinstance(verified_benchmark_snapshot, dict):
            benchmark_path = Path(str(verified_benchmark_snapshot.get("path", "")))
            try:
                benchmark_snapshot_after = _validated_benchmark_snapshot(benchmark_path)
            except Exception as exc:
                payload["issues"].append(
                    "benchmark evidence could not be revalidated after verification: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if benchmark_snapshot_after != verified_benchmark_snapshot:
                    payload["issues"].append(
                        "benchmark evidence changed during verification"
                    )
        if bundle_snapshot_before != _bundle_evidence_snapshot(output_dir):
            payload["issues"].append("smoke bundle evidence changed during verification")
        for run_dir in run_dirs:
            if snapshots_before[run_dir] != _run_evidence_snapshot(run_dir):
                payload["issues"].append(
                    f"{run_dir.name}: run evidence changed during verification"
                )
        payload["verified_snapshot_sha256"] = _hash_payload(
            {
                "bundle": bundle_snapshot_before,
                "benchmark": verified_benchmark_snapshot,
                "runs": {
                    str(run_dir.relative_to(output_dir)): snapshot
                    for run_dir, snapshot in snapshots_before.items()
                },
            }
        )
        payload["passed"] = not payload["issues"]
        return payload
    finally:
        for handle in reversed(lock_handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _verify_llm_smoke_output_locked(output_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    verified_benchmark_snapshot: dict[str, object] | None = None
    plan = _read_json_or_issue(output_dir / "real_llm_smoke_plan.json", issues)
    manifest = _read_json_or_issue(output_dir / "real_llm_smoke_manifest.json", issues)
    runs_csv = output_dir / "real_llm_smoke_runs.csv"
    report_path = output_dir / "real_llm_smoke_report.md"
    if not report_path.exists():
        issues.append("missing real_llm_smoke_report.md")
    if not runs_csv.exists():
        issues.append("missing real_llm_smoke_runs.csv; dry-run outputs are not real-smoke evidence")

    rows = _read_rows(runs_csv, issues) if runs_csv.exists() else []
    if runs_csv.exists() and not rows:
        issues.append("real_llm_smoke_runs.csv must contain paired run rows")
    manifest_call_range: tuple[int, int] | None = None
    if manifest:
        manifest_issues, manifest_call_range = _manifest_schema_issues(manifest, output_dir)
        issues.extend(manifest_issues)
        if report_path.exists() and manifest.get("report_sha256") != _file_sha256(report_path):
            issues.append("manifest report_sha256 does not match real_llm_smoke_report.md")
        if runs_csv.exists() and manifest.get("runs_csv_sha256") != _file_sha256(runs_csv):
            issues.append("manifest runs_csv_sha256 does not match real_llm_smoke_runs.csv")
    if plan:
        if plan.get("schema_version") != 1:
            issues.append("plan schema_version must be 1")
        expected_hash = _hash_payload(plan)
        if manifest and manifest.get("plan_hash") != expected_hash:
            issues.append("manifest plan_hash does not match plan payload")
        if manifest and manifest.get("config_hash") != expected_hash:
            issues.append("manifest config_hash does not match plan payload")
        if manifest and manifest.get("bundle_id") != plan.get("bundle_id"):
            issues.append("manifest bundle_id does not match plan bundle_id")
        if plan.get("execution_mode") != "real":
            issues.append("plan execution_mode must be real for smoke verification")
        if Path(str(plan.get("output_dir", ""))).resolve() != output_dir.resolve():
            issues.append("plan output_dir does not match verification bundle")
        plan_benchmark = plan.get("benchmark")
        if not isinstance(plan_benchmark, dict):
            issues.append("plan benchmark must be an object")
        else:
            benchmark_path = Path(str(plan_benchmark.get("path", ""))).resolve()
            if Path(str(plan.get("benchmark_dir", ""))).resolve() != benchmark_path:
                issues.append("plan benchmark_dir does not match benchmark.path")
            try:
                current_benchmark = _validated_benchmark_snapshot(benchmark_path)
            except Exception as exc:
                issues.append(
                    "could not validate planned benchmark: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if current_benchmark != plan_benchmark:
                    issues.append("planned benchmark snapshot is stale or does not match current benchmark")
                else:
                    verified_benchmark_snapshot = current_benchmark
        variants = [str(entry.get("variant")) for entry in plan.get("runs", []) if isinstance(entry, dict)]
        try:
            _require_paired_contrast(variants)
        except ValueError as exc:
            issues.append(str(exc))
        row_variants = [str(row.get("variant", "")) for row in rows]
        if row_variants:
            try:
                _require_paired_contrast(row_variants)
            except ValueError as exc:
                issues.append(f"runs CSV variants invalid: {exc}")

    plan_runs = {
        str(entry.get("variant")): entry
        for entry in plan.get("runs", [])
        if isinstance(entry, dict)
    }
    expected_run_dirs = _planned_run_directories(plan, output_dir, issues)
    recomputed_rows: list[dict[str, Any]] = []
    for row in rows:
        variant = str(row.get("variant", ""))
        run_dir = Path(str(row.get("run_dir", "")))
        seed = _parse_strict_int(row.get("seed", 0), f"{variant}: row seed", issues)
        if seed is None:
            continue
        plan_entry = plan_runs.get(variant)
        if plan_entry is None:
            issues.append(f"{variant}: row variant is not present in plan")
            continue
        plan_seed = _parse_strict_int(plan.get("seed", -1), "plan seed", issues)
        if plan_seed is not None and seed != plan_seed:
            issues.append(f"{variant}: row seed {seed} does not match plan seed {plan.get('seed')}")
        expected_run_dir = expected_run_dirs.get(variant)
        if expected_run_dir is None:
            continue
        if run_dir.resolve() != expected_run_dir:
            issues.append(f"{variant}: run_dir {run_dir} does not match expected {expected_run_dir}")
            continue
        if not run_dir.exists():
            issues.append(f"{variant}: run_dir does not exist: {run_dir}")
            continue
        try:
            recomputed = _smoke_row(run_dir, variant, seed)
        except Exception as exc:
            issues.append(f"{variant}: could not recompute smoke row: {type(exc).__name__}: {exc}")
            continue
        recomputed_rows.append(recomputed)
        issues.extend(
            _run_binding_issues(
                run_dir,
                variant=variant,
                seed=seed,
                plan_entry=plan_entry,
                plan_benchmark=plan.get("benchmark"),
            )
        )
        if not _csv_row_matches_recomputed(row, recomputed):
            issues.append(f"{variant}: runs CSV row is stale or does not match recomputed run evidence")
        if not _truthy(recomputed["smoke_gate_passed"]):
            issues.append(f"{variant}: smoke gate failed: {recomputed['smoke_gate_issues']}")
        issues.extend(_parallel_trace_issues(run_dir, variant, plan))

    paired_gate = _paired_contrast_gate(recomputed_rows)
    if not paired_gate["passed"]:
        issues.append(f"paired contrast gate failed: {'; '.join(paired_gate['issues'])}")
    if manifest and recomputed_rows:
        call_count = 0
        prompt_tokens = 0
        output_tokens = 0
        for row in recomputed_rows:
            row_variant = str(row.get("variant", "unknown"))
            row_call_count = _parse_strict_int(row.get("llm_calls"), f"{row_variant}: recomputed llm_calls", issues)
            if row_call_count is not None:
                call_count += row_call_count
            _verify_row_ledger(row, manifest, issues)
            ledger_entries, _ = _read_llm_call_ledger(Path(str(row.get("run_dir", ""))), row_variant)
            for entry in ledger_entries:
                prompt_field = (
                    "prompt_tokens_accounted"
                    if "prompt_tokens_accounted" in entry
                    else "prompt_token_estimate"
                )
                prompt_value = _parse_strict_int(
                    entry.get(prompt_field),
                    f"{row_variant}: ledger {prompt_field}",
                    issues,
                )
                output_value = _parse_strict_int(
                    entry.get("response_token_estimate"),
                    f"{row_variant}: ledger response_token_estimate",
                    issues,
                )
                if prompt_value is not None:
                    prompt_tokens += prompt_value
                if output_value is not None:
                    output_tokens += output_value
        if call_count <= 0:
            issues.append("recomputed LLM call count must be positive")
        initial_budget = manifest.get("token_budget")
        final_budget = manifest.get("token_budget_final")
        if isinstance(initial_budget, dict) and isinstance(final_budget, dict):
            for field in (
                "max_prompt_tokens",
                "max_output_tokens",
                "max_total_tokens",
                "max_calls",
                "max_cost_usd",
                "cost_per_1k_tokens_usd",
            ):
                if final_budget.get(field) != initial_budget.get(field):
                    issues.append(f"manifest token_budget_final.{field} does not match token_budget")
            final_calls = _parse_strict_int(
                final_budget.get("calls_used"),
                "manifest token_budget_final.calls_used",
                issues,
            )
            if final_calls is not None and final_calls != call_count:
                issues.append(
                    "manifest token_budget_final.calls_used does not match recomputed LLM call count"
                )
            final_prompt = _parse_strict_int(
                final_budget.get("prompt_tokens_used"),
                "manifest token_budget_final.prompt_tokens_used",
                issues,
            )
            final_output = _parse_strict_int(
                final_budget.get("output_tokens_used"),
                "manifest token_budget_final.output_tokens_used",
                issues,
            )
            if final_prompt is not None and final_prompt != prompt_tokens:
                issues.append("manifest token_budget_final.prompt_tokens_used does not match LLM ledger")
            if final_output is not None and final_output != output_tokens:
                issues.append("manifest token_budget_final.output_tokens_used does not match LLM ledger")
            cost_rate = (
                _parse_finite_number(
                    final_budget.get("cost_per_1k_tokens_usd"),
                    "manifest token_budget_final.cost_per_1k_tokens_usd",
                    issues,
                )
                if final_budget.get("cost_per_1k_tokens_usd") is not None
                else None
            )
            final_cost = _parse_finite_number(
                final_budget.get("estimated_cost_usd"),
                "manifest token_budget_final.estimated_cost_usd",
                issues,
            )
            expected_cost = (prompt_tokens + output_tokens) / 1000.0 * cost_rate if cost_rate else 0.0
            if final_cost is not None and not math.isclose(
                final_cost,
                expected_cost,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                issues.append("manifest token_budget_final.estimated_cost_usd does not match LLM ledger")
            for field, used in (
                ("max_calls", call_count),
                ("max_prompt_tokens", prompt_tokens),
                ("max_output_tokens", output_tokens),
                ("max_total_tokens", prompt_tokens + output_tokens),
            ):
                limit_value = final_budget.get(field)
                limit = (
                    _parse_strict_int(
                        limit_value,
                        f"manifest token_budget_final.{field}",
                        issues,
                    )
                    if limit_value is not None
                    else None
                )
                if limit is not None and used > limit:
                    issues.append(f"manifest token_budget_final exceeds {field}")
            max_cost = (
                _parse_finite_number(
                    final_budget.get("max_cost_usd"),
                    "manifest token_budget_final.max_cost_usd",
                    issues,
                )
                if final_budget.get("max_cost_usd") is not None
                else None
            )
            if max_cost is not None and expected_cost > max_cost:
                issues.append("manifest token_budget_final exceeds max_cost_usd")
        if manifest_call_range is not None:
            min_calls, max_calls = manifest_call_range
        if manifest_call_range is not None and not (min_calls <= call_count <= max_calls):
            issues.append(
                f"recomputed LLM call count {call_count} is outside expected range [{min_calls}, {max_calls}]"
            )

    return {
        "schema_version": 1,
        "passed": not issues,
        "issues": issues,
        "checked_artifacts": {
            "plan": str(output_dir / "real_llm_smoke_plan.json"),
            "manifest": str(output_dir / "real_llm_smoke_manifest.json"),
            "report": str(report_path),
            "runs_csv": str(runs_csv),
        },
        "verified_benchmark_snapshot": verified_benchmark_snapshot,
        "recomputed_rows": recomputed_rows,
    }


def _planned_run_directories(
    plan: dict[str, Any],
    output_dir: Path,
    issues: list[str],
) -> dict[str, Path]:
    entries = plan.get("runs")
    if not isinstance(entries, list):
        issues.append("plan runs must be an array")
        return {}
    expected_by_variant: dict[str, Path] = {}
    experiment_ids: list[str] = []
    valid_entry_count = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"plan runs[{index}] must be an object")
            continue
        valid_entry_count += 1
        variant = entry.get("variant")
        experiment_id = entry.get("experiment_id")
        if not isinstance(variant, str) or not variant:
            issues.append(f"plan runs[{index}].variant must be a non-empty string")
            continue
        if (
            not isinstance(experiment_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id) is None
            or experiment_id in {".", ".."}
        ):
            issues.append(
                f"plan runs[{index}].experiment_id must be a safe single path segment"
            )
            continue
        if experiment_id in experiment_ids:
            issues.append(f"plan experiment_id must be unique: {experiment_id}")
            continue
        if variant in expected_by_variant:
            issues.append(f"plan variant must be unique: {variant}")
            continue
        experiment_ids.append(experiment_id)
        expected_by_variant[variant] = (output_dir / "runs" / experiment_id).resolve()

    actual_members: dict[str, str] = {}
    runs_root = output_dir / "runs"
    if runs_root.is_dir():
        for path in runs_root.iterdir():
            actual_members[path.name] = "directory" if path.is_dir() else "non-directory"
    if len(expected_by_variant) == valid_entry_count:
        expected_names = set(experiment_ids)
        actual_names = set(actual_members)
        if actual_names != expected_names:
            issues.append(
                "smoke bundle run directory set does not match plan: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        wrong_types = sorted(
            name
            for name in expected_names & actual_names
            if actual_members[name] != "directory"
        )
        if wrong_types:
            issues.append(
                "smoke bundle planned run members must be directories: "
                + ", ".join(wrong_types)
            )
    return expected_by_variant


def _failed_smoke_verification_payload(
    output_dir: Path,
    issues: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "passed": False,
        "issues": list(issues),
        "checked_artifacts": {
            "plan": str(output_dir / "real_llm_smoke_plan.json"),
            "manifest": str(output_dir / "real_llm_smoke_manifest.json"),
            "report": str(output_dir / "real_llm_smoke_report.md"),
            "runs_csv": str(output_dir / "real_llm_smoke_runs.csv"),
        },
        "recomputed_rows": [],
    }


def _run_evidence_snapshot(run_dir: Path) -> dict[str, tuple[str, str]]:
    snapshot: dict[str, tuple[str, str]] = {}
    for path in sorted(run_dir.rglob("*"), key=lambda item: str(item.relative_to(run_dir))):
        relative = str(path.relative_to(run_dir))
        if relative == ".invocation.lock":
            continue
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[relative] = ("file", _file_sha256(path))
        elif path.is_dir():
            snapshot[relative] = ("directory", "")
        else:
            snapshot[relative] = ("other", "")
    return snapshot


def _bundle_evidence_snapshot(output_dir: Path) -> dict[str, tuple[str, str]]:
    snapshot: dict[str, tuple[str, str]] = {}
    for name in (
        "real_llm_smoke_plan.json",
        "real_llm_smoke_manifest.json",
        "real_llm_smoke_runs.csv",
        "real_llm_smoke_report.md",
    ):
        path = output_dir / name
        if path.is_symlink():
            snapshot[name] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[name] = ("file", _file_sha256(path))
        elif path.exists():
            snapshot[name] = ("other", "")
        else:
            snapshot[name] = ("missing", "")
    runs_dir = output_dir / "runs"
    if runs_dir.is_symlink():
        snapshot["runs"] = ("symlink", os.readlink(runs_dir))
    elif runs_dir.is_dir():
        snapshot["runs"] = ("directory", "")
        for path in sorted(runs_dir.iterdir(), key=lambda item: item.name):
            key = f"runs/{path.name}"
            if path.is_symlink():
                snapshot[key] = ("symlink", os.readlink(path))
            elif path.is_dir():
                snapshot[key] = ("directory", "")
            elif path.is_file():
                snapshot[key] = ("file", _file_sha256(path))
            else:
                snapshot[key] = ("other", "")
    elif runs_dir.exists():
        snapshot["runs"] = ("other", "")
    else:
        snapshot["runs"] = ("missing", "")
    return snapshot


def _read_json_or_issue(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing {path.name}")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"invalid {path.name}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(data, dict):
        issues.append(f"{path.name} must contain a JSON object")
        return {}
    return data


def _read_rows(path: Path, issues: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        issues.append(f"could not read {path.name}: {type(exc).__name__}: {exc}")
        return []


def _csv_row_matches_recomputed(row: dict[str, str], recomputed: dict[str, Any]) -> bool:
    if set(row) != set(recomputed):
        return False
    order_insensitive_trace_fields = {
        "llm_ledger_call_ids",
        "llm_ledger_methods",
        "llm_ledger_schema_names",
        "llm_trace_call_ids",
        "llm_trace_methods",
        "llm_trace_schema_names",
    }
    return all(
        str(row[key]) == str(recomputed[key])
        for key in recomputed
        if key not in order_insensitive_trace_fields
    )


def _run_binding_issues(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
    plan_entry: dict[str, Any],
    plan_benchmark: Any,
) -> list[str]:
    issues: list[str] = []
    config = _read_json_or_issue(run_dir / "config.json", issues)
    metadata = _read_json_or_issue(run_dir / "run_metadata.json", issues)
    contract = _read_json_or_issue(run_dir / "evaluation_contract.json", issues)
    if not isinstance(plan_benchmark, dict):
        return issues
    expected_benchmark_path = Path(str(plan_benchmark.get("path", ""))).resolve()
    if config:
        if config.get("experiment_id") != plan_entry.get("experiment_id"):
            issues.append(f"{variant}: run config experiment_id does not match plan")
        if Path(str(config.get("benchmark_dir", ""))).resolve() != expected_benchmark_path:
            issues.append(f"{variant}: run config benchmark_dir does not match plan benchmark")
        if config.get("use_mock") is not False:
            issues.append(f"{variant}: run config use_mock must be false")
        evolution = config.get("evolution")
        if not isinstance(evolution, dict):
            issues.append(f"{variant}: run config evolution must be an object")
        else:
            expected_evolution = {
                "max_iterations": plan_entry.get("max_iterations"),
                "parallel_mutations": plan_entry.get("parallel_mutations"),
                "random_seed": seed,
                "use_branch_context": plan_entry.get("use_branch_context"),
            }
            for field, expected in expected_evolution.items():
                if evolution.get(field) != expected:
                    issues.append(f"{variant}: run config evolution.{field} does not match plan")
    if metadata:
        if metadata.get("run_state") not in {"completed", "exported", "finalized"}:
            issues.append(f"{variant}: run_metadata run_state is not exported")
        if metadata.get("benchmark_name") != plan_benchmark.get("name"):
            issues.append(f"{variant}: run_metadata benchmark_name does not match plan")
        if metadata.get("benchmark_fidelity_level") != plan_benchmark.get("fidelity_level"):
            issues.append(f"{variant}: run_metadata benchmark fidelity does not match plan")
        if metadata.get("llm_mode") != "real":
            issues.append(f"{variant}: run_metadata llm_mode must be real")
    if contract:
        if contract.get("benchmark_name") != plan_benchmark.get("name"):
            issues.append(f"{variant}: evaluation contract benchmark_name does not match plan")
        if contract.get("contract_hash") != plan_benchmark.get("contract_hash"):
            issues.append(f"{variant}: evaluation contract hash does not match planned benchmark")
        if (
            contract.get("benchmark_source_manifest_digest")
            != plan_benchmark.get("benchmark_source_manifest_digest")
        ):
            issues.append(f"{variant}: evaluation contract source manifest does not match plan")
    return issues


def _read_llm_call_ledger(run_dir: Path, variant: str) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    path = run_dir / "llm_call_ledger.jsonl"
    if not path.exists():
        return [], [f"{variant}: missing llm_call_ledger.jsonl"]
    entries: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{variant}: invalid ledger JSON on line {line_number}: {exc.msg}")
            continue
        if not isinstance(entry, dict):
            issues.append(f"{variant}: ledger line {line_number} must be a JSON object")
            continue
        issues.extend(_validate_llm_call_ledger_entry(entry, seen_call_ids, line_number, variant))
        entries.append(entry)
    if not entries:
        issues.append(f"{variant}: llm_call_ledger.jsonl must contain at least one call")
    else:
        issues.extend(_ledger_call_id_sequence_issues(entries, variant))
    return entries, issues


def _validate_llm_call_ledger_entry(
    entry: dict[str, Any],
    seen_call_ids: set[str],
    line_number: int,
    variant: str,
) -> list[str]:
    issues: list[str] = []
    prefix = f"{variant}: ledger line {line_number}"
    success = entry.get("success")
    allowed_base = {
        "schema_version",
        "call_id",
        "provider",
        "model",
        "adapter_type",
        "provider_capabilities",
        "method",
        "schema_name",
        "span_kind",
        "prompt_hash",
        "system_hash",
        "prompt_token_estimate",
        "prompt_tokens_accounted",
        "prompt_token_source",
        "response_token_estimate",
        "response_token_source",
        "temperature",
        "reasoning_effort",
        "started_at_unix",
        "success",
        "duration_s",
    }
    allowed_keys = allowed_base | (
        {"response_hash"}
        if success is True
        else {"error_type", "underlying_error_type"}
        if success is False
        else {"response_hash", "error_type", "underlying_error_type"}
    )
    extra_keys = sorted(set(entry) - allowed_keys)
    if extra_keys:
        issues.append(f"{prefix} contains unknown ledger field(s): {', '.join(extra_keys)}")
    forbidden = {
        "prompt",
        "system",
        "response",
        "raw_prompt",
        "raw_response",
        "raw_messages",
        "messages",
        "request_payload",
        "completion_text",
    }
    leaked = sorted(forbidden & set(entry))
    if leaked:
        issues.append(f"{prefix} contains forbidden raw field(s): {', '.join(leaked)}")
    if entry.get("schema_version") != 1:
        issues.append(f"{prefix} schema_version must be 1")
    call_id = entry.get("call_id")
    if not isinstance(call_id, str) or re.fullmatch(r"llm_call_\d{6}", call_id) is None:
        issues.append(f"{prefix} call_id must match llm_call_000001 format")
    if isinstance(call_id, str):
        if call_id in seen_call_ids:
            issues.append(f"{prefix} call_id must be unique")
        seen_call_ids.add(call_id)
    _required_non_empty_string(entry.get("provider"), f"{prefix} provider", issues)
    _required_non_empty_string(entry.get("model"), f"{prefix} model", issues)
    _required_non_empty_string(entry.get("adapter_type"), f"{prefix} adapter_type", issues)
    _validate_provider_capabilities(entry.get("provider_capabilities"), f"{prefix} provider_capabilities", issues)
    method = _required_non_empty_string(entry.get("method"), f"{prefix} method", issues)
    if method and method not in {
        "complete_text",
        "complete_json",
        "complete_json_with_images",
    }:
        issues.append(
            f"{prefix} method must be complete_text, complete_json, "
            "or complete_json_with_images"
        )
    if entry.get("span_kind") != "generation_span":
        issues.append(f"{prefix} span_kind must be generation_span")
    schema_name = entry.get("schema_name")
    if method == "complete_text" and schema_name is not None:
        issues.append(f"{prefix} schema_name must be null for complete_text")
    if method in {"complete_json", "complete_json_with_images"}:
        _required_non_empty_string(schema_name, f"{prefix} schema_name", issues)
    for field in ("prompt_hash", "system_hash"):
        _validate_sha256_hex(entry.get(field), f"{prefix} {field}", issues)
    _validate_non_negative_finite_number(entry.get("prompt_token_estimate"), f"{prefix} prompt_token_estimate", issues)
    accounting_fields = {
        "prompt_tokens_accounted",
        "prompt_token_source",
        "response_token_source",
    }
    has_accounting_fields = bool(accounting_fields & set(entry))
    if has_accounting_fields and "prompt_tokens_accounted" not in entry:
        issues.append(f"{prefix} prompt_tokens_accounted is required with token sources")
    if has_accounting_fields and "prompt_token_source" not in entry:
        issues.append(f"{prefix} prompt_token_source is required with token accounting")
    if (
        has_accounting_fields
        and "response_token_estimate" in entry
        and "response_token_source" not in entry
    ):
        issues.append(f"{prefix} response_token_source is required with token accounting")
    if "response_token_source" in entry and "response_token_estimate" not in entry:
        issues.append(f"{prefix} response_token_estimate is required with response_token_source")
    if "prompt_tokens_accounted" in entry:
        _validate_non_negative_finite_number(
            entry.get("prompt_tokens_accounted"),
            f"{prefix} prompt_tokens_accounted",
            issues,
        )
    for field in ("prompt_token_source", "response_token_source"):
        if field in entry and entry.get(field) not in {
            "local_estimate",
            "provider_usage",
        }:
            issues.append(
                f"{prefix} {field} must be local_estimate or provider_usage"
            )
    if (
        entry.get("prompt_token_source") == "local_estimate"
        and entry.get("prompt_tokens_accounted") != entry.get("prompt_token_estimate")
    ):
        issues.append(
            f"{prefix} local_estimate prompt accounting must equal prompt_token_estimate"
        )
    if not isinstance(success, bool):
        issues.append(f"{prefix} success must be boolean")
    elif success:
        _validate_sha256_hex(entry.get("response_hash"), f"{prefix} response_hash", issues)
        _validate_non_negative_finite_number(entry.get("response_token_estimate"), f"{prefix} response_token_estimate", issues)
    else:
        issues.append(f"{prefix} success must be true for completed smoke evidence")
        _required_non_empty_string(entry.get("error_type"), f"{prefix} error_type", issues)
        if "underlying_error_type" in entry:
            _required_non_empty_string(
                entry.get("underlying_error_type"),
                f"{prefix} underlying_error_type",
                issues,
            )
    _validate_finite_number(entry.get("temperature"), f"{prefix} temperature", issues)
    reasoning_effort = entry.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }:
        issues.append(f"{prefix} reasoning_effort is invalid")
    _validate_non_negative_finite_number(entry.get("duration_s"), f"{prefix} duration_s", issues)
    _validate_positive_finite_number(entry.get("started_at_unix"), f"{prefix} started_at_unix", issues)
    return issues


def _ledger_call_id_sequence_issues(entries: list[dict[str, Any]], variant: str) -> list[str]:
    numbers: list[int] = []
    for entry in entries:
        call_id = entry.get("call_id")
        if not isinstance(call_id, str):
            continue
        match = re.fullmatch(r"llm_call_(\d{6})", call_id)
        if match:
            numbers.append(int(match.group(1)))
    expected = list(range(1, len(entries) + 1))
    if sorted(numbers) != expected:
        return [f"{variant}: ledger call_id sequence must be contiguous from llm_call_000001"]
    return []


def _validate_sha256_hex(value: Any, label: str, issues: list[str]) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        issues.append(f"{label} must be a lowercase sha256 hex string")


def _validate_non_negative_finite_number(value: Any, label: str, issues: list[str]) -> None:
    number = _parse_finite_number(value, label, issues)
    if number is not None and number < 0:
        issues.append(f"{label} must be >= 0")


def _validate_finite_number(value: Any, label: str, issues: list[str]) -> None:
    _parse_finite_number(value, label, issues)


def _validate_positive_finite_number(value: Any, label: str, issues: list[str]) -> None:
    number = _parse_finite_number(value, label, issues)
    if number is not None and number <= 0:
        issues.append(f"{label} must be > 0")


def _parse_finite_number(value: Any, label: str, issues: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(f"{label} must be a finite number")
        return None
    number = float(value)
    if not math.isfinite(number):
        issues.append(f"{label} must be a finite number")
        return None
    return number


def _generation_span_metadata(run_dir: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for event in _read_trace_events(run_dir):
        if event.get("event_type") != "generation_span":
            continue
        metadata = event.get("metadata", {})
        spans.append(metadata if isinstance(metadata, dict) else {})
    return spans


def _llm_token_accounting_issues(
    ledger_entries: list[dict[str, Any]],
    generation_spans: list[dict[str, Any]],
    variant: str,
) -> list[str]:
    issues: list[str] = []
    spans_by_call_id = {
        span.get("llm_call_id"): span
        for span in generation_spans
        if isinstance(span.get("llm_call_id"), str)
    }
    for entry in ledger_entries:
        call_id = entry.get("call_id")
        span = spans_by_call_id.get(call_id)
        usage = span.get("usage") if isinstance(span, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        if not {
            "prompt_tokens_accounted",
            "prompt_token_source",
            "response_token_source",
        } & set(entry):
            # Legacy rows are accepted only with legacy traces. A provider
            # usage trace proves that current accounting fields should exist;
            # silently downgrading that row would permit budget undercounting.
            if any(
                isinstance(usage.get(field), int)
                and not isinstance(usage.get(field), bool)
                and usage[field] >= 0
                for field in ("prompt_tokens", "completion_tokens")
            ):
                issues.append(
                    f"{variant}: ledger {call_id} is missing token accounting "
                    "fields despite provider usage trace"
                )
            continue
        usage_values = [
            usage.get(field)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        ]
        if all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in usage_values
        ) and usage_values[2] != usage_values[0] + usage_values[1]:
            issues.append(
                f"{variant}: trace {call_id} usage.total_tokens does not equal "
                "prompt_tokens + completion_tokens"
            )
        for source_field, ledger_field, usage_field in (
            (
                "prompt_token_source",
                "prompt_tokens_accounted",
                "prompt_tokens",
            ),
            (
                "response_token_source",
                "response_token_estimate",
                "completion_tokens",
            ),
        ):
            source = entry.get(source_field)
            provider_value = usage.get(usage_field)
            has_provider_value = (
                isinstance(provider_value, int)
                and not isinstance(provider_value, bool)
                and provider_value >= 0
            )
            if source == "provider_usage":
                if not has_provider_value:
                    issues.append(
                        f"{variant}: ledger {call_id} {source_field} requires "
                        f"trace usage.{usage_field}"
                    )
                elif entry.get(ledger_field) != provider_value:
                    issues.append(
                        f"{variant}: ledger {call_id} {ledger_field} does not "
                        f"match trace usage.{usage_field}"
                    )
            elif source == "local_estimate" and has_provider_value:
                issues.append(
                    f"{variant}: ledger {call_id} {source_field} uses local_estimate "
                    f"despite trace usage.{usage_field}"
                )
    return issues


def _join_sequence(values: Any) -> str:
    return ",".join("__none__" if value is None else str(value) for value in values)


def _split_sequence(value: Any) -> list[str]:
    if not value:
        return []
    return [item for item in str(value).split(",")]


def _call_fingerprints(entries: list[dict[str, Any]], *, id_key: str) -> list[str]:
    fingerprints: list[str] = []
    for entry in entries:
        call_id = entry.get(id_key)
        fingerprints.append(
            "|".join(
                "__none__" if value is None else str(value)
                for value in (
                    call_id,
                    entry.get("provider"),
                    entry.get("model"),
                    entry.get("method"),
                    entry.get("schema_name"),
                )
            )
        )
    return fingerprints


def _verify_row_ledger(row: dict[str, Any], manifest: dict[str, Any], issues: list[str]) -> None:
    variant = str(row.get("variant", "unknown"))
    ledger_calls = _parse_strict_int(row.get("llm_ledger_calls"), f"{variant}: llm_ledger_calls", issues)
    llm_calls = _parse_strict_int(row.get("llm_calls"), f"{variant}: row llm_calls", issues)
    generation_spans = _parse_strict_int(row.get("generation_span_count"), f"{variant}: generation_span_count", issues)
    if ledger_calls is not None and llm_calls is not None and ledger_calls != llm_calls:
        issues.append(f"{variant}: ledger call count {ledger_calls} does not match row llm_calls {llm_calls}")
    if ledger_calls is not None and generation_spans is not None and ledger_calls != generation_spans:
        issues.append(
            f"{variant}: ledger call count {ledger_calls} does not match generation span count {generation_spans}"
        )
    ledger_call_ids = _split_sequence(row.get("llm_ledger_call_ids"))
    trace_call_ids = _split_sequence(row.get("llm_trace_call_ids"))
    if len(trace_call_ids) != len(set(trace_call_ids)):
        issues.append(f"{variant}: trace llm_call_id values must be unique")
    if "__none__" in trace_call_ids:
        issues.append(f"{variant}: every trace generation span must include llm_call_id")
    if sorted(ledger_call_ids) != sorted(trace_call_ids):
        issues.append(f"{variant}: ledger call_id set does not match trace llm_call_id set")
    ledger_fingerprints = _split_sequence(row.get("llm_ledger_call_fingerprints"))
    trace_fingerprints = _split_sequence(row.get("llm_trace_call_fingerprints"))
    if ledger_fingerprints != trace_fingerprints:
        issues.append(f"{variant}: ledger call fingerprints do not match trace call fingerprints")
    expected_provider = manifest.get("provider")
    expected_model = manifest.get("model")
    providers = {item for item in str(row.get("llm_ledger_providers", "")).split(",") if item}
    models = {item for item in str(row.get("llm_ledger_models", "")).split(",") if item}
    trace_providers = {item for item in str(row.get("llm_trace_providers", "")).split(",") if item}
    trace_models = {item for item in str(row.get("llm_trace_models", "")).split(",") if item}
    if expected_provider and providers != {expected_provider}:
        issues.append(
            f"{variant}: ledger providers {sorted(providers)} do not match manifest provider {expected_provider}"
        )
    if expected_model and models != {expected_model}:
        issues.append(f"{variant}: ledger models {sorted(models)} do not match manifest model {expected_model}")
    if providers != trace_providers:
        issues.append(f"{variant}: ledger providers {sorted(providers)} do not match trace providers {sorted(trace_providers)}")
    if models != trace_models:
        issues.append(f"{variant}: ledger models {sorted(models)} do not match trace models {sorted(trace_models)}")


def _manifest_schema_issues(manifest: dict[str, Any], output_dir: Path) -> tuple[list[str], tuple[int, int] | None]:
    issues: list[str] = []
    call_range_bounds: tuple[int, int] | None = None
    if manifest.get("schema_version") != 1:
        issues.append("manifest schema_version must be 1")
    if manifest.get("execution_mode") != "real":
        issues.append("manifest execution_mode must be real for smoke verification")
    if Path(str(manifest.get("output_dir", ""))).resolve() != output_dir.resolve():
        issues.append("manifest output_dir does not match verification bundle")
    if manifest.get("real_mode_explicit") is not True:
        issues.append("manifest real_mode_explicit must be true for smoke verification")
    _required_non_empty_string(manifest.get("bundle_id"), "manifest bundle_id", issues)
    if manifest.get("status") != "completed":
        issues.append("manifest status must be completed for smoke verification")
    if manifest.get("report_status") != "passed":
        issues.append("manifest report_status must be passed for smoke verification")
    if manifest.get("status") == "completed":
        if "failure_kind" in manifest or "error_type" in manifest:
            issues.append("completed manifest must not contain failure fields")
        preflight = manifest.get("budget_preflight")
        if (
            not isinstance(preflight, dict)
            or preflight.get("passed") is not True
            or preflight.get("status") != "ready"
        ):
            issues.append("completed manifest budget_preflight must be ready and passed")
    for field in ("report_sha256", "runs_csv_sha256"):
        _validate_sha256_hex(manifest.get(field), f"manifest {field}", issues)
    _required_non_empty_string(manifest.get("provider"), "manifest provider", issues)
    _required_non_empty_string(manifest.get("model"), "manifest model", issues)
    _required_non_empty_string(manifest.get("adapter_type"), "manifest adapter_type", issues)
    _validate_provider_capabilities(manifest.get("provider_capabilities"), "manifest provider_capabilities", issues)
    _validate_budget_schema(manifest.get("token_budget"), "manifest token_budget", issues)
    _validate_budget_schema(manifest.get("token_budget_final"), "manifest token_budget_final", issues)

    call_range = manifest.get("expected_llm_call_range", {})
    if not isinstance(call_range, dict):
        issues.append("manifest expected_llm_call_range must be an object")
        return issues, None

    if "min" not in call_range:
        issues.append("manifest expected_llm_call_range.min is required")
    if "max" not in call_range:
        issues.append("manifest expected_llm_call_range.max is required")
    min_calls = _parse_strict_int(call_range.get("min"), "manifest expected_llm_call_range.min", issues)
    max_calls = _parse_strict_int(call_range.get("max"), "manifest expected_llm_call_range.max", issues)
    if min_calls is not None and min_calls < 1:
        issues.append("manifest expected_llm_call_range.min must be >= 1")
    if max_calls is not None and max_calls < 1:
        issues.append("manifest expected_llm_call_range.max must be >= 1")
    if min_calls is not None and max_calls is not None and max_calls < min_calls:
        issues.append("manifest expected_llm_call_range.max must be >= min")
    if min_calls is not None and max_calls is not None and min_calls >= 1 and max_calls >= min_calls:
        call_range_bounds = (min_calls, max_calls)
    return issues, call_range_bounds


def _parallel_trace_issues(run_dir: Path, variant: str, plan: dict[str, Any]) -> list[str]:
    parallel_mutations = 1
    issues: list[str] = []
    for entry in plan.get("runs", []):
        if isinstance(entry, dict) and entry.get("variant") == variant:
            parsed_parallel_mutations = _parse_strict_int(
                entry.get("parallel_mutations", 1),
                f"{variant}: plan parallel_mutations",
                issues,
            )
            if parsed_parallel_mutations is None:
                return issues
            parallel_mutations = parsed_parallel_mutations
            break
    if parallel_mutations <= 1:
        return []
    starts = [
        event
        for event in _read_trace_events(run_dir)
        if event.get("name") == "agenticsciml.parallel_children.start"
    ]
    if not starts:
        return [f"{variant}: missing parallel_children.start trace"]
    has_parallel_workers = False
    for event in starts:
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict) or metadata.get("execution_mode") != "parallel":
            continue
        max_workers = _parse_strict_int(metadata.get("max_workers", 0), f"{variant}: trace max_workers", issues)
        if max_workers is not None and max_workers >= 2:
            has_parallel_workers = True
    if not has_parallel_workers:
        issues.append(f"{variant}: no parallel child trace with max_workers >= 2")
    return issues


def _parse_strict_int(value: Any, label: str, issues: list[str]) -> int | None:
    if isinstance(value, bool):
        issues.append(f"{label} must be an integer, not bool")
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    if value is None:
        issues.append(f"{label} must be an integer")
        return None
    if isinstance(value, float):
        issues.append(f"{label} must be an integer, not float")
        return None
    if isinstance(value, str):
        issues.append(f"{label} must be an integer string")
        return None
    issues.append(f"{label} must be an integer")
    return None


def _required_non_empty_string(value: Any, label: str, issues: list[str]) -> str | None:
    if not isinstance(value, str):
        issues.append(f"{label} must be a non-empty string")
        return None
    if not value.strip():
        issues.append(f"{label} must be a non-empty string")
        return None
    return value


def _llm_call_count(metadata: dict[str, Any], variant: str, issues: list[str]) -> int | None:
    llm_calls = metadata.get("llm_calls")
    if not isinstance(llm_calls, dict):
        issues.append(f"{variant}: llm_calls must be an object")
        return None
    total = _parse_strict_int(llm_calls.get("total"), f"{variant}: llm_calls.total", issues)
    if total is not None and total <= 0:
        issues.append(f"{variant}: llm_calls.total must be positive in real smoke")
    return total


def _validate_smoke_arguments(
    *,
    benchmark_dir: Path,
    timeout_s: int,
    max_iterations: int,
    parallel_mutations: int,
) -> None:
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")
    if parallel_mutations < 1:
        raise ValueError("parallel_mutations must be >= 1")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if not benchmark_dir.is_dir():
        raise ValueError(f"Benchmark directory does not exist: {benchmark_dir}")


def _validated_benchmark_snapshot(benchmark_dir: Path) -> dict[str, object]:
    bundle = ProblemBundle.load(benchmark_dir)
    contract = BenchmarkContractFactory.create_contract(bundle)
    return {
        "name": bundle.benchmark_name,
        "path": str(benchmark_dir.resolve()),
        "family": bundle.benchmark_spec.family,
        "fidelity_level": bundle.benchmark_spec.fidelity_level,
        "metric": bundle.benchmark_spec.metric,
        "contract_hash": contract.contract_hash,
        "benchmark_source_manifest_digest": contract.benchmark_source_manifest_digest,
    }


def _is_completed_smoke_bundle(output_dir: Path) -> bool:
    manifest_path = output_dir / "real_llm_smoke_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("status") == "completed"
        and manifest.get("report_status") == "passed"
        and (output_dir / "real_llm_smoke_plan.json").exists()
        and (output_dir / "real_llm_smoke_report.md").exists()
        and (output_dir / "real_llm_smoke_runs.csv").exists()
    )


def _validate_smoke_variants(variants: list[str]) -> None:
    allowed = set(DEFAULT_SMOKE_VARIANTS)
    unknown = sorted({variant for variant in variants if variant not in allowed})
    if unknown:
        valid = ", ".join(DEFAULT_SMOKE_VARIANTS)
        raise ValueError(f"Unknown LLM smoke variant(s): {', '.join(unknown)}. Valid variants: {valid}")


def _require_paired_contrast(variants: list[str]) -> None:
    required = set(DEFAULT_SMOKE_VARIANTS)
    present = set(variants)
    if present != required or len(variants) != len(DEFAULT_SMOKE_VARIANTS):
        missing = ", ".join(sorted(required - present))
        extra = ", ".join(sorted(present - required))
        duplicate = len(variants) != len(set(variants))
        details = []
        if missing:
            details.append(f"missing: {missing}")
        if extra:
            details.append(f"extra: {extra}")
        if duplicate:
            details.append("duplicates are not allowed")
        raise ValueError(
            "Real LLM smoke requires the exact paired variants "
            f"{', '.join(DEFAULT_SMOKE_VARIANTS)}. " + "; ".join(details)
        )


def _smoke_gate(
    run_dir: Path,
    variant: str,
    metadata: dict[str, Any],
    trace_summary: dict[str, Any],
    nodes: list[dict[str, Any]],
    branch_tags: set[str],
    *,
    ledger_issues: list[str] | None = None,
    ledger_call_count: int | None = None,
    generation_span_count: int | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    issues.extend(ledger_issues or [])
    expected_branch_context = variant != "no_branch_context"
    if bool(metadata.get("branch_context_enabled", False)) is not expected_branch_context:
        issues.append("branch_context_enabled does not match variant")
    if not trace_summary.get("quality_gate", {}).get("passed", False):
        issues.append("trace_summary quality gate failed")
    llm_calls_total = _llm_call_count(metadata, variant, issues)
    if ledger_call_count is not None and llm_calls_total is not None and llm_calls_total != ledger_call_count:
        issues.append(
            f"{variant}: llm_calls.total {llm_calls_total} does not match ledger count {ledger_call_count}"
        )
    if generation_span_count is not None and ledger_call_count is not None and generation_span_count != ledger_call_count:
        issues.append(
            f"{variant}: generation span count {generation_span_count} does not match ledger count {ledger_call_count}"
        )

    child_events = [
        event
        for event in _read_trace_events(run_dir)
        if event.get("name") == "agenticsciml.child_mutation.start"
    ]
    if expected_branch_context:
        if not branch_tags:
            issues.append("branch_context variant produced no branch method tags")
        if not child_events:
            issues.append("branch_context variant produced no child mutation trace events")
        if not any(event.get("metadata", {}).get("branch_context", {}).get("branch_intent") for event in child_events):
            issues.append("branch_context trace has no branch_intent evidence")
        issues.extend(_branch_prompt_delivery_issues(run_dir, nodes))
    else:
        if branch_tags:
            issues.append("no_branch_context variant produced branch method tags")
        if any(event.get("metadata", {}).get("branch_context") for event in child_events):
            issues.append("no_branch_context trace contains branch context")
        forbidden = ("branch_intent", "sibling_branch_ids", "diversity_instruction")
        transcript_text = _child_transcript_prompts(run_dir, nodes)
        leaked = [token for token in forbidden if token in transcript_text]
        if leaked:
            issues.append(f"no_branch_context transcripts leaked tokens: {', '.join(leaked)}")
    return {"passed": not issues, "issues": issues}


def _branch_prompt_delivery_issues(run_dir: Path, nodes: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for node in nodes:
        if node.get("parent_id") is None:
            continue
        node_id = str(node["node_id"])
        context_path = run_dir / "solutions" / node_id / "branch_context.json"
        if not context_path.exists():
            issues.append(f"{node_id} missing branch_context.json")
            continue
        context = json.loads(context_path.read_text(encoding="utf-8"))
        prompt_text = _node_request_prompts(run_dir, node_id)
        missing: list[str] = []
        branch_intent = context.get("branch_intent")
        if not isinstance(branch_intent, str) or branch_intent not in prompt_text:
            missing.append("branch_intent")
        siblings = context.get("sibling_branch_ids")
        if not isinstance(siblings, list) or not all(str(sibling) in prompt_text for sibling in siblings):
            missing.append("sibling_branch_ids")
        diversity_instruction = context.get("diversity_instruction")
        if not isinstance(diversity_instruction, str) or diversity_instruction not in prompt_text:
            missing.append("diversity_instruction")
        if missing:
            issues.append(f"{node_id} request prompts missing branch context fields: {', '.join(missing)}")
    return issues


def _paired_contrast_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    by_variant = {str(row["variant"]): row for row in rows}
    for required in DEFAULT_SMOKE_VARIANTS:
        if required not in by_variant:
            issues.append(f"missing required variant {required}")
    if issues:
        return {"passed": False, "issues": issues}

    branch = by_variant["branch_context"]
    no_branch = by_variant["no_branch_context"]
    if not _truthy(branch.get("smoke_gate_passed")):
        issues.append("branch_context row gate failed")
    if not _truthy(no_branch.get("smoke_gate_passed")):
        issues.append("no_branch_context row gate failed")
    if not _truthy(branch.get("branch_context_enabled")):
        issues.append("branch_context row did not enable branch context")
    if _truthy(no_branch.get("branch_context_enabled")):
        issues.append("no_branch_context row enabled branch context")
    branch_calls = _parse_strict_int(branch.get("llm_calls", 0), "branch_context row llm_calls", issues)
    no_branch_calls = _parse_strict_int(no_branch.get("llm_calls", 0), "no_branch_context row llm_calls", issues)
    if (
        branch_calls is None
        or no_branch_calls is None
        or branch_calls <= 0
        or no_branch_calls <= 0
    ):
        issues.append("both paired variants must record positive LLM call counts")
    if not str(branch.get("branch_intents", "")):
        issues.append("branch_context row has no branch intents")
    if str(no_branch.get("branch_intents", "")):
        issues.append("no_branch_context row has branch intents")
    return {"passed": not issues, "issues": issues}


def _read_trace_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def _child_transcript_prompts(run_dir: Path, nodes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for node in nodes:
        if node.get("parent_id") is None:
            continue
        parts.append(_node_request_prompts(run_dir, str(node["node_id"])))
    return "\n".join(parts)


def _node_request_prompts(run_dir: Path, node_id: str) -> str:
    transcript_dir = run_dir / "solutions" / node_id / "transcripts"
    prompts: list[str] = []
    for name in ("proposal_debate.json", "engineer.json"):
        path = transcript_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict) and item.get("role") in {"proposer", "engineer"}:
                prompt = item.get("prompt")
                if isinstance(prompt, str):
                    prompts.append(prompt)
    return "\n".join(prompts)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _build_manifest(plan: dict[str, Any], *, llm_client: LLMClient | None, budget: LLMBudget) -> dict[str, Any]:
    plan_hash = _hash_payload(plan)
    model = getattr(llm_client, "model", None) or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    provider = _llm_provider_name(llm_client) if llm_client is not None else _default_provider_capabilities().provider
    adapter_type = getattr(llm_client, "adapter_type", None) or (
        type(llm_client).__name__ if llm_client is not None else _default_provider_capabilities().adapter_type
    )
    provider_capabilities = _llm_provider_capabilities(llm_client) if llm_client is not None else _default_provider_capabilities().to_dict()
    expected_llm_call_range = dict(plan.get("expected_llm_call_range") or {})
    budget_preflight = llm_call_budget_preflight(
        budget=budget,
        expected_llm_call_range=expected_llm_call_range,
    )
    return {
        "schema_version": 1,
        "bundle_id": plan["bundle_id"],
        "status": "dry_run" if plan["execution_mode"] == "dry_run" else "running",
        "report_status": "not_executed" if plan["execution_mode"] == "dry_run" else "pending",
        "execution_mode": plan["execution_mode"],
        "real_mode_explicit": plan["real_mode_explicit"],
        "provider": provider,
        "model": model,
        "adapter_type": adapter_type,
        "provider_capabilities": provider_capabilities,
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "seed": plan["seed"],
        "run_count": len(plan["runs"]),
        "max_iterations": [entry["max_iterations"] for entry in plan["runs"]],
        "parallel_mutations": [entry["parallel_mutations"] for entry in plan["runs"]],
        "timeout_s": plan["timeout_s"],
        "timeout_scope": plan["timeout_scope"],
        "output_dir": plan["output_dir"],
        "plan_hash": plan_hash,
        "config_hash": plan_hash,
        "token_budget": budget.to_dict(),
        "expected_llm_call_range": expected_llm_call_range,
        "budget_preflight": budget_preflight,
        "claim_boundary": plan["claim_boundary"],
    }


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_provider_capabilities() -> Any:
    return capabilities_for_openai_compatible(os.environ.get("OPENAI_BASE_URL"))


def _llm_provider_name(llm_client: LLMClient | None) -> str:
    if llm_client is None:
        return _default_provider_capabilities().provider
    provider_name = getattr(llm_client, "provider_name", None)
    if isinstance(provider_name, str) and provider_name:
        return provider_name
    return type(llm_client).__name__


def _llm_provider_capabilities(llm_client: LLMClient | None) -> dict[str, object]:
    capabilities = getattr(llm_client, "provider_capabilities", None)
    if hasattr(capabilities, "to_dict"):
        return capabilities.to_dict()
    if isinstance(capabilities, dict):
        return capabilities
    provider = _llm_provider_name(llm_client)
    return {
        "provider": provider,
        "adapter_type": getattr(llm_client, "adapter_type", type(llm_client).__name__ if llm_client is not None else "unknown"),
        "supports_responses": False,
        "supports_structured_outputs": False,
        "supports_image_inputs": False,
        "supports_usage": False,
        "supports_trace_export": False,
        "supports_prompt_cache": False,
    }


def _validate_provider_capabilities(value: Any, label: str, issues: list[str]) -> None:
    required = {
        "provider": str,
        "adapter_type": str,
        "supports_responses": bool,
        "supports_structured_outputs": bool,
        "supports_image_inputs": bool,
        "supports_usage": bool,
        "supports_trace_export": bool,
        "supports_prompt_cache": bool,
    }
    if not isinstance(value, dict):
        issues.append(f"{label} must be an object")
        return
    unknown = sorted(set(value) - set(required))
    if unknown:
        issues.append(f"{label} contains unknown field(s): {', '.join(unknown)}")
    for field, expected_type in required.items():
        item = value.get(field)
        if expected_type is str:
            _required_non_empty_string(item, f"{label}.{field}", issues)
        elif not isinstance(item, bool):
            issues.append(f"{label}.{field} must be boolean")


def _validate_budget_schema(value: Any, label: str, issues: list[str]) -> None:
    required = {
        "max_prompt_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "max_calls",
        "max_cost_usd",
        "cost_per_1k_tokens_usd",
        "calls_used",
        "prompt_tokens_used",
        "output_tokens_used",
        "estimated_cost_usd",
    }
    if not isinstance(value, dict):
        issues.append(f"{label} must be an object")
        return
    unknown = sorted(set(value) - required)
    if unknown:
        issues.append(f"{label} contains unknown field(s): {', '.join(unknown)}")
    for field in required:
        if field not in value:
            issues.append(f"{label}.{field} is required")
    for field in ("max_prompt_tokens", "max_output_tokens", "max_total_tokens", "max_calls"):
        item = value.get(field)
        if item is not None:
            parsed = _parse_strict_int(item, f"{label}.{field}", issues)
            if parsed is not None and parsed <= 0:
                issues.append(f"{label}.{field} must be positive when configured")
    for field in ("max_cost_usd", "cost_per_1k_tokens_usd", "estimated_cost_usd"):
        item = value.get(field)
        if item is not None:
            _validate_non_negative_finite_number(item, f"{label}.{field}", issues)
    for field in ("calls_used", "prompt_tokens_used", "output_tokens_used"):
        parsed = _parse_strict_int(value.get(field), f"{label}.{field}", issues)
        if parsed is not None and parsed < 0:
            issues.append(f"{label}.{field} must be non-negative")


def _package_versions() -> dict[str, str]:
    names = ["agenticsciml-repro", "numpy", "openai"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions
