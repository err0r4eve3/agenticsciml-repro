from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
from dataclasses import dataclass
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
from agenticsciml.llm.budget import LLMBudget
from agenticsciml.llm.capabilities import capabilities_for_openai_compatible
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.llm_smoke import _RecordingLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


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
    llm_client: LLMClient | None = None,
) -> AblationResult:
    if mock and dry_run:
        raise ValueError("ablation dry-run is only supported with real LLM mode.")
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
            llm_client=llm_client,
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

    runs_csv = output_dir / "ablation_runs.csv"
    _write_csv(runs_csv, run_rows)
    summary_rows = _aggregate(run_rows)
    summary_csv = output_dir / "ablation_summary.csv"
    _write_csv(summary_csv, summary_rows)
    report_md = output_dir / "ablation_report.md"
    report_md.write_text(_render_report(summary_rows), encoding="utf-8")
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
    plan_path = output_dir / "real_llm_ablation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    manifest = _build_real_ablation_manifest(plan, llm_client=llm_client, budget=budget)
    manifest_path = output_dir / "real_llm_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    report_md = output_dir / "ablation_report.md"
    if dry_run:
        report_md.write_text(_render_real_dry_run_report(plan), encoding="utf-8")
        return AblationResult(
            summary_csv=None,
            report_md=report_md,
            plan_json=plan_path,
            manifest_json=manifest_path,
        )

    try:
        inner_llm = llm_client or OpenAIAdapter(timeout_s=llm_timeout_s, max_retries=llm_max_retries)
    except Exception as exc:
        report_md.write_text(_render_real_failure_report(plan, "adapter_init_error", exc), encoding="utf-8")
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
        report_md.write_text(_render_real_failure_report(plan, "orchestrator_error", exc), encoding="utf-8")
        raise

    runs_csv = output_dir / "ablation_runs.csv"
    _write_csv(runs_csv, run_rows)
    summary_rows = _aggregate(run_rows)
    summary_csv = output_dir / "ablation_summary.csv"
    _write_csv(summary_csv, summary_rows)
    report_md.write_text(_render_report(summary_rows), encoding="utf-8")
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
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE if mock else EVIDENCE_MODE_REAL_LLM_ABLATION,
        "llm_mode": run_llm_mode,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "run_evidence_mode": run_evidence_mode,
        "run_scientific_claim": run_scientific_claim,
        "run_dir": str(run_dir),
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
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "experiment_id": f"{variant}-seed-{seed}",
                    "evolution": config.to_dict(),
                }
            )
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
    return {
        "schema_version": 1,
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "seeds": list(seeds),
        "variants": list(variants),
        "timeout_s": timeout_s,
        "timeout_scope": "generated solution subprocesses; provider HTTP request timeout is adapter-specific",
        "execution_mode": "dry_run" if dry_run else "real",
        "real_mode_explicit": not dry_run,
        "provider_calls_enabled": not dry_run,
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_ABLATION,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "runs": runs,
        "expected_artifacts": [
            "real_llm_ablation_plan.json",
            "real_llm_ablation_manifest.json",
            "ablation_report.md",
            "ablation_runs.csv",
            "ablation_summary.csv",
            *expected_run_artifacts,
        ],
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


def _build_real_ablation_manifest(
    plan: dict[str, Any],
    *,
    llm_client: LLMClient | None,
    budget: LLMBudget,
) -> dict[str, Any]:
    provider_capabilities = _provider_capabilities(llm_client)
    run_count = len(plan.get("runs", []))
    return {
        "schema_version": 1,
        "execution_mode": plan["execution_mode"],
        "real_mode_explicit": plan["real_mode_explicit"],
        "provider": provider_capabilities["provider"],
        "model": getattr(llm_client, "model", None) or os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "adapter_type": getattr(llm_client, "adapter_type", None) or provider_capabilities["adapter_type"],
        "provider_capabilities": provider_capabilities,
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "seeds": plan["seeds"],
        "variants": plan["variants"],
        "run_count": run_count,
        "timeout_s": plan["timeout_s"],
        "timeout_scope": plan["timeout_scope"],
        "output_dir": plan["output_dir"],
        "plan_hash": _hash_payload(plan),
        "config_hash": _hash_payload({"runs": plan["runs"], "benchmark_dir": plan["benchmark_dir"]}),
        "token_budget": budget.to_dict(),
        "expected_llm_call_range": {
            "min": max(1, run_count * 6),
            "max": max(1, run_count * 80),
        },
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
        summary.append(
            {
                "variant": variant,
                "evidence_mode": _joined_unique(row.get("evidence_mode", "") for row in rows)
                or EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
                "llm_mode": _joined_unique(row.get("llm_mode", "") for row in rows),
                "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
                "run_evidence_mode": _joined_unique(row.get("run_evidence_mode", "") for row in rows),
                "run_scientific_claim": _joined_unique(row.get("run_scientific_claim", "") for row in rows),
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
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
