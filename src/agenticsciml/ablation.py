from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


DEFAULT_VARIANTS = ("root_only", "no_kb", "kb", "random_kb", "no_critic", "no_debugger")


@dataclass(frozen=True, slots=True)
class AblationResult:
    summary_csv: Path
    report_md: Path


def run_ablation(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    seeds: list[int],
    variants: list[str] | None = None,
    mock: bool = True,
) -> AblationResult:
    if not mock:
        raise ValueError("run_ablation currently supports mock mode only.")
    selected_variants = variants or list(DEFAULT_VARIANTS)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        for seed in seeds:
            run_rows.append(_run_variant(benchmark_dir, output_dir, variant, seed))

    runs_csv = output_dir / "ablation_runs.csv"
    _write_csv(runs_csv, run_rows)
    summary_rows = _aggregate(run_rows)
    summary_csv = output_dir / "ablation_summary.csv"
    _write_csv(summary_csv, summary_rows)
    report_md = output_dir / "ablation_report.md"
    report_md.write_text(_render_report(summary_rows), encoding="utf-8")
    return AblationResult(summary_csv=summary_csv, report_md=report_md)


def _run_variant(benchmark_dir: Path, output_dir: Path, variant: str, seed: int) -> dict[str, Any]:
    config = _variant_config(variant, seed)
    experiment_id = f"{variant}-seed-{seed}"
    run_config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=benchmark_dir,
        output_dir=output_dir / "runs",
        evolution=config,
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(run_config, MockLLMClient()).run()
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
    return {
        "variant": variant,
        "seed": seed,
        "evidence_mode": "mock_workflow_shape",
        "llm_mode": "mock",
        "scientific_claim": "not_supported",
        "run_dir": str(run_dir),
        "champion_node_id": champion["node_id"],
        "champion_score": champion_score,
        "root_score": root_score,
        "champion/root improvement": improvement,
        "valid_solution_rate": evaluated_count / len(nodes) if nodes else 0.0,
        "timeout_count": timeout_count,
        "debug_success_count": debug_success_count,
        "llm_calls": metadata.get("llm_calls", {}).get("total", 0),
        "wall_time_s": metadata.get("wall_time_s", 0.0),
    }


def _variant_config(variant: str, seed: int) -> EvolutionConfig:
    common = {
        "parallel_mutations": 1,
        "max_debug_retries": 1,
        "random_seed": seed,
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
    raise ValueError(f"Unknown ablation variant: {variant}")


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
        summary.append(
            {
                "variant": variant,
                "evidence_mode": "mock_workflow_shape",
                "scientific_claim": "not_supported",
                "runs": len(rows),
                "valid_runs": sum(1 for row in rows if float(row["valid_solution_rate"]) > 0),
                "champion_score_median": _median(champion_scores),
                "champion_score_mean": _mean(champion_scores),
                "champion_score_std": _std(champion_scores),
                "champion_score_best": min(champion_scores) if champion_scores else "",
                "champion_score_worst": max(champion_scores) if champion_scores else "",
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
    lines = [
        "# Ablation Report",
        "",
        "Evidence mode: `mock_workflow_shape`.",
        "",
        "Mock-mode ablation checks workflow behavior, variant switches, and artifact production. It cannot prove emergent discovery or paper-score reproduction.",
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
