from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.evidence import SCIENTIFIC_CLAIM_NOT_SUPPORTED
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


DEFAULT_SMOKE_VARIANTS = ("branch_context", "no_branch_context")
REAL_LLM_SMOKE_EVIDENCE_MODE = "real_llm_smoke"


@dataclass(frozen=True, slots=True)
class LLMSmokeResult:
    plan_json: Path
    report_md: Path
    runs_csv: Path | None = None


def run_llm_smoke(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    variants: list[str] | None = None,
    seed: int = 0,
    dry_run: bool = True,
    timeout_s: int = 60,
) -> LLMSmokeResult:
    selected_variants = variants or list(DEFAULT_SMOKE_VARIANTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(benchmark_dir, output_dir, selected_variants, seed, timeout_s)
    plan_path = output_dir / "real_llm_smoke_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    if dry_run:
        report_path = output_dir / "real_llm_smoke_report.md"
        report_path.write_text(_render_dry_run_report(plan), encoding="utf-8")
        return LLMSmokeResult(plan_json=plan_path, report_md=report_path)

    try:
        llm = OpenAIAdapter()
    except RuntimeError as exc:
        raise RuntimeError("real LLM smoke requires OPENAI_API_KEY or --dry-run") from exc

    rows: list[dict[str, Any]] = []
    for entry in plan["runs"]:
        variant = str(entry["variant"])
        config = ExperimentConfig(
            experiment_id=str(entry["experiment_id"]),
            benchmark_dir=benchmark_dir,
            output_dir=output_dir / "runs",
            evolution=_smoke_variant_config(variant, seed=seed, timeout_s=timeout_s),
            use_mock=False,
        )
        run_dir = AgenticSciMLOrchestrator(config, llm).run()
        rows.append(_smoke_row(run_dir, variant, seed))

    runs_csv = output_dir / "real_llm_smoke_runs.csv"
    _write_csv(runs_csv, rows)
    report_path = output_dir / "real_llm_smoke_report.md"
    report_path.write_text(_render_real_report(plan, rows), encoding="utf-8")
    return LLMSmokeResult(plan_json=plan_path, report_md=report_path, runs_csv=runs_csv)


def _build_plan(
    benchmark_dir: Path,
    output_dir: Path,
    variants: list[str],
    seed: int,
    timeout_s: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "timeout_s": timeout_s,
        "evidence_mode": REAL_LLM_SMOKE_EVIDENCE_MODE,
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
            "runs/<variant>-seed-<seed>/tree.json",
            "runs/<variant>-seed-<seed>/trace_summary.json",
            "runs/<variant>-seed-<seed>/solutions/<solution_id>/proposal.md",
            "runs/<variant>-seed-<seed>/solutions/<solution_id>/engineering_summary.md",
        ],
        "runs": [
            {
                "variant": variant,
                "experiment_id": f"smoke-{variant}-seed-{seed}",
                "max_iterations": 1,
                "parallel_mutations": 2,
                "use_branch_context": variant != "no_branch_context",
            }
            for variant in variants
        ],
        "claim_boundary": (
            "This smoke can provide prompt-delivery and behavioral-difference evidence only. "
            "It is not a paper-scale scientific reproduction."
        ),
    }


def _smoke_variant_config(variant: str, *, seed: int, timeout_s: int) -> EvolutionConfig:
    if variant not in DEFAULT_SMOKE_VARIANTS:
        raise ValueError(f"Unknown LLM smoke variant: {variant}")
    return EvolutionConfig(
        max_iterations=1,
        parallel_mutations=2,
        max_debug_retries=1,
        timeout_s=timeout_s,
        random_seed=seed,
        use_kb=True,
        use_branch_context=variant != "no_branch_context",
    )


def _smoke_row(run_dir: Path, variant: str, seed: int) -> dict[str, Any]:
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    nodes = tree["nodes"]
    branch_tags = sorted(
        {
            tag
            for node in nodes
            for tag in node.get("method_tags", [])
            if isinstance(tag, str) and tag.startswith("branch:")
        }
    )
    proposal_titles = _proposal_titles(run_dir, nodes)
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "evidence_mode": REAL_LLM_SMOKE_EVIDENCE_MODE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "branch_context_enabled": bool(metadata.get("branch_context_enabled", False)),
        "solution_count": len(nodes),
        "branch_intents": ",".join(tag.replace("branch:", "", 1) for tag in branch_tags),
        "proposal_titles": " | ".join(proposal_titles),
        "llm_calls": metadata.get("llm_calls", {}).get("total", 0),
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
        "- API calls: none; dry run only.\n\n"
        "## Planned Runs\n\n"
        f"{run_lines}\n\n"
        "## Boundary\n\n"
        f"{plan['claim_boundary']}\n"
    )


def _render_real_report(plan: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    row_lines = "\n".join(
        f"- `{row['variant']}`: branch_context={row['branch_context_enabled']}, "
        f"solutions={row['solution_count']}, intents={row['branch_intents'] or 'none'}"
        for row in rows
    )
    return (
        "# Real LLM Smoke Report\n\n"
        f"- Evidence mode: `{plan['evidence_mode']}`\n"
        f"- Scientific claim: `{plan['scientific_claim']}`\n\n"
        "## Runs\n\n"
        f"{row_lines}\n\n"
        "## Boundary\n\n"
        f"{plan['claim_boundary']}\n"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
