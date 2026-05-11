from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.evidence import EVIDENCE_MODE_REAL_LLM_SMOKE, SCIENTIFIC_CLAIM_NOT_SUPPORTED
from agenticsciml.llm.base import LLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


DEFAULT_SMOKE_VARIANTS = ("branch_context", "no_branch_context")


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
    max_iterations: int = 1,
    parallel_mutations: int = 2,
    llm_client: LLMClient | None = None,
) -> LLMSmokeResult:
    selected_variants = variants or list(DEFAULT_SMOKE_VARIANTS)
    _validate_smoke_variants(selected_variants)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(
        benchmark_dir,
        output_dir,
        selected_variants,
        seed,
        timeout_s,
        max_iterations,
        parallel_mutations,
        dry_run=dry_run,
    )
    plan_path = output_dir / "real_llm_smoke_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    if dry_run:
        report_path = output_dir / "real_llm_smoke_report.md"
        report_path.write_text(_render_dry_run_report(plan), encoding="utf-8")
        return LLMSmokeResult(plan_json=plan_path, report_md=report_path)

    llm = llm_client or OpenAIAdapter()

    rows: list[dict[str, Any]] = []
    gate_issues: list[str] = []
    for entry in plan["runs"]:
        variant = str(entry["variant"])
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
        )
        run_dir = AgenticSciMLOrchestrator(config, llm).run()
        row = _smoke_row(run_dir, variant, seed)
        rows.append(row)
        if not _truthy(row["smoke_gate_passed"]):
            gate_issues.append(f"{variant}: {row['smoke_gate_issues']}")

    runs_csv = output_dir / "real_llm_smoke_runs.csv"
    _write_csv(runs_csv, rows)
    report_path = output_dir / "real_llm_smoke_report.md"
    report_path.write_text(_render_real_report(plan, rows), encoding="utf-8")
    if gate_issues:
        raise RuntimeError(f"Real LLM smoke gate failed; see {report_path}: {'; '.join(gate_issues)}")
    return LLMSmokeResult(plan_json=plan_path, report_md=report_path, runs_csv=runs_csv)


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
) -> dict[str, Any]:
    runs = [
        {
            "variant": variant,
            "experiment_id": f"smoke-{variant}-seed-{seed}",
            "max_iterations": max_iterations,
            "parallel_mutations": parallel_mutations,
            "use_branch_context": variant != "no_branch_context",
        }
        for variant in variants
    ]
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
        "benchmark_dir": str(benchmark_dir),
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
    trace_summary = json.loads((run_dir / "trace_summary.json").read_text(encoding="utf-8"))
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
    gate = _smoke_gate(run_dir, variant, metadata, trace_summary, nodes, branch_tags)
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "evidence_mode": EVIDENCE_MODE_REAL_LLM_SMOKE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "branch_context_enabled": bool(metadata.get("branch_context_enabled", False)),
        "solution_count": len(nodes),
        "branch_intents": ",".join(tag.replace("branch:", "", 1) for tag in branch_tags),
        "proposal_titles": " | ".join(proposal_titles),
        "llm_calls": metadata.get("llm_calls", {}).get("total", 0),
        "trace_quality_gate_passed": bool(trace_summary.get("quality_gate", {}).get("passed", False)),
        "smoke_gate_passed": gate["passed"],
        "smoke_gate_issues": "; ".join(gate["issues"]),
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
        "## Planned Runs\n\n"
        f"{run_lines}\n\n"
        "## Boundary\n\n"
        f"{plan['claim_boundary']}\n"
    )


def _render_real_report(plan: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    row_lines = "\n".join(
        f"- `{row['variant']}`: branch_context={row['branch_context_enabled']}, "
        f"solutions={row['solution_count']}, intents={row['branch_intents'] or 'none'}, "
        f"trace_gate={row['trace_quality_gate_passed']}, smoke_gate={row['smoke_gate_passed']}"
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


def _validate_smoke_variants(variants: list[str]) -> None:
    allowed = set(DEFAULT_SMOKE_VARIANTS)
    unknown = sorted({variant for variant in variants if variant not in allowed})
    if unknown:
        valid = ", ".join(DEFAULT_SMOKE_VARIANTS)
        raise ValueError(f"Unknown LLM smoke variant(s): {', '.join(unknown)}. Valid variants: {valid}")


def _smoke_gate(
    run_dir: Path,
    variant: str,
    metadata: dict[str, Any],
    trace_summary: dict[str, Any],
    nodes: list[dict[str, Any]],
    branch_tags: set[str],
) -> dict[str, Any]:
    issues: list[str] = []
    expected_branch_context = variant != "no_branch_context"
    if bool(metadata.get("branch_context_enabled", False)) is not expected_branch_context:
        issues.append("branch_context_enabled does not match variant")
    if not trace_summary.get("quality_gate", {}).get("passed", False):
        issues.append("trace_summary quality gate failed")
    if int(metadata.get("llm_calls", {}).get("total", 0) or 0) <= 0:
        issues.append("llm_calls.total must be positive in real smoke")

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
    else:
        if branch_tags:
            issues.append("no_branch_context variant produced branch method tags")
        if any(event.get("metadata", {}).get("branch_context") for event in child_events):
            issues.append("no_branch_context trace contains branch context")
        forbidden = ("branch_intent", "sibling_branch_ids", "diversity_instruction")
        transcript_text = _child_transcript_text(run_dir, nodes)
        leaked = [token for token in forbidden if token in transcript_text]
        if leaked:
            issues.append(f"no_branch_context transcripts leaked tokens: {', '.join(leaked)}")
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


def _child_transcript_text(run_dir: Path, nodes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for node in nodes:
        if node.get("parent_id") is None:
            continue
        transcript_dir = run_dir / "solutions" / str(node["node_id"]) / "transcripts"
        for name in ("proposal_debate.json", "engineer.json"):
            path = transcript_dir / name
            if path.exists():
                parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)
