from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import sys
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
    manifest_json: Path
    runs_csv: Path | None = None


@dataclass(frozen=True, slots=True)
class LLMSmokeVerification:
    verification_json: Path
    passed: bool


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
    if not dry_run:
        _require_paired_contrast(selected_variants)
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
    manifest = _build_manifest(plan, llm_client=llm_client)
    manifest_path = output_dir / "real_llm_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if dry_run:
        report_path = output_dir / "real_llm_smoke_report.md"
        report_path.write_text(_render_dry_run_report(plan), encoding="utf-8")
        return LLMSmokeResult(plan_json=plan_path, report_md=report_path, manifest_json=manifest_path)

    try:
        llm = llm_client or OpenAIAdapter()
    except Exception as exc:
        report_path = output_dir / "real_llm_smoke_report.md"
        report_path.write_text(_render_failure_report(plan, "adapter_init_error", exc), encoding="utf-8")
        raise

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
        try:
            run_dir = AgenticSciMLOrchestrator(config, llm).run()
        except Exception as exc:
            report_path = output_dir / "real_llm_smoke_report.md"
            report_path.write_text(_render_failure_report(plan, "orchestrator_error", exc), encoding="utf-8")
            raise
        row = _smoke_row(run_dir, variant, seed)
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
        raise RuntimeError(f"Real LLM smoke gate failed; see {report_path}: {'; '.join(gate_issues)}")
    return LLMSmokeResult(plan_json=plan_path, report_md=report_path, manifest_json=manifest_path, runs_csv=runs_csv)


def verify_llm_smoke_output(output_dir: Path) -> LLMSmokeVerification:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _verify_llm_smoke_output(output_dir)
    path = output_dir / "real_llm_smoke_verification.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
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
        "## Manifest\n\n"
        "- `real_llm_smoke_manifest.json`\n\n"
        "## Planned Runs\n\n"
        f"{run_lines}\n\n"
        "## Boundary\n\n"
        f"{plan['claim_boundary']}\n"
    )


def _render_real_report(plan: dict[str, Any], rows: list[dict[str, Any]], paired_gate: dict[str, Any]) -> str:
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
        "## Manifest\n\n"
        "- `real_llm_smoke_manifest.json`\n\n"
        "## Runs\n\n"
        f"{row_lines}\n\n"
        "## Paired Contrast Gate\n\n"
        f"- paired_contrast_passed: `{paired_gate['passed']}`\n"
        f"- issues: `{'; '.join(paired_gate['issues']) or 'none'}`\n\n"
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _verify_llm_smoke_output(output_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    plan = _read_json_or_issue(output_dir / "real_llm_smoke_plan.json", issues)
    manifest = _read_json_or_issue(output_dir / "real_llm_smoke_manifest.json", issues)
    runs_csv = output_dir / "real_llm_smoke_runs.csv"
    report_path = output_dir / "real_llm_smoke_report.md"
    if not report_path.exists():
        issues.append("missing real_llm_smoke_report.md")
    if not runs_csv.exists():
        issues.append("missing real_llm_smoke_runs.csv; dry-run outputs are not real-smoke evidence")

    rows = _read_rows(runs_csv, issues) if runs_csv.exists() else []
    if plan:
        expected_hash = _hash_payload(plan)
        if manifest and manifest.get("plan_hash") != expected_hash:
            issues.append("manifest plan_hash does not match plan payload")
        if plan.get("execution_mode") != "real":
            issues.append("plan execution_mode must be real for smoke verification")
        variants = [str(entry.get("variant")) for entry in plan.get("runs", []) if isinstance(entry, dict)]
        try:
            _require_paired_contrast(variants)
        except ValueError as exc:
            issues.append(str(exc))

    recomputed_rows: list[dict[str, Any]] = []
    for row in rows:
        variant = str(row.get("variant", ""))
        run_dir = Path(str(row.get("run_dir", "")))
        seed = int(row.get("seed", 0) or 0)
        if not run_dir.exists():
            issues.append(f"{variant}: run_dir does not exist: {run_dir}")
            continue
        try:
            recomputed = _smoke_row(run_dir, variant, seed)
        except Exception as exc:
            issues.append(f"{variant}: could not recompute smoke row: {type(exc).__name__}: {exc}")
            continue
        recomputed_rows.append(recomputed)
        if not _truthy(recomputed["smoke_gate_passed"]):
            issues.append(f"{variant}: smoke gate failed: {recomputed['smoke_gate_issues']}")
        issues.extend(_parallel_trace_issues(run_dir, variant, plan))

    if recomputed_rows:
        paired_gate = _paired_contrast_gate(recomputed_rows)
        if not paired_gate["passed"]:
            issues.append(f"paired contrast gate failed: {'; '.join(paired_gate['issues'])}")

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
        "recomputed_rows": recomputed_rows,
    }


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


def _parallel_trace_issues(run_dir: Path, variant: str, plan: dict[str, Any]) -> list[str]:
    parallel_mutations = 1
    for entry in plan.get("runs", []):
        if isinstance(entry, dict) and entry.get("variant") == variant:
            parallel_mutations = int(entry.get("parallel_mutations", 1) or 1)
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
    if not any(
        event.get("metadata", {}).get("execution_mode") == "parallel"
        and int(event.get("metadata", {}).get("max_workers", 0) or 0) >= 2
        for event in starts
    ):
        return [f"{variant}: no parallel child trace with max_workers >= 2"]
    return []


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
    if int(branch.get("llm_calls", 0) or 0) <= 0 or int(no_branch.get("llm_calls", 0) or 0) <= 0:
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


def _build_manifest(plan: dict[str, Any], *, llm_client: LLMClient | None) -> dict[str, Any]:
    plan_hash = _hash_payload(plan)
    model = getattr(llm_client, "model", None) or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    provider = type(llm_client).__name__ if llm_client is not None else "OpenAIAdapter"
    run_count = len(plan.get("runs", []))
    return {
        "schema_version": 1,
        "execution_mode": plan["execution_mode"],
        "real_mode_explicit": plan["real_mode_explicit"],
        "provider": provider,
        "model": model,
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "seed": plan["seed"],
        "max_iterations": [entry["max_iterations"] for entry in plan["runs"]],
        "parallel_mutations": [entry["parallel_mutations"] for entry in plan["runs"]],
        "timeout_s": plan["timeout_s"],
        "timeout_scope": plan["timeout_scope"],
        "output_dir": plan["output_dir"],
        "plan_hash": plan_hash,
        "config_hash": plan_hash,
        "token_budget": {
            "prompt_token_ceiling": "not_configured",
            "completion_token_ceiling": "not_configured",
            "cost_ceiling_usd": "not_configured",
        },
        "expected_llm_call_range": {
            "min": max(1, run_count * 8),
            "max": max(1, run_count * 80),
        },
        "claim_boundary": plan["claim_boundary"],
    }


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_versions() -> dict[str, str]:
    names = ["agenticsciml-repro", "numpy", "openai"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions
