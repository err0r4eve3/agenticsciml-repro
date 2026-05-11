from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
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


class _RecordingLLMClient(LLMClient):
    def __init__(self, inner: LLMClient, ledger_path: Path):
        self.inner = inner
        self.ledger_path = ledger_path
        self.provider = type(inner).__name__
        self.model = getattr(inner, "model", None) or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        self._call_count = 0

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return self._record_call(
            method="complete_text",
            schema_name=None,
            prompt=prompt,
            system=system,
            temperature=temperature,
            call=lambda: self.inner.complete_text(prompt, system=system, temperature=temperature),
        )

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return self._record_call(
            method="complete_json",
            schema_name=schema_name,
            prompt=prompt,
            system=system,
            temperature=temperature,
            call=lambda: self.inner.complete_json(prompt, schema_name, system=system, temperature=temperature),
        )

    def _record_call(
        self,
        *,
        method: str,
        schema_name: str | None,
        prompt: str,
        system: str | None,
        temperature: float,
        call: Any,
    ) -> Any:
        self._call_count += 1
        call_id = f"llm_call_{self._call_count:06d}"
        started_wall = time.time()
        started = time.monotonic()
        record: dict[str, Any] = {
            "schema_version": 1,
            "call_id": call_id,
            "provider": self.provider,
            "model": self.model,
            "method": method,
            "schema_name": schema_name,
            "prompt_hash": _hash_text(prompt),
            "system_hash": _hash_text(system or ""),
            "temperature": temperature,
            "started_at_unix": started_wall,
        }
        try:
            response = call()
        except Exception as exc:
            record.update(
                {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "duration_s": time.monotonic() - started,
                }
            )
            self._append_ledger(record)
            raise
        record.update(
            {
                "success": True,
                "response_hash": _hash_payload(response) if isinstance(response, dict) else _hash_text(str(response)),
                "duration_s": time.monotonic() - started,
            }
        )
        self._append_ledger(record)
        return response

    def _append_ledger(self, record: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


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
        expected_run_dir = output_dir / "runs" / str(entry["experiment_id"])
        ledger_path = expected_run_dir / "llm_call_ledger.jsonl"
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
        )
        try:
            recording_llm = _RecordingLLMClient(llm, ledger_path)
            run_dir = AgenticSciMLOrchestrator(config, recording_llm).run()
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
    llm_calls_total = _llm_call_count(metadata, variant, [])
    ledger_entries, ledger_issues = _read_llm_call_ledger(run_dir, variant)
    ledger_call_count = len(ledger_entries)
    generation_span_count = _generation_span_count(run_dir)
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
        "branch_intents": ",".join(tag.replace("branch:", "", 1) for tag in branch_tags),
        "proposal_titles": " | ".join(proposal_titles),
        "llm_calls": llm_calls_total if llm_calls_total is not None else 0,
        "llm_ledger_calls": ledger_call_count,
        "llm_ledger_providers": ",".join(sorted({str(entry.get("provider", "")) for entry in ledger_entries})),
        "llm_ledger_models": ",".join(sorted({str(entry.get("model", "")) for entry in ledger_entries})),
        "generation_span_count": generation_span_count,
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
    if runs_csv.exists() and not rows:
        issues.append("real_llm_smoke_runs.csv must contain paired run rows")
    manifest_call_range: tuple[int, int] | None = None
    if manifest:
        manifest_issues, manifest_call_range = _manifest_schema_issues(manifest, output_dir)
        issues.extend(manifest_issues)
    if plan:
        expected_hash = _hash_payload(plan)
        if manifest and manifest.get("plan_hash") != expected_hash:
            issues.append("manifest plan_hash does not match plan payload")
        if manifest and manifest.get("config_hash") != expected_hash:
            issues.append("manifest config_hash does not match plan payload")
        if plan.get("execution_mode") != "real":
            issues.append("plan execution_mode must be real for smoke verification")
        if Path(str(plan.get("output_dir", ""))).resolve() != output_dir.resolve():
            issues.append("plan output_dir does not match verification bundle")
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
        expected_run_dir = (Path(str(plan.get("output_dir", ""))) / "runs" / str(plan_entry.get("experiment_id"))).resolve()
        if run_dir.resolve() != expected_run_dir:
            issues.append(f"{variant}: run_dir {run_dir} does not match expected {expected_run_dir}")
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

    paired_gate = _paired_contrast_gate(recomputed_rows)
    if not paired_gate["passed"]:
        issues.append(f"paired contrast gate failed: {'; '.join(paired_gate['issues'])}")
    if manifest and recomputed_rows:
        call_count = 0
        for row in recomputed_rows:
            row_variant = str(row.get("variant", "unknown"))
            row_call_count = _parse_strict_int(row.get("llm_calls"), f"{row_variant}: recomputed llm_calls", issues)
            if row_call_count is not None:
                call_count += row_call_count
            _verify_row_ledger(row, manifest, issues)
        if call_count <= 0:
            issues.append("recomputed LLM call count must be positive")
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
        expected_call_id = f"llm_call_{len(entries) + 1:06d}"
        issues.extend(_validate_llm_call_ledger_entry(entry, expected_call_id, seen_call_ids, line_number, variant))
        entries.append(entry)
    if not entries:
        issues.append(f"{variant}: llm_call_ledger.jsonl must contain at least one call")
    return entries, issues


def _validate_llm_call_ledger_entry(
    entry: dict[str, Any],
    expected_call_id: str,
    seen_call_ids: set[str],
    line_number: int,
    variant: str,
) -> list[str]:
    issues: list[str] = []
    prefix = f"{variant}: ledger line {line_number}"
    forbidden = {"prompt", "system", "response", "raw_prompt", "raw_response", "messages"}
    leaked = sorted(forbidden & set(entry))
    if leaked:
        issues.append(f"{prefix} contains forbidden raw field(s): {', '.join(leaked)}")
    if entry.get("schema_version") != 1:
        issues.append(f"{prefix} schema_version must be 1")
    call_id = entry.get("call_id")
    if call_id != expected_call_id:
        issues.append(f"{prefix} call_id must be {expected_call_id}")
    if isinstance(call_id, str):
        if call_id in seen_call_ids:
            issues.append(f"{prefix} call_id must be unique")
        seen_call_ids.add(call_id)
    _required_non_empty_string(entry.get("provider"), f"{prefix} provider", issues)
    _required_non_empty_string(entry.get("model"), f"{prefix} model", issues)
    method = _required_non_empty_string(entry.get("method"), f"{prefix} method", issues)
    if method and method not in {"complete_text", "complete_json"}:
        issues.append(f"{prefix} method must be complete_text or complete_json")
    schema_name = entry.get("schema_name")
    if method == "complete_text" and schema_name is not None:
        issues.append(f"{prefix} schema_name must be null for complete_text")
    if method == "complete_json":
        _required_non_empty_string(schema_name, f"{prefix} schema_name", issues)
    for field in ("prompt_hash", "system_hash"):
        _validate_sha256_hex(entry.get(field), f"{prefix} {field}", issues)
    success = entry.get("success")
    if not isinstance(success, bool):
        issues.append(f"{prefix} success must be boolean")
    elif success:
        _validate_sha256_hex(entry.get("response_hash"), f"{prefix} response_hash", issues)
    else:
        issues.append(f"{prefix} success must be true for completed smoke evidence")
        _required_non_empty_string(entry.get("error_type"), f"{prefix} error_type", issues)
    _validate_non_negative_finite_number(entry.get("duration_s"), f"{prefix} duration_s", issues)
    _validate_positive_finite_number(entry.get("started_at_unix"), f"{prefix} started_at_unix", issues)
    return issues


def _validate_sha256_hex(value: Any, label: str, issues: list[str]) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        issues.append(f"{label} must be a lowercase sha256 hex string")


def _validate_non_negative_finite_number(value: Any, label: str, issues: list[str]) -> None:
    number = _parse_finite_number(value, label, issues)
    if number is not None and number < 0:
        issues.append(f"{label} must be >= 0")


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


def _generation_span_count(run_dir: Path) -> int:
    return sum(1 for event in _read_trace_events(run_dir) if event.get("event_type") == "generation_span")


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
    expected_provider = manifest.get("provider")
    expected_model = manifest.get("model")
    providers = {item for item in str(row.get("llm_ledger_providers", "")).split(",") if item}
    models = {item for item in str(row.get("llm_ledger_models", "")).split(",") if item}
    if expected_provider and providers != {expected_provider}:
        issues.append(
            f"{variant}: ledger providers {sorted(providers)} do not match manifest provider {expected_provider}"
        )
    if expected_model and models != {expected_model}:
        issues.append(f"{variant}: ledger models {sorted(models)} do not match manifest model {expected_model}")


def _manifest_schema_issues(manifest: dict[str, Any], output_dir: Path) -> tuple[list[str], tuple[int, int] | None]:
    issues: list[str] = []
    call_range_bounds: tuple[int, int] | None = None
    if manifest.get("execution_mode") != "real":
        issues.append("manifest execution_mode must be real for smoke verification")
    if Path(str(manifest.get("output_dir", ""))).resolve() != output_dir.resolve():
        issues.append("manifest output_dir does not match verification bundle")
    if manifest.get("real_mode_explicit") is not True:
        issues.append("manifest real_mode_explicit must be true for smoke verification")
    _required_non_empty_string(manifest.get("provider"), "manifest provider", issues)
    _required_non_empty_string(manifest.get("model"), "manifest model", issues)

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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str]:
    names = ["agenticsciml-repro", "numpy", "openai"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions
