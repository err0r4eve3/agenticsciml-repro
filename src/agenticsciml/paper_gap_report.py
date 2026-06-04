from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from agenticsciml.benchmarks import BenchmarkSpec, benchmark_for_path, list_benchmarks
from agenticsciml.evidence import LLM_MODE_REAL
from agenticsciml.storage import _atomic_write_text


PAPER_GAP_REPORT_SCHEMA_VERSION = 1
COMPLETED_RUN_STATES = {"completed", "exported", "finalized"}
CLAIM_GATE_ALLOWED = "allowed"


def build_paper_gap_report(
    *,
    benchmark_dirs: Sequence[Path] | None = None,
    run_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    specs = _benchmark_specs(benchmark_dirs)
    run_evidence = [_run_evidence_summary(Path(run_dir)) for run_dir in run_dirs or []]
    runs_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for run in run_evidence:
        benchmark_name = run.get("benchmark_name")
        if isinstance(benchmark_name, str) and benchmark_name:
            runs_by_benchmark.setdefault(benchmark_name, []).append(run)

    benchmark_reports = [
        _benchmark_gap_report(spec, runs_by_benchmark.get(spec.name, []))
        for spec in specs
    ]
    unmatched_runs = [
        run
        for run in run_evidence
        if run.get("benchmark_name") not in {spec.name for spec in specs}
    ]
    open_gap_count = sum(
        int(report["summary"]["open_gap_count"])
        for report in benchmark_reports
        if isinstance(report.get("summary"), dict)
    )
    status = "blocked" if open_gap_count or unmatched_runs else "paper_claim_review_ready"
    return {
        "schema_version": PAPER_GAP_REPORT_SCHEMA_VERSION,
        "report_type": "agenticsciml_paper_gap_report",
        "status": status,
        "summary": {
            "benchmark_count": len(benchmark_reports),
            "run_evidence_count": len(run_evidence),
            "open_gap_count": open_gap_count,
            "unmatched_run_count": len(unmatched_runs),
            "fidelity_counts": _fidelity_counts(specs),
        },
        "benchmarks": benchmark_reports,
        "run_evidence": run_evidence,
        "unmatched_runs": unmatched_runs,
        "claim_boundary": (
            "Paper gap reports inventory benchmark fidelity and run evidence. They do not "
            "upgrade a run to a scientific or paper-score claim unless the underlying "
            "claim gate, readiness, trace quality, and replication evidence all pass."
        ),
    }


def write_paper_gap_report(
    *,
    output_dir: Path,
    benchmark_dirs: Sequence[Path] | None = None,
    run_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    report = build_paper_gap_report(benchmark_dirs=benchmark_dirs, run_dirs=run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paper_gap_report.json"
    markdown_path = output_dir / "paper_gap_report.md"
    _atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
    )
    _atomic_write_text(markdown_path, render_paper_gap_report_markdown(report))
    return {
        "report": report,
        "paths": {
            "report_json": str(json_path),
            "report_md": str(markdown_path),
        },
    }


def render_paper_gap_report_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Paper Gap Report",
        "",
        f"- status: {report.get('status')}",
        f"- benchmark count: {summary.get('benchmark_count')}",
        f"- run evidence count: {summary.get('run_evidence_count')}",
        f"- open gap count: {summary.get('open_gap_count')}",
        "",
        "## Benchmark Gaps",
    ]
    benchmarks = report.get("benchmarks") if isinstance(report.get("benchmarks"), list) else []
    if benchmarks:
        for benchmark in benchmarks:
            benchmark_summary = (
                benchmark.get("summary") if isinstance(benchmark.get("summary"), dict) else {}
            )
            lines.append(
                f"- `{benchmark.get('name')}`: fidelity={benchmark.get('fidelity_level')}, "
                f"runs={benchmark_summary.get('run_count')}, "
                f"open_gaps={benchmark_summary.get('open_gap_count')}"
            )
            for item in benchmark.get("gap_items", []):
                if not isinstance(item, dict) or item.get("status") == "satisfied":
                    continue
                lines.append(f"  - {item.get('check_id')}: {item.get('blocker')}")
    else:
        lines.append("- none")

    lines.extend(["", "## Run Evidence"])
    run_evidence = (
        report.get("run_evidence") if isinstance(report.get("run_evidence"), list) else []
    )
    if run_evidence:
        for run in run_evidence:
            lines.append(
                f"- `{run.get('path')}`: benchmark={run.get('benchmark_name')}, "
                f"state={run.get('run_state')}, llm={run.get('llm_mode')}, "
                f"trace_passed={run.get('trace_quality_gate_passed')}"
            )
    else:
        lines.append("- no run directories supplied")

    lines.extend(["", "## Claim Boundary", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def _benchmark_specs(benchmark_dirs: Sequence[Path] | None) -> list[BenchmarkSpec]:
    if not benchmark_dirs:
        return list_benchmarks()
    specs: list[BenchmarkSpec] = []
    seen: set[str] = set()
    for benchmark_dir in benchmark_dirs:
        spec = benchmark_for_path(Path(benchmark_dir))
        if spec is None:
            raise ValueError(f"Unknown benchmark: {benchmark_dir}")
        if spec.name not in seen:
            specs.append(spec)
            seen.add(spec.name)
    return specs


def _benchmark_gap_report(spec: BenchmarkSpec, runs: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = _fidelity_matrix(spec)
    gap_items = [
        _gap_item(
            check_id="benchmark_fidelity",
            passed=spec.fidelity_level == "paper-like",
            requirement="paper-like benchmark fidelity for paper-score comparison",
            evidence=f"catalog fidelity_level={spec.fidelity_level}",
            blocker=f"benchmark is {spec.fidelity_level}, not paper-like",
        ),
        _gap_item(
            check_id="paper_benchmark_equivalence",
            passed=matrix.get("paper_benchmark_equivalent") is True,
            requirement="paper-equivalent data, evaluator, metric, and budget dossier",
            evidence=matrix,
            blocker="catalog fidelity matrix does not mark the benchmark paper-equivalent",
        ),
        _gap_item(
            check_id="completed_run_artifacts",
            passed=any(run.get("completed_run_artifacts") is True for run in runs),
            requirement="completed/exported/finalized run metadata for this benchmark",
            evidence=_run_refs(runs),
            blocker="no supplied run has completed/exported/finalized run_metadata.json",
        ),
        _gap_item(
            check_id="trace_quality_gate",
            passed=any(run.get("trace_quality_gate_passed") is True for run in runs),
            requirement="passing trace_summary.json quality gate",
            evidence=_run_refs(runs),
            blocker="no supplied run has a passing trace_summary.json quality gate",
        ),
        _gap_item(
            check_id="real_llm_execution",
            passed=any(run.get("llm_mode") == LLM_MODE_REAL for run in runs),
            requirement="real LLM execution, not mock or dry-run evidence",
            evidence=_run_refs(runs),
            blocker="no supplied run records llm_mode=real",
        ),
        _gap_item(
            check_id="multi_seed_ablation",
            passed=any(run.get("multi_seed_ablation_verified") is True for run in runs),
            requirement="verified multi-seed ablation evidence",
            evidence=_run_refs(runs),
            blocker="no supplied run attaches verified multi-seed ablation evidence",
        ),
        _gap_item(
            check_id="scientific_readiness",
            passed=any(run.get("scientific_readiness_supported") is True for run in runs),
            requirement="scientific_discovery_readiness supports a scientific claim",
            evidence=_run_refs(runs),
            blocker="scientific discovery readiness is missing or blocked",
        ),
        _gap_item(
            check_id="claim_gate_support",
            passed=any(run.get("paper_level_claim_supported") is True for run in runs),
            requirement="claim_gate supports paper-level and scientific claims",
            evidence=_run_refs(runs),
            blocker="claim_gate does not support paper-level claims",
        ),
    ]
    open_gaps = [item for item in gap_items if item["status"] != "satisfied"]
    return {
        "name": spec.name,
        "path": str(spec.path),
        "paper_section": spec.paper_section,
        "paper_task_name": spec.paper_task_name,
        "family": spec.family,
        "metric": spec.metric,
        "description": spec.description,
        "fidelity_level": spec.fidelity_level,
        "expected_runtime_s": spec.expected_runtime_s,
        "requires_torch": spec.requires_torch,
        "requires_gpu": spec.requires_gpu,
        "paper_gap_notes": spec.paper_gap_notes,
        "fidelity_matrix": matrix,
        "claim_boundaries": spec.claim_boundaries(),
        "gap_items": gap_items,
        "summary": {
            "status": "blocked" if open_gaps else "paper_claim_review_ready",
            "run_count": len(runs),
            "open_gap_count": len(open_gaps),
            "satisfied_count": len(gap_items) - len(open_gaps),
        },
    }


def _run_evidence_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir.resolve()
    issues: list[str] = []
    metadata = _read_required_json(path / "run_metadata.json", issues)
    trace_summary = _read_required_json(path / "trace_summary.json", issues)
    scientific_card = _read_optional_json(path / "reports" / "scientific_result_card.json", issues)
    readiness = _read_optional_json(
        path / "reports" / "scientific_discovery_readiness.json",
        issues,
    )
    ablation = _read_optional_json(path / "reports" / "multi_seed_ablation_evidence.json", issues)

    claim_gate = _dict_value(metadata, "claim_gate") or _dict_value(trace_summary, "claim_gate")
    quality_gate = _dict_value(trace_summary, "quality_gate")
    benchmark_name = _first_str(
        metadata.get("benchmark_name"),
        scientific_card.get("benchmark_name"),
    )
    readiness_supported = (
        readiness.get("scientific_claim_supported") is True
        or _nested_bool(metadata, "scientific_discovery_readiness", "scientific_claim_supported")
    )
    paper_level_supported = (
        claim_gate.get("status") == CLAIM_GATE_ALLOWED
        and claim_gate.get("paper_level_claim_supported") is True
        and claim_gate.get("scientific_claim_supported") is True
    )
    scientific_card_support = _dict_value(scientific_card, "claim_support")
    ablation_verified = (
        _nested_bool(metadata, "multi_seed_ablation", "verified_multi_seed_ablation")
        or ablation.get("verified_multi_seed_ablation") is True
    )
    run_state = _first_str(metadata.get("run_state"))
    return {
        "path": str(path),
        "exists": path.exists(),
        "benchmark_name": benchmark_name,
        "run_state": run_state,
        "completed_run_artifacts": run_state in COMPLETED_RUN_STATES,
        "champion": _first_str(metadata.get("champion")),
        "solution_count": _int_or_none(metadata.get("solution_count")),
        "llm_mode": _first_str(metadata.get("llm_mode")),
        "evidence_mode": _first_str(metadata.get("evidence_mode")),
        "scientific_claim": _first_str(metadata.get("scientific_claim")),
        "trace_quality_gate_passed": quality_gate.get("passed") is True,
        "claim_gate_status": claim_gate.get("status"),
        "scientific_claim_supported": claim_gate.get("scientific_claim_supported") is True,
        "paper_level_claim_supported": paper_level_supported,
        "scientific_readiness_supported": readiness_supported,
        "multi_seed_ablation_verified": ablation_verified,
        "scientific_result_card": {
            "present": bool(scientific_card),
            "evidence_grade": scientific_card.get("evidence_grade"),
            "scientific_claim_supported": scientific_card_support.get("scientific_claim_supported") is True,
            "uncertainty_flag_count": len(scientific_card.get("uncertainty_flags", []))
            if isinstance(scientific_card.get("uncertainty_flags"), list)
            else None,
        },
        "issues": issues,
        "artifact_refs": {
            "run_metadata": "run_metadata.json",
            "trace_summary": "trace_summary.json",
            "scientific_result_card": "reports/scientific_result_card.json",
            "scientific_discovery_readiness": "reports/scientific_discovery_readiness.json",
            "multi_seed_ablation_evidence": "reports/multi_seed_ablation_evidence.json",
        },
    }


def _fidelity_matrix(spec: BenchmarkSpec) -> dict[str, Any]:
    paper_equivalent = spec.fidelity_level == "paper-like"
    return {
        "paper_scale_target": spec.paper_task_name,
        "local_fixture_scope": spec.fidelity_level,
        "metric_delta": "paper-equivalent" if paper_equivalent else spec.paper_gap_notes,
        "hidden_label_protocol": "trusted local evaluator with hidden validation labels",
        "training_budget_delta": "paper-scale" if paper_equivalent else "reduced local budget",
        "solver_dependency_delta": {
            "requires_torch": spec.requires_torch,
            "requires_gpu": spec.requires_gpu,
        },
        "missing_requirements": [] if paper_equivalent else [spec.paper_gap_notes],
        "paper_benchmark_equivalent": paper_equivalent,
    }


def _gap_item(
    *,
    check_id: str,
    passed: bool,
    requirement: str,
    evidence: Any,
    blocker: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "satisfied" if passed else "gap",
        "requirement": requirement,
        "evidence": evidence,
        "blocker": None if passed else blocker,
    }


def _run_refs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": run.get("path"),
            "run_state": run.get("run_state"),
            "llm_mode": run.get("llm_mode"),
            "trace_quality_gate_passed": run.get("trace_quality_gate_passed"),
            "claim_gate_status": run.get("claim_gate_status"),
            "issues": run.get("issues", []),
        }
        for run in runs
    ]


def _fidelity_counts(specs: list[BenchmarkSpec]) -> dict[str, int]:
    counts = {"proxy": 0, "faithful-small": 0, "paper-like": 0}
    for spec in specs:
        counts[spec.fidelity_level] = counts.get(spec.fidelity_level, 0) + 1
    return counts


def _read_required_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing {path.name}")
        return {}
    return _read_json(path, issues)


def _read_optional_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path, issues)


def _read_json(path: Path, issues: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"invalid JSON in {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{path.name} must contain a JSON object")
        return {}
    return payload


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _nested_bool(payload: dict[str, Any], key: str, nested_key: str) -> bool:
    nested = payload.get(key)
    if not isinstance(nested, dict):
        return False
    return nested.get(nested_key) is True


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
