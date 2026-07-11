import csv
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agenticsciml.evidence import EVIDENCE_MODE_REAL_LLM_SMOKE
from agenticsciml.llm.budget import LLMBudget
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm_smoke import (
    _RecordingLLMClient,
    _hash_payload,
    _llm_token_accounting_issues,
    _paired_contrast_gate,
    _render_real_report,
    _score_diagnostics,
    _smoke_gate,
    run_llm_smoke,
    verify_llm_smoke_output,
)

_REAL_SMOKE_BUNDLE_TEMPLATE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _build_real_smoke_bundle_template(tmp_path_factory: pytest.TempPathFactory) -> None:
    global _REAL_SMOKE_BUNDLE_TEMPLATE
    output_dir = tmp_path_factory.mktemp("real-smoke-template")
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=output_dir,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _REAL_SMOKE_BUNDLE_TEMPLATE = output_dir


def _copy_real_smoke_bundle(output_dir: Path) -> None:
    assert _REAL_SMOKE_BUNDLE_TEMPLATE is not None
    shutil.copytree(_REAL_SMOKE_BUNDLE_TEMPLATE, output_dir, dirs_exist_ok=True)
    _retarget_smoke_bundle(output_dir)


def _retarget_smoke_bundle(output_dir: Path) -> None:
    plan_path = output_dir / "real_llm_smoke_plan.json"
    manifest_path = output_dir / "real_llm_smoke_manifest.json"
    rows_path = output_dir / "real_llm_smoke_runs.csv"

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["output_dir"] = str(output_dir)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    plan_hash = _hash_payload(plan)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_dir"] = str(output_dir)
    manifest["plan_hash"] = plan_hash
    manifest["config_hash"] = plan_hash
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    rows = list(csv.DictReader(rows_path.open(encoding="utf-8", newline="")))
    run_by_variant = {
        str(entry["variant"]): output_dir / "runs" / str(entry["experiment_id"])
        for entry in plan["runs"]
    }
    for row in rows:
        row["run_dir"] = str(run_by_variant[str(row["variant"])])
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    manifest["runs_csv_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    report_path = output_dir / "real_llm_smoke_report.md"
    manifest["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_parallel_child_max_workers(run_dir: Path, value: object) -> None:
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = False
    for event in events:
        if event.get("name") == "agenticsciml.parallel_children.start":
            event.setdefault("metadata", {})["max_workers"] = value
            changed = True
    assert changed, "expected parallel_children.start trace event"
    trace_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )


def _rewrite_first_generation_span_metadata(run_dir: Path, field: str, value: object) -> None:
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = False
    for event in events:
        if event.get("event_type") == "generation_span":
            event.setdefault("metadata", {})[field] = value
            changed = True
            break
    assert changed, "expected generation_span trace event"
    trace_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )


def _rewrite_second_generation_span_metadata_from_first(run_dir: Path, field: str) -> None:
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generation_events = [event for event in events if event.get("event_type") == "generation_span"]
    assert len(generation_events) >= 2, "expected at least two generation_span trace events"
    generation_events[1].setdefault("metadata", {})[field] = generation_events[0].get("metadata", {}).get(field)
    trace_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )


def _reverse_generation_span_order(run_dir: Path) -> None:
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reversed_generation_events = list(reversed([event for event in events if event.get("event_type") == "generation_span"]))
    rewritten = []
    for event in events:
        if event.get("event_type") == "generation_span":
            rewritten.append(reversed_generation_events.pop(0))
        else:
            rewritten.append(event)
    trace_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in rewritten) + "\n",
        encoding="utf-8",
    )


def _rewrite_run_metadata_llm_calls_total(run_dir: Path, value: object) -> None:
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("llm_calls", {})["total"] = value
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_first_ledger_field(run_dir: Path, field: str, value: object) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries, "expected at least one ledger entry"
    entries[0][field] = value
    ledger_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _delete_first_ledger_field(run_dir: Path, field: str) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries, "expected at least one ledger entry"
    entries[0].pop(field, None)
    ledger_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _duplicate_second_ledger_call_id(run_dir: Path) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) >= 2, "expected at least two ledger entries"
    entries[1]["call_id"] = entries[0]["call_id"]
    ledger_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _delete_first_ledger_entry(run_dir: Path) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "expected at least one ledger entry"
    ledger_path.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""), encoding="utf-8")


def test_llm_smoke_dry_run_writes_plan_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=True,
    )
    plan = json.loads(result.plan_json.read_text(encoding="utf-8"))
    report = result.report_md.read_text(encoding="utf-8")

    assert result.runs_csv is None
    assert result.manifest_json.exists()
    assert plan["evidence_mode"] == EVIDENCE_MODE_REAL_LLM_SMOKE
    assert plan["scientific_claim"] == "not_supported"
    assert [entry["variant"] for entry in plan["runs"]] == ["branch_context", "no_branch_context"]
    assert plan["runs"][0]["use_branch_context"] is True
    assert plan["runs"][1]["use_branch_context"] is False
    assert plan["runs"][0]["experiment_id"] == "smoke-branch_context-seed-0"
    assert "runs/smoke-branch_context-seed-0/tree.json" in plan["expected_artifacts"]
    assert "API calls: none" in report
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "dry_run"
    assert manifest["run_count"] == 2
    assert manifest["expected_llm_call_range"]["min"] > 0
    assert manifest["expected_llm_call_range"]["max"] == 80
    assert manifest["budget_preflight"]["status"] == "ready"
    assert manifest["provider_capabilities"]["provider"] == "openai"
    assert manifest["provider_capabilities"]["supports_structured_outputs"] is True
    assert manifest["provider_capabilities"]["supports_image_inputs"] is True
    assert manifest["token_budget"]["max_calls"] is None


def test_llm_smoke_dry_run_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown LLM smoke variant"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["unknown_variant"],
            dry_run=True,
        )


def test_llm_smoke_real_mode_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context"],
            dry_run=False,
        )


def test_llm_smoke_real_mode_requires_paired_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact paired variants"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )


def test_llm_smoke_real_mode_rejects_duplicate_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicates are not allowed"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context", "branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )


def test_llm_smoke_real_gate_with_scripted_llm(tmp_path: Path) -> None:
    progress_events: list[dict[str, object]] = []
    result = run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
        progress_callback=progress_events.append,
    )
    rows = list(csv.DictReader(result.runs_csv.open(encoding="utf-8")))  # type: ignore[union-attr]

    assert {row["variant"] for row in rows} == {"branch_context", "no_branch_context"}
    assert all(row["smoke_gate_passed"] == "True" for row in rows)
    assert all(row["trace_quality_gate_passed"] == "True" for row in rows)
    assert all(int(row["llm_calls"]) > 0 for row in rows)
    assert {event["event"] for event in progress_events} == {
        "llm_call_started",
        "llm_call_finished",
    }
    assert {event["variant"] for event in progress_events} == {
        "branch_context",
        "no_branch_context",
    }
    assert len(progress_events) == 2 * sum(int(row["llm_calls"]) for row in rows)
    for row in rows:
        assert row["metric"]
        assert row["higher_is_better"] in {"True", "False"}
        assert int(row["evaluated_solution_count"]) > 0
        assert int(row["failed_solution_count"]) >= 0
        assert row["failed_solution_kinds"] == ""
        assert int(row["debug_attempted_node_count"]) == 0
        assert int(row["debug_attempt_count"]) == 0
        assert int(row["debug_recovered_node_count"]) == 0
        assert int(row["debug_failed_node_count"]) == 0
        assert row["debug_recovery_rate"] == ""
        float(row["root_score"])
        float(row["best_child_score"])
        float(row["best_child_improvement_vs_root"])
        assert row["mutation_improved"] in {"True", "False"}
    assert result.manifest_json.exists()
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["provider_capabilities"]["provider"] == "MockLLMClient"
    assert manifest["token_budget"]["calls_used"] == 0
    no_branch = next(row for row in rows if row["variant"] == "no_branch_context")
    assert no_branch["branch_context_enabled"] == "False"
    assert no_branch["branch_intents"] == ""
    report = result.report_md.read_text(encoding="utf-8")
    assert "performance_comparison_supported: `false`" in report
    assert "independently generated" in report
    assert "debug_recovery_rate=not_applicable" in report

    verification = verify_llm_smoke_output(tmp_path)
    verification_payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))
    assert verification.passed is True
    assert verification_payload["passed"] is True
    assert len(verification_payload["recomputed_rows"]) == 2
    for row in verification_payload["recomputed_rows"]:
        assert int(row["llm_ledger_calls"]) == int(row["llm_calls"])
        assert int(row["generation_span_count"]) == int(row["llm_calls"])
        assert row["llm_ledger_providers"] == "MockLLMClient"
        assert row["llm_ledger_call_fingerprints"] == row["llm_trace_call_fingerprints"]
    run_metadata = json.loads(
        (tmp_path / "runs" / "smoke-branch_context-seed-0" / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_metadata["llm_provider_capabilities"]["provider"] == "MockLLMClient"
    assert run_metadata["llm_budget"]["calls_used"] > 0
    assert run_metadata["llm_ledger_usage"]["calls_used"] == int(
        next(row["llm_calls"] for row in rows if row["variant"] == "branch_context")
    )


def test_real_report_surfaces_failure_diagnostics() -> None:
    report = _render_real_report(
        {
            "evidence_mode": EVIDENCE_MODE_REAL_LLM_SMOKE,
            "scientific_claim": "not_supported",
            "claim_boundary": "workflow evidence only",
        },
        [
            {
                "variant": "branch_context",
                "branch_context_enabled": True,
                "solution_count": 2,
                "branch_intents": "features_or_architecture",
                "metric": "validation_mse",
                "higher_is_better": False,
                "root_score": "",
                "best_child_score": 0.5,
                "best_child_improvement_vs_root": "",
                "mutation_improved": "",
                "evaluated_solution_count": 1,
                "failed_solution_count": 1,
                "failed_solution_kinds": "runtime_error",
                "debug_attempted_node_count": 1,
                "debug_attempt_count": 2,
                "debug_recovered_node_count": 0,
                "debug_failed_node_count": 1,
                "debug_recovery_rate": 0.0,
                "trace_quality_gate_passed": False,
                "smoke_gate_passed": False,
                "smoke_gate_issues": "debugger call timed out; trace_summary quality gate failed",
            }
        ],
        {"passed": False, "issues": ["branch_context row gate failed"]},
    )

    assert "failure_kinds=runtime_error" in report
    assert "gate_issues=debugger call timed out; trace_summary quality gate failed" in report
    assert "higher_is_better=False" in report
    assert "mutation_improved=unknown" in report
    assert "debug_attempts=2" in report
    assert "debug_nodes=1" in report
    assert "debug_recovered=0" in report
    assert "debug_failed=1" in report
    assert "debug_recovery_rate=0.0" in report


@pytest.mark.parametrize(
    ("higher_is_better", "root_score", "child_scores", "expected_best", "expected_improvement"),
    [
        (False, 10.0, [8.0, 7.0], 7.0, 3.0),
        (True, 0.5, [0.6, 0.7], 0.7, 0.2),
    ],
)
def test_score_diagnostics_respects_metric_direction(
    higher_is_better: bool,
    root_score: float,
    child_scores: list[float],
    expected_best: float,
    expected_improvement: float,
) -> None:
    def node(node_id: str, parent_id: str | None, score: float) -> dict[str, object]:
        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "status": "evaluated",
            "score": {
                "metric": "research_metric",
                "value": score,
                "higher_is_better": higher_is_better,
            },
        }

    diagnostics = _score_diagnostics(
        [
            node("solution_000", None, root_score),
            node("solution_001", "solution_000", child_scores[0]),
            node("solution_002", "solution_000", child_scores[1]),
        ]
    )

    assert diagnostics["higher_is_better"] is higher_is_better
    assert diagnostics["root_score"] == root_score
    assert diagnostics["best_child_score"] == expected_best
    assert diagnostics["best_child_improvement_vs_root"] == pytest.approx(expected_improvement)
    assert diagnostics["mutation_improved"] is True


def test_score_diagnostics_handles_failed_unscored_root() -> None:
    diagnostics = _score_diagnostics(
        [
            {
                "node_id": "solution_000",
                "parent_id": None,
                "status": "failed",
                "score": None,
                "failure_kind": "runtime_error",
                "num_debug_attempts": 1,
            },
            {
                "node_id": "solution_001",
                "parent_id": "solution_000",
                "status": "evaluated",
                "num_debug_attempts": 2,
                "score": {
                    "metric": "validation_mse",
                    "value": 0.5,
                    "higher_is_better": False,
                },
            },
        ]
    )

    assert diagnostics["failed_solution_kinds"] == "runtime_error"
    assert diagnostics["root_score"] == ""
    assert diagnostics["best_child_score"] == 0.5
    assert diagnostics["best_child_improvement_vs_root"] == ""
    assert diagnostics["mutation_improved"] == ""
    assert diagnostics["debug_attempted_node_count"] == 2
    assert diagnostics["debug_attempt_count"] == 3
    assert diagnostics["debug_recovered_node_count"] == 1
    assert diagnostics["debug_failed_node_count"] == 1
    assert diagnostics["debug_recovery_rate"] == 0.5


def test_llm_smoke_real_mode_enforces_llm_call_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "1")

    with pytest.raises(RuntimeError, match="LLM call budget preflight failed"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )
    report = (tmp_path / "real_llm_smoke_report.md").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "real_llm_smoke_manifest.json").read_text(encoding="utf-8"))
    assert "blocked_by_budget" in report
    assert manifest["run_count"] == 2
    assert manifest["status"] == "failed"
    assert manifest["report_status"] == "failed"
    assert manifest["failure_kind"] == "blocked_by_budget"
    assert manifest["error_type"] == "LLMBudgetPreflightError"
    assert manifest["budget_preflight"]["status"] == "blocked_by_budget"
    assert manifest["budget_preflight"]["expected_max_llm_calls"] == 80


def test_llm_budget_rejects_nonfinite_cost_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for field in ("AGENTICSCIML_MAX_COST_USD", "AGENTICSCIML_COST_PER_1K_TOKENS_USD"):
        monkeypatch.setenv("AGENTICSCIML_MAX_COST_USD", "1")
        monkeypatch.setenv("AGENTICSCIML_COST_PER_1K_TOKENS_USD", "0.01")
        monkeypatch.setenv(field, "nan")
        with pytest.raises(RuntimeError, match="finite positive number"):
            LLMBudget.from_env()


def test_llm_smoke_marks_manifest_failed_when_orchestrator_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("agenticsciml.llm_smoke.AgenticSciMLOrchestrator.run", fail_run)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )

    manifest = json.loads((tmp_path / "real_llm_smoke_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["report_status"] == "failed"
    assert manifest["failure_kind"] == "orchestrator_error"
    assert manifest["error_type"] == "RuntimeError"
    assert manifest["report_sha256"] == hashlib.sha256(
        (tmp_path / "real_llm_smoke_report.md").read_bytes()
    ).hexdigest()


def test_llm_smoke_marks_manifest_failed_when_run_artifacts_are_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agenticsciml.llm_smoke.AgenticSciMLOrchestrator.run",
        lambda *_args, **_kwargs: tmp_path / "missing-run",
    )

    with pytest.raises(FileNotFoundError):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )

    manifest = json.loads((tmp_path / "real_llm_smoke_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["report_status"] == "failed"
    assert manifest["failure_kind"] == "run_artifact_error"
    assert manifest["error_type"] == "FileNotFoundError"


def test_recording_llm_reserves_call_budget_across_parallel_calls(tmp_path: Path) -> None:
    class SlowLLM:
        model = "slow-model"
        adapter_type = "test-compatible"

        def complete_text(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
            time.sleep(0.05)
            return "ok"

        def complete_json(
            self,
            prompt: str,
            schema_name: str,
            system: str | None = None,
            temperature: float = 0.0,
        ) -> dict[str, str]:
            time.sleep(0.05)
            return {"ok": "true"}

    budget = LLMBudget(max_calls=1)
    recording = _RecordingLLMClient(SlowLLM(), tmp_path / "ledger.jsonl", budget)  # type: ignore[arg-type]
    results: list[str] = []
    errors: list[str] = []

    def worker() -> None:
        try:
            results.append(recording.complete_text("parallel prompt"))
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["ok"]
    assert len(errors) == 1
    assert "LLM call budget exceeded" in errors[0]
    assert budget.calls_used == 1
    assert len((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_verify_llm_smoke_output_rejects_dry_run_only(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=True,
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("dry-run outputs are not real-smoke evidence" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_header_only_runs_csv(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    header = rows_path.read_text(encoding="utf-8").splitlines()[0]
    rows_path.write_text(header + "\n", encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("must contain paired run rows" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_stale_external_run_dir(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    rows = list(csv.DictReader(rows_path.open(encoding="utf-8")))
    rows[0]["run_dir"] = str(tmp_path / "external-run")
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("does not match expected" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_manifest_mode_mismatch(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_mode"] = "dry_run"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("manifest execution_mode must be real" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_schema_and_final_budget_drift(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    plan_path = tmp_path / "real_llm_smoke_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["schema_version"] = 999
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest["plan_hash"] = _hash_payload(plan)
    manifest["config_hash"] = manifest["plan_hash"]
    manifest["failure_kind"] = "orchestrator_error"
    manifest["error_type"] = "RuntimeError"
    manifest["budget_preflight"]["passed"] = False
    manifest["budget_preflight"]["status"] = "blocked_by_budget"
    manifest["token_budget"]["max_calls"] = "1"
    manifest["token_budget_final"]["max_calls"] = "1"
    manifest["token_budget_final"]["calls_used"] = 0
    manifest["token_budget_final"]["prompt_tokens_used"] = -1
    manifest["token_budget_final"]["output_tokens_used"] = -1
    manifest["token_budget_final"]["estimated_cost_usd"] = 123.0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert "plan schema_version must be 1" in payload["issues"]
    assert "manifest schema_version must be 1" in payload["issues"]
    assert "completed manifest must not contain failure fields" in payload["issues"]
    assert "completed manifest budget_preflight must be ready and passed" in payload["issues"]
    assert any("token_budget_final.calls_used does not match" in issue for issue in payload["issues"])
    assert "manifest token_budget_final.prompt_tokens_used must be non-negative" in payload["issues"]
    assert "manifest token_budget_final.output_tokens_used must be non-negative" in payload["issues"]
    assert any("prompt_tokens_used does not match LLM ledger" in issue for issue in payload["issues"])
    assert any("output_tokens_used does not match LLM ledger" in issue for issue in payload["issues"])
    assert any("estimated_cost_usd does not match LLM ledger" in issue for issue in payload["issues"])
    assert "manifest token_budget_final exceeds max_calls" in payload["issues"]


def test_verify_llm_smoke_output_rejects_stale_run_config(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    config_path = tmp_path / "runs" / "smoke-branch_context-seed-0" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evolution"]["random_seed"] = 999
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("evolution.random_seed does not match plan" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_binds_completed_report(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    report_path = tmp_path / "real_llm_smoke_report.md"
    report_path.write_text(report_path.read_text(encoding="utf-8") + "\nstale\n", encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("report_sha256" in issue for issue in payload["issues"])


def test_failed_smoke_rerun_preserves_previous_valid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_real_smoke_bundle(tmp_path)
    original_plan = (tmp_path / "real_llm_smoke_plan.json").read_bytes()
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "1")

    with pytest.raises(RuntimeError, match="LLM call budget preflight failed"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )

    assert (tmp_path / "real_llm_smoke_plan.json").read_bytes() == original_plan
    assert verify_llm_smoke_output(tmp_path).passed is True


def test_verify_llm_smoke_output_rejects_output_dir_mismatch(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    plan_path = tmp_path / "real_llm_smoke_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["output_dir"] = str(tmp_path / "other-bundle")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("plan output_dir does not match" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_manifest_output_dir_mismatch(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_dir"] = str(tmp_path / "other-bundle")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("manifest output_dir does not match" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_non_string_manifest_provider_model(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provider"] = True
    manifest["model"] = ""
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("manifest provider must be a non-empty string" in issue for issue in payload["issues"])
    assert any("manifest model must be a non-empty string" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_reports_malformed_seed_without_crashing(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    rows = list(csv.DictReader(rows_path.open(encoding="utf-8")))
    rows[0]["seed"] = "not-an-int"
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("row seed must be an integer" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_non_integer_plan_seed(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    plan_path = tmp_path / "real_llm_smoke_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["seed"] = 1.7
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("plan seed must be an integer, not float" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_malformed_expected_call_range(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_llm_call_range"] = {"min": "abc", "max": 0}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("expected_llm_call_range.min must be an integer string" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.max must be >= 1" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_missing_expected_call_range_bounds(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_llm_call_range"] = {}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("expected_llm_call_range.min is required" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.max is required" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_reports_manifest_schema_when_rows_are_damaged(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    header = rows_path.read_text(encoding="utf-8").splitlines()[0]
    rows_path.write_text(header + "\n", encoding="utf-8")
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_llm_call_range"] = {"min": "bad", "max": 0}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("must contain paired run rows" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.min must be an integer string" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.max must be >= 1" in issue for issue in payload["issues"])


@pytest.mark.parametrize("bad_value", [1.5, True])
def test_verify_llm_smoke_output_rejects_non_integer_parallel_mutations(
    tmp_path: Path,
    bad_value: object,
) -> None:
    _copy_real_smoke_bundle(tmp_path)
    plan_path = tmp_path / "real_llm_smoke_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for entry in plan["runs"]:
        if entry["variant"] == "branch_context":
            entry["parallel_mutations"] = bad_value
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("branch_context: plan parallel_mutations must be an integer" in issue for issue in payload["issues"])


@pytest.mark.parametrize("bad_value", ["abc", True])
def test_verify_llm_smoke_output_rejects_malformed_trace_max_workers(
    tmp_path: Path,
    bad_value: object,
) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_parallel_child_max_workers(tmp_path / "runs" / "smoke-branch_context-seed-0", bad_value)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("branch_context: trace max_workers must be an integer" in issue for issue in payload["issues"])


@pytest.mark.parametrize(
    ("bad_value", "expected_issue"),
    [
        ("bad", "branch_context: llm_calls.total must be an integer string"),
        (True, "branch_context: llm_calls.total must be an integer, not bool"),
        (0, "branch_context: llm_calls.total must be positive in real smoke"),
    ],
)
def test_verify_llm_smoke_output_rejects_malformed_llm_calls_total(
    tmp_path: Path,
    bad_value: object,
    expected_issue: str,
) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_run_metadata_llm_calls_total(tmp_path / "runs" / "smoke-branch_context-seed-0", bad_value)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any(expected_issue in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_deleted_ledger_call(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _delete_first_ledger_entry(tmp_path / "runs" / "smoke-branch_context-seed-0")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger call count" in issue and "does not match" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_ledger_provider_mismatch(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_first_ledger_field(tmp_path / "runs" / "smoke-branch_context-seed-0", "provider", "OtherProvider")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger providers" in issue and "manifest provider" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_manifest_model_mismatch(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = "other-model"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger models" in issue and "manifest model" in issue for issue in payload["issues"])


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("prompt_hash", "not-a-sha", "prompt_hash must be a lowercase sha256 hex string"),
        ("provider", "", "provider must be a non-empty string"),
        ("success", False, "success must be true for completed smoke evidence"),
        ("prompt", "raw prompt text", "contains forbidden raw field"),
        ("raw_messages", [], "contains forbidden raw field"),
        ("completion_text", "raw completion text", "contains forbidden raw field"),
        ("request_payload", {}, "contains forbidden raw field"),
        ("unknown_key", "extra", "contains unknown ledger field"),
        ("duration_s", float("nan"), "duration_s must be a finite number"),
        ("temperature", float("inf"), "temperature must be a finite number"),
        ("span_kind", "tool_span", "span_kind must be generation_span"),
        ("prompt_tokens_accounted", -1, "prompt_tokens_accounted must be >= 0"),
        ("prompt_token_source", "untrusted", "prompt_token_source must be local_estimate or provider_usage"),
        ("response_token_source", "untrusted", "response_token_source must be local_estimate or provider_usage"),
    ],
)
def test_verify_llm_smoke_output_rejects_invalid_ledger_entry_schema(
    tmp_path: Path,
    field: str,
    value: object,
    expected_issue: str,
) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_first_ledger_field(tmp_path / "runs" / "smoke-branch_context-seed-0", field, value)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any(expected_issue in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_accounted_prompt_mismatch_with_trace(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    run_dir = tmp_path / "runs" / "smoke-branch_context-seed-0"
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    entries = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extra_prompt_tokens = 9
    entries[0]["prompt_tokens_accounted"] += extra_prompt_tokens
    entries[0]["prompt_token_source"] = "provider_usage"
    ledger_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_budget = manifest["token_budget_final"]
    final_budget["prompt_tokens_used"] += extra_prompt_tokens
    cost_rate = final_budget.get("cost_per_1k_tokens_usd")
    if cost_rate is not None:
        final_budget["estimated_cost_usd"] = (
            final_budget["prompt_tokens_used"] + final_budget["output_tokens_used"]
        ) / 1000.0 * cost_rate
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any(
        "prompt_tokens_accounted does not match trace usage.prompt_tokens" in issue
        or "prompt_token_source requires trace usage.prompt_tokens" in issue
        for issue in payload["issues"]
    )


def test_verify_llm_smoke_output_rejects_partial_token_accounting_fields(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _delete_first_ledger_field(
        tmp_path / "runs" / "smoke-branch_context-seed-0",
        "prompt_token_source",
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("prompt_token_source is required" in issue for issue in payload["issues"])


def test_llm_token_accounting_accepts_matching_provider_trace_and_legacy_trace() -> None:
    matching_entry = {
        "call_id": "llm_call_000001",
        "prompt_tokens_accounted": 7,
        "prompt_token_source": "provider_usage",
        "response_token_estimate": 11,
        "response_token_source": "provider_usage",
    }
    span = {
        "llm_call_id": "llm_call_000001",
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 11,
            "total_tokens": 18,
        },
    }

    assert _llm_token_accounting_issues([matching_entry], [span], "branch_context") == []
    assert _llm_token_accounting_issues(
        [{"call_id": "llm_call_000001", "prompt_token_estimate": 1}],
        [{"llm_call_id": "llm_call_000001"}],
        "branch_context",
    ) == []


def test_llm_token_accounting_rejects_legacy_downgrade_with_provider_trace() -> None:
    issues = _llm_token_accounting_issues(
        [{"call_id": "llm_call_000001", "prompt_token_estimate": 1}],
        [
            {
                "llm_call_id": "llm_call_000001",
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 11,
                    "total_tokens": 18,
                },
            }
        ],
        "branch_context",
    )

    assert any("missing token accounting fields" in issue for issue in issues)


def test_llm_token_accounting_rejects_inconsistent_provider_total() -> None:
    issues = _llm_token_accounting_issues(
        [
            {
                "call_id": "llm_call_000001",
                "prompt_tokens_accounted": 7,
                "prompt_token_source": "provider_usage",
                "response_token_estimate": 11,
                "response_token_source": "provider_usage",
            }
        ],
        [
            {
                "llm_call_id": "llm_call_000001",
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 11,
                    "total_tokens": 99,
                },
            }
        ],
        "branch_context",
    )

    assert any("usage.total_tokens does not equal" in issue for issue in issues)


def test_verify_llm_smoke_output_rejects_missing_ledger_hash(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _delete_first_ledger_field(tmp_path / "runs" / "smoke-branch_context-seed-0", "response_hash")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("response_hash must be a lowercase sha256 hex string" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_duplicate_ledger_call_id(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _duplicate_second_ledger_call_id(tmp_path / "runs" / "smoke-branch_context-seed-0")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("call_id must be unique" in issue for issue in payload["issues"])
    assert any("call_id sequence must be contiguous" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_trace_call_id_mismatch(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_first_generation_span_metadata(
        tmp_path / "runs" / "smoke-branch_context-seed-0",
        "llm_call_id",
        "llm_call_999999",
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger call_id set does not match trace" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_trace_provider_mismatch(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_first_generation_span_metadata(
        tmp_path / "runs" / "smoke-branch_context-seed-0",
        "provider",
        "OtherProvider",
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger providers" in issue and "trace providers" in issue for issue in payload["issues"])
    assert any("ledger call fingerprints do not match trace" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_accepts_reordered_generation_spans(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _reverse_generation_span_order(tmp_path / "runs" / "smoke-branch_context-seed-0")

    verification = verify_llm_smoke_output(tmp_path)

    assert verification.passed is True


def test_verify_llm_smoke_output_rejects_missing_trace_call_id(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_first_generation_span_metadata(
        tmp_path / "runs" / "smoke-branch_context-seed-0",
        "llm_call_id",
        None,
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("every trace generation span must include llm_call_id" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_duplicate_trace_call_id(tmp_path: Path) -> None:
    _copy_real_smoke_bundle(tmp_path)
    _rewrite_second_generation_span_metadata_from_first(
        tmp_path / "runs" / "smoke-branch_context-seed-0",
        "llm_call_id",
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("trace llm_call_id values must be unique" in issue for issue in payload["issues"])


def test_cli_verify_smoke_llm_command(tmp_path: Path, cli_env: dict[str, str]) -> None:
    _copy_real_smoke_bundle(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-smoke-llm",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert Path(result.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_verification.json"


def test_smoke_gate_requires_branch_context_prompt_delivery(tmp_path: Path) -> None:
    run_dir = tmp_path
    child = run_dir / "solutions" / "solution_001" / "transcripts"
    child.mkdir(parents=True)
    (run_dir / "solutions" / "solution_001" / "branch_context.json").write_text(
        json.dumps(
            {
                "branch_intent": "features_or_architecture",
                "sibling_branch_ids": ["solution_002"],
                "diversity_instruction": "distinct branch",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (child / "proposal_debate.json").write_text(
        json.dumps(
            [
                {
                    "role": "proposer",
                    "prompt": "No branch context was delivered in this request.",
                    "response": "branch_intent sibling_branch_ids diversity_instruction",
                }
            ]
        ),
        encoding="utf-8",
    )
    (child / "engineer.json").write_text("[]", encoding="utf-8")
    (run_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "name": "agenticsciml.child_mutation.start",
                "metadata": {"branch_context": {"branch_intent": "features_or_architecture"}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    gate = _smoke_gate(
        run_dir,
        "branch_context",
        {"branch_context_enabled": True, "llm_calls": {"total": 1}},
        {"quality_gate": {"passed": True}},
        [{"node_id": "solution_001", "parent_id": "solution_000"}],
        {"branch:features_or_architecture"},
    )

    assert gate["passed"] is False
    assert any("request prompts missing" in issue for issue in gate["issues"])


def test_paired_contrast_gate_rejects_single_variant() -> None:
    gate = _paired_contrast_gate(
        [
            {
                "variant": "branch_context",
                "smoke_gate_passed": True,
                "branch_context_enabled": True,
                "llm_calls": 1,
                "branch_intents": "features_or_architecture",
            }
        ]
    )

    assert gate["passed"] is False
    assert "missing required variant no_branch_context" in gate["issues"]


def test_cli_smoke_llm_dry_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = {**cli_env}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context,no_branch_context",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    report_path = Path(result.stdout.strip().splitlines()[-1])
    assert report_path.name == "real_llm_smoke_report.md"
    assert (tmp_path / "real_llm_smoke_plan.json").exists()


def test_cli_smoke_llm_defaults_to_dry_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = {**cli_env}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert Path(result.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_report.md"
    plan = json.loads((tmp_path / "real_llm_smoke_plan.json").read_text(encoding="utf-8"))
    assert plan["execution_mode"] == "dry_run"
