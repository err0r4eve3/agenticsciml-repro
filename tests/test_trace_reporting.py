import json
from pathlib import Path

import pytest

from agenticsciml.config import ExperimentConfig, EvolutionConfig
from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    LLM_MODE_MOCK,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
    claim_gate_for_run,
)
from agenticsciml.llm.budget import LLMBudget, RecordingLLMClient
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.reporting.sdk_trace_export import write_sdk_trace_export
from agenticsciml.reporting.trace_summary import summarize_trace, write_trace_summary


def _write_events(path: Path, events: list[dict]) -> None:
    for index, event in enumerate(events, start=1):
        event.setdefault("event_seq", index)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_trace_summary_passes_when_required_spans_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "start", "metadata": {}},
            {"event_type": "agent_span", "name": "proposer", "metadata": {"spec_role": "proposer"}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {"mode": "json"}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {"exit_code": 0}},
            {"event_type": "guardrail_span", "name": "proposer:structured_output", "metadata": {"passed": True}},
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["quality_gate"]["passed"] is True
    assert summary["quality_gate"]["status"] == "pass"
    assert summary["trace_quality_status"] == "pass"
    assert summary["event_counts"]["guardrail_span"] == 1
    assert summary["agent_roles"] == ["proposer"]


def test_trace_summary_fails_on_missing_spans_or_guardrail_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "start", "metadata": {}},
            {"event_type": "guardrail_span", "name": "sandbox", "metadata": {"passed": False}},
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["quality_gate"]["passed"] is False
    assert summary["quality_gate"]["status"] == "failed_incomplete"
    assert "agent_span" in summary["quality_gate"]["missing_event_types"]
    assert summary["guardrail_failures"][0]["name"] == "sandbox"
    assert summary["hard_guardrail_failures"][0]["name"] == "sandbox"


def test_trace_summary_marks_recovered_structured_output_as_degraded(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "start", "metadata": {}},
            {"event_type": "agent_span", "name": "engineer", "metadata": {"spec_role": "engineer"}},
            {"event_type": "generation_span", "name": "engineer", "metadata": {"mode": "json"}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {"exit_code": 0}},
            {
                "event_type": "guardrail_span",
                "name": "engineer:engineer:structured_output",
                "metadata": {
                    "passed": False,
                    "error": (
                        "LLM JSON call failed for engineer: RuntimeError: "
                        "Model did not return valid JSON for engineer"
                    ),
                },
            },
            {
                "event_type": "guardrail_span",
                "name": "engineer:engineer:structured_output",
                "metadata": {"passed": True},
            },
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["quality_gate"]["passed"] is True
    assert summary["quality_gate"]["status"] == "degraded_recovered"
    assert summary["trace_quality_status"] == "degraded_recovered"
    assert summary["quality_gate"]["structured_output_retry_count"] == 1
    assert summary["quality_gate"]["hard_guardrail_failure_count"] == 0
    assert summary["recoverable_guardrail_failures"][0]["name"] == "engineer:engineer:structured_output"


def test_trace_summary_fails_when_exported_real_run_ledger_is_missing(tmp_path: Path) -> None:
    experiment_id = "trace-missing-real-ledger"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    ledger_path.unlink()

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any(
        "ledger" in issue and "trace" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_ledger_and_metadata_downgrade(tmp_path: Path) -> None:
    experiment_id = "trace-ledger-metadata-downgrade"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    ledger_path.unlink()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("llm_calls")
    metadata.pop("llm_ledger_usage")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["checked"] is True
    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert summary["hard_guardrail_failures"] == []


@pytest.mark.parametrize("missing_field", ["llm_calls", "llm_ledger_usage"])
def test_trace_summary_rejects_current_real_metadata_field_removal(
    tmp_path: Path,
    missing_field: str,
) -> None:
    experiment_id = f"trace-real-metadata-missing-{missing_field}"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop(missing_field)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["checked"] is True
    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any(
        missing_field in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_real_llm_call_role_metadata_drift(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-call-role-drift"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["llm_calls"]["total"] == 4
    metadata["llm_calls"]["by_role"] = {"root_engineer": 4}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any(
        "run_metadata.llm_calls.by_role does not match trace" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_per_call_real_llm_role_swap(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-call-role-swap"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    evaluator = next(event for event in events if event.get("name") == "evaluator")
    root_engineer = next(event for event in events if event.get("name") == "root_engineer")
    evaluator["metadata"]["spec_role"] = "root_engineer"
    root_engineer["metadata"]["spec_role"] = "evaluator"
    _write_events(trace_path, events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any(
        "spec_role does not match event name" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_joint_real_llm_role_identity_swap(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-call-joint-role-swap"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    evaluator = next(event for event in events if event.get("name") == "evaluator")
    root_engineer = next(event for event in events if event.get("name") == "root_engineer")
    evaluator["name"] = evaluator["metadata"]["spec_role"] = "root_engineer"
    root_engineer["name"] = root_engineer["metadata"]["spec_role"] = "evaluator"
    _write_events(trace_path, events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert any(
        "role-call contract mismatch" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_current_calls_moved_before_schema_start(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-current-calls-before-schema-start"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in ledger_rows:
        row.pop("prompt_tokens_accounted", None)
        row.pop("prompt_token_source", None)
        row.pop("response_token_source", None)
    ledger_path.write_text(
        "\n".join(json.dumps(row) for row in ledger_rows) + "\n",
        encoding="utf-8",
    )
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    evaluator = next(event for event in events if event.get("name") == "evaluator")
    root_engineer = next(event for event in events if event.get("name") == "root_engineer")
    evaluator["name"] = evaluator["metadata"]["spec_role"] = "root_engineer"
    root_engineer["name"] = root_engineer["metadata"]["spec_role"] = "evaluator"
    for event in events:
        if event.get("event_type") == "generation_span":
            event["metadata"].pop("usage", None)
    start = next(event for event in events if event.get("name") == "agenticsciml.run.start")
    events.remove(start)
    last_generation_index = max(
        index
        for index, event in enumerate(events)
        if event.get("event_type") == "generation_span"
    )
    events.insert(last_generation_index + 1, start)
    for event_seq, event in enumerate(events, start=1):
        event["event_seq"] = event_seq
    _write_events(trace_path, events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert any(
        "current-accounting call precedes current evidence schema" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    [
        ("missing_trace_floor", "invalid ledger_calls_before"),
        ("history_floor_drift", "first v2 invocation must start before ledger call 1"),
        ("schema_downgrade", "trace/history invocation IDs do not match"),
        ("invocation_id_drift", "invalid invocation IDs"),
        ("final_after_drift", "final ledger boundary does not match ledger call count"),
    ],
)
def test_trace_summary_rejects_v2_invocation_ledger_boundary_drift(
    tmp_path: Path,
    mutation: str,
    issue_fragment: str,
) -> None:
    experiment_id = f"trace-v2-invocation-boundary-{mutation}"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    if mutation in {"missing_trace_floor", "schema_downgrade", "invocation_id_drift"}:
        trace_path = run_dir / "trace.jsonl"
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        start = next(event for event in events if event.get("name") == "agenticsciml.run.start")
        if mutation == "missing_trace_floor":
            start["metadata"].pop("ledger_calls_before")
        elif mutation == "schema_downgrade":
            start["metadata"]["llm_evidence_schema_version"] = 1
            metadata_path = run_dir / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["llm_evidence_schema_version"] = 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        else:
            start["metadata"]["invocation_id"] = "invocation_000002"
        _write_events(trace_path, events)
    if mutation in {"history_floor_drift", "invocation_id_drift", "final_after_drift"}:
        history_path = run_dir / "invocation_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if mutation == "history_floor_drift":
            history["invocations"][0]["ledger_calls_before"] = 1
        elif mutation == "invocation_id_drift":
            history["invocations"][0]["invocation_id"] = "invocation_000002"
        else:
            history["invocations"][0]["ledger_calls_after"] = 0
        history_path.write_text(json.dumps(history), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert any(
        issue_fragment in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_missing_current_real_llm_spec_role(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-call-role-missing"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    evaluator = next(event for event in events if event.get("name") == "evaluator")
    evaluator["metadata"].pop("spec_role")
    _write_events(trace_path, events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert any(
        "missing a valid spec_role" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_each_current_real_llm_call_summary_drift(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-call-summary-drift"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    baseline = json.loads(metadata_path.read_text(encoding="utf-8"))
    drifts = {
        "total": 5,
        "by_role": {"root_engineer": 4},
        "generation_attempt_count": 5,
        "generation_attempt_duration_s": baseline["llm_calls"]["generation_attempt_duration_s"] + 1,
        "unbound_generation_attempt_count": 1,
        "pre_provider_rejection_count": 1,
        "prompt_token_estimate": baseline["llm_calls"]["prompt_token_estimate"] + 1,
        "response_token_estimate": baseline["llm_calls"]["response_token_estimate"] + 1,
        "provider_usage": {
            **baseline["llm_calls"]["provider_usage"],
            "call_count": 3,
        },
        "duration_s": baseline["llm_calls"]["duration_s"] + 1,
    }

    for field, invalid_value in drifts.items():
        metadata = json.loads(json.dumps(baseline))
        metadata["llm_calls"][field] = invalid_value
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        summary = summarize_trace(run_dir)

        assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
        assert any(
            f"run_metadata.llm_calls.{field} does not match trace" in issue
            for issue in summary["artifact_consistency"]["issues"]
        )


def test_trace_summary_rejects_real_llm_evidence_schema_downgrade(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-call-summary-schema-downgrade"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["llm_calls"]["by_role"] = {"root_engineer": 4}
    metadata.pop("llm_evidence_schema_version")
    workflow_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in workflow_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("name") == "agenticsciml.run.start":
            event["metadata"].pop("llm_evidence_schema_version")
    _write_events(workflow_path, events)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any(
        "evidence schema version is missing or unsupported" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )
    assert any(
        "run_metadata.llm_calls.by_role does not match trace" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


@pytest.mark.parametrize("downgraded_mode", [None, "mock"])
def test_trace_summary_rejects_real_llm_mode_downgrade(
    tmp_path: Path,
    downgraded_mode: str | None,
) -> None:
    experiment_id = f"trace-real-mode-downgrade-{downgraded_mode}"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if downgraded_mode is None:
        metadata.pop("llm_mode")
    else:
        metadata["llm_mode"] = downgraded_mode
    for event in events:
        if event.get("name") == "agenticsciml.run.start":
            if downgraded_mode is None:
                event["metadata"].pop("llm_mode")
            else:
                event["metadata"]["llm_mode"] = downgraded_mode
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_events(trace_path, events)

    summary = summarize_trace(run_dir)

    ledger_trace = summary["artifact_consistency"]["real_llm_ledger_trace"]
    assert ledger_trace["checked"] is True
    assert ledger_trace["passed"] is False
    assert any(
        "real-only evidence requires llm_mode=real" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_removing_all_declared_real_llm_call_evidence(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-evidence-removed"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    ledger_path.unlink()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("llm_calls")
    metadata.pop("llm_ledger_usage")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        event.get("metadata", {}).pop("llm_call_id", None)
    _write_events(trace_path, events)

    summary = summarize_trace(run_dir)

    ledger_trace = summary["artifact_consistency"]["real_llm_ledger_trace"]
    assert ledger_trace["checked"] is True
    assert ledger_trace["passed"] is False
    assert summary["quality_gate"]["passed"] is False


def test_trace_summary_rejects_real_llm_call_summary_numeric_type_drift(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-call-summary-type-drift"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    baseline = json.loads(metadata_path.read_text(encoding="utf-8"))

    for field, invalid_value in (
        ("total", float(baseline["llm_calls"]["total"])),
        ("duration_s", float("nan")),
    ):
        metadata = json.loads(json.dumps(baseline))
        metadata["llm_calls"][field] = invalid_value
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        summary = summarize_trace(run_dir)
        assert any(
            f"run_metadata.llm_calls.{field} does not match trace" in issue
            for issue in summary["artifact_consistency"]["issues"]
        )

    metadata = json.loads(json.dumps(baseline))
    metadata["llm_calls"]["provider_usage"]["total_tokens"] = float(
        metadata["llm_calls"]["provider_usage"]["total_tokens"]
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    summary = summarize_trace(run_dir)
    assert any(
        "run_metadata.llm_calls.provider_usage does not match trace" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_non_finite_generation_duration(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-real-non-finite-generation-duration"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    generation = next(event for event in events if event.get("event_type") == "generation_span")
    generation["metadata"]["duration_s"] = float("inf")
    _write_events(trace_path, events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert any(
        "non-finite JSON constant" in issue
        or "Invalid trace" in issue
        or "generation trace cannot be summarized" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_rejects_root_only_evidence_for_evaluated_child(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-child-history-truncated"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=30)),
    ).run()
    ledger_rows = ledger_path.read_text(encoding="utf-8").splitlines()
    ledger_path.write_text("\n".join(ledger_rows[:4]) + "\n", encoding="utf-8")
    retained_call_ids = {f"llm_call_{index:06d}" for index in range(1, 5)}
    trace_path = run_dir / "trace.jsonl"
    trace_events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_events(
        trace_path,
        [
            event
            for event in trace_events
            if not (
                event.get("event_type") == "generation_span"
                and isinstance(event.get("metadata", {}).get("llm_call_id"), str)
                and event["metadata"]["llm_call_id"] not in retained_call_ids
            )
        ],
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["llm_calls"]["total"] = 4
    metadata["llm_ledger_usage"]["calls_used"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    history_path = run_dir / "invocation_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["invocations"][0]["ledger_calls_after"] = 4
    history_path.write_text(json.dumps(history), encoding="utf-8")

    summary = summarize_trace(run_dir)

    ledger_trace = summary["artifact_consistency"]["real_llm_ledger_trace"]
    assert ledger_trace["historical_call_floor"]["minimum_calls"] == 13
    assert ledger_trace["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert "at least 13" in ledger_trace["error"]


def test_trace_summary_accepts_legacy_call_floor_without_role_map(tmp_path: Path) -> None:
    experiment_id = "trace-legacy-call-floor"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["llm_historical_call_floor"]["schema_version"] = 1
    metadata["llm_historical_call_floor"].pop("minimum_calls_by_role")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True


@pytest.mark.parametrize("invalid_schema_version", [True, 1.0, 2.0])
def test_trace_summary_rejects_non_integer_call_floor_schema(
    tmp_path: Path,
    invalid_schema_version: object,
) -> None:
    experiment_id = f"trace-invalid-call-floor-{type(invalid_schema_version).__name__}"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["llm_historical_call_floor"]["schema_version"] = invalid_schema_version
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False


def test_trace_summary_rejects_local_token_usage_drift_from_ledger(
    tmp_path: Path,
) -> None:
    experiment_id = "trace-local-token-usage-drift"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["llm_ledger_usage"]["prompt_tokens_used"] += 1
    metadata["llm_ledger_usage"]["total_tokens_used"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any(
        "prompt_tokens_used does not match ledger" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


@pytest.mark.parametrize(
    "mutation",
    ["local_cost", "offset_calls", "aggregate_prompt", "cost_rate"],
)
def test_trace_summary_rejects_cost_or_aggregate_usage_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    experiment_id = f"trace-budget-usage-drift-{mutation}"
    config = ExperimentConfig(
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
    )
    ledger_path = tmp_path / experiment_id / "llm_call_ledger.jsonl"
    run_dir = AgenticSciMLOrchestrator(
        config,
        RecordingLLMClient(MockLLMClient(), ledger_path, LLMBudget(max_calls=10)),
    ).run()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "local_cost":
        metadata["llm_ledger_usage"]["estimated_cost_usd"] = 1.0
    elif mutation == "offset_calls":
        metadata["llm_ledger_usage"]["aggregate_offset"]["calls_used"] += 1
    elif mutation == "aggregate_prompt":
        metadata["llm_budget"]["prompt_tokens_used"] += 1
    else:
        metadata["llm_budget"]["cost_per_1k_tokens_usd"] = 0.01
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["real_llm_ledger_trace"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False


def test_trace_summary_counts_recovered_provider_timeout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "start", "metadata": {}},
            {"event_type": "agent_span", "name": "root_engineer", "metadata": {"spec_role": "root_engineer"}},
            {"event_type": "generation_span", "name": "root_engineer", "metadata": {"mode": "json"}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {"exit_code": 0}},
            {
                "event_type": "guardrail_span",
                "name": "root_engineer:root_engineer:structured_output",
                "metadata": {
                    "passed": False,
                    "error": "LLM JSON call failed for root_engineer: APITimeoutError: Request timed out.",
                },
            },
            {
                "event_type": "guardrail_span",
                "name": "root_engineer:root_engineer:structured_output",
                "metadata": {"passed": True},
            },
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["quality_gate"]["passed"] is True
    assert summary["quality_gate"]["status"] == "degraded_recovered"
    assert summary["quality_gate"]["provider_timeout_count"] == 1
    assert summary["quality_gate"]["structured_output_retry_count"] == 1


def test_trace_summary_treats_unrecovered_structured_output_as_hard_failure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "start", "metadata": {}},
            {"event_type": "agent_span", "name": "engineer", "metadata": {}},
            {"event_type": "generation_span", "name": "engineer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {
                "event_type": "guardrail_span",
                "name": "engineer:engineer:structured_output",
                "metadata": {
                    "passed": False,
                    "attempt": 1,
                    "error": "Model did not return valid JSON for engineer",
                },
            },
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["quality_gate"]["passed"] is False
    assert summary["quality_gate"]["status"] == "failed_hard"
    assert summary["recoverable_guardrail_failures"] == []
    assert summary["hard_guardrail_failures"][0]["name"] == (
        "engineer:engineer:structured_output"
    )


def test_trace_summary_requires_later_recovery_on_same_guardrail_boundary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    failed_name = "engineer:engineer:structured_output"
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "start", "metadata": {}},
            {"event_type": "agent_span", "name": "engineer", "metadata": {}},
            {"event_type": "generation_span", "name": "engineer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {
                "event_type": "guardrail_span",
                "name": failed_name,
                "metadata": {"passed": True, "attempt": 2},
            },
            {
                "event_type": "guardrail_span",
                "name": failed_name,
                "metadata": {
                    "passed": False,
                    "attempt": 1,
                    "error": "Model did not return valid JSON for engineer",
                },
            },
            {
                "event_type": "guardrail_span",
                "name": "critic:critic:structured_output",
                "metadata": {"passed": True, "attempt": 2},
            },
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["quality_gate"]["passed"] is False
    assert summary["quality_gate"]["status"] == "failed_hard"
    assert summary["recoverable_guardrail_failures"] == []


def test_write_trace_summary_creates_json_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "start", "metadata": {}},
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )

    path = write_trace_summary(run_dir)

    assert path == run_dir / "trace_summary.json"
    assert json.loads(path.read_text(encoding="utf-8"))["quality_gate"]["passed"] is True


def test_sdk_trace_export_preserves_numeric_token_counts_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_events(
        run_dir / "trace.jsonl",
        [
            {
                "event_type": "generation_span",
                "name": "engineer",
                "metadata": {
                    "prompt": "private prompt",
                    "api_token": "private-token",
                    "prompt_token_estimate": 17,
                    "response_token_estimate": 9,
                    "usage": {"input_tokens": 16, "output_tokens": 8},
                    "token_budget": {"max_total_tokens": 100},
                },
            }
        ],
    )

    payload = json.loads(write_sdk_trace_export(run_dir).read_text(encoding="utf-8"))
    metadata = payload["spans"][0]["metadata"]

    assert metadata["prompt"] == "<redacted>"
    assert metadata["api_token"] == "<redacted>"
    assert metadata["prompt_token_estimate"] == 17
    assert metadata["response_token_estimate"] == 9
    assert metadata["usage"] == {"input_tokens": 16, "output_tokens": 8}
    assert metadata["token_budget"] == {"max_total_tokens": 100}


def test_trace_summary_checks_run_artifact_evidence_consistency(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_valid_data_analysis(run_dir)
    metadata = {
        "llm_mode": "mock",
        "benchmark_fidelity_level": "proxy",
        "evidence_mode": "mock_workflow_shape",
        "scientific_claim": "not_supported",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "solutions" / "solution_000").mkdir(parents=True)
    (run_dir / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "benchmark_fidelity": {
                    "schema_version": 1,
                    "paper_task_name": "Proxy",
                    "paper_section": "S1.1",
                    "fidelity_level": "proxy",
                    "expected_runtime_s": 20,
                    "requires_torch": False,
                    "requires_gpu": False,
                    "paper_gap_notes": "proxy task",
                }
            }
        ),
        encoding="utf-8",
    )
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "agenticsciml.run.start", "metadata": metadata},
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["checked"] is True
    assert summary["artifact_consistency"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True


def test_trace_summary_fails_on_run_artifact_evidence_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "llm_mode": "mock",
                "benchmark_fidelity_level": "paper-like",
                "evidence_mode": "mock_workflow_shape",
                "scientific_claim": "not_supported",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "benchmark_fidelity": {
                    "schema_version": 1,
                    "paper_task_name": "Proxy",
                    "paper_section": "S1.1",
                    "fidelity_level": "proxy",
                    "expected_runtime_s": 20,
                    "requires_torch": False,
                    "requires_gpu": False,
                    "paper_gap_notes": "proxy task",
                }
            }
        ),
        encoding="utf-8",
    )
    _write_events(
        run_dir / "trace.jsonl",
        [
            {
                "event_type": "workflow_span",
                "name": "agenticsciml.run.start",
                "metadata": {
                    "llm_mode": "mock",
                    "benchmark_fidelity_level": "proxy",
                    "evidence_mode": "mock_workflow_shape",
                    "scientific_claim": "not_supported",
                },
            },
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["checked"] is True
    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("benchmark_fidelity_level" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_claim_gate_overclaim(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    claim_gate = claim_gate_for_run(
        claim_level="workflow_proxy",
        use_mock=True,
        fidelity_level="proxy",
    )
    metadata = {
        "llm_mode": "mock",
        "benchmark_fidelity_level": "proxy",
        "evidence_mode": "mock_workflow_shape",
        "scientific_claim": "not_supported",
        "claim_gate": claim_gate,
        "paper_level_claim_supported": True,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "benchmark_fidelity": {
                    "schema_version": 1,
                    "paper_task_name": "Proxy",
                    "paper_section": "S1.1",
                    "fidelity_level": "proxy",
                    "expected_runtime_s": 20,
                    "requires_torch": False,
                    "requires_gpu": False,
                    "paper_gap_notes": "proxy task",
                }
            }
        ),
        encoding="utf-8",
    )
    _write_events(
        run_dir / "trace.jsonl",
        [
            {"event_type": "workflow_span", "name": "agenticsciml.run.start", "metadata": metadata},
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["claim_gate"] == claim_gate
    assert summary["artifact_consistency"]["passed"] is False
    assert any("paper_level_claim_supported overclaims" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_detects_claim_gate_status_and_scientific_support_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_valid_data_analysis(run_dir)
    claim_gate = claim_gate_for_run(
        claim_level="workflow_proxy",
        use_mock=True,
        fidelity_level="proxy",
    )
    metadata = {
        "llm_mode": LLM_MODE_MOCK,
        "benchmark_fidelity_level": "proxy",
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "claim_gate": claim_gate,
    }
    workflow_gate = dict(claim_gate)
    workflow_gate["status"] = "blocked"
    workflow_gate["scientific_claim_supported"] = True
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "evaluation_contract.json").write_text(
        json.dumps({"benchmark_fidelity": {"fidelity_level": "proxy"}}),
        encoding="utf-8",
    )
    _write_events(
        run_dir / "trace.jsonl",
        [
            {
                "event_type": "workflow_span",
                "name": "agenticsciml.run.start",
                "metadata": {**metadata, "claim_gate": workflow_gate},
            },
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )

    summary = summarize_trace(run_dir)
    issues = summary["artifact_consistency"]["issues"]

    assert summary["quality_gate"]["passed"] is False
    assert any("claim_gate status" in issue for issue in issues)
    assert any("claim_gate scientific_claim_supported" in issue for issue in issues)


def test_trace_summary_exposes_workflow_claim_gate_without_run_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    claim_gate = claim_gate_for_run(
        claim_level="workflow_proxy",
        use_mock=True,
        fidelity_level="proxy",
    )
    _write_events(
        run_dir / "trace.jsonl",
        [
            {
                "event_type": "workflow_span",
                "name": "agenticsciml.run.start",
                "metadata": {"claim_gate": claim_gate},
            },
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )

    summary = summarize_trace(run_dir)

    assert summary["claim_gate"] == claim_gate
    assert summary["quality_gate"]["passed"] is True


def test_trace_summary_reports_data_analysis_specificity_warnings(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "reports").mkdir()
    (run_dir / "reports" / "data_analysis_structured.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_name": "function_approx",
                "training_array_keys": [],
                "task_specific_observations": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "llm_mode": LLM_MODE_MOCK,
                "benchmark_fidelity_level": "proxy",
                "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
                "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
            }
        ),
        encoding="utf-8",
    )
    _write_events(
        run_dir / "trace.jsonl",
        [
            {
                "event_type": "workflow_span",
                "name": "agenticsciml.run.start",
                "metadata": {
                    "llm_mode": LLM_MODE_MOCK,
                    "benchmark_fidelity_level": "proxy",
                    "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
                    "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
                },
            },
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )

    summary = summarize_trace(run_dir)

    specificity = summary["artifact_consistency"]["data_analysis_specificity"]
    assert specificity["checked"] is True
    assert specificity["passed"] is False
    assert "training_array_keys is missing" in specificity["warnings"]
    assert "task_specific_observations is missing" in specificity["warnings"]
    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("data_analysis_specificity" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_checks_tree_and_checkpoint_contract_consistency(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["checked"] is True
    assert summary["artifact_consistency"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True


def test_trace_summary_fails_when_data_analysis_specificity_is_missing(
    tmp_path: Path,
) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    (run_dir / "reports" / "data_analysis_structured.json").unlink()

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert summary["artifact_consistency"]["data_analysis_specificity"]["checked"] is False
    assert any(
        "reports/data_analysis_structured.json is missing" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_fails_on_tree_contract_mismatch(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    tree["nodes"][0]["contract_hash"] = "wrong-contract"
    (run_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("tree.json node solution_000 contract_hash" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_checkpoint_tree_node_set_mismatch(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["nodes"] = []
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("checkpoint.json nodes" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_missing_solution_tree_parent(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["parent_id"] = "solution_999"
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("parent_id references missing node" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_solution_node_required_field_is_missing(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        del artifact["nodes"][0]["status"]
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("missing required field status" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_solution_node_status_is_invalid(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["status"] = "done"
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("invalid status" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_solution_node_score_shape_is_invalid(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["score"] = {"metric": "", "value": "bad", "higher_is_better": "no"}
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("score.value must be a finite number" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_solution_node_score_is_not_finite(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["score"] = {
            "metric": "validation_mse",
            "value": float("inf"),
            "higher_is_better": False,
        }
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("score.value must be a finite number" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_solution_node_status_semantics_are_invalid(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["score"] = None
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("evaluated node must have a score" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_solution_node_has_unknown_fields(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["extra"] = "drift"
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("unknown fields: extra" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_unsupported_solution_tree_schema_version(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["schema_version"] = "solution_tree.v999"
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("unsupported schema_version" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_solution_node_artifact_path_escapes_run(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["analysis_path"] = "/etc/passwd"
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any(
        "analysis_path must be inside node workspace" in issue
        for issue in summary["artifact_consistency"]["issues"]
    )


def test_trace_summary_fails_on_invalid_solution_tree_child_link(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["children"] = ["solution_999"]
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("children references missing node" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_children_field_is_missing(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        del artifact["nodes"][0]["children"]
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("children is missing" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_duplicate_solution_tree_child_link(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    child_node = {
        "node_id": "solution_001",
        "parent_id": "solution_000",
        "workspace": str(run_dir / "solutions" / "solution_001"),
        "score": {"metric": "validation_mse", "value": 0.2, "higher_is_better": False},
        "children": [],
        "status": "evaluated",
        "proposal_path": None,
        "analysis_path": None,
        "error": None,
        "benchmark_name": "function_approx",
        "contract_hash": "a" * 64,
        "method_tags": [],
        "failure_kind": None,
        "score_delta_from_parent": None,
        "num_debug_attempts": 0,
    }
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["children"] = ["solution_001", "solution_001"]
        artifact["nodes"].append(child_node)
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["solution_count"] = 2
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("children contains duplicate node id" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_child_is_missing_from_parent_children(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    child_node = {
        "node_id": "solution_001",
        "parent_id": "solution_000",
        "workspace": str(run_dir / "solutions" / "solution_001"),
        "score": {"metric": "validation_mse", "value": 0.3, "higher_is_better": False},
        "children": [],
        "status": "evaluated",
        "proposal_path": None,
        "analysis_path": None,
        "error": None,
        "benchmark_name": "function_approx",
        "contract_hash": "a" * 64,
        "method_tags": [],
        "failure_kind": None,
        "score_delta_from_parent": None,
        "num_debug_attempts": 0,
    }
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"].append(child_node)
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["solution_count"] = 2
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("parent children does not include node exactly once" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_solution_tree_parent_cycle(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    child_node = {
        "node_id": "solution_001",
        "parent_id": "solution_000",
        "workspace": str(run_dir / "solutions" / "solution_001"),
        "score": {"metric": "validation_mse", "value": 0.4, "higher_is_better": False},
        "children": ["solution_000"],
        "status": "evaluated",
        "proposal_path": None,
        "analysis_path": None,
        "error": None,
        "benchmark_name": "function_approx",
        "contract_hash": "a" * 64,
        "method_tags": [],
        "failure_kind": None,
        "score_delta_from_parent": None,
        "num_debug_attempts": 0,
    }
    for artifact_name in ("tree.json", "checkpoint.json"):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact["nodes"][0]["parent_id"] = "solution_001"
        artifact["nodes"][0]["children"] = ["solution_001"]
        artifact["nodes"].append(child_node)
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["solution_count"] = 2
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("parent links contain a cycle" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_completed_run_is_missing_tree(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    (run_dir / "tree.json").unlink()

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("tree.json is required" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_completed_run_is_missing_checkpoint(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    (run_dir / "checkpoint.json").unlink()

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("checkpoint.json is required" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_requires_solution_artifacts_for_exported_run_state(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["run_state"] = "exported"
    del metadata["solution_count"]
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "tree.json").unlink()

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("tree.json is required" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_does_not_require_solution_artifacts_for_partial_run_state(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["run_state"] = "partial"
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for event in events:
        if event.get("name") == "agenticsciml.run.end":
            event["metadata"]["run_state"] = "partial"
    _write_events(run_dir / "trace.jsonl", events)
    (run_dir / "tree.json").unlink()
    (run_dir / "checkpoint.json").unlink()

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True


def test_trace_summary_fails_on_unknown_run_state(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["run_state"] = "done-ish"
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("run_state" in issue and "invalid" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_workflow_end_run_state_mismatch(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.run.end",
            "metadata": {"run_state": "partial"},
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("run_state" in issue and "workflow end" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_exported_run_is_missing_workflow_end(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [event for event in events if event.get("name") != "agenticsciml.run.end"]
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("workflow end" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_workflow_end_precedes_start(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = sorted(events, key=lambda event: 0 if event.get("name") == "agenticsciml.run.end" else 1)
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("workflow end precedes workflow start" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_conflicting_workflow_end_states(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.insert(
        1,
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.run.end",
            "metadata": {"run_state": "finalized"},
        },
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("conflicting workflow end run_state" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_exported_run_trace_is_missing_event_seq(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    del events[0]["event_seq"]
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("event_seq" in issue and "missing" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_on_non_monotonic_event_seq(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events[1]["event_seq"] = events[0]["event_seq"]
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("event_seq" in issue and "monotonic" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_trace_references_unknown_solution_node(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "tool_span",
            "name": "train_and_evaluate",
            "metadata": {"solution_id": "solution_999"},
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("unknown solution node" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_parent_child_map_references_unknown_node(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.parallel_children.start",
            "metadata": {"parent_to_child": {"solution_000": "solution_999"}},
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("unknown solution node" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_parent_children_map_references_unknown_node(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.parallel_children.start",
            "metadata": {"parent_to_children": {"solution_000": ["solution_001", "solution_999"]}},
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("unknown solution node" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_parent_children_map_is_malformed(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.parallel_children.start",
            "metadata": {
                "parent_ids": ["solution_000"],
                "child_ids": ["solution_001"],
                "parent_to_children": {"solution_000": "solution_001"},
            },
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("parent_to_children" in issue and "must be a list" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_parent_child_edges_mismatch_child_ids(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.parallel_children.start",
            "metadata": {
                "parent_ids": ["solution_000"],
                "unique_parent_ids": ["solution_000"],
                "child_ids": ["solution_001"],
                "parent_child_edges": [
                    {"slot_index": 0, "parent_id": "solution_000", "child_id": "solution_002"}
                ],
                "parent_to_children": {"solution_000": ["solution_001"]},
                "parent_to_child": {"solution_000": "solution_001"},
            },
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("parent_child_edges child order mismatch" in issue for issue in summary["artifact_consistency"]["issues"])
    assert any("parent_to_children mismatch" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_legacy_parent_to_child_is_not_last_child(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.parallel_children.start",
            "metadata": {
                "parent_ids": ["solution_000", "solution_000"],
                "unique_parent_ids": ["solution_000"],
                "child_ids": ["solution_001", "solution_002"],
                "parent_child_edges": [
                    {"slot_index": 0, "parent_id": "solution_000", "child_id": "solution_001"},
                    {"slot_index": 1, "parent_id": "solution_000", "child_id": "solution_002"},
                ],
                "parent_to_children": {"solution_000": ["solution_001", "solution_002"]},
                "parent_to_child": {"solution_000": "solution_001"},
            },
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("legacy mapping mismatch" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_parallel_fanout_schema_is_incomplete(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.parallel_children.start",
            "metadata": {
                "parent_ids": ["solution_000"],
                "child_ids": ["solution_001"],
                "parent_to_children": {"solution_000": ["solution_001"]},
            },
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("requires unique_parent_ids" in issue for issue in summary["artifact_consistency"]["issues"])
    assert any("requires parent_child_edges" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_legacy_parent_to_child_lacks_canonical_map(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.parallel_children.start",
            "metadata": {
                "parent_ids": ["solution_000"],
                "child_ids": ["solution_001"],
                "unique_parent_ids": ["solution_000"],
                "parent_child_edges": [
                    {"slot_index": 0, "parent_id": "solution_000", "child_id": "solution_001"}
                ],
                "parent_to_child": {"solution_000": "solution_001"},
            },
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("requires parent_to_children" in issue for issue in summary["artifact_consistency"]["issues"])
    assert any("requires canonical parent_to_children" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_ignores_non_solution_parent_id_metadata(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "tool_span",
            "name": "process.spawn",
            "metadata": {"parent_id": "pid-123", "child_id": "pid-456"},
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True
    assert summary["artifact_consistency"]["trace_node_reference_events_checked"] == 1
    assert summary["artifact_consistency"]["trace_node_reference_events_skipped"] == 7


def test_trace_summary_reports_trace_node_reference_check_counts(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["trace_node_reference_events_checked"] == 1
    assert summary["artifact_consistency"]["trace_node_reference_events_skipped"] == 6
    assert summary["artifact_consistency"]["trace_node_references_checked"] == 1
    assert summary["artifact_consistency"]["trace_node_reference_events_with_references"] == 1
    assert summary["artifact_consistency"]["trace_node_references_checked_by_name"] == {
        "train_and_evaluate": 1
    }
    assert summary["artifact_consistency"]["trace_node_reference_node_coverage"] == {
        "total_nodes": 1,
        "referenced": ["solution_000"],
        "unreferenced": [],
        "nodes": {
            "solution_000": {
                "referenced_by_event_names": ["train_and_evaluate"],
                "reference_keys": ["solution_id"],
                "self_reference_count": 1,
                "relation_reference_count": 0,
            }
        },
    }
    assert summary["artifact_consistency"]["trace_node_lifecycle_stage_coverage"] == {
        "nodes": {
            "solution_000": {
                "stages": ["evaluated", "materialized"],
                "stage_events": {
                    "evaluated": ["train_and_evaluate"],
                    "materialized": ["root_engineer"],
                },
                "stage_event_seqs": {
                    "evaluated": [6],
                    "materialized": [5],
                },
            }
        }
    }
    assert summary["artifact_consistency"]["trace_node_reference_events_checked_by_name"] == {
        "train_and_evaluate": 1
    }
    assert summary["artifact_consistency"]["trace_node_reference_events_skipped_by_name"] == {
        "agenticsciml.run.end": 1,
        "agenticsciml.run.start": 1,
        "guard": 1,
        "proposer": 2,
        "root_engineer": 1,
    }


def test_trace_summary_fails_when_exported_run_checks_zero_solution_reference_events(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [event for event in events if event.get("name") != "train_and_evaluate"]
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert summary["artifact_consistency"]["trace_node_reference_events_checked"] == 0
    assert any("no solution-reference trace events" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_exported_node_is_not_referenced_by_trace(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    child_node = {
        "node_id": "solution_001",
        "parent_id": "solution_000",
        "workspace": str(run_dir / "solutions" / "solution_001"),
        "score": None,
        "children": [],
        "status": "failed",
        "proposal_path": None,
        "analysis_path": None,
        "error": "mutation failed before trace reference",
        "benchmark_name": "function_approx",
        "contract_hash": "a" * 64,
        "method_tags": [],
        "failure_kind": "mutation_error",
        "score_delta_from_parent": None,
        "num_debug_attempts": 0,
    }
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    tree["nodes"].append(child_node)
    (run_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["nodes"].append(child_node)
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["solution_count"] = 2
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert summary["artifact_consistency"]["trace_node_reference_node_coverage"]["unreferenced"] == [
        "solution_001"
    ]
    assert any("solution nodes have no allowlisted trace references" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_exported_node_only_has_relation_reference(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    child_node = {
        "node_id": "solution_001",
        "parent_id": "solution_000",
        "workspace": str(run_dir / "solutions" / "solution_001"),
        "score": {"metric": "validation_mse", "value": 0.1, "higher_is_better": False},
        "children": [],
        "status": "evaluated",
        "proposal_path": None,
        "analysis_path": None,
        "error": None,
        "benchmark_name": "function_approx",
        "contract_hash": "a" * 64,
        "method_tags": [],
        "failure_kind": None,
        "score_delta_from_parent": None,
        "num_debug_attempts": 0,
    }
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    tree["nodes"].append(child_node)
    (run_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["nodes"].append(child_node)
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["solution_count"] = 2
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.child_mutation.end",
            "metadata": {"parent_id": "solution_001"},
        }
    )
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    child_detail = summary["artifact_consistency"]["trace_node_reference_node_coverage"]["nodes"]["solution_001"]
    assert child_detail["relation_reference_count"] == 1
    assert child_detail["self_reference_count"] == 0
    assert any("solution nodes have no self trace references" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_evaluated_node_has_no_evaluated_stage(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [event for event in events if event.get("name") != "train_and_evaluate"]
    for event in events:
        event.pop("event_seq", None)
    events.append(
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.child_mutation.start",
            "metadata": {"solution_id": "solution_000"},
        }
    )
    events.append({"event_type": "tool_span", "name": "process.spawn", "metadata": {}})
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert summary["artifact_consistency"]["trace_node_lifecycle_stage_coverage"]["nodes"]["solution_000"][
        "stages"
    ] == ["created", "materialized"]
    assert any("evaluated solution nodes have no evaluated trace stage" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_evaluated_root_has_no_materialized_stage(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [event for event in events if event.get("name") != "root_engineer"]
    for event in events:
        event.pop("event_seq", None)
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert summary["artifact_consistency"]["trace_node_lifecycle_stage_coverage"]["nodes"]["solution_000"][
        "stages"
    ] == ["evaluated"]
    assert any("solution nodes missing required lifecycle stages" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_lifecycle_stage_order_is_invalid(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root_engineer_event = next(event for event in events if event.get("name") == "root_engineer")
    events = [event for event in events if event.get("name") != "root_engineer"]
    for event in events:
        event.pop("event_seq", None)
    root_engineer_event.pop("event_seq", None)
    events.append(root_engineer_event)
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert any("solution nodes have invalid lifecycle stage order" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_fails_when_exported_run_checks_no_actual_solution_node_references(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for event in events:
        if event.get("name") == "train_and_evaluate":
            event["metadata"] = {}
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is False
    assert summary["quality_gate"]["passed"] is False
    assert summary["artifact_consistency"]["trace_node_reference_events_checked"] == 1
    assert summary["artifact_consistency"]["trace_node_references_checked"] == 0
    assert any("no solution node references were checked" in issue for issue in summary["artifact_consistency"]["issues"])


def test_trace_summary_allows_partial_run_with_zero_solution_reference_events(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["run_state"] = "partial"
    del metadata["solution_count"]
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [
        event
        for event in events
        if event.get("name") not in {"train_and_evaluate", "agenticsciml.run.end"}
    ]
    for event in events:
        event.pop("event_seq", None)
    events.append({"event_type": "workflow_span", "name": "agenticsciml.run.end", "metadata": {"run_state": "partial"}})
    events.append({"event_type": "tool_span", "name": "process.spawn", "metadata": {}})
    _write_events(run_dir / "trace.jsonl", events)

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True
    assert summary["artifact_consistency"]["trace_node_reference_events_checked"] == 0
    assert summary["artifact_consistency"]["trace_node_references_checked"] == 0
    assert not any("no solution-reference trace events" in issue for issue in summary["artifact_consistency"]["issues"])


def _write_consistent_run_artifacts(run_dir: Path) -> Path:
    run_dir.mkdir()
    _write_valid_data_analysis(run_dir)
    contract_hash = "a" * 64
    benchmark_name = "function_approx"
    metadata = {
        "run_state": "exported",
        "llm_mode": LLM_MODE_MOCK,
        "benchmark_fidelity_level": "proxy",
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "solution_count": 1,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "solutions" / "solution_000").mkdir(parents=True)
    (run_dir / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "benchmark_name": benchmark_name,
                "contract_hash": contract_hash,
                "benchmark_fidelity": {
                    "schema_version": 1,
                    "paper_task_name": "Proxy",
                    "paper_section": "S1.1",
                    "fidelity_level": "proxy",
                    "expected_runtime_s": 20,
                    "requires_torch": False,
                    "requires_gpu": False,
                    "paper_gap_notes": "proxy task",
                },
            }
        ),
        encoding="utf-8",
    )
    node = {
        "node_id": "solution_000",
        "parent_id": None,
        "workspace": str(run_dir / "solutions" / "solution_000"),
        "score": {"metric": "validation_mse", "value": 0.1, "higher_is_better": False},
        "children": [],
        "status": "evaluated",
        "proposal_path": None,
        "analysis_path": None,
        "error": None,
        "benchmark_name": benchmark_name,
        "contract_hash": contract_hash,
        "method_tags": ["root_baseline"],
        "failure_kind": None,
        "score_delta_from_parent": None,
        "num_debug_attempts": 0,
    }
    (run_dir / "tree.json").write_text(
        json.dumps({"schema_version": "solution_tree.v1", "nodes": [node]}),
        encoding="utf-8",
    )
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "phase": "completed",
                "schema_version": "solution_tree.v1",
                "experiment_id": "trace-run",
                "benchmark_name": benchmark_name,
                "contract_hash": contract_hash,
                "nodes": [node],
                "analysis_node_ids": [],
            }
        ),
        encoding="utf-8",
    )
    _write_events(
        run_dir / "trace.jsonl",
        [
            {
                "event_type": "workflow_span",
                "name": "agenticsciml.run.start",
                "metadata": {**metadata, "run_state": "partial"},
            },
            {"event_type": "workflow_span", "name": "agenticsciml.run.end", "metadata": {"run_state": "exported"}},
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "agent_span", "name": "root_engineer", "metadata": {"solution_id": "solution_000"}},
            {
                "event_type": "tool_span",
                "name": "train_and_evaluate",
                "metadata": {"solution_id": "solution_000"},
            },
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )
    return run_dir


def _write_valid_data_analysis(run_dir: Path) -> None:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "data_analysis_structured.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_name": "function_approx",
                "benchmark_family": "regression",
                "evaluation_metric": "validation_mse",
                "training_array_keys": ["x_train", "y_train"],
                "task_specific_observations": [
                    "function_approx regression uses x_train and y_train with validation_mse"
                ],
            }
        ),
        encoding="utf-8",
    )
