import json
from pathlib import Path

from agenticsciml.reporting.trace_summary import summarize_trace, write_trace_summary


def _write_events(path: Path, events: list[dict]) -> None:
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
    assert "agent_span" in summary["quality_gate"]["missing_event_types"]
    assert summary["guardrail_failures"][0]["name"] == "sandbox"


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
