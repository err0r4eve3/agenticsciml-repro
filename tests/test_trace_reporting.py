import json
from pathlib import Path

from agenticsciml.evidence import EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE, LLM_MODE_MOCK, SCIENTIFIC_CLAIM_NOT_SUPPORTED
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


def test_trace_summary_checks_run_artifact_evidence_consistency(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = {
        "llm_mode": "mock",
        "benchmark_fidelity_level": "proxy",
        "evidence_mode": "mock_workflow_shape",
        "scientific_claim": "not_supported",
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


def test_trace_summary_checks_tree_and_checkpoint_contract_consistency(tmp_path: Path) -> None:
    run_dir = _write_consistent_run_artifacts(tmp_path / "run")

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["checked"] is True
    assert summary["artifact_consistency"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True


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
    (run_dir / "tree.json").unlink()
    (run_dir / "checkpoint.json").unlink()

    summary = summarize_trace(run_dir)

    assert summary["artifact_consistency"]["passed"] is True
    assert summary["quality_gate"]["passed"] is True


def _write_consistent_run_artifacts(run_dir: Path) -> Path:
    run_dir.mkdir()
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
        "score": None,
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
    (run_dir / "tree.json").write_text(json.dumps({"nodes": [node]}), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "phase": "completed",
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
            {"event_type": "workflow_span", "name": "agenticsciml.run.start", "metadata": metadata},
            {"event_type": "agent_span", "name": "proposer", "metadata": {}},
            {"event_type": "generation_span", "name": "proposer", "metadata": {}},
            {"event_type": "tool_span", "name": "train_and_evaluate", "metadata": {}},
            {"event_type": "guardrail_span", "name": "guard", "metadata": {"passed": True}},
        ],
    )
    return run_dir
