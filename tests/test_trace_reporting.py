import json
from pathlib import Path

from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    LLM_MODE_MOCK,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
    claim_gate_for_run,
)
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

    assert summary["artifact_consistency"]["passed"] is False
    assert any("paper_level_claim_supported overclaims" in issue for issue in summary["artifact_consistency"]["issues"])


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
