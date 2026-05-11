import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agenticsciml.benchmarks import BenchmarkContractFactory
from agenticsciml.config import ExperimentConfig, EvolutionConfig
from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
)
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.state import SolutionNode, SolutionScore


FAILING_TRAIN_SOLUTION = r'''
from __future__ import annotations

import argparse

import numpy as np

MODEL_CHECKPOINT = "model.pkl"


class MODEL:
    def predict(self, x):
        return np.zeros((len(x), 1), dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    if args.mode == "validate":
        MODEL().predict(np.zeros((2, 1)))
        return
    if args.mode == "train":
        raise RuntimeError("boom during train")
    if args.mode == "predict":
        data = np.load(args.input)
        np.savez(args.output, predictions=MODEL().predict(data["x_val"]))


if __name__ == "__main__":
    main()
'''.strip()


class MalformedDebuggerLLM(MockLLMClient):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if schema_name == "root_engineer":
            return {"proposal": "failing train fixture", "code": FAILING_TRAIN_SOLUTION}
        if schema_name == "debugger":
            match = re.search(r"parent_digest:\s*([a-f0-9]{64})", prompt)
            return {
                "summary": "malformed patch fixture",
                "failure_kind": "runtime_error",
                "minimal_fix": True,
                "parent_digest": match.group(1) if match else "",
                "patch": "not a unified patch",
                "files_changed": ["solution.py"],
                "risks": ["fixture intentionally malformed"],
            }
        return super().complete_json(prompt, schema_name, system=system, temperature=temperature)


class MalformedEngineerLLM(MockLLMClient):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if schema_name == "engineer":
            match = re.search(r"parent_digest:\s*([a-f0-9]{64})", prompt)
            return {
                "mutation_summary": "malformed patch fixture",
                "expected_effect": "none",
                "risks": ["fixture intentionally malformed"],
                "parent_digest": match.group(1) if match else "",
                "patch": "not a unified patch",
                "files_changed": ["solution.py"],
            }
        return super().complete_json(prompt, schema_name, system=system, temperature=temperature)


def test_full_mock_pipeline_generates_tree_and_champion(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="mock-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = {event["event_type"] for event in trace_events}
    run_start = next(event for event in trace_events if event["name"] == "agenticsciml.run.start")
    run_end = next(event for event in trace_events if event["name"] == "agenticsciml.run.end")

    assert len(tree["nodes"]) >= 2
    child_nodes = [node for node in tree["nodes"] if node["parent_id"] is not None]
    assert child_nodes
    assert child_nodes[0]["method_tags"]
    assert child_nodes[0]["benchmark_name"] == "function_approx"
    assert child_nodes[0]["contract_hash"]
    assert "num_debug_attempts" in child_nodes[0]
    assert (run_dir / "leaderboard.csv").exists()
    assert (run_dir / "champion" / "solution.py").exists()
    assert (run_dir / "tree.mmd").exists()
    assert (run_dir / "trace_summary.json").exists()
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["run_state"] == "exported"
    assert run_metadata["evidence_mode"] == EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE
    assert run_metadata["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED
    assert run_metadata["benchmark_fidelity_level"] == "proxy"
    assert run_metadata["llm_calls"]["total"] >= 1
    assert run_metadata["llm_calls"]["by_role"]["proposer"] >= 1
    assert run_metadata["llm_calls"]["prompt_token_estimate"] > 0
    assert run_metadata["llm_calls"]["response_token_estimate"] > 0
    assert {"workflow_span", "tool_span", "agent_span", "generation_span", "guardrail_span"} <= event_types
    assert run_start["metadata"]["run_state"] == "partial"
    assert run_end["metadata"]["run_state"] == "exported"
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["phase"] == "completed"
    assert len(checkpoint["nodes"]) == len(tree["nodes"])


def test_parallel_mutations_run_as_parallel_child_jobs(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="parallel-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=2, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    parallel_starts = [
        event
        for event in trace_events
        if event["name"] == "agenticsciml.parallel_children.start"
        and event["metadata"].get("execution_mode") == "parallel"
    ]
    node_ids = [node["node_id"] for node in tree["nodes"]]

    assert len(tree["nodes"]) >= 4
    assert len(node_ids) == len(set(node_ids))
    assert parallel_starts
    assert parallel_starts[-1]["metadata"]["child_count"] == 2
    assert parallel_starts[-1]["metadata"]["max_workers"] == 2


def test_parallel_child_jobs_respect_parallel_mutation_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(
        experiment_id="parallel-budget-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=2, max_debug_retries=0),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)
    orchestrator.contract = contract
    orchestrator.nodes = [
        SolutionNode(
            node_id=f"solution_{index:03d}",
            parent_id=None,
            workspace=str(tmp_path / f"solution_{index:03d}"),
            score=SolutionScore("validation_mse", float(index + 1), higher_is_better=False),
            status="evaluated",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        )
        for index in range(4)
    ]

    def fake_create_child(
        parent: SolutionNode,
        contract_arg,
        solution_id: str | None = None,
    ) -> SolutionNode:
        assert contract_arg.contract_hash == contract.contract_hash
        assert solution_id is not None
        return SolutionNode(
            node_id=solution_id,
            parent_id=parent.node_id,
            workspace=str(tmp_path / solution_id),
            score=None,
            status="failed",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        )

    monkeypatch.setattr(orchestrator, "_create_child", fake_create_child)

    children = orchestrator._create_children_for_parents(orchestrator.nodes, contract)
    trace_events = [
        json.loads(line)
        for line in (orchestrator.storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start = next(event for event in trace_events if event["name"] == "agenticsciml.parallel_children.start")

    assert len(children) == 2
    assert [child.node_id for _, child in children] == ["solution_004", "solution_005"]
    assert start["metadata"]["child_count"] == 2
    assert start["metadata"]["max_workers"] == 2


def test_failed_child_exception_writes_minimum_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="failed-child-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)
    parent = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace=str(tmp_path / "solution_000"),
        score=None,
        status="evaluated",
        benchmark_name=contract.benchmark_name,
        contract_hash=contract.contract_hash,
    )

    child = orchestrator._failed_child_from_exception(parent, "solution_001", contract, RuntimeError("boom"))

    assert child.status == "failed"
    assert child.failure_kind == "orchestration_error"
    assert Path(child.proposal_path).exists()
    assert Path(child.analysis_path).exists()
    assert (Path(child.workspace) / "orchestration_error.md").exists()


def test_parallel_child_jobs_keep_mixed_success_failure_artifacts_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(
        experiment_id="parallel-mixed-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=2, max_debug_retries=0),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)
    orchestrator.contract = contract
    orchestrator.nodes = [
        SolutionNode(
            node_id=f"solution_{index:03d}",
            parent_id=None,
            workspace=str(tmp_path / f"solution_{index:03d}"),
            score=SolutionScore("validation_mse", float(index + 1), higher_is_better=False),
            status="evaluated",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        )
        for index in range(2)
    ]

    def fake_create_child(
        parent: SolutionNode,
        contract_arg,
        solution_id: str | None = None,
    ) -> SolutionNode:
        assert solution_id is not None
        if parent.node_id == "solution_000":
            raise RuntimeError("synthetic child failure")
        return SolutionNode(
            node_id=solution_id,
            parent_id=parent.node_id,
            workspace=str(tmp_path / solution_id),
            score=None,
            status="evaluated",
            benchmark_name=contract_arg.benchmark_name,
            contract_hash=contract_arg.contract_hash,
        )

    monkeypatch.setattr(orchestrator, "_create_child", fake_create_child)

    children = orchestrator._create_children_for_parents(orchestrator.nodes, contract)
    trace_events = [
        json.loads(line)
        for line in (orchestrator.storage.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    end_events = [
        event
        for event in trace_events
        if event["name"] == "agenticsciml.child_mutation.end"
    ]
    by_child = {child.node_id: child for _, child in children}

    assert [child.node_id for _, child in children] == ["solution_002", "solution_003"]
    assert by_child["solution_002"].status == "failed"
    assert by_child["solution_003"].status == "evaluated"
    assert Path(by_child["solution_002"].proposal_path).exists()
    assert (Path(by_child["solution_002"].workspace) / "orchestration_error.md").exists()
    assert {event["metadata"]["solution_id"] for event in end_events} == {"solution_002", "solution_003"}
    assert any(event["metadata"]["status"] == "failed" for event in end_events)
    assert any(event["metadata"]["status"] == "evaluated" for event in end_events)


def test_resume_continues_existing_solution_tree_without_rebuilding_root(tmp_path: Path) -> None:
    first_config = ExperimentConfig(
        experiment_id="resume-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
    )
    first_run_dir = AgenticSciMLOrchestrator(first_config, MockLLMClient()).run()
    root_solution = first_run_dir / "solutions" / "solution_000" / "solution.py"
    root_mtime = root_solution.stat().st_mtime_ns

    second_config = ExperimentConfig(
        experiment_id="resume-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
        resume=True,
    )
    second_run_dir = AgenticSciMLOrchestrator(second_config, MockLLMClient()).run()
    tree = json.loads((second_run_dir / "tree.json").read_text(encoding="utf-8"))
    trace_text = (second_run_dir / "trace.jsonl").read_text(encoding="utf-8")

    assert second_run_dir == first_run_dir
    assert root_solution.stat().st_mtime_ns == root_mtime
    assert len(tree["nodes"]) == 2
    assert "agenticsciml.resume.loaded" in trace_text


def test_resume_rejects_stale_evaluation_contract(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmarks" / "function_approx"
    shutil.copytree(Path("examples/function_approx").resolve(), benchmark_dir)
    first_config = ExperimentConfig(
        experiment_id="stale-contract-run",
        benchmark_dir=benchmark_dir,
        output_dir=tmp_path / "runs",
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    AgenticSciMLOrchestrator(first_config, MockLLMClient()).run()

    evaluate_path = benchmark_dir / "evaluate.py"
    evaluate_path.write_text(
        evaluate_path.read_text(encoding="utf-8") + "\n# stale contract detector\n",
        encoding="utf-8",
    )
    resume_config = ExperimentConfig(
        experiment_id="stale-contract-run",
        benchmark_dir=benchmark_dir,
        output_dir=tmp_path / "runs",
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="stale"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_requires_existing_evaluation_contract(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="missing-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    (run_dir / "evaluation_contract.json").unlink()
    resume_config = ExperimentConfig(
        experiment_id="missing-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="Cannot resume without evaluation contract"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_checkpoint_contract_mismatch(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="checkpoint-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["contract_hash"] = "wrong"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="checkpoint-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="Checkpoint contract hash mismatch"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_invalid_solution_node_schema(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="invalid-node-schema-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"][0]["status"] = "done"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="invalid-node-schema-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="Invalid checkpoint solution tree"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_invalid_solution_tree_graph(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="invalid-node-graph-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"][0]["parent_id"] = "solution_999"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="invalid-node-graph-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="parent_id references missing node"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_empty_solution_tree(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="empty-node-tree-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"] = []
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="empty-node-tree-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="must contain at least one node"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_invalid_solution_node_status_semantics(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="invalid-node-status-semantics-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"][0]["score"] = None
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="invalid-node-status-semantics-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="evaluated node must have a score"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_solution_node_artifact_path_escape(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="invalid-node-path-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"][0]["analysis_path"] = "/etc/passwd"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="invalid-node-path-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="analysis_path must be inside node workspace"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_solution_node_workspace_path_escape(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="invalid-workspace-path-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"][0]["workspace"] = str(tmp_path / "outside" / "solution_000")
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="invalid-workspace-path-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="workspace must be under run solutions directory"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_node_contract_mismatch(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="node-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"][0]["contract_hash"] = "wrong"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="node-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="Node solution_000 contract hash mismatch"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_missing_node_contract_hash(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="missing-node-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    del checkpoint["nodes"][0]["contract_hash"]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="missing-node-contract-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="missing required field contract_hash"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_missing_node_benchmark_name(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="missing-node-benchmark-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    del checkpoint["nodes"][0]["benchmark_name"]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="missing-node-benchmark-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="missing required field benchmark_name"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_cli_run_mock_pipeline(tmp_path: Path, cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "run",
            "examples/function_approx",
            "--mock",
            "--max-iterations",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    run_dir = Path(result.stdout.strip().splitlines()[-1])
    assert (run_dir / "leaderboard.csv").exists()


def test_cli_resume_existing_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    base_cmd = [
        sys.executable,
        "-m",
        "agenticsciml.cli",
        "run",
        "examples/function_approx",
        "--mock",
        "--max-iterations",
        "0",
        "--output-dir",
        str(tmp_path),
        "--experiment-id",
        "cli-resume",
    ]
    subprocess.run(base_cmd, check=True, text=True, capture_output=True, env=cli_env)
    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "run",
            "examples/function_approx",
            "--mock",
            "--resume",
            "--max-iterations",
            "1",
            "--output-dir",
            str(tmp_path),
            "--experiment-id",
            "cli-resume",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    run_dir = Path(resumed.stdout.strip().splitlines()[-1])
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    assert len(tree["nodes"]) == 2


def test_cli_trace_summary_prints_quality_gate(tmp_path: Path, cli_env: dict[str, str]) -> None:
    config = ExperimentConfig(
        experiment_id="trace-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    result = subprocess.run(
        [sys.executable, "-m", "agenticsciml.cli", "trace-summary", str(run_dir)],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    summary = json.loads(result.stdout)

    assert summary["quality_gate"]["passed"] is True
    assert summary["event_counts"]["workflow_span"] >= 1


def test_orchestrator_no_kb_mode_skips_retrieved_entry(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="no-kb-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(
            max_iterations=1,
            parallel_mutations=1,
            max_debug_retries=1,
            use_kb=False,
        ),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    child_workspace = run_dir / "solutions" / "solution_001"

    assert (child_workspace / "retrieval_query.txt").exists()
    assert not (child_workspace / "retrieved_kb.md").exists()


def test_orchestrator_random_kb_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    def run(experiment_id: str) -> str:
        config = ExperimentConfig(
            experiment_id=experiment_id,
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            evolution=EvolutionConfig(
                max_iterations=1,
                parallel_mutations=1,
                max_debug_retries=1,
                random_kb=True,
                random_seed=11,
            ),
            use_mock=True,
        )
        run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
        return (run_dir / "solutions" / "solution_001" / "retrieved_kb.md").read_text(
            encoding="utf-8"
        )

    assert run("random-kb-a") == run("random-kb-b")


def test_debugger_patch_error_is_recorded_without_aborting_run(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="bad-debugger-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=1),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MalformedDebuggerLLM()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    debugger_error = run_dir / "solutions" / "solution_000" / "debugger_error.md"
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")

    assert tree["nodes"][0]["status"] == "failed"
    assert tree["nodes"][0]["num_debug_attempts"] == 1
    assert debugger_error.exists()
    assert "PatchApplicationError" in debugger_error.read_text(encoding="utf-8")
    assert "not a unified patch" in (
        run_dir / "solutions" / "solution_000" / "transcripts" / "debugger.json"
    ).read_text(encoding="utf-8")
    assert "debugger:patch_application" in trace_text


def test_engineer_patch_error_creates_failed_child_without_aborting_run(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="bad-engineer-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=1),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MalformedEngineerLLM()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    child = next(node for node in tree["nodes"] if node["parent_id"] is not None)
    engineering_error = run_dir / "solutions" / child["node_id"] / "engineering_error.md"
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")

    assert child["status"] == "failed"
    assert child["failure_kind"] == "engineering_error"
    assert engineering_error.exists()
    assert "PatchApplicationError" in engineering_error.read_text(encoding="utf-8")
    assert "engineer:patch_application" in trace_text
