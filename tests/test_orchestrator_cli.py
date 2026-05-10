import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agenticsciml.config import ExperimentConfig, EvolutionConfig
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


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
    assert run_metadata["llm_calls"]["total"] >= 1
    assert run_metadata["llm_calls"]["by_role"]["proposer"] >= 1
    assert run_metadata["llm_calls"]["prompt_token_estimate"] > 0
    assert run_metadata["llm_calls"]["response_token_estimate"] > 0
    assert {"workflow_span", "tool_span", "agent_span", "generation_span", "guardrail_span"} <= event_types
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["phase"] == "completed"
    assert len(checkpoint["nodes"]) == len(tree["nodes"])


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
