from pathlib import Path
from typing import Any

import pytest

from agenticsciml.agents.debugger import DebuggerAgent
from agenticsciml.agents.engineer import EngineerAgent
from agenticsciml.agents.root_engineer import RootEngineerAgent
from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.llm.base import LLMClient
from agenticsciml.patching import (
    PatchApplicationError,
    apply_unified_patch,
    make_unified_patch,
    solution_digest,
)
from agenticsciml.state import AnalysisReport, Proposal
from agenticsciml.storage import ExperimentStorage


class RecordingContextLLM(LLMClient):
    def __init__(self):
        self.prompts: list[str] = []

    def complete_text(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        return "ok"

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.prompts.append(prompt)
        if schema_name == "root_engineer":
            return {"proposal": "baseline", "code": "class MODEL:\n    pass\n"}
        if schema_name == "debugger":
            return {
                "summary": "replace broken line",
                "failure_kind": "runtime_error",
                "minimal_fix": "replace broken line",
                "parent_digest": solution_digest("broken\n"),
                "patch": make_unified_patch("broken\n", "fixed\n"),
                "files_changed": ["solution.py"],
                "risks": ["minimal fixture patch only"],
            }
        return {
            "mutation_summary": "replace implementation",
            "expected_effect": "lower score",
            "risks": ["patch may fail"],
            "parent_digest": solution_digest("old\n"),
            "patch": "--- solution.py\n+++ solution.py\n@@ -1 +1 @@\n-old\n+new\n",
            "files_changed": ["solution.py"],
        }


def _bundle_and_contract():
    bundle = ProblemBundle.load(Path("examples/function_approx").resolve())
    contract = BenchmarkContractFactory.create_contract(bundle)
    guidelines = (bundle.benchmark_dir / "guidelines.md").read_text(encoding="utf-8")
    return bundle, contract, guidelines


def test_root_engineer_prompt_includes_problem_contract_guidelines_and_data_report(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingContextLLM()
    bundle, contract, guidelines = _bundle_and_contract()

    RootEngineerAgent(llm, storage).generate(
        solution_id="solution_000",
        problem_bundle=bundle,
        contract=contract,
        guidelines=guidelines,
        data_report="Data report: discontinuity and oscillation.",
    )

    prompt = llm.prompts[-1]
    assert "Problem.md" in prompt
    assert "EvaluationContract JSON" in prompt
    assert contract.contract_hash in prompt
    assert "Data report: discontinuity" in prompt
    assert "solution.py must define class MODEL" in prompt


def test_engineer_prompt_includes_context_and_applies_patch(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingContextLLM()
    bundle, contract, guidelines = _bundle_and_contract()
    workspace = storage.create_solution_workspace("solution_001")

    code = EngineerAgent(llm, storage).mutate(
        solution_id="solution_001",
        parent_code="old\n",
        proposal=Proposal(
            title="change",
            diagnosis="old is weak",
            mutation_plan=["replace old with new"],
            expected_effect="better",
            risks=["none"],
        ),
        problem_bundle=bundle,
        contract=contract,
        guidelines=guidelines,
        parent_analysis=AnalysisReport(node_id="solution_000", summary="Parent underfits."),
    )

    prompt = llm.prompts[-1]
    assert "parent_digest:" in prompt
    assert contract.contract_hash in prompt
    assert "Parent underfits" in prompt
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "new\n"
    assert code == "new\n"


def test_engineer_rejects_wrong_parent_digest(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = EngineerAgent(RecordingContextLLM(), storage)

    with pytest.raises(PatchApplicationError, match="parent_digest"):
        agent.apply_mutation_output(
            solution_id="solution_001",
            parent_code="old\n",
            response={
                "mutation_summary": "bad",
                "expected_effect": "none",
                "risks": [],
                "parent_digest": "wrong",
                "patch": "--- solution.py\n+++ solution.py\n@@ -1 +1 @@\n-old\n+new\n",
                "files_changed": ["solution.py"],
            },
        )


def test_malformed_patch_fails_clearly() -> None:
    with pytest.raises(PatchApplicationError):
        apply_unified_patch("old\n", "not a unified patch")


def test_debugger_prompt_includes_contract_context_and_applies_patch(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    llm = RecordingContextLLM()
    bundle, contract, guidelines = _bundle_and_contract()
    workspace = storage.create_solution_workspace("solution_001")
    (workspace / "solution.py").write_text("broken\n", encoding="utf-8")

    changed = DebuggerAgent(llm, storage).debug(
        "solution_001",
        workspace,
        "RuntimeError: broken training",
        problem_bundle=bundle,
        contract=contract,
        guidelines=guidelines,
        failure_phase="train",
    )

    prompt = llm.prompts[-1]
    assert changed is True
    assert "Failure Phase" in prompt
    assert "parent_digest:" in prompt
    assert contract.contract_hash in prompt
    assert "ProblemBundle Summary" in prompt
    assert (workspace / "solution.py").read_text(encoding="utf-8") == "fixed\n"


def test_debugger_rejects_wrong_parent_digest(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = DebuggerAgent(RecordingContextLLM(), storage)

    with pytest.raises(PatchApplicationError, match="parent_digest"):
        agent.apply_debug_output(
            "solution_001",
            "old\n",
            {
                "summary": "bad",
                "failure_kind": "runtime_error",
                "minimal_fix": "bad digest fixture",
                "parent_digest": "wrong",
                "patch": make_unified_patch("old\n", "new\n"),
                "files_changed": ["solution.py"],
                "risks": [],
            },
        )


def test_debugger_rejects_malformed_patch(tmp_path: Path) -> None:
    storage = ExperimentStorage.create(tmp_path, "demo")
    agent = DebuggerAgent(RecordingContextLLM(), storage)

    with pytest.raises(PatchApplicationError):
        agent.apply_debug_output(
            "solution_001",
            "old\n",
            {
                "summary": "bad",
                "failure_kind": "runtime_error",
                "minimal_fix": "malformed patch fixture",
                "parent_digest": solution_digest("old\n"),
                "patch": "not a unified patch",
                "files_changed": ["solution.py"],
                "risks": [],
            },
        )
