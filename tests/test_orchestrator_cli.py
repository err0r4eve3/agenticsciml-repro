import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agenticsciml.benchmarks import BenchmarkContractFactory
from agenticsciml.config import AgentConfig, ExperimentConfig, EvolutionConfig
from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
)
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.capabilities import ProviderCapabilities, capabilities_for_openai_compatible
from agenticsciml.orchestrator import AgenticSciMLOrchestrator, EvaluationApprovalRequired
from agenticsciml.state import AnalysisReport, SolutionNode, SolutionScore


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
                "minimal_fix": "fixture intentionally returns a malformed unified diff",
                "parent_digest": match.group(1) if match else "",
                "patch": "not a unified patch",
                "files_changed": ["solution.py"],
                "risks": ["fixture intentionally malformed"],
                "full_file_map": {"not_solution.py": "still malformed"},
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
                "full_file_map": {"not_solution.py": "still malformed"},
            }
        return super().complete_json(prompt, schema_name, system=system, temperature=temperature)


class DuplicateEngineerLLM(MockLLMClient):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if schema_name == "engineer":
            match = re.search(r"parent_digest:\s*([a-f0-9]{64})", prompt)
            parent_code = prompt.split("Parent code:\n", 1)[1] if "Parent code:\n" in prompt else ""
            return {
                "mutation_summary": "Duplicate parent code for plateau audit fixture.",
                "expected_effect": "No score movement expected.",
                "risks": ["Deliberately duplicate code"],
                "parent_digest": match.group(1) if match else "",
                "patch": "",
                "files_changed": ["solution.py"],
                "full_file_map": {"solution.py": parent_code if parent_code.endswith("\n") else parent_code + "\n"},
                "implemented_kb_points": [],
            }
        return super().complete_json(prompt, schema_name, system=system, temperature=temperature)


class CommentOnlyEngineerLLM(MockLLMClient):
    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if schema_name == "engineer":
            match = re.search(r"parent_digest:\s*([a-f0-9]{64})", prompt)
            parent_code = prompt.split("Parent code:\n", 1)[1] if "Parent code:\n" in prompt else ""
            child_code = parent_code if parent_code.endswith("\n") else parent_code + "\n"
            child_code += "# plateau audit fixture: code digest changes but behavior does not.\n"
            return {
                "mutation_summary": "Comment-only mutation for plateau audit fixture.",
                "expected_effect": "No score movement expected despite a changed digest.",
                "risks": ["Deliberately behavior-preserving code change"],
                "parent_digest": match.group(1) if match else "",
                "patch": "",
                "files_changed": ["solution.py"],
                "full_file_map": {"solution.py": child_code},
                "implemented_kb_points": [],
            }
        return super().complete_json(prompt, schema_name, system=system, temperature=temperature)


class VisionAuditLLM(MockLLMClient):
    model = "fake-vision-real"
    provider_name = "FakeVisionProvider"
    adapter_type = "openai_native_responses"
    provider_capabilities = ProviderCapabilities(
        provider="FakeVisionProvider",
        adapter_type="openai_native_responses",
        supports_responses=True,
        supports_structured_outputs=True,
        supports_image_inputs=True,
        supports_usage=True,
        supports_trace_export=True,
        supports_prompt_cache=False,
    )

    def __init__(self) -> None:
        self.image_calls: list[list[Path]] = []
        self.last_call_metadata: dict[str, Any] | None = None

    def complete_json_with_images(
        self,
        prompt: str,
        schema_name: str,
        image_paths: list[Path],
        system: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        assert schema_name == "visual_audit"
        self.image_calls.append(list(image_paths))
        self.last_call_metadata = {
            "provider": self.provider_name,
            "model": self.model,
            "method": "complete_json_with_images",
            "schema_name": schema_name,
            "adapter_type": self.adapter_type,
            "provider_capabilities": self.provider_capabilities.to_dict(),
            "image_input_count": len(image_paths),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return {
            "summary": "Fake vision provider reviewed the diagnostic image artifacts.",
            "physical_consistency_checks": ["image_input_received", "prediction_only_no_private_labels"],
            "visual_artifacts_reviewed": [path.name for path in image_paths],
            "warnings": [],
            "actual_image_inputs_used": True,
            "analysis_mode": "real_visual_provider_image_input",
        }


class ProviderAwareMockLLM(MockLLMClient):
    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: str = "test-key",
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.provider_capabilities = capabilities_for_openai_compatible(base_url)
        self.provider_name = self.provider_capabilities.provider
        self.adapter_type = self.provider_capabilities.adapter_type


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
    assert (run_dir / "reports" / "data_observations.json").exists()
    assert (run_dir / "reports" / "data_overview.svg").exists()
    assert (run_dir / "reports" / "data_eda.py").exists()
    assert (run_dir / "reports" / "data_eda.json").exists()
    assert (run_dir / "reports" / "data_analysis_structured.json").exists()
    assert (run_dir / "reports" / "evolution_health.json").exists()
    assert (run_dir / "reports" / "innovation_report.json").exists()
    assert (run_dir / "reports" / "innovation_report.md").exists()
    assert (run_dir / "reports" / "scientific_discovery_readiness.json").exists()
    assert (run_dir / "reports" / "scientific_discovery_readiness.md").exists()
    assert (run_dir / "reports" / "scientific_result_card.json").exists()
    assert (run_dir / "reports" / "scientific_result_card.md").exists()
    assert (run_dir / "reports" / "visual_audit_manifest.json").exists()
    assert (run_dir / "reports" / "domain_approval.json").exists()
    assert (run_dir / "reports" / "paper_like_benchmark_dossier.json").exists()
    assert (run_dir / "reports" / "selector_heterogeneity.json").exists()
    assert (run_dir / "reports" / "multi_seed_ablation_evidence.json").exists()
    assert (run_dir / "reports" / "method_experience_cache.json").exists()
    assert (run_dir / "run_inputs" / "manifest.json").exists()
    assert (run_dir / "solutions" / "solution_000" / "solution_observations.json").exists()
    assert (run_dir / "solutions" / "solution_000" / "visual_audit_report.json").exists()
    assert (run_dir / "solutions" / "solution_000" / "method_experience_record.json").exists()
    assert (run_dir / "solutions" / "solution_000" / "prediction_overview.svg").exists()
    assert (run_dir / "solutions" / "solution_000" / "emergence_report.json").exists()
    assert not (run_dir / "solutions" / "solution_000" / "private_eval").exists()
    assert not (run_dir / "solutions" / "solution_000" / "val_data.npz").exists()
    assert (run_dir / "trace_summary.json").exists()
    assert (run_dir / "openai_sdk_trace.json").exists()
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
    child_emergence = json.loads(
        (run_dir / "solutions" / child_nodes[0]["node_id"] / "emergence_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert child_emergence["auditor_version"] == "emergence_audit.v1"
    assert child_emergence["claim_level"] != "proved_emergent_discovery"
    child_workspace = run_dir / "solutions" / child_nodes[0]["node_id"]
    kb_report = json.loads((child_workspace / "kb_application_report.json").read_text(encoding="utf-8"))
    mutation_report = json.loads((child_workspace / "mutation_effect_report.json").read_text(encoding="utf-8"))
    operator_assignment = json.loads((child_workspace / "operator_assignment.json").read_text(encoding="utf-8"))
    evolution_health = json.loads((run_dir / "reports" / "evolution_health.json").read_text(encoding="utf-8"))
    innovation_report = json.loads((run_dir / "reports" / "innovation_report.json").read_text(encoding="utf-8"))
    scientific_result_card = json.loads(
        (run_dir / "reports" / "scientific_result_card.json").read_text(encoding="utf-8")
    )
    assert kb_report["status"] in {"retrieved_only", "proposed", "implemented", "unverified"}
    assert mutation_report["status"] in {"changed_score_moved", "changed_but_score_plateau", "duplicate_parent"}
    assert operator_assignment["scheduler_mode"] == "auto-audited"
    assert mutation_report["operator_id"] == operator_assignment["operator_id"]
    assert mutation_report["mutation_axis"] == operator_assignment["mutation_axis"]
    assert any(tag.startswith("operator:") for tag in child_nodes[0]["method_tags"])
    assert any(tag.startswith("axis:") for tag in child_nodes[0]["method_tags"])
    assert evolution_health["solution_count"] == len(tree["nodes"])
    assert operator_assignment["operator_id"] in evolution_health["operator_health"]
    assert evolution_health["operator_scheduler_mode"] == "auto-audited"
    assert evolution_health["operator_assignment_count"] == len(child_nodes)
    assert evolution_health["operator_assignment_expected_count"] == len(child_nodes)
    assert evolution_health["missing_operator_assignment_nodes"] == []
    assert evolution_health["operator_method_tag_mismatch_nodes"] == []
    assert innovation_report["innovation_claim_level"] == "workflow_exploration_only"
    assert innovation_report["scientific_novelty_supported"] is False
    assert innovation_report["evidence_summary"]["solution_count"] == len(tree["nodes"])
    assert innovation_report["evidence_summary"]["operator_count"] >= 1
    assert scientific_result_card["evidence_grade"] == "workflow_evidence_only"
    assert scientific_result_card["claim_support"]["scientific_claim_supported"] is False
    assert scientific_result_card["champion"]["node_id"] == run_metadata["champion"]
    assert scientific_result_card["score"]["score_source"] == "benchmark evaluator artifact"
    assert scientific_result_card["uncertainty_flags"]
    assert run_metadata["innovation_report"]["innovation_claim_level"] == "workflow_exploration_only"
    assert run_metadata["scientific_result_card"]["evidence_grade"] == "workflow_evidence_only"
    assert run_metadata["scientific_result_card"]["scientific_claim_supported"] is False
    assert run_metadata["operator_scheduler"]["mode"] == "auto-audited"
    assert run_metadata["evolution_health"]["operator_assignment_count"] == len(child_nodes)
    assert run_metadata["evolution_health"]["missing_operator_assignment_count"] == 0
    assert run_metadata["input_layout"]["layout"] == "run_level_inputs_v1"
    readiness = json.loads((run_dir / "reports" / "scientific_discovery_readiness.json").read_text(encoding="utf-8"))
    visual_manifest = json.loads((run_dir / "reports" / "visual_audit_manifest.json").read_text(encoding="utf-8"))
    domain_approval = json.loads((run_dir / "reports" / "domain_approval.json").read_text(encoding="utf-8"))
    paper_dossier = json.loads((run_dir / "reports" / "paper_like_benchmark_dossier.json").read_text(encoding="utf-8"))
    selector_heterogeneity = json.loads(
        (run_dir / "reports" / "selector_heterogeneity.json").read_text(encoding="utf-8")
    )
    multi_seed_evidence = json.loads(
        (run_dir / "reports" / "multi_seed_ablation_evidence.json").read_text(encoding="utf-8")
    )
    method_record = json.loads(
        (run_dir / "solutions" / "solution_000" / "method_experience_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["status"] == "blocked"
    assert readiness["scientific_claim_supported"] is False
    assert any(blocker["check_id"] == "real_llm" for blocker in readiness["blockers"])
    assert visual_manifest["actual_image_inputs_used"] is False
    assert visual_manifest["audited_solution_count"] == len(tree["nodes"])
    assert domain_approval["approved"] is False
    assert paper_dossier["paper_like_ready"] is False
    assert selector_heterogeneity["heterogeneous_selector_evidence"] is False
    assert multi_seed_evidence["verified_multi_seed_ablation"] is False
    assert method_record["experience_scope"]["exact_fingerprint_retrieval"] is True
    assert method_record["experience_scope"]["metric_space_self_improvement_claimed"] is False
    assert method_record["failure_attribution"]["classification"] in {"success", "failure"}


def test_duplicate_child_code_is_marked_in_mutation_and_evolution_health(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="duplicate-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, DuplicateEngineerLLM()).run()

    mutation_report = json.loads(
        (run_dir / "solutions" / "solution_001" / "mutation_effect_report.json").read_text(
            encoding="utf-8"
        )
    )
    evolution_health = json.loads((run_dir / "reports" / "evolution_health.json").read_text(encoding="utf-8"))

    assert mutation_report["status"] == "duplicate_parent"
    assert mutation_report["duplicate_of"] == "solution_000"
    assert evolution_health["duplicate_code_count"] >= 1
    assert evolution_health["warnings"]


def test_cylinder_faithful_small_mock_run_writes_scientific_readiness_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="cylinder-readiness-run",
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        visual_audit_mode="mock",
        resource_constraints={"cpu": "local", "timeout_s": 60, "gpu": False},
        expert_blueprint_id="fluid_pde",
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    readiness = json.loads((run_dir / "reports" / "scientific_discovery_readiness.json").read_text(encoding="utf-8"))
    visual_report = json.loads(
        (run_dir / "solutions" / "solution_000" / "visual_audit_report.json").read_text(encoding="utf-8")
    )
    method_record = json.loads(
        (run_dir / "solutions" / "solution_000" / "method_experience_record.json").read_text(
            encoding="utf-8"
        )
    )

    assert readiness["benchmark"]["name"] == "cylinder_wake_reconstruction_faithful_small"
    assert readiness["status"] == "blocked"
    assert readiness["scientific_claim_supported"] is False
    assert visual_report["visual_audit_mode"] == "mock"
    assert visual_report["actual_image_inputs_used"] is False
    assert visual_report["privacy_boundary"] == "prediction_only_no_validation_labels"
    assert "private_validation_labels_not_loaded" in visual_report["physical_consistency_checks"]
    assert (run_dir / "solutions" / "solution_000" / "visual_field_diagnostic.svg").exists()
    assert method_record["benchmark_family"] == "inverse reconstruction"
    assert method_record["experience_scope"]["benchmark_family_retrieval"] is True


def test_real_visual_audit_records_actual_image_input_with_capable_provider(tmp_path: Path) -> None:
    llm = VisionAuditLLM()
    config = ExperimentConfig(
        experiment_id="real-visual-audit-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=False,
        visual_audit_mode="real",
        resource_constraints={"cpu": "local", "timeout_s": 60},
        expert_blueprint_id="piml",
    )

    run_dir = AgenticSciMLOrchestrator(config, llm).run()

    visual_report = json.loads(
        (run_dir / "solutions" / "solution_000" / "visual_audit_report.json").read_text(
            encoding="utf-8"
        )
    )
    visual_manifest = json.loads((run_dir / "reports" / "visual_audit_manifest.json").read_text(encoding="utf-8"))
    readiness = json.loads((run_dir / "reports" / "scientific_discovery_readiness.json").read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert llm.image_calls
    assert visual_report["actual_image_inputs_used"] is True
    assert visual_report["analysis_mode"] == "real_visual_provider_image_input"
    assert visual_report["visual_provider_output"]["actual_image_inputs_used"] is True
    assert visual_manifest["actual_image_inputs_used"] is True
    actual_image_check = next(check for check in readiness["checks"] if check["check_id"] == "actual_image_inputs")
    assert actual_image_check["passed"] is True
    assert readiness["scientific_claim_supported"] is False
    assert any(
        event["name"] == "visual_audit"
        and event["event_type"] == "generation_span"
        and event["metadata"].get("actual_image_inputs_used") is True
        for event in trace_events
    )


def test_run_writes_domain_selector_paper_and_multiseed_readiness_artifacts(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="readiness-modules-run",
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        domain_evaluator_approved=True,
        domain_reviewer="fluid-reviewer",
        domain_review_notes="Approved evaluator boundary for local faithful-small smoke only.",
        paper_benchmark_approved=True,
        visual_audit_mode="mock",
        resource_constraints={"cpu": "local", "gpu": False, "timeout_s": 60},
        expert_blueprint_id="fluid_pde",
        multi_seed_ablation={
            "seed_count": 2,
            "ablation_count": 1,
            "verified": True,
            "verified_by": "ablation-reviewer",
            "seeds": [0, 1],
            "variants": ["branch_context"],
        },
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    domain_report = json.loads((run_dir / "reports" / "domain_approval.json").read_text(encoding="utf-8"))
    paper_dossier = json.loads(
        (run_dir / "reports" / "paper_like_benchmark_dossier.json").read_text(encoding="utf-8")
    )
    selector_report = json.loads((run_dir / "reports" / "selector_heterogeneity.json").read_text(encoding="utf-8"))
    multi_seed_report = json.loads(
        (run_dir / "reports" / "multi_seed_ablation_evidence.json").read_text(encoding="utf-8")
    )
    readiness = json.loads((run_dir / "reports" / "scientific_discovery_readiness.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert domain_report["approved"] is True
    assert domain_report["domain_review_notes_sha256"]
    assert paper_dossier["paper_like_ready"] is False
    assert paper_dossier["fidelity_level"] == "faithful-small"
    assert selector_report["heterogeneous_selector_evidence"] is False
    assert multi_seed_report["verified_multi_seed_ablation"] is True
    assert multi_seed_report["verifier"] == "ablation-reviewer"
    assert multi_seed_report["attached_artifact_paths"] == [
        "reports/multi_seed_ablation_declared_manifest.json"
    ]
    assert (run_dir / "reports" / "multi_seed_ablation_declared_manifest.json").exists()
    checks = {check["check_id"]: check for check in readiness["checks"]}
    assert checks["domain_review"]["passed"] is True
    assert checks["multi_seed_ablation"]["passed"] is True
    assert checks["paper_like_benchmark"]["passed"] is False
    assert readiness["scientific_claim_supported"] is False
    assert run_metadata["domain_approval"]["approved"] is True
    assert run_metadata["multi_seed_ablation"]["verified_multi_seed_ablation"] is True


def test_default_mock_engineer_generates_distinct_sequential_children(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="distinct-mock-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=2, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    first = json.loads(
        (run_dir / "solutions" / "solution_001" / "mutation_effect_report.json").read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        (run_dir / "solutions" / "solution_002" / "mutation_effect_report.json").read_text(
            encoding="utf-8"
        )
    )
    evolution_health = json.loads((run_dir / "reports" / "evolution_health.json").read_text(encoding="utf-8"))

    assert first["code_digest"] != second["code_digest"]
    assert first["status"] != "duplicate_parent"
    assert second["status"] != "duplicate_parent"
    assert evolution_health["duplicate_code_count"] == 0
    assert evolution_health["unique_code_count"] == evolution_health["solution_count"]


def test_plateau_without_duplicate_code_is_explained_in_mutation_and_evolution_health(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="plateau-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, CommentOnlyEngineerLLM()).run()

    mutation_report = json.loads(
        (run_dir / "solutions" / "solution_001" / "mutation_effect_report.json").read_text(
            encoding="utf-8"
        )
    )
    evolution_health = json.loads((run_dir / "reports" / "evolution_health.json").read_text(encoding="utf-8"))

    assert mutation_report["status"] == "changed_but_score_plateau"
    assert mutation_report["duplicate_of"] is None
    assert mutation_report["code_changed_from_parent"] is True
    assert mutation_report["diff_line_count"] > 0
    assert "solution_001" in evolution_health["score_plateau_nodes"]
    assert evolution_health["mutation_status_counts"]["changed_but_score_plateau"] == 1
    assert any("Score plateau" in warning for warning in evolution_health["warnings"])
    method_record = json.loads(
        (run_dir / "solutions" / "solution_001" / "method_experience_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert method_record["failure_attribution"]["classification"] == "plateau"


def test_manual_strategy_lock_inspector_blocks_unfaithful_solution(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="policy-lock-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        readiness_report={
            "manual_strategy_locks": [
                {
                    "lock_id": "lock_lbfgs",
                    "required": True,
                    "inspection": {"required_terms": ["LBFGS"]},
                }
            ]
        },
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    report = json.loads(
        (run_dir / "solutions" / "solution_000" / "policy_fidelity_report.json").read_text(
            encoding="utf-8"
        )
    )
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["status"] == "blocked"
    assert report["execution_allowed"] is False
    assert "Required strategy term not found" in report["checks"][0]["message"]
    assert tree["nodes"][0]["status"] == "failed"
    assert tree["nodes"][0]["failure_kind"] == "guardrail_error"
    method_record = json.loads(
        (run_dir / "solutions" / "solution_000" / "method_experience_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert method_record["failure_attribution"]["classification"] == "policy_fidelity_mismatch"
    assert any(
        event["name"] == "strategy_fidelity_inspector"
        and event["metadata"]["passed"] is False
        for event in trace_events
    )


def test_evaluation_approval_gate_pauses_before_root_generation(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="approval-pending-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        auto_approve_evaluation=False,
    )

    with pytest.raises(EvaluationApprovalRequired, match="Evaluation approval required"):
        AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    run_dir = tmp_path / "approval-pending-run"
    approval = json.loads((run_dir / "evaluation_approval.json").read_text(encoding="utf-8"))
    assert approval["status"] == "pending"
    assert approval["approval_required"] is True
    assert approval["contract_hash"]
    assert (run_dir / "evaluation_contract.json").exists()
    assert not (run_dir / "solutions" / "solution_000").exists()
    assert not (run_dir / "checkpoint.json").exists()


def test_evaluation_approval_resume_creates_root_after_manual_approval(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="approval-resume-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        auto_approve_evaluation=False,
    )
    with pytest.raises(EvaluationApprovalRequired):
        AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    run_dir = tmp_path / "approval-resume-run"
    approval_path = run_dir / "evaluation_approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["status"] = "approved"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="approval-resume-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        auto_approve_evaluation=False,
        resume=True,
    )

    resumed_run_dir = AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()

    tree = json.loads((resumed_run_dir / "tree.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((resumed_run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert tree["nodes"][0]["node_id"] == "solution_000"
    assert checkpoint["phase"] == "completed"
    assert checkpoint["nodes"][0]["node_id"] == "solution_000"


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


def test_parallel_mutation_budget_fans_out_single_parent(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="parallel-fanout-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    root = next(node for node in tree["nodes"] if node["node_id"] == "solution_000")
    trace_events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start = next(event for event in trace_events if event["name"] == "agenticsciml.parallel_children.start")

    assert len(tree["nodes"]) == 3
    assert root["children"] == ["solution_001", "solution_002"]
    assert start["metadata"]["execution_mode"] == "parallel"
    assert start["metadata"]["child_count"] == 2
    assert start["metadata"]["parent_ids"] == ["solution_000", "solution_000"]
    assert start["metadata"]["unique_parent_ids"] == ["solution_000"]
    assert start["metadata"]["parent_child_edges"] == [
        {"slot_index": 0, "parent_id": "solution_000", "child_id": "solution_001"},
        {"slot_index": 1, "parent_id": "solution_000", "child_id": "solution_002"},
    ]
    assert start["metadata"]["parent_to_children"] == {
        "solution_000": ["solution_001", "solution_002"]
    }


def test_parallel_fanout_records_branch_context(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="branch-context-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    child_nodes = [node for node in tree["nodes"] if node["parent_id"] == "solution_000"]
    contexts = {
        node["node_id"]: json.loads(
            (run_dir / "solutions" / node["node_id"] / "branch_context.json").read_text(
                encoding="utf-8"
            )
        )
        for node in child_nodes
    }
    trace_events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    child_starts = [
        event
        for event in trace_events
        if event["name"] == "agenticsciml.child_mutation.start"
    ]
    proposal_transcript = json.loads(
        (
            run_dir
            / "solutions"
            / "solution_001"
            / "transcripts"
            / "proposal_debate.json"
        ).read_text(encoding="utf-8")
    )

    assert contexts["solution_001"]["branch_intent"] == "features_or_architecture"
    assert contexts["solution_002"]["branch_intent"] == "training_stability"
    assert contexts["solution_001"]["operator_focus"]["operator_id"]
    assert contexts["solution_001"]["operator_focus"]["mutation_axis"]
    assert contexts["solution_001"]["sibling_branch_ids"] == ["solution_002"]
    assert contexts["solution_002"]["sibling_branch_ids"] == ["solution_001"]
    assert any(
        node["node_id"] == "solution_001"
        and "branch:features_or_architecture" in node["method_tags"]
        for node in child_nodes
    )
    assert {event["metadata"]["branch_context"]["branch_intent"] for event in child_starts} == {
        "features_or_architecture",
        "training_stability",
    }
    assert "Branch context" in proposal_transcript[0]["prompt"]
    assert "features_or_architecture" in proposal_transcript[0]["prompt"]


def test_parallel_mutation_fanout_respects_max_children_per_node(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="parallel-fanout-limit-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(
            max_iterations=0,
            parallel_mutations=3,
            max_children_per_node=2,
            max_debug_retries=0,
        ),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)
    parent = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace=str(tmp_path / "solution_000"),
        score=SolutionScore("validation_mse", 1.0, higher_is_better=False),
        status="evaluated",
        benchmark_name=contract.benchmark_name,
        contract_hash=contract.contract_hash,
    )
    parent.children = ["solution_001"]

    slots = orchestrator._mutation_parent_slots([parent], mutation_budget=3)

    assert [slot.node_id for slot in slots] == ["solution_000"]


def test_configured_selector_panel_records_member_provenance(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="selector-panel-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(
            max_iterations=0,
            parallel_mutations=2,
            selector_vote_count=3,
            max_debug_retries=0,
        ),
        use_mock=True,
        selector_panel=[
            AgentConfig(role="selector_alpha", model="gpt-5-mini", temperature=0.05, reasoning_effort="high"),
            AgentConfig(role="selector_beta", model="deepseek-v4-pro", temperature=0.1, reasoning_effort="xhigh"),
        ],
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)
    orchestrator.nodes = [
        SolutionNode(
            node_id="solution_000",
            parent_id=None,
            workspace=str(tmp_path / "solution_000"),
            score=SolutionScore("validation_mse", 0.5, higher_is_better=False),
            status="evaluated",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        ),
        SolutionNode(
            node_id="solution_001",
            parent_id=None,
            workspace=str(tmp_path / "solution_001"),
            score=SolutionScore("validation_mse", 0.8, higher_is_better=False),
            status="evaluated",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        ),
        SolutionNode(
            node_id="solution_002",
            parent_id=None,
            workspace=str(tmp_path / "solution_002"),
            score=SolutionScore("validation_mse", 0.1, higher_is_better=False),
            status="evaluated",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        ),
    ]

    selected = orchestrator._select_parents()

    artifact = json.loads(
        (orchestrator.storage.run_dir / "reports" / "selector_votes.json").read_text(
            encoding="utf-8"
        )
    )
    assert [node.node_id for node in selected][:2] == ["solution_002", "solution_000"]
    assert artifact["ensemble_mode"] == "configured_selector_panel"
    assert [member["member_id"] for member in artifact["selector_panel_members"]] == [
        "selector_alpha",
        "selector_beta",
    ]
    assert [vote["configured_model"] for vote in artifact["votes"]] == [
        "gpt-5-mini",
        "deepseek-v4-pro",
    ]
    assert len(artifact["votes"]) == 2
    assert all(vote["actual_model"] == "mock" for vote in artifact["votes"])
    assert all(vote["adapter_type"] == "mock_local" for vote in artifact["votes"])
    assert all(vote["provider_capabilities"]["supports_image_inputs"] is False for vote in artifact["votes"])
    assert artifact["selector_diversity"]["actual_vote_count"] == 2
    assert artifact["selector_diversity"]["panel_member_count"] == 2
    assert artifact["selector_diversity"]["mock_evidence"] is True
    assert artifact["selector_diversity"]["provider_diversity"] is False
    assert artifact["selector_diversity"]["panel_repeated_members"] is False
    assert artifact["selector_diversity"]["heterogeneous_selector_evidence"] is False
    assert "only heterogeneous provider evidence" in artifact["claim_boundary"]


def test_real_selector_panel_uses_per_member_base_url_for_heterogeneous_evidence(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="real-selector-panel-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(
            max_iterations=2,
            parallel_mutations=1,
            selector_vote_count=3,
            max_debug_retries=0,
        ),
        use_mock=False,
        selector_panel=[
            AgentConfig(role="selector_openai", model="gpt-5-mini", temperature=0.05),
            AgentConfig(
                role="selector_compatible",
                model="deepseek-v4-pro",
                temperature=0.05,
                base_url="https://api.deepseek.com",
            ),
        ],
    )

    run_dir = AgenticSciMLOrchestrator(config, ProviderAwareMockLLM()).run()

    votes = json.loads((run_dir / "reports" / "selector_votes.json").read_text(encoding="utf-8"))
    selector_report = json.loads((run_dir / "reports" / "selector_heterogeneity.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert votes["selector_diversity"]["heterogeneous_selector_evidence"] is True
    assert votes["selector_diversity"]["mock_evidence"] is False
    assert votes["selector_diversity"]["unique_providers"] == ["api.deepseek.com", "openai"]
    assert [member["configured_base_url"] for member in votes["selector_panel_members"]] == [
        None,
        "https://api.deepseek.com",
    ]
    assert selector_report["heterogeneous_selector_evidence"] is True
    assert selector_report["status"] == "ready"
    assert metadata["selector_heterogeneity"]["heterogeneous_selector_evidence"] is True


def test_selector_vote_history_keeps_each_selection_artifact(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="selector-history-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(
            max_iterations=3,
            parallel_mutations=1,
            selector_vote_count=3,
            max_debug_retries=0,
        ),
        use_mock=True,
        selector_panel=[
            AgentConfig(role="selector_alpha", model="gpt-5-mini", temperature=0.05),
            AgentConfig(role="selector_beta", model="deepseek-v4-pro", temperature=0.05),
        ],
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    vote_files = sorted((run_dir / "reports" / "selector_votes").glob("selection_*.json"))
    latest = json.loads((run_dir / "reports" / "selector_votes.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert [path.name for path in vote_files] == ["selection_000001.json", "selection_000002.json"]
    assert latest["selection_index"] == 2
    assert latest["selector_policy_digest"] == metadata["selector_policy_digest"]
    assert metadata["selector_panel"]["selector_voting_exercised"] is True
    assert metadata["selector_panel"]["selector_vote_events"] == 2


def test_configured_selector_panel_without_selection_is_not_exercised(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="selector-not-exercised-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        selector_panel=[
            AgentConfig(role="selector_alpha", model="gpt-5-mini", temperature=0.05),
            AgentConfig(role="selector_beta", model="deepseek-v4-pro", temperature=0.05),
        ],
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert metadata["selector_panel"]["selector_voting_exercised"] is False
    assert metadata["selector_panel"]["selector_vote_events"] == 0
    assert not (run_dir / "reports" / "selector_votes.json").exists()
    assert checkpoint["selector_policy_digest"] == metadata["selector_policy_digest"]


def test_resume_rejects_selector_policy_change_without_overwriting_config(tmp_path: Path) -> None:
    first_config = ExperimentConfig(
        experiment_id="selector-policy-resume-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        selector_panel=[
            AgentConfig(role="selector_alpha", model="gpt-5-mini", temperature=0.05),
        ],
    )
    run_dir = AgenticSciMLOrchestrator(first_config, MockLLMClient()).run()
    original_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    resume_config = ExperimentConfig(
        experiment_id="selector-policy-resume-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
        selector_panel=[
            AgentConfig(role="selector_beta", model="deepseek-v4-pro", temperature=0.05),
        ],
    )

    with pytest.raises(ValueError, match="Checkpoint selector policy mismatch"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()

    preserved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert preserved_config["selector_panel"] == original_config["selector_panel"]


def test_analysis_context_includes_parent_sibling_and_uncle_reports(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="analysis-context-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)

    def node(node_id: str, parent_id: str | None) -> SolutionNode:
        workspace = orchestrator.storage.create_solution_workspace(node_id)
        return SolutionNode(
            node_id=node_id,
            parent_id=parent_id,
            workspace=str(workspace),
            score=SolutionScore("validation_mse", 1.0, higher_is_better=False),
            status="evaluated",
            analysis_path=str(workspace / "analysis.md"),
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        )

    root = node("solution_000", None)
    parent = node("solution_001", "solution_000")
    uncle = node("solution_002", "solution_000")
    sibling = node("solution_003", "solution_001")
    root.children = ["solution_001", "solution_002"]
    parent.children = ["solution_003"]
    orchestrator.nodes = [root, parent, uncle, sibling]
    orchestrator.analysis_by_node = {
        "solution_001": AnalysisReport("solution_001", "parent improved smoothness"),
        "solution_002": AnalysisReport("solution_002", "uncle explored regularization"),
        "solution_003": AnalysisReport("solution_003", "sibling overfit high frequencies"),
    }

    context = orchestrator._analysis_context_for_parent(parent, "solution_004")
    formatted = orchestrator._format_analysis_context(context)

    assert context["parent_report"]["node_id"] == "solution_001"  # type: ignore[index]
    assert [item["node_id"] for item in context["sibling_reports"]] == ["solution_003"]  # type: ignore[index]
    assert [item["node_id"] for item in context["uncle_reports"]] == ["solution_002"]  # type: ignore[index]
    assert context["omitted_reports"] == []
    assert "Parent analysis" in formatted
    assert "Sibling analyses" in formatted
    assert "Uncle analyses" in formatted


def test_analysis_context_records_missing_reports(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="analysis-context-missing-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)
    root = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace=str(orchestrator.storage.create_solution_workspace("solution_000")),
        score=SolutionScore("validation_mse", 1.0, higher_is_better=False),
        status="evaluated",
        benchmark_name=contract.benchmark_name,
        contract_hash=contract.contract_hash,
    )
    parent = SolutionNode(
        node_id="solution_001",
        parent_id="solution_000",
        workspace=str(orchestrator.storage.create_solution_workspace("solution_001")),
        score=SolutionScore("validation_mse", 0.9, higher_is_better=False),
        status="evaluated",
        benchmark_name=contract.benchmark_name,
        contract_hash=contract.contract_hash,
    )
    root.children = ["solution_001", "solution_002"]
    parent.children = ["solution_003"]
    orchestrator.nodes = [root, parent]
    orchestrator.analysis_by_node = {
        "solution_001": AnalysisReport("solution_001", "parent report exists"),
    }

    context = orchestrator._analysis_context_for_parent(parent, "solution_004")
    omitted = context["omitted_reports"]

    assert {"relationship": "sibling", "node_id": "solution_003", "reason": "node_missing"} in omitted
    assert {"relationship": "uncle", "node_id": "solution_002", "reason": "node_missing"} in omitted


def test_child_mutation_writes_analysis_context_artifact(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="analysis-context-artifact-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )

    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    context = json.loads(
        (run_dir / "solutions" / "solution_001" / "analysis_context.json").read_text(
            encoding="utf-8"
        )
    )
    transcript = json.loads(
        (run_dir / "solutions" / "solution_001" / "transcripts" / "proposal_debate.json").read_text(
            encoding="utf-8"
        )
    )

    assert context["schema_version"] == 1
    assert context["parent_report"]["node_id"] == "solution_000"
    assert context["sibling_reports"] == []
    assert context["uncle_reports"] == []
    assert "Analysis Base context" in transcript[0]["prompt"]
    assert "Parent analysis" in transcript[0]["prompt"]


def test_solution_id_allocator_uses_max_existing_suffix(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="id-allocator-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    contract = BenchmarkContractFactory.create_contract(orchestrator.problem_bundle)
    orchestrator.nodes = [
        SolutionNode(
            node_id="solution_000",
            parent_id=None,
            workspace=str(tmp_path / "solution_000"),
            score=SolutionScore("validation_mse", 1.0, higher_is_better=False),
            status="evaluated",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        ),
        SolutionNode(
            node_id="solution_002",
            parent_id="solution_000",
            workspace=str(tmp_path / "solution_002"),
            score=SolutionScore("validation_mse", 0.9, higher_is_better=False),
            status="evaluated",
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
        ),
    ]
    orchestrator.storage.create_solution_workspace("solution_004")

    assert orchestrator._reserve_solution_ids(2) == ["solution_005", "solution_006"]


def test_solution_id_allocator_rejects_malformed_existing_node_id(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="bad-id-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    orchestrator = AgenticSciMLOrchestrator(config, MockLLMClient())
    orchestrator.nodes = [
        SolutionNode(
            node_id="candidate_a",
            parent_id=None,
            workspace=str(tmp_path / "candidate_a"),
            score=None,
            status="created",
            benchmark_name="function_approx",
            contract_hash="hash",
        )
    ]

    with pytest.raises(ValueError, match="malformed node_id"):
        orchestrator._reserve_solution_ids(1)


def test_resume_solution_id_allocation_skips_existing_workspace_suffix(tmp_path: Path) -> None:
    first_config = ExperimentConfig(
        experiment_id="resume-id-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
    )
    first_run_dir = AgenticSciMLOrchestrator(first_config, MockLLMClient()).run()
    (first_run_dir / "solutions" / "solution_002").mkdir(parents=True)

    second_config = ExperimentConfig(
        experiment_id="resume-id-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=2, max_debug_retries=1),
        use_mock=True,
        resume=True,
    )
    second_run_dir = AgenticSciMLOrchestrator(second_config, MockLLMClient()).run()
    tree = json.loads((second_run_dir / "tree.json").read_text(encoding="utf-8"))
    root = next(node for node in tree["nodes"] if node["node_id"] == "solution_000")

    assert root["children"] == ["solution_003", "solution_004"]
    assert {node["node_id"] for node in tree["nodes"]} == {
        "solution_000",
        "solution_003",
        "solution_004",
    }


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
        branch_context: dict[str, object] | None = None,
        operator_assignment=None,
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
        branch_context: dict[str, object] | None = None,
        operator_assignment=None,
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
    root = next(node for node in tree["nodes"] if node["node_id"] == "solution_000")
    assert len(tree["nodes"]) == 3
    assert root["children"] == ["solution_001", "solution_002"]
    assert "agenticsciml.resume.loaded" in trace_text


def test_non_resume_rejects_existing_run_directory(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="existing-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    AgenticSciMLOrchestrator(config, MockLLMClient()).run()

    with pytest.raises(FileExistsError, match="Use --resume"):
        AgenticSciMLOrchestrator(config, MockLLMClient()).run()


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


def test_resume_rejects_solution_node_unknown_fields(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="unknown-node-field-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["nodes"][0]["extra"] = "drift"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="unknown-node-field-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="unknown fields: extra"):
        AgenticSciMLOrchestrator(resume_config, MockLLMClient()).run()


def test_resume_rejects_unsupported_solution_tree_schema_version(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="unsupported-tree-schema-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=0, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
    )
    run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["schema_version"] = "solution_tree.v999"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resume_config = ExperimentConfig(
        experiment_id="unsupported-tree-schema-run",
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        evolution=EvolutionConfig(max_iterations=1, parallel_mutations=1, max_debug_retries=0),
        use_mock=True,
        resume=True,
    )

    with pytest.raises(ValueError, match="unsupported schema_version"):
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
    root = next(node for node in tree["nodes"] if node["node_id"] == "solution_000")
    assert len(tree["nodes"]) == 3
    assert root["children"] == ["solution_001", "solution_002"]


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
    method_record = json.loads(
        (run_dir / "solutions" / "solution_000" / "method_experience_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert method_record["failure_attribution"]["classification"] == "failure"


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
    method_record = json.loads(
        (run_dir / "solutions" / child["node_id"] / "method_experience_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert method_record["failure_attribution"]["classification"] == "failure"
