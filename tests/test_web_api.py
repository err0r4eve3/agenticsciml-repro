from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from agenticsciml.web.app import create_app


def test_web_benchmarks_match_catalog() -> None:
    client = TestClient(create_app())

    response = client.get("/api/benchmarks")

    assert response.status_code == 200
    payload = response.json()
    names = {item["name"] for item in payload["benchmarks"]}
    assert "function_approx" in names
    first = payload["benchmarks"][0]
    assert "fidelity_matrix" in first
    assert "paper_benchmark_equivalent" in first["fidelity_matrix"]
    assert first["fidelity_matrix"]["hidden_label_protocol"] == "trusted local evaluator with hidden validation labels"


def test_web_algorithms_expose_professional_catalog() -> None:
    client = TestClient(create_app())

    response = client.get("/api/algorithms")

    assert response.status_code == 200
    payload = response.json()
    algorithms = {item["id"]: item for item in payload["algorithms"]}
    assert "Algorithm catalog entries are planning" in payload["claim_boundary"]
    assert "pinn_residual_minimizer" in algorithms
    assert "weak_form_pinn" in algorithms
    assert "xpinn_domain_decomposition" in algorithms
    assert "deeponet_operator" in algorithms
    assert "kernel_surrogate_regression" in algorithms
    assert "sindy_sparse_discovery" in algorithms
    assert "paper_sigmoid_moe_gate" in algorithms
    assert "paper_linear_bias_free_deeponet" in algorithms
    assert algorithms["paper_sigmoid_moe_gate"]["status"] == "reference_implementation"
    assert algorithms["paper_sigmoid_moe_gate"]["implementation_path"]
    assert algorithms["pinn_residual_minimizer"]["status"] == "strategy_blueprint"
    assert len(algorithms) >= 20
    assert "paper-level" in algorithms["sparse_sensor_reconstructor"]["safety_notes"]
    for algorithm in algorithms.values():
        assert algorithm["description"]
        assert algorithm["description_zh"]
        assert algorithm["features"]
        assert algorithm["features_zh"]
        assert algorithm["problem_fit"]
        assert algorithm["problem_fit_zh"]
        assert algorithm["safety_notes_zh"]
        assert len(algorithm["features"]) == len(algorithm["features_zh"])
        assert len(algorithm["problem_fit"]) == len(algorithm["problem_fit_zh"])


def test_paper_tasks_expose_s1_mapping() -> None:
    client = TestClient(create_app())

    response = client.get("/api/paper-tasks")

    assert response.status_code == 200
    tasks = response.json()["tasks"]
    assert [task["paper_section"] for task in tasks] == ["S1.1", "S1.2", "S1.3", "S1.4", "S1.5", "S1.6"]
    by_section = {task["paper_section"]: task for task in tasks}
    s16 = by_section["S1.6"]
    assert "Cylinder Wake" in s16["title"]
    assert any(
        benchmark["name"] == "cylinder_wake_reconstruction_faithful_small"
        for benchmark in s16["benchmarks"]
    )
    assert all("fidelity_matrix" in benchmark for benchmark in s16["benchmarks"])
    assert s16["algorithms"][0]["id"] == "paper_cylinder_bandlimited_filter"
    assert "not paper-score evidence" in s16["claim_boundary"]
    assert "reports/data_overview.svg" in s16["local_artifact_figures"]


def test_solver_settings_expose_mode_defaults() -> None:
    client = TestClient(create_app())

    response = client.get("/api/solver/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_assistant_mode"] == "ask"
    assert payload["reasoning_efforts"] == ["low", "medium", "high", "xhigh"]
    assert payload["temperature_range"] == [0.0, 2.0]
    assert payload["assistant_modes"] == {
        "ask": {"reasoning_effort": "medium", "temperature": 0.2},
        "plan": {"reasoning_effort": "high", "temperature": 0.35},
        "agent": {"reasoning_effort": "high", "temperature": 0.1},
    }
    assert payload["agent_role_defaults"]["proposer"] == {
        "temperature": 0.55,
        "reasoning_effort": "xhigh",
        "rationale": "Proposal generation is the main creative search step and benefits from deeper reasoning.",
    }
    assert payload["agent_role_defaults"]["debugger"]["temperature"] == 0.05
    assert payload["agent_role_defaults"]["debugger"]["reasoning_effort"] == "xhigh"


def test_agent_roles_expose_layered_model_contract() -> None:
    client = TestClient(create_app())

    response = client.get("/api/agent-roles")

    assert response.status_code == 200
    payload = response.json()
    roles = {item["role"]: item for item in payload["roles"]}
    assert roles["data_analyst"]["label"] == "Data Analyst"
    assert roles["engineer"]["kind"] == "patch"
    assert roles["selector"]["label"] == "Selector"
    assert roles["proposer"]["default_model_settings"]["temperature"] == 0.55
    assert roles["evaluator"]["default_model_settings"]["temperature"] == 0.0
    assert roles["engineer"]["default_model_settings"]["reasoning_effort"] == "xhigh"
    assert "reasoning_effort" in payload["reasoning_effort_note"]


def test_problem_intake_plans_benchmark_and_strategy_seeds() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/problem-intake/plan",
        json={
            "problem_statement": (
                "Reconstruct two-dimensional cylinder wake vorticity fields from sparse noisy temporal sensors. "
                "The model should use lagged sensor history and preserve smooth band-limited spatial structure."
            ),
            "requirements": "No future sensor leakage; local deterministic evaluation.",
            "evaluation_criteria": "Mean relative L2 on private vorticity fields.",
            "data_description": "Eight sparse sensors over five time steps.",
            "mode": "mock",
            "target_solution_count": 7,
            "parallel_mutations": 3,
            "selected_algorithm_ids": ["paper_cylinder_bandlimited_filter"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommended_benchmark"]["name"] == "cylinder_wake_reconstruction_faithful_small"
    assert payload["problem_intake"]["problem_statement"].startswith("Reconstruct two-dimensional")
    assert payload["planner_snapshot"]["planner_version"] == "problem_intake_keyword_planner.v1"
    assert payload["planner_snapshot"]["selected_seed_snapshot"][0]["id"] == "paper_cylinder_bandlimited_filter"
    assert payload["run_config"]["max_iterations"] == 2
    assert payload["run_config"]["planned_solution_budget"] == 7
    assert "paper_cylinder_bandlimited_filter" in payload["selected_algorithm_ids"]
    assert payload["actions"][0]["payload"]["benchmark"] == "cylinder_wake_reconstruction_faithful_small"
    assert payload["actions"][0]["payload"]["selected_algorithm_ids"] == payload["selected_algorithm_ids"]
    assert payload["actions"][0]["payload"]["problem_intake"] == payload["problem_intake"]
    assert payload["actions"][0]["payload"]["planner_snapshot"] == payload["planner_snapshot"]
    assert any(
        item["algorithm"]["id"] == "paper_cylinder_bandlimited_filter" and item["selected"]
        for item in payload["algorithm_rankings"]
    )


def test_problem_intake_custom_benchmark_generates_runnable_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    response = client.post(
        "/api/problem-intake/plan",
        json={
            "problem_statement": (
                "I want to solve a new inverse scattering SciML problem with complex-valued sensor fields, "
                "unknown material coefficients, and a private finite-element validation metric."
            ),
            "requirements": "Do not map this to an existing toy benchmark; outline a new evaluator bundle.",
            "evaluation_criteria": "Private relative error on hidden complex fields and coefficient recovery.",
            "data_description": "Synthetic source/receiver pairs and measured boundary responses.",
            "allow_custom_benchmark": True,
            "account_id": "custom-user",
            "selected_algorithm_ids": ["pinn_residual_minimizer"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    scaffold = payload["custom_problem_package"]
    assert payload["run_allowed"] is True
    assert payload["actions"][0]["type"] == "start_run"
    assert payload["actions"][0]["payload"]["account_id"] == "custom-user"
    assert payload["actions"][0]["payload"]["benchmark"] == scaffold["benchmark"]
    assert scaffold["status"] == "custom_proxy_benchmark_scaffolded"
    assert scaffold["synthesis_level"] == "autonomous_workflow_proxy_evaluator_synthesis"
    assert scaffold["run_allowed"] is True
    assert scaffold["workflow_run_approval_required"] is False
    assert scaffold["evaluator_trust_level"] == "synthetic_proxy"
    assert scaffold["domain_evaluator_present"] is False
    assert scaffold["metric_validated_by_domain_expert"] is False
    assert scaffold["paper_benchmark_equivalent"] is False
    assert scaffold["requires_replacement_for_scientific_claim"] is True
    assert scaffold["domain_evidence_review_required"] is True
    assert scaffold["paper_level_claim_supported"] is False
    assert scaffold["scientific_claim_supported"] is False
    assert scaffold["approval_scope"] == "workflow_proxy_run_only"
    assert scaffold["evidence_level"] == "workflow_proxy"
    generated = payload["planner_snapshot"]["generated_custom_benchmark"]
    assert generated["evidence_level"] == "workflow_proxy"
    assert generated["domain_evidence_review_required"] is True
    assert generated["paper_level_claim_supported"] is False
    benchmark_dir = Path(scaffold["benchmark_dir"])
    assert benchmark_dir.exists()
    assert "evaluate.py" in scaffold["required_files"]
    assert "evaluator_synthesis.json" in scaffold["required_files"]
    assert "eda/data_eda.py" in scaffold["created_files"]
    assert "eda/data_eda_seed0.json" in scaffold["created_files"]
    synthesis = json.loads((benchmark_dir / "evaluator_synthesis.json").read_text(encoding="utf-8"))
    assert synthesis["synthesis_level"] == "autonomous_workflow_proxy_evaluator_synthesis"
    assert synthesis["evaluator_trust_level"] == "synthetic_proxy"
    assert synthesis["domain_evaluator_present"] is False
    assert synthesis["metric_validated_by_domain_expert"] is False
    assert synthesis["paper_benchmark_equivalent"] is False
    assert synthesis["requires_replacement_for_scientific_claim"] is True
    assert synthesis["problem_class"] == "inverse_reconstruction"
    assert synthesis["metric"]["primary"] == "custom_proxy_relative_l2"
    assert synthesis["data_schema"]["prediction_input"] == "x_val"
    assert synthesis["quality_gates"]["human_domain_review_required"] is True
    assert synthesis["review_boundary"]["workflow_proxy_run_requires_human_review"] is False
    assert synthesis["review_boundary"]["scientific_claim_requires_human_domain_review"] is True
    assert synthesis["review_boundary"]["paper_level_claim_supported"] is False
    seed_eda_path = benchmark_dir / "eda" / "data_eda_seed0.json"
    seed_eda = json.loads(seed_eda_path.read_text(encoding="utf-8"))
    assert seed_eda["privacy_boundary"] == "training_data_only_no_private_labels"
    assert "val_data" not in seed_eda_path.read_text(encoding="utf-8")
    assert scaffold["strategy_seed_suggestions"][0]["id"] == "pinn_residual_minimizer"
    assert any("workflow-proxy evaluator scaffold" in warning for warning in payload["warnings"])
    assert "workflow-proxy" in scaffold["claim_boundary"]

    run_payload = dict(payload["actions"][0]["payload"])
    run_payload.update(
        {
            "mode": "mock",
            "max_iterations": 0,
            "target_solution_count": None,
            "background": False,
            "output_dir": "runs",
        }
    )
    run_response = client.post("/api/runs", json=run_payload)

    assert run_response.status_code == 200
    run = run_response.json()
    run_dir = Path(run["run_dir"])
    assert (run_dir / "evaluation_contract.json").exists()
    assert (run_dir / "solutions" / "solution_000" / "eval.json").exists()
    contract = json.loads((run_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    assert "evaluator_synthesis.json" in contract["benchmark_source_manifest"]["artifacts"]
    assert "eda/data_eda.py" in contract["benchmark_source_manifest"]["artifacts"]
    assert "eda/data_eda_seed0.json" in contract["benchmark_source_manifest"]["artifacts"]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["benchmark_fidelity_level"] == "proxy"
    assert metadata["scientific_claim"] == "not_supported"
    assert metadata["claim_gate"]["status"] == "allowed"
    assert metadata["claim_gate"]["evaluator_trust_level"] == "synthetic_proxy"
    assert metadata["claim_gate"]["paper_level_claim_supported"] is False


def test_problem_intake_rejects_unknown_algorithm() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/problem-intake/plan",
        json={
            "problem_statement": "Solve a discontinuous function approximation problem with a local evaluator.",
            "selected_algorithm_ids": ["not-a-real-algorithm"],
        },
    )

    assert response.status_code == 400
    assert "Unknown algorithm id" in response.json()["detail"]


def test_run_readiness_preview_audits_proxy_claims_and_strategy_seeds() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/run-readiness/preview",
        json={
            "benchmark": "function_approx",
            "mode": "mock",
            "target_solution_count": 5,
            "parallel_mutations": 2,
            "selected_algorithm_ids": ["piecewise_local_basis"],
            "manual_strategy_locks": [
                {
                    "lock_id": "lock_piecewise",
                    "kind": "mathematical_intuition",
                    "text": "Preserve local basis structure for discontinuities.",
                    "scope": "all_branches",
                    "inspection": {"required_terms": ["piecewise"]},
                }
            ],
            "branch_context": {"expected_inherited_lock_ids": ["lock_piecewise"]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["readiness_version"] == "run_readiness.v1"
    assert payload["status"] == "ready_with_warnings"
    assert payload["launch_allowed"] is True
    assert payload["run_budget"]["max_iterations"] == 2
    assert payload["run_budget"]["planned_solution_budget"] == 5
    assert payload["claim_gate"]["status"] == "allowed"
    assert payload["claim_gate"]["paper_level_claim_supported"] is False
    assert payload["kb_manifest"]["coverage_status"] in {"missing", "local_kb_seed"}
    check_ids = {check["check_id"] for check in payload["checks"]}
    assert "claim-boundary.proxy-warning" in check_ids
    assert "claim-gate.workflow-proxy" in check_ids
    assert "algorithm-selection.catalog-seeds" in check_ids
    assert payload["algorithm_seed_preview"][0]["algorithm_id"] == "piecewise_local_basis"
    assert payload["algorithm_seed_preview"][0]["catalog_role"] == "strategy_seed"
    assert payload["algorithm_seed_preview"][0]["is_evaluated_implementation"] is False
    assert payload["strategy_lock_preview"][0]["lock_id"] == "lock_piecewise"
    assert payload["strategy_lock_preview"][0]["auditable_criteria_count"] == 1
    assert payload["manual_strategy_locks"][0]["inspection"] == {"required_terms": ["piecewise"]}
    assert "planning/readiness_report.json" in payload["artifact_capture_requirements"]
    assert "solutions/*/policy_fidelity_report.json" in payload["artifact_capture_requirements"]
    assert "solutions/*/emergence_report.json" in payload["artifact_capture_requirements"]
    serialized = json.dumps(payload)
    assert "chain_of_thought" not in serialized
    assert "private_reasoning" not in serialized


def test_run_readiness_preview_blocks_unknown_algorithm_and_real_without_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTICSCIML_ENABLE_REAL_WEB_RUNS", raising=False)
    client = TestClient(create_app())

    response = client.post(
        "/api/run-readiness/preview",
        json={
            "benchmark": "function_approx",
            "mode": "real",
            "selected_algorithm_ids": ["not-a-real-algorithm"],
            "real_confirmed": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["launch_allowed"] is False
    failed_blockers = {
        check["check_id"]
        for check in payload["checks"]
        if check["severity"] == "blocker" and not check["passed"]
    }
    assert failed_blockers == {
        "algorithm-selection.unknown",
        "real-mode.requires-confirmation",
        "real-mode.server-disabled",
    }
    assert payload["real_mode_gates"]["blocked_reasons"] == [
        "real-mode.requires-confirmation",
        "real-mode.server-disabled",
    ]


def test_paper_workflow_claim_gate_blocks_readiness_and_launch(tmp_path: Path) -> None:
    client = TestClient(create_app())

    readiness_response = client.post(
        "/api/run-readiness/preview",
        json={
            "benchmark": "function_approx_faithful_small",
            "mode": "mock",
            "claim_level": "paper_workflow",
            "domain_evaluator_approved": True,
            "paper_benchmark_approved": True,
            "selector_panel": [
                {"model": "gpt-5-mini", "temperature": 0.05},
                {"model": "deepseek-v4-pro", "temperature": 0.05},
            ],
        },
    )

    assert readiness_response.status_code == 200
    readiness = readiness_response.json()
    assert readiness["status"] == "blocked"
    assert readiness["launch_allowed"] is False
    assert readiness["claim_gate"]["status"] == "blocked"
    assert "paper_workflow requires real LLM mode" in readiness["claim_gate"]["reasons"]
    assert any(
        check["check_id"] == "claim-gate.paper-workflow" and not check["passed"]
        for check in readiness["checks"]
    )

    run_response = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx_faithful_small",
            "mode": "mock",
            "claim_level": "paper_workflow",
            "experiment_id": "blocked-paper-workflow",
            "output_dir": str(tmp_path),
        },
    )

    assert run_response.status_code == 400
    assert "claim gate blocked launch" in run_response.json()["detail"]


def test_web_mock_run_writes_required_artifacts(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "mock",
            "max_iterations": 0,
            "parallel_mutations": 1,
            "experiment_id": "web-test",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    run_dir = tmp_path / "web-test"
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "trace_summary.json").exists()
    assert payload["metadata"]["run_state"] == "exported"
    assert payload["trace_summary"]["quality_gate"]["passed"] is True
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["agent_models"]["proposer"]["source"] == "role_default"
    assert metadata["agent_models"]["proposer"]["temperature"] == 0.55
    assert metadata["agent_models"]["proposer"]["reasoning_effort"] == "xhigh"


def test_web_mock_run_persists_readiness_report(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "mock",
            "target_solution_count": 1,
            "experiment_id": "readiness-run",
            "output_dir": str(tmp_path),
            "selected_algorithm_ids": ["piecewise_local_basis"],
            "manual_strategy_locks": [
                    {
                        "lock_id": "lock_piecewise",
                        "kind": "constraint",
                        "text": "Keep the piecewise local basis intent visible.",
                        "scope": "all_branches",
                        "inspection": {"required_terms": ["MODEL"]},
                    }
                ],
            },
    )

    assert response.status_code == 200
    run_dir = tmp_path / "readiness-run"
    readiness = json.loads((run_dir / "planning" / "readiness_report.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert readiness["readiness_version"] == "run_readiness.v1"
    assert readiness["selected_algorithm_ids"] == ["piecewise_local_basis"]
    assert readiness["manual_strategy_locks"][0]["lock_id"] == "lock_piecewise"
    assert readiness["manual_strategy_locks"][0]["inspection"] == {"required_terms": ["MODEL"]}
    assert config["readiness_report"]["readiness_id"] == readiness["readiness_id"]
    assert metadata["readiness_summary"]["readiness_id"] == readiness["readiness_id"]
    assert metadata["readiness_summary"]["status"] == "ready_with_warnings"
    policy_report = json.loads(
        (run_dir / "solutions" / "solution_000" / "policy_fidelity_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy_report["inspector_version"] == "strategy_fidelity.v1"
    assert policy_report["execution_allowed"] is True
    assert policy_report["summary"]["auditable_lock_count"] == 1


def test_web_mock_run_persists_agent_model_overrides(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "mock",
            "target_solution_count": 1,
            "experiment_id": "role-model-test",
            "output_dir": str(tmp_path),
            "agent_models": {
                "engineer": {
                    "model": "deepseek-v4-pro",
                    "temperature": 0.15,
                    "reasoning_effort": "xhigh",
                },
                "selector": {
                    "model": "gpt-5-mini",
                    "temperature": 0.05,
                    "reasoning_effort": "medium",
                },
            },
            "selector_panel": [
                {
                    "model": "gpt-5-mini",
                    "temperature": 0.05,
                    "reasoning_effort": "high",
                },
                {
                    "model": "deepseek-v4-pro",
                    "temperature": 0.1,
                    "reasoning_effort": "xhigh",
                },
            ],
            "selected_algorithm_ids": ["fourier_feature_mlp", "piecewise_local_basis"],
        },
    )

    assert response.status_code == 200
    run_dir = tmp_path / "role-model-test"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert config["agents"]["engineer"] == {
        "role": "engineer",
        "model": "deepseek-v4-pro",
        "temperature": 0.15,
        "reasoning_effort": "xhigh",
    }
    assert metadata["agent_models"]["engineer"]["model"] == "deepseek-v4-pro"
    assert metadata["agent_models"]["engineer"]["actual_model"] == "mock"
    assert [item["model"] for item in config["selector_panel"]] == ["gpt-5-mini", "deepseek-v4-pro"]
    assert metadata["selector_panel"]["ensemble_mode"] == "configured_selector_panel"
    assert [item["configured_model"] for item in metadata["selector_panel"]["members"]] == [
        "gpt-5-mini",
        "deepseek-v4-pro",
    ]
    assert config["strategy_seed_ids"] == ["fourier_feature_mlp", "piecewise_local_basis"]
    assert metadata["strategy_seed_ids"] == ["fourier_feature_mlp", "piecewise_local_basis"]


def test_web_mock_run_persists_problem_intake_and_prompt_context(tmp_path: Path) -> None:
    client = TestClient(create_app())
    plan = client.post(
        "/api/problem-intake/plan",
        json={
            "problem_statement": (
                "Solve a discontinuous function approximation problem with a local evaluator. "
                "Prefer compact basis functions and preserve the private validation boundary."
            ),
            "requirements": "No validation leakage; keep generated solution deterministic.",
            "evaluation_criteria": "validation_mse from evaluator-owned labels.",
            "data_description": "Training samples from a discontinuous scalar function.",
            "selected_algorithm_ids": ["piecewise_local_basis"],
            "target_solution_count": 1,
        },
    )
    assert plan.status_code == 200
    action_payload = plan.json()["actions"][0]["payload"]
    response = client.post(
        "/api/runs",
        json={
            **action_payload,
            "mode": "mock",
            "experiment_id": "problem-context-run",
            "output_dir": str(tmp_path),
            "target_solution_count": 1,
            "background": False,
        },
    )

    assert response.status_code == 200
    run_dir = tmp_path / "problem-context-run"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    planning = json.loads((run_dir / "planning" / "problem_intake.json").read_text(encoding="utf-8"))
    transcript = json.loads(
        (run_dir / "solutions" / "solution_000" / "transcripts" / "root_engineer.json").read_text(
            encoding="utf-8"
        )
    )
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")

    assert config["problem_intake"]["problem_statement"].startswith("Solve a discontinuous")
    assert metadata["problem_intake"] == config["problem_intake"]
    assert planning["planner_snapshot"]["planner_version"] == "problem_intake_keyword_planner.v1"
    assert "piecewise_local_basis" in metadata["planner_snapshot"]["selected_algorithm_ids"]
    assert "problem_intake_summary" in trace_text
    prompt = transcript[0]["prompt"]
    assert "User Problem Intake Context (Non-Contract)" in prompt
    assert "non-authoritative run context" in prompt
    assert "Solve a discontinuous function approximation problem" in prompt
    assert "Root baseline isolation" in prompt
    assert "non-authoritative strategy seeds" not in prompt


def test_web_resume_preserves_seed_models_and_problem_context(tmp_path: Path) -> None:
    client = TestClient(create_app())
    problem_intake = {
        "schema_version": 1,
        "problem_statement": "Solve a discontinuous function approximation problem with compact local bases.",
        "problem_summary": "discontinuous compact local bases",
    }
    planner_snapshot = {
        "schema_version": 1,
        "planner_version": "problem_intake_keyword_planner.v1",
        "selected_algorithm_ids": ["piecewise_local_basis"],
        "selected_seed_snapshot": [
            {
                "id": "piecewise_local_basis",
                "name": "Piecewise Local Basis",
                "family": "classical_ml",
                "status": "strategy_blueprint",
                "compatible_benchmark_families": ["function_approximation"],
                "benchmark_examples": ["function_approx"],
                "description": "Local basis seed.",
                "claim_boundary": "planning seed only",
                "safety_notes": "not evidence",
                "source_scope": "catalog",
                "implementation_path": None,
            }
        ],
    }
    first = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "mock",
            "target_solution_count": 1,
            "experiment_id": "resume-preserve",
            "output_dir": str(tmp_path),
            "agent_models": {
                "engineer": {
                    "model": "deepseek-v4-pro",
                    "temperature": 0.15,
                    "reasoning_effort": "high",
                }
            },
            "selected_algorithm_ids": ["piecewise_local_basis"],
            "problem_intake": problem_intake,
            "planner_snapshot": planner_snapshot,
        },
    )
    assert first.status_code == 200

    resumed = client.post(
        "/api/runs/resume-preserve/resume",
        json={
            "mode": "mock",
            "experiment_id": "resume-preserve",
            "output_dir": str(tmp_path),
            "max_iterations": 1,
            "parallel_mutations": 1,
        },
    )

    assert resumed.status_code == 200
    run_dir = tmp_path / "resume-preserve"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert config["strategy_seed_ids"] == ["piecewise_local_basis"]
    assert config["agents"]["engineer"]["model"] == "deepseek-v4-pro"
    assert config["problem_intake"] == problem_intake
    assert config["planner_snapshot"]["selected_algorithm_ids"] == ["piecewise_local_basis"]
    assert metadata["strategy_seed_ids"] == ["piecewise_local_basis"]
    assert metadata["problem_intake"] == problem_intake


def test_web_run_rejects_unknown_agent_role(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "dry_run",
            "experiment_id": "bad-role",
            "output_dir": str(tmp_path),
            "agent_models": {
                "global_boss": {
                    "model": "deepseek-v4-pro",
                    "temperature": 0.2,
                }
            },
        },
    )

    assert response.status_code == 400
    assert "Unknown agent role" in response.json()["detail"]


def test_web_run_budget_fields_are_applied(tmp_path: Path) -> None:
    client = TestClient(create_app())

    dry_run = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "dry_run",
            "target_solution_count": 5,
            "parallel_mutations": 2,
            "selector_vote_count": 4,
            "max_children_per_node": 3,
            "experiment_id": "budget-dry-run",
            "output_dir": str(tmp_path),
        },
    )

    assert dry_run.status_code == 200
    assert dry_run.json()["run_budget"] == {
        "target_solution_count": 5,
        "planned_solution_budget": 5,
        "max_iterations": 2,
        "parallel_mutations": 2,
        "selector_vote_count": 4,
        "max_children_per_node": 3,
        "selector_panel_member_count": 0,
    }

    mock_run = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "mock",
            "target_solution_count": 1,
            "parallel_mutations": 2,
            "selector_vote_count": 4,
            "max_children_per_node": 3,
            "experiment_id": "budget-mock-run",
            "output_dir": str(tmp_path),
        },
    )

    assert mock_run.status_code == 200
    config = json.loads((tmp_path / "budget-mock-run" / "config.json").read_text(encoding="utf-8"))
    assert config["evolution"]["max_iterations"] == 0
    assert config["evolution"]["parallel_mutations"] == 2
    assert config["evolution"]["selector_vote_count"] == 4
    assert config["evolution"]["max_children_per_node"] == 3


def test_selector_votes_and_solutions_are_read_only_evidence(tmp_path: Path) -> None:
    client = TestClient(create_app())
    run_dir = tmp_path / "paper-run"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "solutions" / "solution_000").mkdir(parents=True)
    (run_dir / "trace.jsonl").write_text("", encoding="utf-8")
    (run_dir / "reports" / "data_overview.svg").write_text("<svg />", encoding="utf-8")
    (run_dir / "solutions" / "solution_000" / "prediction_overview.svg").write_text("<svg />", encoding="utf-8")
    (run_dir / "solutions" / "solution_000" / "eval.json").write_text(
        json.dumps(
            {
                "metric": "validation_mse",
                "score": 0.25,
                "higher_is_better": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "solutions" / "solution_000" / "emergence_report.json").write_text(
        json.dumps(
            {
                "auditor_version": "emergence_audit.v1",
                "claim_level": "not_assessed",
                "blocking_gaps": ["root_solution_not_emergent"],
                "claim_boundary": "candidate only",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "solutions" / "solution_000" / "kb_application_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "implemented",
                "retrieved_entry_id": "fourier_features",
                "retrieved_title": "Fourier Features",
                "actionable_points": [{"id": "feature_expansion"}],
                "engineer_implemented_points": ["feature_expansion"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "solutions" / "solution_000" / "mutation_effect_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "changed_score_moved",
                "code_changed_from_parent": True,
                "diff_line_count": 7,
                "duplicate_of": None,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "reports" / "evolution_health.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "unique_code_count": 1,
                "duplicate_code_count": 0,
                "max_plateau_length": 1,
                "best_improvement": 0.05,
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_inputs" / "private_eval").mkdir(parents=True)
    (run_dir / "run_inputs" / "private_eval" / "val_data.npz").write_text("private", encoding="utf-8")
    (run_dir / "tree.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root_id": "solution_000",
                "nodes": [
                    {
                        "node_id": "solution_000",
                        "parent_id": None,
                        "children": [],
                        "status": "evaluated",
                        "score": {"metric": "validation_mse", "value": 0.3, "higher_is_better": False},
                        "method_tags": ["baseline"],
                        "score_delta_from_parent": None,
                        "num_debug_attempts": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "leaderboard.csv").write_text(
        "rank,node_id,parent_id,metric,score,status\n1,solution_000,,validation_mse,0.25,evaluated\n",
        encoding="utf-8",
    )

    empty_votes = client.get("/api/runs/paper-run/selector-votes", params={"output_dir": str(tmp_path)})
    assert empty_votes.status_code == 200
    assert empty_votes.json()["available"] is False
    assert empty_votes.json()["votes"] == []
    assert empty_votes.json()["selector_voting_exercised"] is False
    assert empty_votes.json()["selector_vote_events"] == 0

    (run_dir / "reports" / "selector_votes").mkdir()
    historical_vote = {
        "schema_version": 2,
        "selection_index": 1,
        "ensemble_mode": "configured_selector_panel",
        "selector_panel_members": [
            {
                "member_id": "selector_alpha",
                "configured_model": "gpt-5-mini",
                "actual_model": "mock",
                "provider": "MockLLMClient",
            }
        ],
        "selector_diversity": {
            "mock_evidence": True,
            "provider_diversity": False,
            "heterogeneous_selector_evidence": False,
        },
        "selected_parent_ids": ["solution_000"],
        "vote_counts": {"solution_000": 3},
        "votes": [{"candidate_id": "solution_000", "rationale": "best loss"}],
        "claim_boundary": "mock evidence only",
    }
    (run_dir / "reports" / "selector_votes" / "selection_000001.json").write_text(
        json.dumps(historical_vote),
        encoding="utf-8",
    )
    (run_dir / "reports" / "selector_votes.json").write_text(
        json.dumps(historical_vote),
        encoding="utf-8",
    )
    votes = client.get("/api/runs/paper-run/selector-votes", params={"output_dir": str(tmp_path)})
    assert votes.status_code == 200
    assert votes.json()["available"] is True
    assert votes.json()["schema_version"] == 2
    assert votes.json()["selector_voting_exercised"] is True
    assert votes.json()["selector_vote_events"] == 1
    assert votes.json()["ensemble_mode"] == "configured_selector_panel"
    assert votes.json()["selector_panel_members"][0]["actual_model"] == "mock"
    assert votes.json()["selector_diversity"]["heterogeneous_selector_evidence"] is False
    assert votes.json()["vote_counts"] == {"solution_000": 3}

    solutions = client.get("/api/runs/paper-run/solutions", params={"output_dir": str(tmp_path)})
    assert solutions.status_code == 200
    payload = solutions.json()
    assert payload["tree"]["node_count"] == 1
    assert payload["solutions"][0]["node_id"] == "solution_000"
    assert payload["solutions"][0]["score"] == 0.25
    assert payload["solutions"][0]["loss"] == 0.25
    assert payload["solutions"][0]["method_tags"] == ["baseline"]
    assert payload["solutions"][0]["policy_fidelity"] == {"available": False}
    assert payload["solutions"][0]["emergence_audit"] == {
        "available": True,
        "auditor_version": "emergence_audit.v1",
        "claim_level": "not_assessed",
        "blocking_gap_count": 1,
        "claim_boundary": "candidate only",
    }
    assert payload["solutions"][0]["kb_application"]["status"] == "implemented"
    assert payload["solutions"][0]["mutation_effect"]["status"] == "changed_score_moved"
    assert payload["evolution_health"]["unique_code_count"] == 1
    assert {figure["path"] for figure in payload["figures"]} == {
        "reports/data_overview.svg",
        "solutions/solution_000/prediction_overview.svg",
    }

    artifacts = client.get("/api/runs/paper-run/artifacts/run_inputs", params={"output_dir": str(tmp_path)})
    assert artifacts.status_code == 200
    assert "private_eval" not in {entry["path"].split("/")[-1] for entry in artifacts.json()["entries"]}
    private = client.get(
        "/api/runs/paper-run/artifacts/run_inputs/private_eval/val_data.npz",
        params={"output_dir": str(tmp_path)},
    )
    assert private.status_code == 403


def test_account_scoped_evidence_endpoints_use_account_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    for account_id, score in (("alice", 0.1), ("bob", 0.9)):
        run_dir = tmp_path / "accounts" / account_id / "runs" / "same-run-id"
        (run_dir / "reports").mkdir(parents=True)
        (run_dir / "solutions" / "solution_000").mkdir(parents=True)
        (run_dir / "trace.jsonl").write_text("", encoding="utf-8")
        (run_dir / "reports" / "selector_votes.json").write_text(
            json.dumps(
                {
                    "selected_parent_ids": ["solution_000"],
                    "vote_counts": {"solution_000": 1 if account_id == "alice" else 2},
                    "votes": [{"candidate_id": "solution_000", "account": account_id}],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "tree.json").write_text(
            json.dumps(
                {
                    "root_id": "solution_000",
                    "nodes": [
                        {
                            "node_id": "solution_000",
                            "parent_id": None,
                            "children": [],
                            "status": "evaluated",
                            "score": {"metric": "validation_mse", "value": score, "higher_is_better": False},
                            "method_tags": [account_id],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    alice_votes = client.get("/api/runs/same-run-id/selector-votes", params={"account_id": "alice"})
    bob_votes = client.get("/api/runs/same-run-id/selector-votes", params={"account_id": "bob"})
    alice_solutions = client.get("/api/runs/same-run-id/solutions", params={"account_id": "alice"})
    bob_solutions = client.get("/api/runs/same-run-id/solutions", params={"account_id": "bob"})

    assert alice_votes.status_code == 200
    assert bob_votes.status_code == 200
    assert alice_votes.json()["vote_counts"] == {"solution_000": 1}
    assert bob_votes.json()["vote_counts"] == {"solution_000": 2}
    assert alice_solutions.json()["solutions"][0]["score"] == 0.1
    assert bob_solutions.json()["solutions"][0]["score"] == 0.9


def test_web_real_run_requires_server_flag_and_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    monkeypatch.delenv("AGENTICSCIML_ENABLE_REAL_WEB_RUNS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app())

    disabled = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "real",
            "account_id": "alice",
            "experiment_id": "real-web-test",
            "real_confirmed": True,
        },
    )
    assert disabled.status_code == 403
    assert "disabled" in disabled.json()["detail"]

    monkeypatch.setenv("AGENTICSCIML_ENABLE_REAL_WEB_RUNS", "1")
    unconfirmed = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "real",
            "account_id": "alice",
            "experiment_id": "real-web-test",
        },
    )
    assert unconfirmed.status_code == 400
    assert "real_confirmed" in unconfirmed.json()["detail"]

    missing_key = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "real",
            "account_id": "alice",
            "experiment_id": "real-web-missing-key",
            "real_confirmed": True,
        },
    )
    assert missing_key.status_code == 200
    missing_key_payload = missing_key.json()
    assert missing_key_payload["status"] == "failed"
    assert "OPENAI_API_KEY" in missing_key_payload["record"]["error"]


def test_account_scoped_runs_reject_path_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    output_override = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "dry_run",
            "account_id": "alice",
            "experiment_id": "path-override",
            "output_dir": str(tmp_path),
        },
    )
    assert output_override.status_code == 400
    assert "account runs directory" in output_override.json()["detail"]

    benchmark_dir_override = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "benchmark_dir": "examples/function_approx",
            "mode": "dry_run",
            "account_id": "alice",
            "experiment_id": "benchmark-dir-override",
        },
    )
    assert benchmark_dir_override.status_code == 400
    assert "benchmark_dir" in benchmark_dir_override.json()["detail"]

    benchmark_path = client.post(
        "/api/runs",
        json={
            "benchmark": "examples/function_approx",
            "mode": "dry_run",
            "account_id": "alice",
            "experiment_id": "benchmark-path",
        },
    )
    assert benchmark_path.status_code == 400
    assert "catalog names" in benchmark_path.json()["detail"]


def test_web_artifacts_reject_path_escape(tmp_path: Path) -> None:
    client = TestClient(create_app())
    run_dir = tmp_path / "web-test"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text("", encoding="utf-8")
    (run_dir / "safe.txt").write_text("ok", encoding="utf-8")

    safe = client.get(
        "/api/runs/web-test/artifacts/safe.txt",
        params={"output_dir": str(tmp_path)},
    )
    assert safe.status_code == 200
    assert safe.json()["content"] == "ok"

    encoded_dotdot = client.get(
        "/api/runs/web-test/artifacts/%2E%2E/pyproject.toml",
        params={"output_dir": str(tmp_path)},
    )
    assert encoded_dotdot.status_code in {400, 404}

    if hasattr(os, "symlink"):
        os.symlink(tmp_path, run_dir / "escape")
        symlink_escape = client.get(
            "/api/runs/web-test/artifacts/escape",
            params={"output_dir": str(tmp_path)},
        )
        assert symlink_escape.status_code == 400


def test_solver_chat_defaults_to_ask_without_actions(tmp_path: Path) -> None:
    client = TestClient(create_app())

    identity = client.post(
        "/api/solver/chat",
        json={
            "message": "你是谁",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert identity.status_code == 200
    identity_payload = identity.json()
    assert identity_payload["assistant_mode"] == "ask"
    assert identity_payload["model_settings"] == {
        "reasoning_effort": "medium",
        "temperature": 0.2,
        "source": "mode_default",
    }
    assert identity_payload["actions"] == []
    assert identity_payload["warnings"] == []
    assert "AgenticSciML 助手" in identity_payload["reply"]

    capabilities = client.post(
        "/api/solver/chat",
        json={
            "message": "你能做什么",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert capabilities.status_code == 200
    capabilities_payload = capabilities.json()
    assert capabilities_payload["assistant_mode"] == "ask"
    assert capabilities_payload["actions"] == []
    assert capabilities_payload["warnings"] == []
    assert "benchmark" in capabilities_payload["reply"]
    assert "Plan 模式" in capabilities_payload["reply"]
    assert "Agent 模式" in capabilities_payload["reply"]

    start = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )
    assert start.status_code == 200
    start_payload = start.json()
    assert start_payload["assistant_mode"] == "ask"
    assert start_payload["actions"] == []
    assert any("Ask mode" in warning for warning in start_payload["warnings"])


def test_solver_chat_mode_model_settings_are_distinct(tmp_path: Path) -> None:
    client = TestClient(create_app())

    expected = {
        "ask": {"reasoning_effort": "medium", "temperature": 0.2, "source": "mode_default"},
        "plan": {"reasoning_effort": "high", "temperature": 0.35, "source": "mode_default"},
        "agent": {"reasoning_effort": "high", "temperature": 0.1, "source": "mode_default"},
    }
    for assistant_mode, settings in expected.items():
        response = client.post(
            "/api/solver/chat",
            json={
                "message": "跑一个 mock 实验",
                "selected_benchmark": "function_approx",
                "mode": "mock",
                "assistant_mode": assistant_mode,
                "workspace_scope": "account",
                "account_id": "alice",
                "output_dir": str(tmp_path),
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["assistant_mode"] == assistant_mode
        assert payload["model_settings"] == settings

    override = client.post(
        "/api/solver/chat",
        json={
            "message": "你能做什么",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "ask",
            "reasoning_effort": "xhigh",
            "temperature": 0.45,
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert override.status_code == 200
    assert override.json()["model_settings"] == {
        "reasoning_effort": "xhigh",
        "temperature": 0.45,
        "source": "request_override",
    }


def test_solver_chat_plan_and_agent_modes_return_structured_actions(tmp_path: Path) -> None:
    client = TestClient(create_app())

    plan = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "plan",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )
    assert plan.status_code == 200
    plan_payload = plan.json()
    assert plan_payload["assistant_mode"] == "plan"
    assert plan_payload["actions"][0]["type"] == "start_run"
    assert any("Plan mode" in warning for warning in plan_payload["warnings"])

    start = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "account_id": "alice",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert start.status_code == 200
    start_payload = start.json()
    assert start_payload["assistant_mode"] == "agent"
    assert start_payload["actions"][0]["type"] == "start_run"
    assert start_payload["actions"][0]["payload"]["account_id"] == "alice"
    assert start_payload["actions"][0]["payload"]["background"] is True


def test_solver_chat_agent_can_plan_benchmark_and_seeded_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    response = client.post(
        "/api/solver/chat",
        json={
            "message": (
                "请用 Agent 模式自主选择 benchmark 和解法求解："
                "Reconstruct two-dimensional cylinder wake vorticity fields from sparse noisy temporal sensors. "
                "Use lagged sensor history and preserve smooth band-limited spatial structure. "
                "运行 mock 实验并保留 trace evidence。"
            ),
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "account_id": "alice",
            "workspace_scope": "account",
            "target_solution_count": 1,
            "parallel_mutations": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["assistant_mode"] == "agent"
    action = payload["actions"][0]
    assert action["type"] == "start_run"
    action_payload = action["payload"]
    assert action_payload["benchmark"] == "cylinder_wake_reconstruction_faithful_small"
    assert action_payload["account_id"] == "alice"
    assert "paper_cylinder_bandlimited_filter" in action_payload["selected_algorithm_ids"]
    assert action_payload["problem_intake"]["problem_statement"].startswith("请用 Agent 模式")
    assert action_payload["planner_snapshot"]["planner_version"] == "problem_intake_keyword_planner.v1"
    assert payload["artifacts"][0]["kind"] == "problem_intake_plan"

    run_response = client.post(
        "/api/runs",
        json={
            **action_payload,
            "experiment_id": "agent-planned-run",
            "background": False,
        },
    )

    assert run_response.status_code == 200
    run_dir = tmp_path / "accounts" / "alice" / "runs" / "agent-planned-run"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert metadata["benchmark_name"] == "cylinder_wake_reconstruction_faithful_small"
    assert "paper_cylinder_bandlimited_filter" in metadata["strategy_seed_ids"]
    assert config["problem_intake"]["problem_statement"].startswith("请用 Agent 模式")
    trace_summary = json.loads((run_dir / "trace_summary.json").read_text(encoding="utf-8"))
    assert trace_summary["quality_gate"]["passed"] is True
    describe_response = client.get("/api/runs/agent-planned-run", params={"account_id": "alice"})
    assert describe_response.status_code == 200
    assert describe_response.json()["metadata"]["champion_node_id"] == metadata["champion"]


def test_solver_chat_agent_does_not_open_code_for_solution_loss_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    response = client.post(
        "/api/solver/chat",
        json={
            "message": (
                "我想求解 cylinder wake sparse sensor reconstruction benchmark。"
                "请自主选择合适算法库策略，启动一个 mock run，解法量 3，"
                "比较 solution loss，并告诉我 champion 和 trace gate。"
            ),
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "account_id": "alice",
            "workspace_scope": "account",
            "target_solution_count": 3,
            "parallel_mutations": 1,
        },
    )

    assert response.status_code == 200
    actions = response.json()["actions"]
    assert [action["type"] for action in actions] == ["start_run"]
    assert actions[0]["payload"]["benchmark"] == "cylinder_wake_reconstruction_faithful_small"


def test_solver_chat_agent_rejects_shared_repo_scope(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/solver/chat",
        json={
            "message": "打开代码",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "account_id": "alice",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["actions"] == []
    assert any("shared repo workspace" in warning for warning in payload["warnings"])


def test_solver_chat_agent_requires_account_id(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["actions"] == []
    assert any("requires account_id" in warning for warning in payload["warnings"])


def test_solver_chat_returns_real_mode_warnings(tmp_path: Path) -> None:
    client = TestClient(create_app())

    real_dry = client.post(
        "/api/solver/chat",
        json={
            "message": "解释 trace",
            "selected_benchmark": "function_approx",
            "mode": "real",
            "assistant_mode": "ask",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )
    assert real_dry.status_code == 200
    warnings = real_dry.json()["warnings"]
    assert any("Real LLM mode" in warning for warning in warnings)
    assert any("Select an active run" in warning for warning in warnings)


def test_code_server_url_rejects_repo_workspace_by_default() -> None:
    client = TestClient(create_app())

    response = client.get("/api/code-server/url", params={"scope": "repo"})

    assert response.status_code == 403
    assert "shared repo workspace is disabled" in response.json()["detail"]


def test_code_server_url_uses_loopback_without_token_when_repo_workspace_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ALLOW_REPO_WORKSPACE", "1")
    client = TestClient(create_app())

    response = client.get("/api/code-server/url", params={"scope": "repo"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["url"].startswith("http://127.0.0.1:8080/")
    assert "PASSWORD=" not in payload["url"]
    assert "PASSWORD=" not in payload["command_hint"]
    assert "--auth none" in payload["command_hint"]
    assert payload["auth_mode"] == "upstream_account"
    assert payload["workspace"].endswith("New project 11")


def test_code_server_workspaces_list_independent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ALLOW_REPO_WORKSPACE", "1")
    client = TestClient(create_app())
    run_dir = tmp_path / "web-test"
    (run_dir / "champion").mkdir(parents=True)
    (run_dir / "solutions" / "solution_000").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        '{"run_state": "exported", "champion_node_id": "solution_000"}',
        encoding="utf-8",
    )

    response = client.get(
        "/api/code-server/workspaces",
        params={"run_id": "web-test", "output_dir": str(tmp_path)},
    )

    assert response.status_code == 200
    workspaces = {item["id"]: item for item in response.json()["workspaces"]}
    assert workspaces["repo"]["workspace"].endswith("New project 11")
    assert workspaces["run:web-test"]["workspace"] == str(run_dir.resolve())
    assert workspaces["champion:web-test"]["workspace"] == str((run_dir / "champion").resolve())
    assert workspaces["solution:web-test:solution_000"]["workspace"] == str(
        (run_dir / "solutions" / "solution_000").resolve()
    )
    for item in workspaces.values():
        assert "PASSWORD=" not in item["url"]
        assert "PASSWORD=" not in item["command_hint"]
        assert "--auth none" in item["command_hint"]
        assert item["auth_mode"] == "upstream_account"


def test_account_workspaces_are_isolated_local_namespaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    created = client.post("/api/accounts", json={"account_id": "alice", "display_name": "Alice"})
    assert created.status_code == 200
    account = created.json()["account"]
    assert account["account_id"] == "alice"
    assert account["auth"] == "not_implemented"
    assert Path(account["workspace_root"]).is_relative_to(tmp_path / "accounts" / "alice")

    run_dir = Path(account["runs_dir"]) / "alice-run"
    (run_dir / "solutions" / "solution_000").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text('{"run_state": "partial"}', encoding="utf-8")

    response = client.get("/api/code-server/workspaces", params={"account_id": "alice"})

    assert response.status_code == 200
    workspaces = {item["id"]: item for item in response.json()["workspaces"]}
    assert "repo" not in workspaces
    assert workspaces["account:alice"]["scope"] == "account"
    assert workspaces["account:alice"]["isolation"] == "account"
    assert workspaces["account:alice"]["workspace"] == account["workspace_root"]
    assert workspaces["run:alice-run"]["workspace"] == str(run_dir.resolve())
    assert workspaces["solution:alice-run:solution_000"]["account_id"] == "alice"

    invalid = client.get("/api/code-server/workspaces", params={"account_id": "../alice"})
    assert invalid.status_code == 400


def test_accounts_endpoint_tolerates_parallel_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    with ThreadPoolExecutor(max_workers=6) as executor:
        responses = list(executor.map(lambda _: client.get("/api/accounts"), range(12)))

    assert {response.status_code for response in responses} == {200}
    assert all(response.json()["accounts"][0]["account_id"] == "local" for response in responses)


def test_account_run_records_do_not_collide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    for account_id in ("alice", "bob"):
        response = client.post(
            "/api/runs",
            json={
                "benchmark": "function_approx",
                "mode": "dry_run",
                "account_id": account_id,
                "experiment_id": "same-run-id",
            },
        )
        assert response.status_code == 200

    alice = client.get("/api/runs/same-run-id", params={"account_id": "alice"})
    bob = client.get("/api/runs/same-run-id", params={"account_id": "bob"})

    assert alice.status_code == 200
    assert bob.status_code == 200
    alice_record = alice.json()["record"]
    bob_record = bob.json()["record"]
    assert alice_record["status"] == "dry_run"
    assert bob_record["status"] == "dry_run"
    assert "/alice/" in alice_record["run_dir"]
    assert "/bob/" in bob_record["run_dir"]
    assert alice_record["run_dir"] != bob_record["run_dir"]
