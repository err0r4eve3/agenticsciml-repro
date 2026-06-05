from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.real_problem_closure import (
    build_real_problem_closure_plan,
    write_real_problem_closure_plan,
)


def test_real_problem_closure_blocks_missing_real_assets() -> None:
    plan = build_real_problem_closure_plan(
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        selector_panel=[],
        resource_constraints={},
        expert_blueprint_id=None,
        env={},
    )

    blocked_module_ids = {item["module_id"] for item in plan["blocked_modules"]}

    assert plan["schema_version"] == 1
    assert plan["closure_version"] == "real_problem_closure.v1"
    assert plan["status"] == "blocked"
    assert plan["multi_agent_real_problem_claim_supported"] is False
    assert "real_llm_execution" in blocked_module_ids
    assert "real_multimodal_input" in blocked_module_ids
    assert "heterogeneous_selector" in blocked_module_ids
    assert "paper_like_benchmark" in blocked_module_ids
    assert "domain_approval" in blocked_module_ids
    assert "completed_run_audit" in blocked_module_ids
    assert "scientific_claim_supported=true" not in json.dumps(plan)


def test_real_problem_closure_includes_reference_capability_matrix() -> None:
    plan = build_real_problem_closure_plan(
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        problem_intake={
            "hypothesis": "Lagged sparse sensors preserve coherent wake modes.",
            "observable": ["sensor_history", "vorticity_field"],
            "metric": "relative_l2",
            "failure_modes": ["phase drift", "boundary artifacts"],
            "physical_constraints": ["boundary consistency", "residual proxy stability"],
            "domain_review_checklist": ["failure samples reviewed", "claim boundary reviewed"],
        },
        expert_blueprint_id="fluid_pde",
        env={},
    )

    matrix = plan["reference_capability_matrix"]
    assert matrix["problem_intake_rubric"]["passed"] is True
    assert matrix["scientific_claim_supported"] is False
    context_pack = plan["llm_problem_context_pack"]
    assert context_pack["status"] == "blocked"
    assert context_pack["llm_execution_required"] is True
    assert context_pack["scientific_claim_supported"] is False
    assert plan["multi_agent_real_problem_claim_supported"] is False


def test_write_real_problem_closure_plan_outputs_json_and_markdown(tmp_path: Path) -> None:
    result = write_real_problem_closure_plan(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        env={},
    )

    plan = json.loads(Path(result["paths"]["plan_json"]).read_text(encoding="utf-8"))

    assert plan["status"] == "blocked"
    assert plan["multi_agent_real_problem_claim_supported"] is False
    assert Path(result["paths"]["plan_md"]).exists()
    assert Path(result["paths"]["paper_workflow_readiness_json"]).exists()


def test_real_problem_closure_accepts_verified_completed_run_audit(tmp_path: Path) -> None:
    completed_run_dir = _write_completed_run_audit(tmp_path / "completed-run")

    plan = build_real_problem_closure_plan(
        benchmark_dir=Path("examples/function_approx").resolve(),
        completed_run_dir=completed_run_dir,
        env={},
    )

    blocked_module_ids = {item["module_id"] for item in plan["blocked_modules"]}

    assert plan["completed_run_audit"]["ready"] is True
    assert plan["completed_run_audit"]["trace_quality_gate_passed"] is True
    assert plan["completed_run_audit"]["scientific_readiness_status"] == "blocked"
    assert "completed_run_audit" not in blocked_module_ids
    assert plan["status"] == "blocked"
    assert plan["multi_agent_real_problem_claim_supported"] is False


def test_real_problem_closure_rejects_inconsistent_completed_run_claim(tmp_path: Path) -> None:
    completed_run_dir = _write_completed_run_audit(tmp_path / "completed-run")
    trace_path = completed_run_dir / "trace_summary.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["claim_gate"]["scientific_claim_supported"] = True
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    plan = build_real_problem_closure_plan(
        benchmark_dir=Path("examples/function_approx").resolve(),
        completed_run_dir=completed_run_dir,
        env={},
    )

    blocked_module_ids = {item["module_id"] for item in plan["blocked_modules"]}

    assert plan["completed_run_audit"]["ready"] is False
    assert (
        "trace_summary claim_gate must not support scientific claims when readiness is blocked"
        in plan["completed_run_audit"]["issues"]
    )
    assert "completed_run_audit" in blocked_module_ids


def test_cli_plan_real_problem_closure_writes_blocked_plan(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    output_dir = tmp_path / "real problem closure"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-real-problem-closure",
            "examples/cylinder_wake_reconstruction_faithful_small",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    plan_path = Path(result.stdout.strip())
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan_path == output_dir / "real_problem_closure_plan.json"
    assert plan["status"] == "blocked"
    assert (output_dir / "real_problem_closure_plan.md").exists()
    assert (output_dir / "paper_workflow_readiness.json").exists()


def test_cli_plan_real_problem_closure_accepts_completed_run_dir(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    completed_run_dir = _write_completed_run_audit(tmp_path / "completed-run")
    output_dir = tmp_path / "closure"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-real-problem-closure",
            "examples/function_approx",
            "--completed-run-dir",
            str(completed_run_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    plan_path = Path(result.stdout.strip())
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["completed_run_audit"]["ready"] is True
    assert "completed_run_audit" not in {item["module_id"] for item in plan["blocked_modules"]}


def test_cli_plan_real_problem_closure_can_fail_on_blockers(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-real-problem-closure",
            "examples/function_approx",
            "--output-dir",
            str(tmp_path),
            "--fail-on-blockers",
        ],
        check=False,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert result.returncode == 1
    assert (tmp_path / "real_problem_closure_plan.json").exists()


def _write_completed_run_audit(run_dir: Path) -> Path:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_state": "exported",
                "llm_mode": "real",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trace_summary.json").write_text(
        json.dumps(
            {
                "quality_gate": {
                    "passed": True,
                },
                "claim_gate": {
                    "scientific_claim_supported": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "scientific_discovery_readiness.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "scientific_claim_supported": False,
            }
        ),
        encoding="utf-8",
    )
    return run_dir
