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
