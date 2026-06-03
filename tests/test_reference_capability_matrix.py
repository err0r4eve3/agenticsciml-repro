from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.reference_capability_matrix import (
    build_reference_capability_matrix,
    write_reference_capability_matrix,
)


def test_reference_capability_matrix_maps_sources_to_engineering_artifacts() -> None:
    matrix = build_reference_capability_matrix(
        problem_intake=_complete_problem_intake(),
        expert_blueprint_id="fluid_pde",
    )

    assert matrix["status"] == "ready_for_offline_planning"
    assert matrix["scientific_claim_supported"] is False
    mechanism_ids = {item["mechanism_id"] for item in matrix["mechanisms"]}
    assert {
        "scientific_problem_decomposition",
        "fluid_pde_visual_physics_audit",
        "athena_graft_action_reward_trace",
        "code_orchestrated_agent_boundary",
        "domain_negative_result_closure",
    }.issubset(mechanism_ids)
    for item in matrix["mechanisms"]:
        assert item["reference_mechanism"]
        assert item["project_module_mapping"]
        assert item["required_artifact"]
        assert item["fail_closed_blocker"]
        assert item["no_key_local_implementation_path"]
        assert item["future_real_run_requirement"]
    assert matrix["problem_intake_rubric"]["status"] == "ready"
    assert matrix["problem_intake_rubric"]["passed"] is True
    assert "scientific discovery" not in matrix["claim_boundary"].lower()


def test_reference_capability_matrix_blocks_incomplete_problem_intake() -> None:
    matrix = build_reference_capability_matrix(
        problem_intake={"hypothesis": "Sparse sensors should reconstruct coherent wake modes."},
        expert_blueprint_id=None,
    )

    assert matrix["status"] == "blocked"
    assert matrix["scientific_claim_supported"] is False
    rubric = matrix["problem_intake_rubric"]
    assert rubric["passed"] is False
    missing = {item["field_id"] for item in rubric["missing_fields"]}
    assert {"observable", "metric", "failure_modes", "physical_constraints", "domain_review_checklist"}.issubset(
        missing
    )
    assert "expert_blueprint" in {item["field_id"] for item in rubric["missing_fields"]}


def test_write_reference_capability_matrix_outputs_json_and_markdown(tmp_path: Path) -> None:
    result = write_reference_capability_matrix(
        output_dir=tmp_path,
        problem_intake=_complete_problem_intake(),
        expert_blueprint_id="fluid_pde",
    )

    matrix_path = Path(result["paths"]["matrix_json"])
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix_path == tmp_path / "reference_capability_matrix.json"
    assert matrix["problem_intake_rubric"]["status"] == "ready"
    assert (tmp_path / "reference_capability_matrix.md").exists()


def test_cli_build_reference_capability_matrix_writes_artifact(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    intake_path = tmp_path / "problem_intake.json"
    intake_path.write_text(json.dumps(_complete_problem_intake()), encoding="utf-8")
    output_dir = tmp_path / "reference capability"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "build-reference-capability-matrix",
            "--output-dir",
            str(output_dir),
            "--problem-intake-json",
            str(intake_path),
            "--expert-blueprint-id",
            "fluid_pde",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    matrix_path = Path(result.stdout.strip())
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix_path == output_dir / "reference_capability_matrix.json"
    assert matrix["status"] == "ready_for_offline_planning"


def _complete_problem_intake() -> dict[str, object]:
    return {
        "problem_summary": "Sparse-sensor cylinder wake reconstruction under a prediction-only evaluator.",
        "hypothesis": "Lagged sparse sensors preserve enough coherent structure for low-rank wake reconstruction.",
        "observable": ["sensor_history", "vorticity_field"],
        "metric": "relative_l2",
        "failure_modes": ["boundary artifacts", "phase drift", "high-frequency noise"],
        "physical_constraints": ["wake smoothness", "boundary consistency", "residual proxy stability"],
        "domain_review_checklist": [
            "hidden labels remain evaluator-only",
            "visual diagnostics reviewed",
            "negative samples audited",
        ],
        "data_source": "local faithful-small synthetic cylinder wake fixture",
    }
