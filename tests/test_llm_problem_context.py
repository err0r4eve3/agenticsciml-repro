from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.llm_problem_context import (
    LLM_PROBLEM_CONTEXT_PACK_JSON,
    LLM_PROBLEM_CONTEXT_PACK_MD,
    build_llm_problem_context_pack,
    write_llm_problem_context_pack,
)


def test_llm_problem_context_pack_assigns_bounded_agent_roles_without_claim() -> None:
    pack = build_llm_problem_context_pack(
        problem_intake=_complete_problem_intake(),
        expert_blueprint_id="fluid_pde",
        resource_constraints=_resource_constraints(),
    )

    assert pack["status"] == "ready_for_llm_context"
    assert pack["llm_execution_required"] is True
    assert pack["offline_validated_only"] is True
    assert pack["scientific_claim_supported"] is False

    roles = {item["role_id"]: item for item in pack["role_task_plan"]}
    assert {
        "data_analyst",
        "root_engineer",
        "proposer",
        "critic",
        "engineer",
        "debugger",
        "selector",
        "result_analyst",
        "visual_audit",
    }.issubset(roles)
    assert roles["root_engineer"]["model_policy"]["reasoning_effort"] == "xhigh"
    assert roles["engineer"]["expected_output_schema"]["type"] == "structured_patch"
    assert roles["visual_audit"]["requires_image_capable_provider"] is True
    assert roles["visual_audit"]["model_policy"]["reasoning_effort"] == "high"
    assert "Unknown role default" not in roles["visual_audit"]["model_policy"]["rationale"]
    assert any(
        "private validation labels" in action
        for role in roles.values()
        for action in role["forbidden_actions"]
    )

    assert "evaluation scoring" in pack["orchestrator_owned_decisions"]
    assert "champion selection" in pack["orchestrator_owned_decisions"]
    assert "claim gate" in pack["orchestrator_owned_decisions"]
    assert pack["problem_decomposition"]["metric"] == "relative_l2"
    prompt_controls = {item["control_id"]: item for item in pack["prompt_quality_controls"]}
    assert "paper_context_is_non_authoritative" in prompt_controls
    assert "diagnostics_match_problem_family" in prompt_controls
    assert "prediction error alone" in prompt_controls["diagnostics_match_problem_family"]["paper_informed_reason"]
    assert "benchmark names" in prompt_controls["bilingual_wiki_preserves_identifiers"]["requirement"]


def test_llm_problem_context_pack_blocks_incomplete_intake_and_resource_limits() -> None:
    pack = build_llm_problem_context_pack(
        problem_intake={"hypothesis": "Sparse sensors should reconstruct wake fields."},
        expert_blueprint_id=None,
        resource_constraints={},
    )

    blocker_ids = {item["blocker_id"] for item in pack["blockers"]}

    assert pack["status"] == "blocked"
    assert "missing_problem_intake_field:observable" in blocker_ids
    assert "missing_problem_intake_field:metric" in blocker_ids
    assert "missing_expert_blueprint" in blocker_ids
    assert "missing_resource_constraint:timeout_s" in blocker_ids
    assert pack["scientific_claim_supported"] is False


def test_write_llm_problem_context_pack_outputs_json_and_markdown(tmp_path: Path) -> None:
    result = write_llm_problem_context_pack(
        output_dir=tmp_path,
        problem_intake=_complete_problem_intake(),
        expert_blueprint_id="fluid_pde",
        resource_constraints=_resource_constraints(),
    )

    pack_path = Path(result["paths"]["pack_json"])
    markdown_path = Path(result["paths"]["pack_md"])
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert pack_path == tmp_path / LLM_PROBLEM_CONTEXT_PACK_JSON
    assert markdown_path == tmp_path / LLM_PROBLEM_CONTEXT_PACK_MD
    assert pack["status"] == "ready_for_llm_context"
    assert "# LLM Problem Context Pack" in markdown
    assert "## Prompt Quality Controls" in markdown
    assert "root_engineer" in markdown


def test_cli_build_llm_problem_context_writes_pack(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    output_dir = tmp_path / "llm context"
    intake_path = tmp_path / "problem_intake.json"
    intake_path.write_text(json.dumps(_complete_problem_intake()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "build-llm-problem-context",
            "--output-dir",
            str(output_dir),
            "--problem-intake-json",
            str(intake_path),
            "--resource-constraints-json",
            json.dumps(_resource_constraints()),
            "--expert-blueprint-id",
            "fluid_pde",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    pack_path = Path(result.stdout.strip())
    pack = json.loads(pack_path.read_text(encoding="utf-8"))

    assert pack_path == output_dir / LLM_PROBLEM_CONTEXT_PACK_JSON
    assert pack["status"] == "ready_for_llm_context"
    assert (output_dir / LLM_PROBLEM_CONTEXT_PACK_MD).exists()


def _complete_problem_intake() -> dict[str, object]:
    return {
        "hypothesis": "Lagged sparse sensors retain coherent wake modes for reconstruction.",
        "observable": ["sensor_history", "vorticity_field"],
        "metric": "relative_l2",
        "failure_modes": ["phase drift", "boundary artifacts"],
        "physical_constraints": ["boundary consistency", "smooth residual proxy"],
        "domain_review_checklist": ["failure samples reviewed", "claim boundary reviewed"],
        "data_source": "local faithful-small synthetic fixture",
    }


def _resource_constraints() -> dict[str, object]:
    return {
        "cpu": "local",
        "gpu": False,
        "timeout_s": 120,
        "dependency_limits": ["numpy", "scipy"],
        "data_limits": "prediction-only faithful-small fixture",
    }
