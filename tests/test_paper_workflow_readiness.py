from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.paper_workflow_readiness import (
    DOMAIN_CHECKLIST_IDS,
    build_paper_workflow_readiness_bundle,
    write_paper_workflow_readiness_bundle,
)


def test_paper_workflow_readiness_blocks_missing_real_assets(tmp_path: Path) -> None:
    bundle = build_paper_workflow_readiness_bundle(
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        selector_panel=[],
        resource_constraints={},
        expert_blueprint_id=None,
        env={},
    )

    assert bundle["status"] == "blocked"
    assert bundle["ready_to_execute_real_run"] is False
    assert bundle["scientific_claim_supported"] is False
    blockers = {item["check_id"] for item in bundle["blockers"]}
    assert "real_llm_credentials" in blockers
    assert "real_multimodal_provider" in blockers
    assert "heterogeneous_real_selector" in blockers
    assert "paper_like_benchmark" in blockers
    assert "domain_approval_packet" in blockers
    assert "multi_seed_ablation_output" in blockers
    assert bundle["provider_readiness"]["secret_policy"]
    assert "sk-test" not in json.dumps(bundle)


def test_paper_workflow_readiness_accepts_available_non_paper_assets_but_keeps_benchmark_blocker(
    tmp_path: Path,
) -> None:
    domain_path = tmp_path / "domain_approval.json"
    domain_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approved": True,
                "reviewer": "fluid-reviewer",
                "review_notes": "Reviewed evaluator, claim boundary, and failure samples.",
                "checklist": {check_id: True for check_id in DOMAIN_CHECKLIST_IDS},
            }
        ),
        encoding="utf-8",
    )
    ablation_dir = tmp_path / "ablation"
    _write_ablation_output(ablation_dir, seeds=[0, 1], variants=["root_only", "kb"])

    bundle = build_paper_workflow_readiness_bundle(
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        selector_panel=[
            {"role": "selector_openai", "model": "gpt-5-mini"},
            {"role": "selector_deepseek", "model": "deepseek-v4-pro", "base_url": "https://api.deepseek.com"},
        ],
        resource_constraints={
            "cpu": "local",
            "gpu": False,
            "timeout_s": 120,
            "dependency_limits": ["numpy"],
            "data_limits": "faithful-small fixture",
        },
        expert_blueprint_id="fluid_pde",
        domain_approval_path=domain_path,
        ablation_output_dir=ablation_dir,
        expected_seeds=[0, 1],
        expected_variants=["root_only", "kb"],
        env={
            "OPENAI_API_KEY": "redacted-test-key",
            "OPENAI_MODEL": "gpt-5-mini",
            "AGENTICSCIML_MAX_LLM_CALLS": "20",
        },
    )

    blockers = {item["check_id"] for item in bundle["blockers"]}
    assert bundle["provider_readiness"]["api_key_present"] is True
    assert bundle["provider_readiness"]["budget_configured"] is True
    assert bundle["selector_readiness"]["heterogeneous_selector_candidate"] is True
    assert bundle["domain_approval_readiness"]["approved"] is True
    assert bundle["multi_seed_ablation_readiness"]["verified_multi_seed_ablation"] is True
    assert "paper_like_benchmark" in blockers
    assert bundle["paper_benchmark_readiness"]["fidelity_level"] == "faithful-small"
    assert "redacted-test-key" not in json.dumps(bundle)


def test_write_paper_workflow_readiness_bundle_outputs_templates(tmp_path: Path) -> None:
    result = write_paper_workflow_readiness_bundle(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        env={},
    )

    plan = json.loads(Path(result["paths"]["plan_json"]).read_text(encoding="utf-8"))
    domain_template = json.loads(
        Path(result["paths"]["domain_approval_template"]).read_text(encoding="utf-8")
    )
    benchmark_template = json.loads(
        Path(result["paths"]["paper_benchmark_manifest_template"]).read_text(encoding="utf-8")
    )

    assert plan["status"] == "blocked"
    assert Path(result["paths"]["plan_md"]).exists()
    assert Path(result["paths"]["commands_md"]).exists()
    assert domain_template["checklist"] == {check_id: False for check_id in DOMAIN_CHECKLIST_IDS}
    assert benchmark_template["benchmark_name"] == "function_approx"
    assert benchmark_template["data_provenance"]["private_label_protocol"]


def test_cli_plan_paper_workflow_writes_blocked_bundle(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    output_dir = tmp_path / "paper workflow"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-paper-workflow",
            "examples/cylinder_wake_reconstruction_faithful_small",
            "--output-dir",
            str(output_dir),
            "--selector-panel-json",
            '[{"model":"gpt-5-mini"},{"model":"deepseek-v4-pro","base_url":"https://api.deepseek.com"}]',
            "--resource-constraints-json",
            '{"cpu":"local","gpu":false,"timeout_s":120,"dependency_limits":["numpy"],"data_limits":"fixture"}',
            "--expert-blueprint-id",
            "fluid_pde",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    plan_path = Path(result.stdout.strip())
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan_path == output_dir / "paper_workflow_readiness.json"
    assert plan["status"] == "blocked"
    assert (output_dir / "paper_workflow_readiness.md").exists()
    assert (output_dir / "domain_approval_template.json").exists()
    assert (output_dir / "paper_benchmark_manifest_template.json").exists()
    assert (output_dir / "paper_workflow_commands.md").exists()


def test_cli_plan_paper_workflow_can_fail_on_blockers(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-paper-workflow",
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
    assert (tmp_path / "paper_workflow_readiness.json").exists()


def _write_ablation_output(output_dir: Path, *, seeds: list[int], variants: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            run_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "evidence_mode": "mock_workflow_shape",
                    "llm_mode": "mock",
                    "scientific_claim": "not_supported",
                    "run_dir": str(output_dir / "runs" / f"{variant}-seed-{seed}"),
                    "valid_solution_rate": 1.0,
                }
            )
    _write_csv(output_dir / "ablation_runs.csv", run_rows)
    _write_csv(
        output_dir / "ablation_summary.csv",
        [
            {
                "variant": variant,
                "evidence_mode": "mock_workflow_shape",
                "scientific_claim": "not_supported",
                "runs": len(seeds),
                "valid_runs": len(seeds),
                "example_run_dir": str(output_dir / "runs" / f"{variant}-seed-{seeds[0]}"),
            }
            for variant in variants
        ],
    )
    (output_dir / "ablation_report.md").write_text("# Ablation Report\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
