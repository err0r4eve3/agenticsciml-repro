from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agenticsciml.ablation import _aggregate, run_ablation
from agenticsciml.ablation_evidence import build_multi_seed_ablation_verified_manifest
from agenticsciml.evidence import (
    EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
    EVIDENCE_MODE_REAL_LLM_ABLATION,
    LLM_MODE_REAL,
    SCIENTIFIC_CLAIM_NOT_SUPPORTED,
)
from agenticsciml.llm.mock import MockLLMClient


def test_ablation_runner_writes_summary_and_report(tmp_path: Path) -> None:
    result = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        seeds=[0],
        variants=[
            "root_only",
            "no_kb",
            "kb",
            "random_kb",
            "no_critic",
            "no_debugger",
            "branch_context",
            "no_branch_context",
        ],
    )

    rows = list(csv.DictReader(result.summary_csv.open(encoding="utf-8")))
    run_rows = list(csv.DictReader(result.runs_csv.open(encoding="utf-8")))
    variants = {row["variant"] for row in rows}
    report = result.report_md.read_text(encoding="utf-8")

    assert variants == {
        "root_only",
        "no_kb",
        "kb",
        "random_kb",
        "no_critic",
        "no_debugger",
        "branch_context",
        "no_branch_context",
    }
    assert result.summary_csv.exists()
    assert result.report_md.exists()
    assert (tmp_path / "ablation_plan.json").exists()
    assert (tmp_path / "ablation_manifest.json").exists()
    assert (tmp_path / "ablation_evidence_bundle.json").exists()
    assert run_rows[0]["search_seed"] == "0"
    assert run_rows[0]["data_seed"] == "0"
    assert run_rows[0]["model_seed"] == ""
    assert run_rows[0]["provider_seed"] == ""
    assert run_rows[0]["seed_provenance_complete"] == "False"
    assert run_rows[0]["execution_mode"] == "mock"
    assert "champion/root improvement" in rows[0]
    assert "valid_runs" in rows[0]
    assert rows[0]["evidence_mode"] == EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE
    assert rows[0]["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED
    assert "Champion/root improvement" in report
    assert "Valid runs" in report
    assert EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE in report
    assert "cannot prove emergent discovery" in report
    for row in rows:
        assert Path(row["example_run_dir"]).exists()

    no_critic_run = Path(next(row["example_run_dir"] for row in rows if row["variant"] == "no_critic"))
    no_critic_events = [
        json.loads(line)
        for line in (no_critic_run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not any(
        event.get("event_type") == "generation_span"
        and event.get("metadata", {}).get("spec_role") == "critic"
        for event in no_critic_events
    )

    no_debugger_run = Path(next(row["example_run_dir"] for row in rows if row["variant"] == "no_debugger"))
    no_debugger_config = json.loads((no_debugger_run / "config.json").read_text(encoding="utf-8"))
    assert no_debugger_config["evolution"]["use_debugger"] is False

    branch_context_row = next(row for row in rows if row["variant"] == "branch_context")
    no_branch_context_row = next(row for row in rows if row["variant"] == "no_branch_context")
    assert int(branch_context_row["branch_context_count_total"]) > 0
    assert branch_context_row["branch_intents"]
    assert int(no_branch_context_row["branch_context_count_total"]) == 0

    no_branch_context_run = Path(no_branch_context_row["example_run_dir"])
    no_branch_context_config = json.loads(
        (no_branch_context_run / "config.json").read_text(encoding="utf-8")
    )
    assert no_branch_context_config["evolution"]["use_branch_context"] is False
    no_branch_metadata = json.loads((no_branch_context_run / "run_metadata.json").read_text(encoding="utf-8"))
    assert no_branch_metadata["branch_context_enabled"] is False
    assert no_branch_context_row["branch_context_enabled"] == "False"

    no_branch_events = [
        json.loads(line)
        for line in (no_branch_context_run / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    child_starts = [
        event
        for event in no_branch_events
        if event["name"] == "agenticsciml.child_mutation.start"
    ]
    assert child_starts
    assert all(event["metadata"]["branch_context_enabled"] is False for event in child_starts)
    assert all(event["metadata"]["branch_context"] == {} for event in child_starts)

    proposal_transcript = (
        no_branch_context_run
        / "solutions"
        / "solution_001"
        / "transcripts"
        / "proposal_debate.json"
    ).read_text(encoding="utf-8")
    engineer_transcript = (
        no_branch_context_run
        / "solutions"
        / "solution_001"
        / "transcripts"
        / "engineer.json"
    ).read_text(encoding="utf-8")
    assert "branch_intent" not in proposal_transcript
    assert "sibling_branch_ids" not in proposal_transcript
    assert "diversity_instruction" not in proposal_transcript
    assert "branch_intent" not in engineer_transcript
    assert "sibling_branch_ids" not in engineer_transcript
    assert "diversity_instruction" not in engineer_transcript


def test_cli_ablate_command_runs_mock_pipeline(tmp_path: Path, cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "ablate",
            "examples/function_approx",
            "--seeds",
            "0",
            "--variants",
            "root_only,kb",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    summary_path = Path(result.stdout.strip().splitlines()[-1])
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))

    assert summary_path.name == "ablation_summary.csv"
    assert {row["variant"] for row in rows} == {"root_only", "kb"}
    assert (summary_path.parent / "ablation_report.md").exists()


def test_mock_ablation_bundle_is_workflow_shape_not_scientific_evidence(tmp_path: Path) -> None:
    run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        seeds=[0, 1],
        variants=["kb"],
    )

    manifest = build_multi_seed_ablation_verified_manifest(
        {"ablation_output_dir": str(tmp_path), "verified_by": "workflow-reviewer"}
    )

    assert manifest["workflow_shape_verified"] is True
    assert manifest["scientific_multi_seed_verified"] is False
    assert manifest["verified"] is False
    assert manifest["execution_mode"] == "mock"


def test_ablation_real_dry_run_writes_plan_without_evidence_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        seeds=[0, 1],
        variants=["root_only", "kb"],
        mock=False,
        dry_run=True,
    )

    assert result.summary_csv is None
    assert result.runs_csv is None
    assert result.plan_json is not None
    assert result.manifest_json is not None
    assert result.plan_json.exists()
    assert result.manifest_json.exists()
    assert result.report_md.exists()
    assert not (tmp_path / "ablation_runs.csv").exists()
    assert not (tmp_path / "ablation_summary.csv").exists()
    assert not (tmp_path / "ablation_evidence_bundle.json").exists()

    plan = json.loads(result.plan_json.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    report = result.report_md.read_text(encoding="utf-8")

    assert plan["execution_mode"] == "dry_run"
    assert plan["real_mode_explicit"] is False
    assert plan["provider_calls_enabled"] is False
    assert plan["evidence_mode"] == EVIDENCE_MODE_REAL_LLM_ABLATION
    assert plan["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED
    assert len(plan["runs"]) == 4
    assert manifest["execution_mode"] == "dry_run"
    assert manifest["run_count"] == 4
    assert manifest["full_stage_run_count"] == 4
    assert manifest["full_stage_plan_hash"] == plan["full_stage_plan_hash"]
    assert manifest["benchmark_content_hash"] == plan["benchmark_content_hash"]
    assert manifest["budget_preflight"]["status"] == "ready"
    assert manifest["expected_llm_call_range"]["max"] == 52
    assert manifest["full_stage_expected_llm_call_range"]["max"] == 52
    assert manifest["budget_batch_plan"]["full_stage_budget_status"] == "not_configured"
    assert manifest["budget_batch_plan"]["batch_plan_status"] == "not_configured"
    assert manifest["budget_batch_plan"]["batching_required"] is False
    assert manifest["budget_batch_plan"]["batch_count"] == 1
    assert manifest["budget_batch_plan"]["coverage_run_count"] == 4
    assert "No provider calls were made" in report
    assert "must not pass `verify-ablation-evidence`" in report


def test_ablation_real_dry_run_records_budget_batches_when_full_stage_exceeds_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "80")

    result = run_ablation(
        benchmark_dir=Path("examples/function_approx_faithful_small").resolve(),
        output_dir=tmp_path,
        seeds=[0, 1, 2, 3, 4],
        variants=["root_only", "no_kb", "kb", "random_kb"],
        mock=False,
        dry_run=True,
    )

    plan = json.loads(result.plan_json.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    report = result.report_md.read_text(encoding="utf-8")

    assert manifest["run_count"] == 20
    assert manifest["budget_preflight"]["status"] == "blocked_by_budget"
    assert manifest["full_stage_plan_hash"] == plan["full_stage_plan_hash"]
    assert manifest["budget_batch_plan"]["full_stage_budget_status"] == "blocked_by_budget"
    assert manifest["budget_batch_plan"]["batch_plan_status"] == "ready"
    assert manifest["budget_batch_plan"]["batching_required"] is True
    assert manifest["budget_batch_plan"]["required_for_execution"] is True
    assert manifest["budget_batch_plan"]["coverage_run_count"] == 20
    assert manifest["budget_batch_plan"]["batch_count"] > 1
    assert plan["budget_batch_plan"]["batches"][0]["expected_llm_call_range"]["max"] <= 80
    assert "## Budget Batches" in report


def test_ablation_full_stage_plan_hash_survives_output_dir_and_batch_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "25")

    result_a = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "a",
        seeds=[0],
        variants=["root_only", "kb"],
        mock=False,
        dry_run=True,
    )
    result_b = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "b",
        seeds=[0],
        variants=["root_only", "kb"],
        mock=False,
        dry_run=True,
    )
    batch = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path / "batch",
        seeds=[0],
        variants=["root_only", "kb"],
        mock=False,
        dry_run=True,
        budget_batch_index=1,
    )

    plan_a = json.loads(result_a.plan_json.read_text(encoding="utf-8"))
    plan_b = json.loads(result_b.plan_json.read_text(encoding="utf-8"))
    batch_plan = json.loads(batch.plan_json.read_text(encoding="utf-8"))
    batch_manifest = json.loads(batch.manifest_json.read_text(encoding="utf-8"))

    assert plan_a["full_stage_plan_hash"] == plan_b["full_stage_plan_hash"]
    assert batch_plan["full_stage_plan_hash"] == plan_a["full_stage_plan_hash"]
    assert batch_manifest["full_stage_plan_hash"] == plan_a["full_stage_plan_hash"]
    assert batch_manifest["plan_hash"] != json.loads(result_a.manifest_json.read_text(encoding="utf-8"))["plan_hash"]


def test_ablation_real_mode_blocks_on_call_budget_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "10")

    with pytest.raises(RuntimeError, match="LLM call budget preflight failed"):
        run_ablation(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            seeds=[0],
            variants=["branch_context", "no_branch_context"],
            mock=False,
            dry_run=False,
            llm_client=MockLLMClient(),
        )

    report = (tmp_path / "ablation_report.md").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "real_llm_ablation_manifest.json").read_text(encoding="utf-8"))
    assert "blocked_by_budget" in report
    assert manifest["budget_preflight"]["status"] == "blocked_by_budget"
    assert manifest["budget_preflight"]["expected_max_llm_calls"] == 80
    assert not (tmp_path / "ablation_runs.csv").exists()


def test_ablation_real_mode_can_execute_selected_budget_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "25")

    result = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        seeds=[0],
        variants=["root_only", "kb"],
        mock=False,
        dry_run=False,
        budget_batch_index=1,
        llm_client=MockLLMClient(),
    )

    manifest = json.loads((tmp_path / "real_llm_ablation_manifest.json").read_text(encoding="utf-8"))
    run_rows = list(csv.DictReader(result.runs_csv.open(encoding="utf-8")))

    assert manifest["full_stage_run_count"] == 2
    assert manifest["run_count"] == 1
    assert manifest["selected_budget_batch_index"] == 1
    assert manifest["budget_preflight"]["status"] == "ready"
    assert manifest["full_stage_expected_llm_call_range"]["max"] == 26
    assert manifest["full_stage_plan_hash"]
    assert manifest["expected_llm_call_range"]["max"] == 4
    assert [row["variant"] for row in run_rows] == ["root_only"]


def test_ablation_real_runner_accepts_injected_llm_without_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_ablation(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        seeds=[0],
        variants=["root_only"],
        mock=False,
        dry_run=False,
        llm_client=MockLLMClient(),
    )

    assert result.summary_csv is not None
    assert result.runs_csv is not None
    run_rows = list(csv.DictReader(result.runs_csv.open(encoding="utf-8")))
    summary_rows = list(csv.DictReader(result.summary_csv.open(encoding="utf-8")))

    assert len(run_rows) == 1
    assert run_rows[0]["evidence_mode"] == EVIDENCE_MODE_REAL_LLM_ABLATION
    assert run_rows[0]["llm_mode"] == LLM_MODE_REAL
    assert run_rows[0]["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED
    assert run_rows[0]["run_evidence_mode"].startswith("real_llm_")
    assert run_rows[0]["search_seed"] == "0"
    assert run_rows[0]["data_seed"] == "0"
    assert run_rows[0]["model_seed"] == ""
    assert run_rows[0]["provider_seed"] == ""
    assert run_rows[0]["seed_provenance_complete"] == "False"
    assert run_rows[0]["execution_mode"] == "real"
    assert summary_rows[0]["evidence_mode"] == EVIDENCE_MODE_REAL_LLM_ABLATION
    assert summary_rows[0]["scientific_claim"] == SCIENTIFIC_CLAIM_NOT_SUPPORTED
    ledger_path = Path(run_rows[0]["run_dir"]) / "llm_call_ledger.jsonl"
    assert ledger_path.exists()
    assert ledger_path.read_text(encoding="utf-8").strip()
    assert (tmp_path / "ablation_evidence_bundle.json").exists()


def test_cli_ablate_real_dry_run_is_no_key_safe(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = dict(cli_env)
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "ablate",
            "examples/function_approx",
            "--real",
            "--dry-run",
            "--seeds",
            "0",
            "--variants",
            "root_only",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    plan_path = Path(result.stdout.strip().splitlines()[-1])
    assert plan_path == tmp_path / "real_llm_ablation_plan.json"
    assert plan_path.exists()
    assert (tmp_path / "real_llm_ablation_manifest.json").exists()
    assert (tmp_path / "ablation_report.md").exists()
    assert not (tmp_path / "ablation_runs.csv").exists()


def test_ablation_aggregate_respects_score_direction() -> None:
    base_row = {
        "variant": "accuracy_metric",
        "evidence_mode": EVIDENCE_MODE_MOCK_WORKFLOW_SHAPE,
        "scientific_claim": SCIENTIFIC_CLAIM_NOT_SUPPORTED,
        "valid_solution_rate": 1.0,
        "timeout_count": 0,
        "debug_success_count": 0,
        "llm_calls": 0,
        "wall_time_s": 0.0,
        "run_dir": "runs/demo",
        "champion/root improvement": 0.0,
    }
    rows = [
        {**base_row, "seed": 0, "higher_is_better": True, "champion_score": 0.8},
        {**base_row, "seed": 1, "higher_is_better": True, "champion_score": 0.9},
    ]

    summary = _aggregate(rows)[0]

    assert summary["higher_is_better"] is True
    assert summary["champion_score_best"] == 0.9
    assert summary["champion_score_worst"] == 0.8
