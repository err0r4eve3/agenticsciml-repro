import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agenticsciml.evidence import EVIDENCE_MODE_REAL_LLM_SMOKE
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm_smoke import _paired_contrast_gate, _smoke_gate, run_llm_smoke


def test_llm_smoke_dry_run_writes_plan_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=True,
    )
    plan = json.loads(result.plan_json.read_text(encoding="utf-8"))
    report = result.report_md.read_text(encoding="utf-8")

    assert result.runs_csv is None
    assert result.manifest_json.exists()
    assert plan["evidence_mode"] == EVIDENCE_MODE_REAL_LLM_SMOKE
    assert plan["scientific_claim"] == "not_supported"
    assert [entry["variant"] for entry in plan["runs"]] == ["branch_context", "no_branch_context"]
    assert plan["runs"][0]["use_branch_context"] is True
    assert plan["runs"][1]["use_branch_context"] is False
    assert plan["runs"][0]["experiment_id"] == "smoke-branch_context-seed-0"
    assert "runs/smoke-branch_context-seed-0/tree.json" in plan["expected_artifacts"]
    assert "API calls: none" in report
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "dry_run"
    assert manifest["expected_llm_call_range"]["min"] > 0


def test_llm_smoke_dry_run_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown LLM smoke variant"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["unknown_variant"],
            dry_run=True,
        )


def test_llm_smoke_real_mode_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context"],
            dry_run=False,
        )


def test_llm_smoke_real_mode_requires_paired_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact paired variants"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )


def test_llm_smoke_real_mode_rejects_duplicate_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicates are not allowed"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context", "branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )


def test_llm_smoke_real_gate_with_scripted_llm(tmp_path: Path) -> None:
    result = run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    rows = list(csv.DictReader(result.runs_csv.open(encoding="utf-8")))  # type: ignore[union-attr]

    assert {row["variant"] for row in rows} == {"branch_context", "no_branch_context"}
    assert all(row["smoke_gate_passed"] == "True" for row in rows)
    assert all(row["trace_quality_gate_passed"] == "True" for row in rows)
    assert all(int(row["llm_calls"]) > 0 for row in rows)
    assert result.manifest_json.exists()
    no_branch = next(row for row in rows if row["variant"] == "no_branch_context")
    assert no_branch["branch_context_enabled"] == "False"
    assert no_branch["branch_intents"] == ""


def test_smoke_gate_requires_branch_context_prompt_delivery(tmp_path: Path) -> None:
    run_dir = tmp_path
    child = run_dir / "solutions" / "solution_001" / "transcripts"
    child.mkdir(parents=True)
    (run_dir / "solutions" / "solution_001" / "branch_context.json").write_text(
        json.dumps(
            {
                "branch_intent": "features_or_architecture",
                "sibling_branch_ids": ["solution_002"],
                "diversity_instruction": "distinct branch",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (child / "proposal_debate.json").write_text(
        json.dumps(
            [
                {
                    "role": "proposer",
                    "prompt": "No branch context was delivered in this request.",
                    "response": "branch_intent sibling_branch_ids diversity_instruction",
                }
            ]
        ),
        encoding="utf-8",
    )
    (child / "engineer.json").write_text("[]", encoding="utf-8")
    (run_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "name": "agenticsciml.child_mutation.start",
                "metadata": {"branch_context": {"branch_intent": "features_or_architecture"}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    gate = _smoke_gate(
        run_dir,
        "branch_context",
        {"branch_context_enabled": True, "llm_calls": {"total": 1}},
        {"quality_gate": {"passed": True}},
        [{"node_id": "solution_001", "parent_id": "solution_000"}],
        {"branch:features_or_architecture"},
    )

    assert gate["passed"] is False
    assert any("request prompts missing" in issue for issue in gate["issues"])


def test_paired_contrast_gate_rejects_single_variant() -> None:
    gate = _paired_contrast_gate(
        [
            {
                "variant": "branch_context",
                "smoke_gate_passed": True,
                "branch_context_enabled": True,
                "llm_calls": 1,
                "branch_intents": "features_or_architecture",
            }
        ]
    )

    assert gate["passed"] is False
    assert "missing required variant no_branch_context" in gate["issues"]


def test_cli_smoke_llm_dry_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = {**cli_env}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context,no_branch_context",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    report_path = Path(result.stdout.strip().splitlines()[-1])
    assert report_path.name == "real_llm_smoke_report.md"
    assert (tmp_path / "real_llm_smoke_plan.json").exists()


def test_cli_smoke_llm_defaults_to_dry_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = {**cli_env}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert Path(result.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_report.md"
    plan = json.loads((tmp_path / "real_llm_smoke_plan.json").read_text(encoding="utf-8"))
    assert plan["execution_mode"] == "dry_run"
