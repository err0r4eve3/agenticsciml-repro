from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.resume import (
    inspect_pre_root_resume_state,
    read_source_revision,
    validate_pre_root_real_llm_evidence,
    validate_resume_conditions_compatible,
)
from agenticsciml.observations import render_structured_data_analysis


def _conditions(*, max_calls: int | None, cost_rate: float | None = 0.01) -> dict[str, object]:
    return {
        "config": {"benchmark_dir": "/benchmark", "use_mock": False},
        "llm_runtime": {
            "adapter_class": "agenticsciml.llm.budget.RecordingLLMClient",
            "budget_limits": {
                "max_prompt_tokens": 100,
                "max_output_tokens": 100,
                "max_total_tokens": 200,
                "max_calls": max_calls,
                "max_cost_usd": 1.0,
                "cost_per_1k_tokens_usd": cost_rate,
            },
        },
        "source_revision": {"commit": "abc", "dirty": False},
    }


def _write_data_ready_artifacts(run_dir: Path) -> None:
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    structured = {
        "schema_version": 1,
        "benchmark_name": "function_approx",
        "training_array_keys": ["x_train", "u_train"],
        "task_specific_observations": ["function approximation"],
        "private_label_boundary": "training_data_only_no_validation_labels",
        "llm_report_summary": "summary",
    }
    (reports / "data_analysis.md").write_text(
        render_structured_data_analysis(structured), encoding="utf-8"
    )
    (reports / "data_analysis_structured.json").write_text(
        json.dumps(structured),
        encoding="utf-8",
    )
    transcripts = run_dir / "transcripts"
    transcripts.mkdir()
    (transcripts / "data_analyst.json").write_text(
        json.dumps(
            [{"role": "data_analyst", "prompt": "analyze", "response": "summary", "metadata": {}}]
        ),
        encoding="utf-8",
    )


def test_resume_conditions_allow_only_budget_ceiling_relaxation() -> None:
    validate_resume_conditions_compatible(
        _conditions(max_calls=4),
        _conditions(max_calls=8),
    )
    validate_resume_conditions_compatible(
        _conditions(max_calls=4),
        _conditions(max_calls=None),
    )

    with pytest.raises(ValueError, match="cannot be tightened"):
        validate_resume_conditions_compatible(
            _conditions(max_calls=4),
            _conditions(max_calls=3),
        )
    with pytest.raises(ValueError, match="cost rate cannot change"):
        validate_resume_conditions_compatible(
            _conditions(max_calls=4),
            _conditions(max_calls=8, cost_rate=0.02),
        )


def test_pre_root_resume_rejects_partial_data_analysis(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "data_analysis.md").write_text("# Data Analysis\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts are incomplete"):
        inspect_pre_root_resume_state(
            tmp_path,
            ProblemBundle.load(Path("examples/function_approx")),
        )


def test_pre_root_resume_accepts_contract_ready_before_approval(tmp_path: Path) -> None:
    bundle = ProblemBundle.load(Path("examples/function_approx"))
    _write_data_ready_artifacts(tmp_path)
    contract = BenchmarkContractFactory.create_contract(bundle)
    (tmp_path / "evaluation_contract.json").write_text(
        json.dumps(contract.to_dict()),
        encoding="utf-8",
    )
    (tmp_path / "reports" / "evaluation_contract.md").write_text(
        BenchmarkContractFactory.create_guidelines(bundle, contract), encoding="utf-8"
    )
    (tmp_path / "transcripts" / "evaluator.json").write_text(
        json.dumps(
            [{"role": "evaluator", "prompt": "contract", "response": "{}", "metadata": {}}]
        ),
        encoding="utf-8",
    )

    state = inspect_pre_root_resume_state(tmp_path, bundle)

    assert state.stage == "contract_ready"
    assert state.completed_llm_calls == 2
    assert state.approval_status is None
    assert state.contract == contract


def test_pre_root_resume_rejects_post_root_artifacts_without_checkpoint(tmp_path: Path) -> None:
    _write_data_ready_artifacts(tmp_path)
    (tmp_path / "tree.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="post-root artifacts"):
        inspect_pre_root_resume_state(
            tmp_path,
            ProblemBundle.load(Path("examples/function_approx")),
        )


def test_pre_root_real_evidence_requires_role_bound_success(tmp_path: Path) -> None:
    bundle = ProblemBundle.load(Path("examples/function_approx"))
    _write_data_ready_artifacts(tmp_path)
    state = inspect_pre_root_resume_state(tmp_path, bundle)
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": "llm_call_000001",
                "schema_name": "evaluator",
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        json.dumps(
            {
                "event_type": "generation_span",
                "name": "data_analyst",
                "metadata": {"llm_call_id": "llm_call_000001"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not bound to a successful"):
        validate_pre_root_real_llm_evidence(tmp_path, state)


def test_source_revision_digest_changes_with_uncommitted_runtime_source(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    before = read_source_revision(tmp_path)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = read_source_revision(tmp_path)

    assert before["dirty"] is False
    assert after["dirty"] is True
    assert before["runtime_source_digest"] != after["runtime_source_digest"]
    assert before["runtime_source_file_count"] == after["runtime_source_file_count"] == 3
