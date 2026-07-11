from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.resume import (
    inspect_pre_root_resume_state,
    read_source_revision,
    real_llm_checkpoint_call_floor,
    validate_pre_root_real_llm_evidence,
    validate_real_llm_ledger_trace_consistency,
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
                "method": "complete_json",
                "schema_name": "evaluator",
                "provider": "test-provider",
                "model": "test-model",
                "adapter_type": "TestAdapter",
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
                "metadata": {
                    "llm_call_id": "llm_call_000001",
                    "method": "complete_json",
                    "schema_name": "data_analyst",
                    "provider": "test-provider",
                    "model": "test-model",
                    "adapter_type": "TestAdapter",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_name mismatch"):
        validate_pre_root_real_llm_evidence(tmp_path, state)


def test_real_llm_ledger_trace_rejects_joint_metadata_downgrade(tmp_path: Path) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    ledger_row = {
        "call_id": "llm_call_000001",
        "success": False,
        **common,
    }
    trace_event = {
        "event_type": "generation_span",
        "name": "data_analyst",
        "metadata": {"llm_call_id": "llm_call_000001", **common},
    }
    ledger_row.pop("provider")
    trace_event["metadata"].pop("provider")
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(ledger_row) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        json.dumps(trace_event) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provider metadata"):
        validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=1)


def test_real_llm_ledger_trace_enforces_checkpoint_role_floor(tmp_path: Path) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": "llm_call_000001",
                "success": True,
                **common,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        json.dumps(
            {
                "event_type": "generation_span",
                "name": "evaluator",
                "metadata": {"llm_call_id": "llm_call_000001", **common},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="1 data_analyst calls; found 0"):
        validate_real_llm_ledger_trace_consistency(
            tmp_path,
            minimum_calls=1,
            minimum_calls_by_role={"data_analyst": 1},
        )


def test_real_llm_ledger_trace_rejects_provider_usage_legacy_downgrade(
    tmp_path: Path,
) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": "llm_call_000001",
                "success": True,
                "prompt_token_estimate": 3,
                "response_token_estimate": 5,
                **common,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "workflow_span",
                        "name": "agenticsciml.run.start",
                        "metadata": {"llm_evidence_schema_version": 1},
                    }
                ),
                json.dumps(
                    {
                        "event_type": "generation_span",
                        "name": "data_analyst",
                        "metadata": {
                            "llm_call_id": "llm_call_000001",
                            "prompt_token_estimate": 3,
                            "response_token_estimate": 5,
                            "usage": {
                                "prompt_tokens": 7,
                                "completion_tokens": 5,
                                "total_tokens": 12,
                            },
                            **common,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot downgrade provider usage"):
        validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=1)


@pytest.mark.parametrize(
    ("trace_usage", "prompt_accounted", "prompt_source", "response_source"),
    [
        (
            {"prompt_tokens": 7, "total_tokens": 99},
            7,
            "provider_usage",
            "local_estimate",
        ),
        (
            {"completion_tokens": 5, "total_tokens": 99},
            3,
            "local_estimate",
            "provider_usage",
        ),
        (
            {"total_tokens": 99},
            3,
            "local_estimate",
            "local_estimate",
        ),
    ],
)
def test_real_llm_ledger_trace_rejects_unreconciled_partial_provider_total(
    tmp_path: Path,
    trace_usage: dict[str, int],
    prompt_accounted: int,
    prompt_source: str,
    response_source: str,
) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": "llm_call_000001",
                "success": True,
                "prompt_token_estimate": 3,
                "prompt_tokens_accounted": prompt_accounted,
                "prompt_token_source": prompt_source,
                "response_token_estimate": 5,
                "response_token_source": response_source,
                **common,
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
                "metadata": {
                    "llm_call_id": "llm_call_000001",
                    "prompt_token_estimate": 3,
                    "response_token_estimate": 5,
                    "usage": trace_usage,
                    **common,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provider total usage mismatch"):
        validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=1)


@pytest.mark.parametrize("post_response_failure", [False, True])
def test_real_llm_ledger_trace_accepts_current_failed_call_accounting(
    tmp_path: Path,
    post_response_failure: bool,
) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    ledger_row: dict[str, object] = {
        "call_id": "llm_call_000001",
        "success": False,
        "prompt_token_estimate": 3,
        "prompt_tokens_accounted": 7 if post_response_failure else 3,
        "prompt_token_source": (
            "provider_usage" if post_response_failure else "local_estimate"
        ),
        **common,
    }
    trace_metadata: dict[str, object] = {
        "llm_call_id": "llm_call_000001",
        "prompt_token_estimate": 3,
        "response_token_estimate": 0,
        **common,
    }
    if post_response_failure:
        ledger_row.update(
            {
                "response_token_estimate": 5,
                "response_token_source": "provider_usage",
            }
        )
        trace_metadata["response_token_estimate"] = 5
        trace_metadata["usage"] = {
            "prompt_tokens": 7,
            "completion_tokens": 5,
            "total_tokens": 12,
        }
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(ledger_row) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "workflow_span",
                        "name": "agenticsciml.run.start",
                        "metadata": {"llm_evidence_schema_version": 1},
                    }
                ),
                json.dumps(
                    {
                        "event_type": "generation_span",
                        "name": "data_analyst",
                        "metadata": trace_metadata,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=1)

    assert result["ledger_usage"] == {
        "calls_used": 1,
        "prompt_tokens_used": 7 if post_response_failure else 3,
        "output_tokens_used": 5 if post_response_failure else 0,
        "total_tokens_used": 12 if post_response_failure else 3,
    }


def test_real_llm_ledger_trace_rejects_current_failed_pre_response_drift(
    tmp_path: Path,
) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": "llm_call_000001",
                "success": False,
                "prompt_token_estimate": 3,
                "prompt_tokens_accounted": 3,
                "prompt_token_source": "local_estimate",
                **common,
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
                "metadata": {
                    "llm_call_id": "llm_call_000001",
                    "prompt_token_estimate": 3,
                    "response_token_estimate": 999,
                    **common,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="failed response estimate mismatch"):
        validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=1)


def test_real_llm_ledger_trace_accepts_mixed_legacy_and_current_resume_rows(
    tmp_path: Path,
) -> None:
    common = {
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    ledger_rows = [
        {
            "call_id": "llm_call_000001",
            "method": "complete_text",
            "schema_name": None,
            "success": True,
            "prompt_token_estimate": 3,
            "response_token_estimate": 2,
            **common,
        },
        {
            "call_id": "llm_call_000002",
            "method": "complete_json",
            "schema_name": "evaluator",
            "success": True,
            "prompt_token_estimate": 4,
            "prompt_tokens_accounted": 4,
            "prompt_token_source": "local_estimate",
            "response_token_estimate": 3,
            "response_token_source": "local_estimate",
            **common,
        },
    ]
    trace_events = [
        {
            "event_type": "generation_span",
            "name": "data_analyst",
            "metadata": {
                "llm_call_id": "llm_call_000001",
                "method": "complete_text",
                "schema_name": None,
                "prompt_token_estimate": 3,
                "response_token_estimate": 2,
                **common,
            },
        },
        {
            "event_type": "workflow_span",
            "name": "agenticsciml.run.start",
            "metadata": {"llm_evidence_schema_version": 1},
        },
        {
            "event_type": "generation_span",
            "name": "evaluator",
            "metadata": {
                "llm_call_id": "llm_call_000002",
                "method": "complete_json",
                "schema_name": "evaluator",
                "prompt_token_estimate": 4,
                "response_token_estimate": 3,
                **common,
            },
        },
    ]
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in trace_events),
        encoding="utf-8",
    )

    result = validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=2)

    assert result["ledger_usage"] == {
        "calls_used": 2,
        "prompt_tokens_used": 7,
        "output_tokens_used": 5,
        "total_tokens_used": 12,
    }


def test_real_llm_ledger_trace_accepts_legacy_failed_pre_response_row(
    tmp_path: Path,
) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(
            {
                "call_id": "llm_call_000001",
                "success": False,
                "prompt_token_estimate": 3,
                **common,
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
                "metadata": {
                    "llm_call_id": "llm_call_000001",
                    "prompt_token_estimate": 3,
                    "response_token_estimate": 0,
                    **common,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=1)

    assert result["ledger_usage"] == {
        "calls_used": 1,
        "prompt_tokens_used": 3,
        "output_tokens_used": 0,
        "total_tokens_used": 3,
    }


@pytest.mark.parametrize("drift_field", ["prompt_token_estimate", "response_token_estimate"])
def test_real_llm_ledger_trace_rejects_legacy_local_estimate_drift(
    tmp_path: Path,
    drift_field: str,
) -> None:
    common = {
        "method": "complete_text",
        "schema_name": None,
        "provider": "test-provider",
        "model": "test-model",
        "adapter_type": "TestAdapter",
    }
    ledger_row = {
        "call_id": "llm_call_000001",
        "success": True,
        "prompt_token_estimate": 3,
        "response_token_estimate": 2,
        **common,
    }
    ledger_row[drift_field] = 1
    (tmp_path / "llm_call_ledger.jsonl").write_text(
        json.dumps(ledger_row) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "workflow_span",
                        "name": "agenticsciml.run.start",
                        "metadata": {"llm_evidence_schema_version": 1},
                    }
                ),
                json.dumps(
                    {
                        "event_type": "generation_span",
                        "name": "data_analyst",
                        "metadata": {
                            "llm_call_id": "llm_call_000001",
                            "prompt_token_estimate": 3,
                            "response_token_estimate": 2,
                            **common,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy .* estimate mismatch"):
        validate_real_llm_ledger_trace_consistency(tmp_path, minimum_calls=1)


@pytest.mark.parametrize(
    (
        "use_critic",
        "child",
        "expected_minimum",
        "expected_child_roles",
        "inflight",
    ),
    [
        (
            True,
            {"status": "evaluated", "failure_kind": None},
            13,
            {"critic": 3, "engineer": 1, "proposer": 4, "result_analyst": 2},
            False,
        ),
        (
            False,
            {"status": "evaluated", "failure_kind": None},
            10,
            {"engineer": 1, "proposer": 4, "result_analyst": 2},
            False,
        ),
        (
            True,
            {"status": "failed", "failure_kind": "runtime_error"},
            13,
            {"critic": 3, "engineer": 1, "proposer": 4, "result_analyst": 2},
            False,
        ),
        (
            True,
            {"status": "failed", "failure_kind": "orchestration_error"},
            5,
            {"result_analyst": 2},
            False,
        ),
        (
            True,
            {"status": "evaluated", "failure_kind": None},
            13,
            {"critic": 3, "engineer": 1, "proposer": 4, "result_analyst": 2},
            True,
        ),
    ],
)
def test_real_llm_checkpoint_call_floor_respects_finalized_child_path(
    tmp_path: Path,
    use_critic: bool,
    child: dict[str, object],
    expected_minimum: int,
    expected_child_roles: dict[str, int],
    inflight: bool,
) -> None:
    conditions = {
        "config": {
            "use_mock": False,
            "evolution": {"use_critic": use_critic},
        }
    }
    encoded = json.dumps(
        conditions,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    conditions_digest = hashlib.sha256(encoded).hexdigest()
    (tmp_path / "experiment_conditions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "conditions": conditions,
                "conditions_digest": conditions_digest,
            }
        ),
        encoding="utf-8",
    )
    root = {"node_id": "solution_000", "parent_id": None}
    child_payload = {
        "node_id": "solution_001",
        "parent_id": "solution_000",
        **child,
    }
    checkpoint: dict[str, object] = {
        "experiment_conditions_digest": conditions_digest,
        "nodes": [root] if inflight else [root, child_payload],
        "inflight_batch": (
            {
                "completed_children": [
                    {"parent_id": "solution_000", "child": child_payload}
                ]
            }
            if inflight
            else None
        ),
    }

    floor = real_llm_checkpoint_call_floor(tmp_path, checkpoint)

    assert floor["minimum_calls"] == expected_minimum
    for role, count in expected_child_roles.items():
        assert floor["minimum_calls_by_role"][role] == count


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
