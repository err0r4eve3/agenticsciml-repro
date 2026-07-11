from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agenticsciml import cli as cli_module
from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.cli import (
    _agent_configs_from_role_payloads,
    _default_experiment_id,
    _resume_expected_llm_call_range,
    build_parser,
)
from agenticsciml.config import ExperimentConfig, EvolutionConfig
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.observations import render_structured_data_analysis


def test_run_cli_defaults_to_mock_and_requires_explicit_real() -> None:
    parser = build_parser()

    assert parser.parse_args(["run", "examples/function_approx"]).mock is True
    assert parser.parse_args(["run", "examples/function_approx", "--mock"]).mock is True
    assert parser.parse_args(["run", "examples/function_approx", "--real"]).mock is False


def test_default_experiment_ids_do_not_collide() -> None:
    assert _default_experiment_id(True) != _default_experiment_id(True)


def test_run_cli_exposes_scientific_claim_and_seed_inputs() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "examples/function_approx",
            "--claim-level",
            "paper_workflow",
            "--domain-evaluator-approved",
            "--domain-reviewer",
            "researcher@example.invalid",
            "--domain-review-notes",
            "reviewed metric and data protocol",
            "--paper-benchmark-approved",
            "--strategy-seed-id",
            "fourier_feature_mlp",
        ]
    )

    assert args.claim_level == "paper_workflow"
    assert args.domain_evaluator_approved is True
    assert args.paper_benchmark_approved is True
    assert args.strategy_seed_id == ["fourier_feature_mlp"]


def test_run_cli_accepts_selector_vote_count() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "examples/function_approx",
            "--mock",
            "--selector-vote-count",
            "3",
        ]
    )

    assert args.selector_vote_count == 3


def test_run_cli_defaults_to_three_selector_votes() -> None:
    parser = build_parser()

    args = parser.parse_args(["run", "examples/function_approx", "--mock"])

    assert args.selector_vote_count == 3


def test_run_cli_accepts_real_llm_timeout_and_retry_budget() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "examples/function_approx",
            "--llm-timeout-s",
            "30",
            "--llm-max-retries",
            "1",
            "--llm-fast-mode",
        ]
    )

    assert args.llm_timeout_s == 30.0
    assert args.llm_max_retries == 1
    assert args.llm_fast_mode is True


def test_run_cli_accepts_agent_model_overrides() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "examples/function_approx",
            "--mock",
            "--agent-models-json",
            (
                '{"root_engineer":{"model":"gpt-5.5","base_url":"https://api.gatexflow.com/v1",'
                '"temperature":0.1,"reasoning_effort":"xhigh"},'
                '"retriever":{"model":"gpt-5.4-mini","temperature":0.0}}'
            ),
        ]
    )
    payload = json.loads(args.agent_models_json)

    assert payload["root_engineer"]["model"] == "gpt-5.5"
    assert payload["root_engineer"]["reasoning_effort"] == "xhigh"
    assert payload["retriever"]["model"] == "gpt-5.4-mini"

    configs = _agent_configs_from_role_payloads(payload)
    assert configs["root_engineer"].base_url == "https://api.gatexflow.com/v1"
    assert configs["root_engineer"].temperature == 0.1
    assert configs["root_engineer"].reasoning_effort == "xhigh"
    assert configs["retriever"].model == "gpt-5.4-mini"
    assert configs["retriever"].temperature == 0.0

    defaulted = _agent_configs_from_role_payloads({"proposer": {"model": "gpt-5.5"}})
    assert defaulted["proposer"].temperature == 0.55
    assert defaulted["proposer"].reasoning_effort is None


def test_run_cli_rejects_unknown_agent_model_role() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        _agent_configs_from_role_payloads({"planner": {"model": "gpt-5.5"}})


def test_init_example_creates_a_runnable_custom_benchmark(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    benchmark_dir = tmp_path / "custom function study"
    initialized = subprocess.run(
        [sys.executable, "-m", "agenticsciml.cli", "init-example", str(benchmark_dir)],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert Path(initialized.stdout.strip()) == benchmark_dir
    spec = json.loads((benchmark_dir / "Benchmark_spec.json").read_text(encoding="utf-8"))
    assert spec["name"] == benchmark_dir.name
    assert ProblemBundle.load(benchmark_dir).benchmark_name == benchmark_dir.name

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "run",
            str(benchmark_dir),
            "--max-iterations",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert run.returncode == 0, run.stderr
    assert (Path(run.stdout.strip().splitlines()[-1]) / "tree.json").exists()


def test_real_run_budget_preflight_blocks_before_adapter_initialization(
    tmp_path: Path,
    cli_env: dict[str, str],
) -> None:
    env = {**cli_env, "AGENTICSCIML_MAX_LLM_CALLS": "1"}
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "run",
            "examples/function_approx",
            "--real",
            "--max-iterations",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 1
    assert "LLM call budget preflight failed" in result.stderr
    assert "OPENAI_API_KEY" not in result.stderr
    assert not list(tmp_path.glob("*/llm_call_ledger.jsonl"))


def test_real_dry_run_rejects_unknown_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "4")
    monkeypatch.setattr(
        cli_module,
        "read_source_revision",
        lambda _path: {
            "commit": "unknown",
            "dirty": "unknown",
            "runtime_source_digest": "unknown",
            "runtime_source_file_count": "unknown",
        },
    )

    exit_code = cli_module.main(
        [
            "run",
            "examples/function_approx",
            "--real",
            "--dry-run",
            "--max-iterations",
            "0",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Git source provenance is unavailable" in captured.err
    assert captured.out == ""


def _write_real_resume_preflight_run(tmp_path: Path, experiment_id: str) -> Path:
    run_dir = tmp_path / experiment_id
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "use_mock": False,
                "evolution": {"parallel_mutations": 1},
            }
        ),
        encoding="utf-8",
    )
    root = {
        "node_id": "solution_000",
        "parent_id": None,
        "workspace": str(run_dir / "solutions" / "solution_000"),
        "score": {"metric": "mse", "value": 1.0, "higher_is_better": False},
        "children": [],
        "status": "evaluated",
        "proposal_path": None,
        "analysis_path": None,
        "error": None,
        "benchmark_name": "function_approx",
        "contract_hash": "contract",
        "method_tags": [],
        "failure_kind": None,
        "score_delta_from_parent": None,
        "num_debug_attempts": 0,
    }
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "phase": "root_created",
                "schema_version": "solution_tree.v1",
                "checkpoint_schema_version": "orchestrator_checkpoint.v2",
                "experiment_id": experiment_id,
                "completed_iterations": 0,
                "target_iterations": 1,
                "inflight_batch": None,
                "nodes": [root],
            }
        ),
        encoding="utf-8",
    )
    calls = (
        ("data_analyst", "complete_text", None),
        ("evaluator", "complete_json", "evaluator"),
        ("root_engineer", "complete_json", "root_engineer"),
        ("result_analyst", "complete_json", "analysis"),
    )
    ledger_rows = []
    trace_events = []
    for index, (role, method, schema_name) in enumerate(calls, start=1):
        call_id = f"llm_call_{index:06d}"
        shared = {
            "call_id": call_id,
            "method": method,
            "schema_name": schema_name,
            "provider": "test-provider",
            "model": "test-model",
            "adapter_type": "test-adapter",
        }
        ledger_rows.append(
            {
                **shared,
                "success": True,
                "prompt_token_estimate": 1,
                "response_token_estimate": 1,
            }
        )
        trace_events.append(
            {
                "event_seq": index,
                "event_type": "generation_span",
                "name": role,
                "metadata": {**shared, "llm_call_id": call_id},
            }
        )
    (run_dir / "llm_call_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger_rows), encoding="utf-8"
    )
    (run_dir / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in trace_events), encoding="utf-8"
    )
    return run_dir


def test_real_resume_budget_preflight_counts_only_remaining_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_id = "resume-budget-run"
    run_dir = _write_real_resume_preflight_run(tmp_path, experiment_id)
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "22")
    _write_pre_root_resume_conditions(run_dir)

    exit_code = cli_module.main(
        [
            "run",
            "examples/function_approx",
            "--real",
            "--resume",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "1",
            "--experiment-id",
            experiment_id,
            "--output-dir",
            str(tmp_path),
        ]
    )

    plan = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert plan["expected_llm_call_range"] == {"min": 6, "max": 18}
    assert plan["llm_budget"]["calls_used"] == 4
    assert plan["budget_preflight"]["projected_max_llm_calls"] == 22
    assert plan["budget_preflight"]["passed"] is True


def test_real_pre_root_approval_resume_keeps_conservative_root_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_id = "resume-pre-root-budget-run"
    run_dir = _write_real_resume_preflight_run(tmp_path, experiment_id)
    (run_dir / "checkpoint.json").unlink()
    _write_pre_root_data_analysis(run_dir)
    ledger_rows = (run_dir / "llm_call_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    (run_dir / "llm_call_ledger.jsonl").write_text(
        "\n".join(ledger_rows[:2]) + "\n",
        encoding="utf-8",
    )

    benchmark_dir = Path("examples/function_approx").resolve()
    contract = BenchmarkContractFactory.create_contract(ProblemBundle.load(benchmark_dir))
    (run_dir / "evaluation_contract.json").write_text(
        json.dumps(contract.to_dict()),
        encoding="utf-8",
    )
    (run_dir / "reports" / "evaluation_contract.md").write_text(
        BenchmarkContractFactory.create_guidelines(
            ProblemBundle.load(benchmark_dir), contract
        ),
        encoding="utf-8",
    )
    (run_dir / "transcripts" / "evaluator.json").write_text(
        json.dumps(
            [{"role": "evaluator", "prompt": "contract", "response": "{}", "metadata": {}}]
        ),
        encoding="utf-8",
    )
    _write_pre_root_real_evidence(run_dir, ("data_analyst", "evaluator"))
    (run_dir / "evaluation_approval.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "approved",
                "approval_required": True,
                "benchmark_name": contract.benchmark_name,
                "contract_hash": contract.contract_hash,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "22")
    _write_pre_root_resume_conditions(run_dir)

    exit_code = cli_module.main(
        [
            "run",
            str(benchmark_dir),
            "--real",
            "--resume",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "1",
            "--experiment-id",
            experiment_id,
            "--output-dir",
            str(tmp_path),
        ]
    )

    plan = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert plan["expected_llm_call_range"] == {"min": 8, "max": 20}
    assert plan["llm_budget"]["calls_used"] == 2
    assert plan["budget_preflight"]["projected_max_llm_calls"] == 22
    assert plan["budget_preflight"]["passed"] is True


def _write_pre_root_resume_conditions(
    run_dir: Path,
    *,
    requested_max_iterations: int = 1,
) -> None:
    benchmark_dir = Path("examples/function_approx").resolve()
    config = ExperimentConfig(
        experiment_id=run_dir.name,
        benchmark_dir=benchmark_dir,
        output_dir=run_dir.parent,
        evolution=EvolutionConfig(
            max_iterations=requested_max_iterations,
            parallel_mutations=1,
        ),
        use_mock=False,
    )
    config_payload = config.to_dict()
    config_payload.pop("experiment_id")
    config_payload.pop("output_dir")
    config_payload.pop("resume")
    assert isinstance(config_payload["evolution"], dict)
    config_payload["evolution"].pop("max_iterations")
    parser_args = build_parser().parse_args(
        [
            "run",
            str(benchmark_dir),
            "--real",
            "--resume",
            "--dry-run",
            "--max-iterations",
            str(requested_max_iterations),
            "--parallel-mutations",
            "1",
            "--experiment-id",
            run_dir.name,
            "--output-dir",
            str(run_dir.parent),
        ]
    )
    budget = cli_module.LLMBudget.from_env()
    conditions = {
        "config": config_payload,
        "llm_runtime": cli_module._dry_run_real_llm_runtime_identity(parser_args, budget),
        "source_revision": cli_module.read_source_revision(benchmark_dir),
    }
    conditions_digest = hashlib.sha256(
        json.dumps(
            conditions,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    (run_dir / "experiment_conditions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "conditions": conditions,
                "conditions_digest": conditions_digest,
                "initial_target_iterations": requested_max_iterations,
            }
        ),
        encoding="utf-8",
    )


def _write_pre_root_data_analysis(run_dir: Path) -> None:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    structured = {
        "schema_version": 1,
        "benchmark_name": "function_approx",
        "training_array_keys": ["x_train", "u_train"],
        "task_specific_observations": ["function approximation training data"],
        "private_label_boundary": "training_data_only_no_validation_labels",
        "llm_report_summary": "function_approx training arrays inspected",
    }
    (reports_dir / "data_analysis.md").write_text(
        render_structured_data_analysis(structured), encoding="utf-8"
    )
    (reports_dir / "data_analysis_structured.json").write_text(
        json.dumps(structured), encoding="utf-8"
    )
    transcripts_dir = run_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    (transcripts_dir / "data_analyst.json").write_text(
        json.dumps(
            [
                {
                    "role": "data_analyst",
                    "prompt": "analyze",
                    "response": structured["llm_report_summary"],
                    "metadata": {},
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_pre_root_real_evidence(run_dir: Path, roles: tuple[str, ...]) -> None:
    rows = []
    trace_events = []
    for index, role in enumerate(roles, start=1):
        call_id = f"llm_call_{index:06d}"
        schema_name = None if role == "data_analyst" else role
        prompt = "analyze" if role == "data_analyst" else "contract"
        response = (
            "function_approx training arrays inspected"
            if role == "data_analyst"
            else "{}"
        )
        rows.append(
            {
                "call_id": call_id,
                "method": "complete_text" if schema_name is None else "complete_json",
                "schema_name": schema_name,
                "provider": "test-provider",
                "model": "test-model",
                "adapter_type": "TestAdapter",
                "success": True,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response_hash": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "prompt_token_estimate": 1,
                "response_token_estimate": 1,
            }
        )
        trace_events.append(
            {
                "event_seq": index,
                "event_type": "generation_span",
                "name": role,
                "metadata": {
                    "llm_call_id": call_id,
                    "method": "complete_text" if schema_name is None else "complete_json",
                    "schema_name": schema_name,
                    "provider": "test-provider",
                    "model": "test-model",
                    "adapter_type": "TestAdapter",
                },
            }
        )
    (run_dir / "llm_call_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in trace_events), encoding="utf-8"
    )


def test_real_data_ready_pre_root_resume_counts_one_completed_call(tmp_path: Path) -> None:
    experiment_id = "resume-data-ready"
    run_dir = _write_real_resume_preflight_run(tmp_path, experiment_id)
    (run_dir / "checkpoint.json").unlink()
    _write_pre_root_resume_conditions(run_dir)
    _write_pre_root_data_analysis(run_dir)
    _write_pre_root_real_evidence(run_dir, ("data_analyst",))

    expected = _resume_expected_llm_call_range(
        run_dir=run_dir,
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        requested_max_iterations=1,
        parallel_mutations=1,
    )

    assert expected == {"min": 9, "max": 21}


def test_real_initialized_pre_root_resume_allows_zero_call_ledger(tmp_path: Path) -> None:
    experiment_id = "resume-initialized"
    run_dir = _write_real_resume_preflight_run(tmp_path, experiment_id)
    (run_dir / "checkpoint.json").unlink()
    (run_dir / "llm_call_ledger.jsonl").unlink()
    (run_dir / "trace.jsonl").unlink()
    _write_pre_root_resume_conditions(run_dir)

    expected = _resume_expected_llm_call_range(
        run_dir=run_dir,
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        requested_max_iterations=1,
        parallel_mutations=1,
    )
    budget = cli_module.LLMBudget(max_calls=22)
    cli_module._load_resume_llm_budget_usage(
        run_dir / "llm_call_ledger.jsonl",
        budget,
        minimum_calls=0,
    )

    assert expected == {"min": 10, "max": 22}
    assert budget.calls_used == 0


def test_real_resume_dry_run_rejects_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_id = "resume-model-drift"
    run_dir = _write_real_resume_preflight_run(tmp_path, experiment_id)
    (run_dir / "checkpoint.json").unlink()
    (run_dir / "llm_call_ledger.jsonl").unlink()
    (run_dir / "trace.jsonl").unlink()
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "22")
    monkeypatch.setenv("OPENAI_MODEL", "model-before")
    _write_pre_root_resume_conditions(run_dir)
    monkeypatch.setenv("OPENAI_MODEL", "model-after")

    exit_code = cli_module.main(
        [
            "run",
            "examples/function_approx",
            "--real",
            "--resume",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "1",
            "--experiment-id",
            experiment_id,
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "incompatible conditions" in captured.err
    assert captured.out == ""


def test_real_resume_call_range_counts_pending_inflight_and_future_slots(tmp_path: Path) -> None:
    experiment_id = "resume-inflight-budget-run"
    run_dir = _write_real_resume_preflight_run(tmp_path, experiment_id)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config["evolution"]["parallel_mutations"] = 2
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["phase"] = "children_inflight"
    checkpoint["target_iterations"] = 2
    child = {
        **checkpoint["nodes"][0],
        "node_id": "solution_001",
        "parent_id": "solution_000",
        "workspace": str(run_dir / "solutions" / "solution_001"),
    }
    checkpoint["inflight_batch"] = {
        "schema_version": 1,
        "iteration_index": 0,
        "jobs": [
            {"parent_id": "solution_000", "solution_id": "solution_001"},
            {"parent_id": "solution_000", "solution_id": "solution_002"},
        ],
        "completed_children": [{"parent_id": "solution_000", "child": child}],
    }
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    call_range = _resume_expected_llm_call_range(
        run_dir=run_dir,
        experiment_id=experiment_id,
        benchmark_dir=Path("examples/function_approx").resolve(),
        requested_max_iterations=2,
        parallel_mutations=2,
    )

    # One pending child remains in the current iteration; one future iteration
    # can still schedule two mutation slots.
    assert call_range == {"min": 18, "max": 54}


@pytest.mark.parametrize("checkpoint_state", ["missing", "malformed", "stale"])
def test_real_resume_rejects_unusable_checkpoint_before_provider_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    checkpoint_state: str,
) -> None:
    experiment_id = f"resume-{checkpoint_state}-checkpoint"
    run_dir = _write_real_resume_preflight_run(tmp_path, experiment_id)
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_state == "missing":
        checkpoint_path.unlink()
    elif checkpoint_state == "malformed":
        checkpoint_path.write_text("{", encoding="utf-8")
    else:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["checkpoint_schema_version"] = "orchestrator_checkpoint.v1"
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    provider_initialized = False

    def adapter(**_kwargs: object) -> MockLLMClient:
        nonlocal provider_initialized
        provider_initialized = True
        return MockLLMClient()

    monkeypatch.setattr(cli_module, "OpenAIAdapter", adapter)
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "22")

    exit_code = cli_module.main(
        [
            "run",
            "examples/function_approx",
            "--real",
            "--resume",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "1",
            "--experiment-id",
            experiment_id,
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert provider_initialized is False
    assert "Cannot preflight real resume" in capsys.readouterr().err


def test_standard_real_run_writes_budgeted_llm_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_MAX_LLM_CALLS", "10")
    monkeypatch.setattr(cli_module, "OpenAIAdapter", lambda **_kwargs: MockLLMClient())

    exit_code = cli_module.main(
        [
            "run",
            "examples/function_approx",
            "--real",
            "--max-iterations",
            "0",
            "--experiment-id",
            "budgeted-real-run",
            "--output-dir",
            str(tmp_path),
        ]
    )

    run_dir = tmp_path / "budgeted-real-run"
    ledger = [
        json.loads(line)
        for line in (run_dir / "llm_call_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert [row["call_id"] for row in ledger] == [
        f"llm_call_{index:06d}" for index in range(1, len(ledger) + 1)
    ]
    assert metadata["llm_budget"]["calls_used"] == len(ledger)


def test_module_cli_smoke_dry_run_is_not_real_evidence(tmp_path: Path, cli_env: dict[str, str]) -> None:
    output_dir = tmp_path / "smoke output with spaces"
    smoke = subprocess.run(
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
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert Path(smoke.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_report.md"
    assert (output_dir / "real_llm_smoke_plan.json").exists()
    assert (output_dir / "real_llm_smoke_manifest.json").exists()

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-smoke-llm",
            str(output_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    payload = json.loads((output_dir / "real_llm_smoke_verification.json").read_text(encoding="utf-8"))

    assert verify.returncode == 1
    assert any("dry-run outputs are not real-smoke evidence" in issue for issue in payload["issues"])
    assert any("plan execution_mode must be real" in issue for issue in payload["issues"])


def test_module_cli_smoke_from_checkout_path_with_spaces(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    shutil.copytree(
        source_root / "src",
        checkout / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (checkout / "examples").mkdir()
    shutil.copytree(
        source_root / "examples" / "function_approx",
        checkout / "examples" / "function_approx",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout / "src")
    output_dir = Path("runs") / "real llm smoke"

    smoke = subprocess.run(
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
            str(output_dir),
        ],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    bundle = checkout / output_dir
    assert Path(smoke.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_report.md"
    assert (bundle / "real_llm_smoke_plan.json").exists()
    assert (bundle / "real_llm_smoke_manifest.json").exists()

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-smoke-llm",
            str(output_dir),
        ],
        cwd=checkout,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    payload = json.loads((bundle / "real_llm_smoke_verification.json").read_text(encoding="utf-8"))

    assert verify.returncode == 1
    assert any("dry-run outputs are not real-smoke evidence" in issue for issue in payload["issues"])
    assert any("plan execution_mode must be real" in issue for issue in payload["issues"])
