from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import sys
import time
import json
import uuid
from pathlib import Path

from agenticsciml.ablation_evidence import build_multi_seed_ablation_verified_manifest
from agenticsciml.ablation_collect import collect_ablation_batches
from agenticsciml.ablation import DEFAULT_VARIANTS, run_ablation
from agenticsciml.algorithm_catalog import list_algorithms
from agenticsciml.benchmarks import (
    BenchmarkContractFactory,
    ProblemBundle,
    benchmark_for_path,
    list_benchmarks,
)
from agenticsciml.config import (
    DEFAULT_AGENT_ROLE_MODEL_SETTINGS,
    EXPERT_BLUEPRINT_IDS,
    VISUAL_AUDIT_MODES,
    AgentConfig,
    EvolutionConfig,
    ExperimentConfig,
)
from agenticsciml.iteration_campaign import (
    record_iteration_round_evidence,
    write_iteration_campaign,
    write_iteration_campaign_verification,
)
from agenticsciml.evidence import CLAIM_LEVELS
from agenticsciml.llm.budget import (
    LLMBudget,
    RecordingLLMClient,
    estimate_orchestrator_llm_call_range,
    load_llm_budget_usage,
    llm_call_budget_preflight,
    require_llm_call_budget_preflight,
)
from agenticsciml.llm_problem_context import write_llm_problem_context_pack
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import (
    OpenAIAdapter,
    _max_retries_from_env,
    _normalize_model_name,
    _timeout_from_env,
)
from agenticsciml.llm.capabilities import capabilities_for_openai_compatible
from agenticsciml.llm_smoke import DEFAULT_SMOKE_VARIANTS, run_llm_smoke, verify_llm_smoke_output
from agenticsciml.orchestrator import (
    CHECKPOINT_SCHEMA_VERSION,
    INFLIGHT_BATCH_SCHEMA_VERSION,
    AgenticSciMLOrchestrator,
)
from agenticsciml.paper_workflow_readiness import write_paper_workflow_readiness_bundle
from agenticsciml.paper_gap_report import write_paper_gap_report
from agenticsciml.paper_source_collect import DEFAULT_ARXIV_QUERY, DEFAULT_SOURCE_LIMIT, write_paper_source_collection
from agenticsciml.real_problem_closure import write_real_problem_closure_plan
from agenticsciml.reference_capability_matrix import write_reference_capability_matrix
from agenticsciml.reporting import write_sdk_trace_export, write_trace_summary
from agenticsciml.resume import (
    MIN_ROOT_REAL_LLM_CALLS,
    inspect_pre_root_resume_state,
    read_source_revision,
    real_llm_checkpoint_call_floor,
    validate_pre_root_real_llm_evidence,
    validate_real_llm_ledger_trace_consistency,
    validate_resume_conditions_compatible,
)
from agenticsciml.selector_evidence import write_selector_evidence_packet
from agenticsciml.secret_hygiene import write_secret_hygiene_report
from agenticsciml.state import validate_solution_tree_artifact_payload
from agenticsciml.storage import _atomic_write_text


def _default_experiment_id(mock: bool) -> str:
    prefix = "mock" if mock else "real"
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:10]}"


def _print_llm_progress(payload: dict[str, object]) -> None:
    print(
        "llm-progress " + json.dumps(payload, sort_keys=True, allow_nan=False),
        file=sys.stderr,
        flush=True,
    )


def cmd_init_example(args: argparse.Namespace) -> int:
    source = Path(__file__).resolve().parents[2] / "examples" / "function_approx"
    target = Path(args.target)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Target already exists and is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name in {"train_data.npz", "val_data.npz"}:
            continue
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)
    source_spec = benchmark_for_path(source)
    if source_spec is None:  # pragma: no cover - checked-in source invariant.
        raise RuntimeError(f"Cannot resolve source benchmark metadata: {source}")
    spec_payload = {
        "schema_version": 1,
        **source_spec.contract_digest_metadata(),
        "name": target.name,
    }
    _atomic_write_text(
        target / "Benchmark_spec.json",
        json.dumps(spec_payload, indent=2, sort_keys=True, allow_nan=False),
    )
    print(target.resolve())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    benchmark_dir = Path(args.benchmark_dir).resolve()
    _validate_run_arguments(args)
    benchmark_snapshot = _validated_benchmark_snapshot(benchmark_dir)
    evolution = EvolutionConfig(
        max_iterations=args.max_iterations,
        parallel_mutations=args.parallel_mutations,
        max_children_per_node=args.max_children_per_node,
        max_debug_retries=args.max_debug_retries,
        timeout_s=args.timeout_s,
        use_kb=not args.no_kb,
        random_kb=args.random_kb,
        random_seed=args.random_seed,
        use_critic=not args.no_critic,
        use_debugger=not args.no_debugger,
        use_branch_context=not args.no_branch_context,
        selector_vote_count=args.selector_vote_count,
    )
    agent_configs = _agent_configs_from_role_payloads(
        _json_object_arg(args.agent_models_json, "--agent-models-json")
    )
    selector_panel = [
        _agent_config_from_selector_payload(index, payload)
        for index, payload in enumerate(_selector_panel_payloads(args), start=1)
    ]
    strategy_seed_ids = _strategy_seed_ids(args)
    _validate_strategy_seed_ids(strategy_seed_ids)
    config = ExperimentConfig(
        experiment_id=args.experiment_id or _default_experiment_id(args.mock),
        benchmark_dir=benchmark_dir,
        output_dir=Path(args.output_dir).resolve(),
        evolution=evolution,
        use_mock=args.mock,
        agents=agent_configs,
        selector_panel=selector_panel,
        strategy_seed_ids=strategy_seed_ids,
        claim_level=args.claim_level,
        domain_evaluator_approved=args.domain_evaluator_approved,
        domain_reviewer=args.domain_reviewer,
        domain_review_notes=args.domain_review_notes,
        paper_benchmark_approved=args.paper_benchmark_approved,
        visual_audit_mode=args.visual_audit_mode,
        resource_constraints=_json_object_arg(args.resource_constraints_json, "--resource-constraints-json"),
        expert_blueprint_id=args.expert_blueprint_id,
        multi_seed_ablation=_json_object_arg(args.multi_seed_ablation_json, "--multi-seed-ablation-json"),
        llm_fast_mode=args.llm_fast_mode,
        auto_approve_evaluation=not args.require_evaluation_approval,
        resume=args.resume,
    )
    expected_call_range = estimate_orchestrator_llm_call_range(
        max_iterations=args.max_iterations,
        parallel_mutations=args.parallel_mutations,
    )
    run_dir = config.output_dir / config.experiment_id
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    resume_minimum_calls: int | None = None
    if args.resume and not args.mock:
        expected_call_range = _resume_expected_llm_call_range(
            run_dir=run_dir,
            experiment_id=config.experiment_id,
            benchmark_dir=benchmark_dir,
            requested_max_iterations=args.max_iterations,
            parallel_mutations=args.parallel_mutations,
        )
        resume_minimum_calls = _resume_minimum_ledger_calls(run_dir, benchmark_dir)
    if args.dry_run:
        plan = _build_run_plan(config, benchmark_snapshot, expected_call_range)
        if not args.mock:
            source_revision = read_source_revision(benchmark_dir)
            if any(value == "unknown" for value in source_revision.values()):
                raise ValueError(
                    "Cannot preflight real run: Git source provenance is unavailable"
                )
            plan["source_revision"] = source_revision
            dry_budget = LLMBudget.from_env()
            if args.resume:
                _load_resume_llm_budget_usage(
                    ledger_path,
                    dry_budget,
                    minimum_calls=resume_minimum_calls or 0,
                )
                _validate_resume_dry_run_conditions(
                    run_dir=run_dir,
                    config=config,
                    args=args,
                    budget=dry_budget,
                )
            plan["llm_budget"] = dry_budget.to_dict()
            plan["budget_preflight"] = llm_call_budget_preflight(
                budget=dry_budget,
                expected_llm_call_range=expected_call_range,
            )
        print(
            json.dumps(
                plan,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0

    if args.mock:
        llm = MockLLMClient()
    else:
        budget = LLMBudget.from_env()
        if args.resume:
            _load_resume_llm_budget_usage(
                ledger_path,
                budget,
                minimum_calls=resume_minimum_calls or 0,
            )
        require_llm_call_budget_preflight(
            llm_call_budget_preflight(
                budget=budget,
                expected_llm_call_range=expected_call_range,
            )
        )
        inner_llm = OpenAIAdapter(timeout_s=args.llm_timeout_s, max_retries=args.llm_max_retries)
        llm = RecordingLLMClient(
            inner_llm,
            ledger_path,
            budget,
            progress_callback=_print_llm_progress,
        )
        require_llm_call_budget_preflight(
            llm_call_budget_preflight(
                budget=budget,
                expected_llm_call_range=expected_call_range,
            )
        )
    run_dir = AgenticSciMLOrchestrator(config, llm).run()
    print(run_dir.resolve())
    issues = _completed_run_issues(run_dir)
    if issues:
        print("run completion gate failed: " + "; ".join(issues), file=sys.stderr)
        return 1
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    path = Path(args.run_dir) / "leaderboard.csv"
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def cmd_export_tree(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if args.format == "mermaid":
        path = run_dir / "tree.mmd"
    else:
        path = run_dir / "tree.json"
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def cmd_trace_summary(args: argparse.Namespace) -> int:
    path = write_trace_summary(Path(args.run_dir))
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("quality_gate", {}).get("passed") is True else 1


def cmd_export_sdk_trace(args: argparse.Namespace) -> int:
    path = write_sdk_trace_export(Path(args.run_dir))
    print(path.resolve())
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    result = run_ablation(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        seeds=args.seeds,
        variants=variants,
        mock=not args.real,
        dry_run=args.dry_run,
        timeout_s=args.timeout_s,
        llm_timeout_s=args.llm_timeout_s,
        llm_max_retries=args.llm_max_retries,
        llm_fast_mode=args.llm_fast_mode,
        budget_batch_index=args.budget_batch_index,
    )
    output_path = result.summary_csv or result.plan_json or result.report_md
    print(output_path.resolve())
    return 0


def cmd_verify_ablation_evidence(args: argparse.Namespace) -> int:
    source = {
        "ablation_output_dir": str(Path(args.output_dir).resolve()),
        "verified_by": args.verified_by,
        "expected_seeds": args.expected_seeds,
        "expected_variants": _split_csv(args.expected_variants),
    }
    manifest = build_multi_seed_ablation_verified_manifest(source)
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else Path(args.output_dir).resolve() / "multi_seed_ablation_verified_manifest.json"
    )
    _atomic_write_text(
        output_json,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
    )
    print(output_json)
    return 0 if manifest["verified"] is True else 1


def cmd_collect_ablation_batches(args: argparse.Namespace) -> int:
    result = collect_ablation_batches(
        stage_plan_dir=Path(args.stage_plan).resolve(),
        batch_dirs=[Path(item).resolve() for item in args.batch_dir],
        output_dir=Path(args.output_dir).resolve(),
        allow_partial=args.allow_partial,
        allow_legacy_missing_full_stage_hash=args.allow_legacy_missing_full_stage_hash,
    )
    print(result.manifest_json.resolve())
    return 0 if result.passed or args.allow_partial else 1


def cmd_plan_paper_workflow(args: argparse.Namespace) -> int:
    selector_panel = _json_array_arg(args.selector_panel_json, "--selector-panel-json")
    result = write_paper_workflow_readiness_bundle(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        selector_panel=selector_panel,
        selector_evidence_path=Path(args.selector_evidence_json).resolve()
        if args.selector_evidence_json
        else None,
        problem_intake=_json_object_file_arg(args.problem_intake_json, "--problem-intake-json")
        if args.problem_intake_json
        else {},
        resource_constraints=_json_object_arg(args.resource_constraints_json, "--resource-constraints-json"),
        expert_blueprint_id=args.expert_blueprint_id,
        domain_approval_path=Path(args.domain_approval_json).resolve()
        if args.domain_approval_json
        else None,
        ablation_output_dir=Path(args.ablation_output_dir).resolve()
        if args.ablation_output_dir
        else None,
        expected_seeds=args.expected_seeds,
        expected_variants=_split_csv(args.expected_variants),
        env=os.environ,
    )
    print(result["paths"]["plan_json"])
    if args.fail_on_blockers and result["bundle"]["status"] == "blocked":
        return 1
    return 0


def cmd_paper_gap_report(args: argparse.Namespace) -> int:
    result = write_paper_gap_report(
        output_dir=Path(args.output_dir).resolve(),
        benchmark_dirs=[Path(item).resolve() for item in args.benchmark_dir],
        run_dirs=[Path(item).resolve() for item in args.run_dir],
    )
    print(result["paths"]["report_json"])
    if args.fail_on_gaps and result["report"]["status"] == "blocked":
        return 1
    return 0


def cmd_generate_selector_evidence(args: argparse.Namespace) -> int:
    result = write_selector_evidence_packet(
        Path(args.run_dir).resolve(),
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(result["paths"]["packet_json"])
    if args.fail_on_blockers and result["packet"]["status"] == "blocked":
        return 1
    return 0


def cmd_build_reference_capability_matrix(args: argparse.Namespace) -> int:
    result = write_reference_capability_matrix(
        output_dir=Path(args.output_dir).resolve(),
        problem_intake=_json_object_file_arg(args.problem_intake_json, "--problem-intake-json")
        if args.problem_intake_json
        else {},
        expert_blueprint_id=args.expert_blueprint_id,
    )
    print(result["paths"]["matrix_json"])
    if args.fail_on_blockers and result["matrix"]["status"] == "blocked":
        return 1
    return 0


def cmd_build_llm_problem_context(args: argparse.Namespace) -> int:
    result = write_llm_problem_context_pack(
        output_dir=Path(args.output_dir).resolve(),
        problem_intake=_json_object_file_arg(args.problem_intake_json, "--problem-intake-json")
        if args.problem_intake_json
        else {},
        expert_blueprint_id=args.expert_blueprint_id,
        resource_constraints=_json_object_arg(args.resource_constraints_json, "--resource-constraints-json"),
    )
    print(result["paths"]["pack_json"])
    if args.fail_on_blockers and result["pack"]["status"] == "blocked":
        return 1
    return 0


def cmd_paper_problem_loop_audit(args: argparse.Namespace) -> int:
    from agenticsciml.paper_problem_loop import (
        LOOP_HEALTH_JSON,
        LOOP_INDEX_JSON,
        update_paper_problem_loop_index,
        verify_paper_problem_loop,
        write_paper_problem_loop_audit,
    )

    if args.rounds is not None and args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.interval_s < 0:
        raise ValueError("--interval-s must be >= 0")
    if args.source_candidate_limit < 0:
        raise ValueError("--source-candidate-limit must be >= 0")

    base_output_dir = Path(args.output_dir).resolve()
    source_collection_path = (
        Path(args.source_collection_json).resolve()
        if args.source_collection_json
        else base_output_dir / "paper_source_collection.json"
    )
    had_issues = False
    round_index = _next_repeat_round_index(base_output_dir) - 1 if args.repeat else 0
    try:
        while True:
            round_index += 1
            if args.refresh_source_collection:
                write_paper_source_collection(
                    output_dir=source_collection_path.parent,
                    query=args.source_query,
                    max_results=args.source_limit,
                    timeout_s=args.source_timeout_s,
                )
            output_dir = (
                base_output_dir
                if not args.repeat
                else base_output_dir / f"round-{round_index:04d}-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            result = write_paper_problem_loop_audit(
                output_dir=output_dir,
                case_limit=args.case_limit,
                source_collection_path=source_collection_path if source_collection_path.exists() else None,
                source_candidate_limit=args.source_candidate_limit,
                source_candidate_offset=(round_index - 1) * args.source_candidate_limit if args.repeat else 0,
            )
            update_paper_problem_loop_index(
                index_path=base_output_dir / LOOP_INDEX_JSON,
                audit_path=Path(result["paths"]["audit_json"]),
                audit=result["audit"],
            )
            health = verify_paper_problem_loop(output_dir=base_output_dir)
            _atomic_write_text(
                base_output_dir / LOOP_HEALTH_JSON,
                json.dumps(health, indent=2, ensure_ascii=False, allow_nan=False),
            )
            result["paths"]["loop_index_json"] = str((base_output_dir / LOOP_INDEX_JSON).resolve())
            result["paths"]["loop_health_json"] = str((base_output_dir / LOOP_HEALTH_JSON).resolve())
            print(result["paths"]["audit_json"], flush=True)
            had_issues = had_issues or bool(result["audit"]["issues"]) or health["status"] != "passed"
            if not args.repeat or (args.rounds is not None and round_index >= args.rounds):
                break
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("paper-problem-loop-audit stopped by user", file=sys.stderr)
        return 130
    return 1 if args.fail_on_issues and had_issues else 0


def cmd_verify_paper_problem_loop(args: argparse.Namespace) -> int:
    from agenticsciml.paper_problem_loop import verify_paper_problem_loop

    if args.max_age_s is not None and args.max_age_s < 0:
        raise ValueError("--max-age-s must be >= 0")
    result = verify_paper_problem_loop(
        output_dir=Path(args.output_dir).resolve(),
        max_age_s=args.max_age_s,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


def _next_repeat_round_index(base_output_dir: Path) -> int:
    if not base_output_dir.exists():
        return 1
    max_index = 0
    for path in base_output_dir.iterdir():
        parts = path.name.split("-", 2)
        if path.is_dir() and len(parts) >= 2 and parts[0] == "round" and parts[1].isdigit():
            max_index = max(max_index, int(parts[1]))
    return max_index + 1


def cmd_collect_paper_sources(args: argparse.Namespace) -> int:
    result = write_paper_source_collection(
        output_dir=Path(args.output_dir).resolve(),
        query=args.query,
        max_results=args.max_results,
        timeout_s=args.timeout_s,
    )
    print(result["paths"]["collection_json"])
    return 1 if args.fail_on_issues and result["collection"]["issues"] else 0


def cmd_plan_real_problem_closure(args: argparse.Namespace) -> int:
    selector_panel = _json_array_arg(args.selector_panel_json, "--selector-panel-json")
    result = write_real_problem_closure_plan(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        selector_panel=selector_panel,
        selector_evidence_path=Path(args.selector_evidence_json).resolve()
        if args.selector_evidence_json
        else None,
        problem_intake=_json_object_file_arg(args.problem_intake_json, "--problem-intake-json")
        if args.problem_intake_json
        else {},
        resource_constraints=_json_object_arg(args.resource_constraints_json, "--resource-constraints-json"),
        expert_blueprint_id=args.expert_blueprint_id,
        domain_approval_path=Path(args.domain_approval_json).resolve()
        if args.domain_approval_json
        else None,
        ablation_output_dir=Path(args.ablation_output_dir).resolve()
        if args.ablation_output_dir
        else None,
        expected_seeds=args.expected_seeds,
        expected_variants=_split_csv(args.expected_variants),
        completed_run_dir=Path(args.completed_run_dir).resolve()
        if args.completed_run_dir
        else None,
        env=os.environ,
    )
    print(result["paths"]["plan_json"])
    if args.fail_on_blockers and result["plan"]["status"] == "blocked":
        return 1
    return 0


def cmd_plan_iteration_campaign(args: argparse.Namespace) -> int:
    selector_panel = _json_array_arg(args.selector_panel_json, "--selector-panel-json")
    result = write_iteration_campaign(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        rounds=args.rounds,
        batch_size=args.batch_size,
        selector_panel=selector_panel,
        selector_evidence_path=Path(args.selector_evidence_json).resolve()
        if args.selector_evidence_json
        else None,
        problem_intake=_json_object_file_arg(args.problem_intake_json, "--problem-intake-json")
        if args.problem_intake_json
        else {},
        resource_constraints=_json_object_arg(args.resource_constraints_json, "--resource-constraints-json"),
        expert_blueprint_id=args.expert_blueprint_id,
        domain_approval_path=Path(args.domain_approval_json).resolve()
        if args.domain_approval_json
        else None,
        ablation_output_dir=Path(args.ablation_output_dir).resolve()
        if args.ablation_output_dir
        else None,
        expected_seeds=args.expected_seeds,
        expected_variants=_split_csv(args.expected_variants),
        env=os.environ,
    )
    print(result["paths"]["campaign_json"])
    return 0


def cmd_record_iteration_round(args: argparse.Namespace) -> int:
    record = record_iteration_round_evidence(
        Path(args.campaign_json).resolve(),
        round_index=args.round_index,
        evidence_path=Path(args.evidence_path),
        validation_command=args.validation_command,
        validation_exit_code=args.validation_exit_code,
        validation_output_path=Path(args.validation_output_path),
        notes=args.notes,
    )
    campaign_dir = Path(args.campaign_json).resolve().parent
    print(campaign_dir / f"iteration_round_{record['round_index']:03d}_record.json")
    return 0


def cmd_verify_iteration_campaign(args: argparse.Namespace) -> int:
    result = write_iteration_campaign_verification(
        Path(args.campaign_json).resolve(),
        require_complete=args.require_complete,
    )
    print(result["path"])
    if args.fail_on_issues and result["verification"]["passed"] is not True:
        return 1
    return 0


def cmd_smoke_llm(args: argparse.Namespace) -> int:
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    result = run_llm_smoke(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        variants=variants,
        seed=args.seed,
        dry_run=args.dry_run,
        timeout_s=args.timeout_s,
        max_iterations=args.max_iterations,
        parallel_mutations=args.parallel_mutations,
        llm_fast_mode=args.llm_fast_mode,
        progress_callback=_print_llm_progress,
    )
    print(result.report_md.resolve())
    return 0


def cmd_verify_smoke_llm(args: argparse.Namespace) -> int:
    result = verify_llm_smoke_output(Path(args.output_dir).resolve())
    print(result.verification_json.resolve())
    return 0 if result.passed else 1


def cmd_secret_hygiene(args: argparse.Namespace) -> int:
    result = write_secret_hygiene_report(
        Path(args.run_dir).resolve(),
        output_json=Path(args.output_json).resolve() if args.output_json else None,
    )
    print(result.report_json.resolve())
    return 1 if args.fail_on_findings and not result.passed else 0


def cmd_web(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "error: install the web extra first, for example `uv run --extra web agenticsciml web`",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(
        "agenticsciml.web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_benchmarks(args: argparse.Namespace) -> int:
    specs = list_benchmarks()
    if args.json:
        print(json.dumps({"benchmarks": [spec.to_dict() for spec in specs]}, indent=2, sort_keys=True))
        return 0
    for spec in specs:
        claim = spec.claim_boundaries()["real_llm"]
        scientific_claim = claim["scientific_claim"] if isinstance(claim, dict) else "unknown"
        print(
            f"{spec.name}\t{spec.paper_section}\t{spec.fidelity_level}\t"
            f"{spec.family}\t{spec.metric}\t{scientific_claim}\t{spec.path}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenticsciml")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-example")
    init.add_argument("target")
    init.set_defaults(func=cmd_init_example)

    run = sub.add_parser("run")
    run.add_argument("benchmark_dir")
    run_mode = run.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--mock",
        dest="mock",
        action="store_true",
        help="use the deterministic mock LLM (default)",
    )
    run_mode.add_argument(
        "--real",
        dest="mock",
        action="store_false",
        help="explicitly enable a real LLM provider",
    )
    run.set_defaults(mock=True)
    run.add_argument("--max-iterations", type=int, default=1)
    run.add_argument("--parallel-mutations", type=int, default=2)
    run.add_argument("--max-children-per-node", type=int, default=10)
    run.add_argument("--max-debug-retries", type=int, default=2)
    run.add_argument("--timeout-s", type=int, default=60)
    run.add_argument(
        "--llm-timeout-s",
        type=float,
        default=None,
        help="real LLM provider HTTP timeout in seconds; defaults to OPENAI_TIMEOUT_S",
    )
    run.add_argument(
        "--llm-max-retries",
        type=int,
        default=None,
        help="real LLM provider retry count; defaults to OPENAI_MAX_RETRIES, which defaults to 0",
    )
    run.add_argument(
        "--llm-fast-mode",
        action="store_true",
        help="route unspecified agent reasoning_effort defaults to low for latency-sensitive real runs",
    )
    run.add_argument("--output-dir", default="runs")
    run.add_argument("--experiment-id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--no-kb", action="store_true")
    run.add_argument("--random-kb", action="store_true")
    run.add_argument("--random-seed", type=int, default=0)
    run.add_argument("--no-critic", action="store_true")
    run.add_argument("--no-debugger", action="store_true")
    run.add_argument("--no-branch-context", action="store_true")
    run.add_argument("--selector-vote-count", type=int, default=3)
    run.add_argument("--claim-level", choices=sorted(CLAIM_LEVELS), default="workflow_proxy")
    run.add_argument(
        "--domain-evaluator-approved",
        action="store_true",
        help="record explicit human domain-evaluator approval for the run claim gate",
    )
    run.add_argument("--domain-reviewer")
    run.add_argument("--domain-review-notes")
    run.add_argument(
        "--paper-benchmark-approved",
        action="store_true",
        help="record explicit approval that this benchmark is paper-equivalent",
    )
    run.add_argument(
        "--strategy-seed-id",
        action="append",
        default=[],
        help="algorithm catalog strategy seed id; may be repeated",
    )
    run.add_argument(
        "--strategy-seed-ids",
        default="",
        help="comma-separated algorithm catalog strategy seed ids",
    )
    run.add_argument("--visual-audit-mode", choices=sorted(VISUAL_AUDIT_MODES), default="off")
    run.add_argument("--expert-blueprint-id", choices=sorted(EXPERT_BLUEPRINT_IDS))
    run.add_argument(
        "--resource-constraints-json",
        default="{}",
        help="JSON object describing CPU/GPU/time/dependency/data limits for readiness artifacts",
    )
    run.add_argument(
        "--multi-seed-ablation-json",
        default="{}",
        help="JSON object describing verified multi-seed or ablation evidence for readiness artifacts",
    )
    run.add_argument(
        "--agent-models-json",
        default="{}",
        help=(
            "JSON object keyed by agent role; each value may include model, base_url, "
            "temperature, and reasoning_effort"
        ),
    )
    run.add_argument(
        "--selector-panel-models",
        default="",
        help="comma-separated selector panel model names; records per-member selector vote provenance",
    )
    run.add_argument(
        "--selector-panel-json",
        default="[]",
        help="JSON array of selector member objects with model, optional base_url, temperature, and reasoning_effort",
    )
    run.add_argument(
        "--require-evaluation-approval",
        action="store_true",
        help="write evaluation_approval.json and pause before root generation until status is approved",
    )
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=cmd_run)

    leaderboard = sub.add_parser("leaderboard")
    leaderboard.add_argument("run_dir")
    leaderboard.set_defaults(func=cmd_leaderboard)

    export = sub.add_parser("export-tree")
    export.add_argument("run_dir")
    export.add_argument("--format", choices=["mermaid", "json"], default="mermaid")
    export.set_defaults(func=cmd_export_tree)

    trace_summary = sub.add_parser("trace-summary")
    trace_summary.add_argument("run_dir")
    trace_summary.set_defaults(func=cmd_trace_summary)

    sdk_trace = sub.add_parser("export-sdk-trace")
    sdk_trace.add_argument("run_dir")
    sdk_trace.set_defaults(func=cmd_export_sdk_trace)

    ablate = sub.add_parser("ablate")
    ablate.add_argument("benchmark_dir")
    ablate.add_argument("--seeds", nargs="+", type=int, default=[0])
    ablate.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ablate.add_argument("--output-dir", default="runs/ablation")
    ablate.add_argument("--real", action="store_true", help="run ablation with a real LLM provider")
    ablate.add_argument(
        "--dry-run",
        action="store_true",
        help="with --real, write the ablation plan and manifest without provider calls",
    )
    ablate.add_argument("--timeout-s", type=int, default=60)
    ablate.add_argument(
        "--llm-timeout-s",
        type=float,
        default=None,
        help="real LLM provider HTTP timeout in seconds; defaults to OPENAI_TIMEOUT_S",
    )
    ablate.add_argument(
        "--llm-max-retries",
        type=int,
        default=None,
        help="real LLM provider retry count; defaults to OPENAI_MAX_RETRIES, which defaults to 0",
    )
    ablate.add_argument(
        "--llm-fast-mode",
        action="store_true",
        help="route unspecified agent reasoning_effort defaults to low for latency-sensitive real ablations",
    )
    ablate.add_argument(
        "--budget-batch-index",
        type=int,
        default=None,
        help=(
            "with --real, execute only one preflight-planned budget batch; "
            "the full-stage batch plan remains recorded in the manifest"
        ),
    )
    ablate.set_defaults(func=cmd_ablate)

    verify_ablation = sub.add_parser("verify-ablation-evidence")
    verify_ablation.add_argument("output_dir")
    verify_ablation.add_argument("--verified-by", required=True)
    verify_ablation.add_argument("--expected-seeds", nargs="+", type=int, default=[])
    verify_ablation.add_argument("--expected-variants", default="")
    verify_ablation.add_argument("--output-json")
    verify_ablation.set_defaults(func=cmd_verify_ablation_evidence)

    collect_ablation = sub.add_parser("collect-ablation-batches")
    collect_ablation.add_argument("--stage-plan", required=True)
    collect_ablation.add_argument("--batch-dir", action="append", required=True)
    collect_ablation.add_argument("--output-dir", required=True)
    collect_ablation.add_argument("--allow-partial", action="store_true")
    collect_ablation.add_argument("--allow-legacy-missing-full-stage-hash", action="store_true")
    collect_ablation.set_defaults(func=cmd_collect_ablation_batches)

    paper_workflow = sub.add_parser("plan-paper-workflow")
    paper_workflow.add_argument("benchmark_dir")
    paper_workflow.add_argument("--output-dir", default="runs/paper-workflow-readiness")
    paper_workflow.add_argument(
        "--selector-panel-json",
        default="[]",
        help="JSON array of selector member objects with model and optional base_url",
    )
    paper_workflow.add_argument(
        "--resource-constraints-json",
        default="{}",
        help="JSON object with cpu, gpu, timeout_s, dependency_limits, and data_limits",
    )
    paper_workflow.add_argument("--expert-blueprint-id", choices=sorted(EXPERT_BLUEPRINT_IDS))
    paper_workflow.add_argument("--selector-evidence-json")
    paper_workflow.add_argument("--problem-intake-json")
    paper_workflow.add_argument("--domain-approval-json")
    paper_workflow.add_argument("--ablation-output-dir")
    paper_workflow.add_argument("--expected-seeds", nargs="+", type=int, default=[])
    paper_workflow.add_argument("--expected-variants", default="")
    paper_workflow.add_argument("--fail-on-blockers", action="store_true")
    paper_workflow.set_defaults(func=cmd_plan_paper_workflow)

    paper_gap = sub.add_parser("paper-gap-report")
    paper_gap.add_argument(
        "--benchmark-dir",
        action="append",
        default=[],
        help="benchmark directory to include; omit to include the catalog",
    )
    paper_gap.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="completed run directory to attach as evidence; may be repeated",
    )
    paper_gap.add_argument("--output-dir", default="runs/paper-gap-report")
    paper_gap.add_argument("--fail-on-gaps", action="store_true")
    paper_gap.set_defaults(func=cmd_paper_gap_report)

    selector_evidence = sub.add_parser("generate-selector-evidence")
    selector_evidence.add_argument("run_dir")
    selector_evidence.add_argument("--output-dir")
    selector_evidence.add_argument("--fail-on-blockers", action="store_true")
    selector_evidence.set_defaults(func=cmd_generate_selector_evidence)

    reference_matrix = sub.add_parser("build-reference-capability-matrix")
    reference_matrix.add_argument("--output-dir", default="runs/reference-capability-matrix")
    reference_matrix.add_argument("--problem-intake-json")
    reference_matrix.add_argument("--expert-blueprint-id", choices=sorted(EXPERT_BLUEPRINT_IDS))
    reference_matrix.add_argument("--fail-on-blockers", action="store_true")
    reference_matrix.set_defaults(func=cmd_build_reference_capability_matrix)

    llm_context = sub.add_parser("build-llm-problem-context")
    llm_context.add_argument("--output-dir", default="runs/llm-problem-context")
    llm_context.add_argument("--problem-intake-json")
    llm_context.add_argument(
        "--resource-constraints-json",
        default="{}",
        help="JSON object with cpu, gpu, timeout_s, dependency_limits, and data_limits",
    )
    llm_context.add_argument("--expert-blueprint-id", choices=sorted(EXPERT_BLUEPRINT_IDS))
    llm_context.add_argument("--fail-on-blockers", action="store_true")
    llm_context.set_defaults(func=cmd_build_llm_problem_context)

    paper_problem_loop = sub.add_parser("paper-problem-loop-audit")
    paper_problem_loop.add_argument("--output-dir", default="runs/paper-problem-loop")
    paper_problem_loop.add_argument("--case-limit", type=int)
    paper_problem_loop.add_argument("--repeat", action="store_true")
    paper_problem_loop.add_argument("--rounds", type=int, help="with --repeat, stop after this many rounds")
    paper_problem_loop.add_argument("--interval-s", type=float, default=60.0)
    paper_problem_loop.add_argument("--source-collection-json")
    paper_problem_loop.add_argument("--refresh-source-collection", action="store_true")
    paper_problem_loop.add_argument("--source-query", default=DEFAULT_ARXIV_QUERY)
    paper_problem_loop.add_argument("--source-limit", type=int, default=DEFAULT_SOURCE_LIMIT)
    paper_problem_loop.add_argument("--source-candidate-limit", type=int, default=3)
    paper_problem_loop.add_argument("--source-timeout-s", type=float, default=15.0)
    paper_problem_loop.add_argument("--fail-on-issues", action="store_true")
    paper_problem_loop.set_defaults(func=cmd_paper_problem_loop_audit)

    verify_paper_problem_loop = sub.add_parser("verify-paper-problem-loop")
    verify_paper_problem_loop.add_argument("output_dir")
    verify_paper_problem_loop.add_argument("--max-age-s", type=float)
    verify_paper_problem_loop.set_defaults(func=cmd_verify_paper_problem_loop)

    collect_sources = sub.add_parser("collect-paper-sources")
    collect_sources.add_argument("--output-dir", default="runs/paper-problem-loop")
    collect_sources.add_argument("--query", default=DEFAULT_ARXIV_QUERY)
    collect_sources.add_argument("--max-results", type=int, default=DEFAULT_SOURCE_LIMIT)
    collect_sources.add_argument("--timeout-s", type=float, default=15.0)
    collect_sources.add_argument("--fail-on-issues", action="store_true")
    collect_sources.set_defaults(func=cmd_collect_paper_sources)

    real_problem = sub.add_parser("plan-real-problem-closure")
    real_problem.add_argument("benchmark_dir")
    real_problem.add_argument("--output-dir", default="runs/real-problem-closure")
    real_problem.add_argument(
        "--selector-panel-json",
        default="[]",
        help="JSON array of selector member objects with model and optional base_url",
    )
    real_problem.add_argument(
        "--resource-constraints-json",
        default="{}",
        help="JSON object with cpu, gpu, timeout_s, dependency_limits, and data_limits",
    )
    real_problem.add_argument("--expert-blueprint-id", choices=sorted(EXPERT_BLUEPRINT_IDS))
    real_problem.add_argument("--selector-evidence-json")
    real_problem.add_argument("--problem-intake-json")
    real_problem.add_argument("--domain-approval-json")
    real_problem.add_argument("--ablation-output-dir")
    real_problem.add_argument("--completed-run-dir")
    real_problem.add_argument("--expected-seeds", nargs="+", type=int, default=[])
    real_problem.add_argument("--expected-variants", default="")
    real_problem.add_argument("--fail-on-blockers", action="store_true")
    real_problem.set_defaults(func=cmd_plan_real_problem_closure)

    campaign = sub.add_parser("plan-iteration-campaign")
    campaign.add_argument("benchmark_dir")
    campaign.add_argument("--rounds", type=int, default=60)
    campaign.add_argument("--batch-size", type=int, default=10)
    campaign.add_argument("--output-dir", default="runs/iteration-campaign")
    campaign.add_argument(
        "--selector-panel-json",
        default="[]",
        help="JSON array of selector member objects with model and optional base_url",
    )
    campaign.add_argument(
        "--resource-constraints-json",
        default="{}",
        help="JSON object with cpu, gpu, timeout_s, dependency_limits, and data_limits",
    )
    campaign.add_argument("--expert-blueprint-id", choices=sorted(EXPERT_BLUEPRINT_IDS))
    campaign.add_argument("--selector-evidence-json")
    campaign.add_argument("--problem-intake-json")
    campaign.add_argument("--domain-approval-json")
    campaign.add_argument("--ablation-output-dir")
    campaign.add_argument("--expected-seeds", nargs="+", type=int, default=[])
    campaign.add_argument("--expected-variants", default="")
    campaign.set_defaults(func=cmd_plan_iteration_campaign)

    record_round = sub.add_parser("record-iteration-round")
    record_round.add_argument("campaign_json")
    record_round.add_argument("--round", dest="round_index", type=int, required=True)
    record_round.add_argument("--evidence-path", required=True)
    record_round.add_argument("--validation-command", required=True)
    record_round.add_argument("--validation-exit-code", type=int, required=True)
    record_round.add_argument("--validation-output-path", required=True)
    record_round.add_argument("--notes", default="")
    record_round.set_defaults(func=cmd_record_iteration_round)

    verify_campaign = sub.add_parser("verify-iteration-campaign")
    verify_campaign.add_argument("campaign_json")
    verify_campaign.add_argument("--require-complete", action="store_true")
    verify_campaign.add_argument("--fail-on-issues", action="store_true")
    verify_campaign.set_defaults(func=cmd_verify_iteration_campaign)

    smoke_llm = sub.add_parser("smoke-llm")
    smoke_llm.add_argument("benchmark_dir")
    smoke_llm.add_argument("--variants", default=",".join(DEFAULT_SMOKE_VARIANTS))
    smoke_llm.add_argument("--seed", type=int, default=0)
    smoke_llm.add_argument("--timeout-s", type=int, default=60)
    smoke_llm.add_argument("--max-iterations", type=int, default=1)
    smoke_llm.add_argument("--parallel-mutations", type=int, default=2)
    smoke_llm.add_argument("--output-dir", default="runs/real-llm-smoke")
    smoke_llm.add_argument(
        "--llm-fast-mode",
        action="store_true",
        help="route unspecified agent reasoning_effort defaults to low for latency-sensitive real smoke runs",
    )
    smoke_mode = smoke_llm.add_mutually_exclusive_group()
    smoke_mode.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    smoke_mode.add_argument("--real", dest="dry_run", action="store_false")
    smoke_llm.set_defaults(func=cmd_smoke_llm)

    verify_smoke_llm = sub.add_parser("verify-smoke-llm")
    verify_smoke_llm.add_argument("output_dir")
    verify_smoke_llm.set_defaults(func=cmd_verify_smoke_llm)

    secret_hygiene = sub.add_parser("secret-hygiene")
    secret_hygiene.add_argument("run_dir")
    secret_hygiene.add_argument("--output-json")
    secret_hygiene.add_argument("--fail-on-findings", action="store_true")
    secret_hygiene.set_defaults(func=cmd_secret_hygiene)

    web = sub.add_parser("web")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--reload", action="store_true")
    web.set_defaults(func=cmd_web)

    benchmarks = sub.add_parser("benchmarks")
    benchmarks.add_argument("--json", action="store_true")
    benchmarks.set_defaults(func=cmd_benchmarks)
    return parser


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resume_expected_llm_call_range(
    *,
    run_dir: Path,
    experiment_id: str,
    benchmark_dir: Path,
    requested_max_iterations: int,
    parallel_mutations: int,
) -> dict[str, int]:
    """Return a fail-closed call range for only the work left by a v2 checkpoint."""

    config_payload = _resume_json_object(run_dir / "config.json", "stored run config")
    if config_payload.get("experiment_id") != experiment_id:
        raise ValueError("Cannot preflight real resume: stored config experiment_id mismatch")
    if config_payload.get("use_mock") is not False:
        raise ValueError("Cannot preflight real resume: stored run is not a real-LLM run")
    stored_evolution = config_payload.get("evolution")
    if not isinstance(stored_evolution, dict):
        raise ValueError("Cannot preflight real resume: stored evolution config is invalid")
    if stored_evolution.get("parallel_mutations") != parallel_mutations:
        raise ValueError(
            "Cannot preflight real resume: stored parallel_mutations does not match the request"
        )

    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return _pre_root_resume_expected_llm_call_range(
            run_dir=run_dir,
            benchmark_dir=benchmark_dir,
            requested_max_iterations=requested_max_iterations,
            parallel_mutations=parallel_mutations,
        )
    try:
        validate_real_llm_ledger_trace_consistency(
            run_dir,
            minimum_calls=MIN_ROOT_REAL_LLM_CALLS,
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot preflight real resume: invalid ledger/trace evidence"
        ) from exc

    checkpoint = _resume_json_object(checkpoint_path, "checkpoint")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Cannot preflight real resume: unsupported checkpoint schema_version; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if checkpoint.get("experiment_id") != experiment_id:
        raise ValueError("Cannot preflight real resume: checkpoint experiment_id mismatch")
    tree_issues = validate_solution_tree_artifact_payload(
        checkpoint,
        context="resume checkpoint",
    )
    if tree_issues:
        raise ValueError(
            "Cannot preflight real resume: invalid checkpoint solution tree: "
            + "; ".join(tree_issues)
        )
    try:
        historical_floor = real_llm_checkpoint_call_floor(run_dir, checkpoint)
        validate_real_llm_ledger_trace_consistency(
            run_dir,
            minimum_calls=int(historical_floor["minimum_calls"]),
            minimum_calls_by_role=dict(historical_floor["minimum_calls_by_role"]),
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot preflight real resume: checkpoint history exceeds ledger/trace evidence"
        ) from exc

    completed_iterations = _resume_non_negative_int(
        checkpoint.get("completed_iterations"),
        "checkpoint completed_iterations",
    )
    target_iterations = _resume_non_negative_int(
        checkpoint.get("target_iterations"),
        "checkpoint target_iterations",
    )
    if target_iterations < completed_iterations:
        raise ValueError(
            "Cannot preflight real resume: checkpoint target_iterations is less than "
            "completed_iterations"
        )

    phase = checkpoint.get("phase")
    valid_phases = {"root_created", "children_inflight", "iteration_completed", "completed", "exported"}
    if phase not in valid_phases:
        raise ValueError(f"Cannot preflight real resume: invalid checkpoint phase {phase!r}")
    if phase == "root_created" and completed_iterations != 0:
        raise ValueError("Cannot preflight real resume: stale root_created checkpoint progress")
    if phase == "iteration_completed" and completed_iterations == 0:
        raise ValueError("Cannot preflight real resume: stale iteration_completed checkpoint progress")

    inflight_batch = checkpoint.get("inflight_batch")
    if phase == "children_inflight":
        pending_current_slots = _resume_pending_inflight_slots(
            inflight_batch,
            completed_iterations=completed_iterations,
            parallel_mutations=parallel_mutations,
        )
    else:
        if inflight_batch is not None:
            raise ValueError(
                "Cannot preflight real resume: inflight_batch is only valid in children_inflight phase"
            )
        pending_current_slots = None

    if phase in {"completed", "exported"}:
        if target_iterations != completed_iterations:
            raise ValueError(
                "Cannot preflight real resume: completed checkpoint target does not match "
                "completed_iterations"
            )
        remaining_mutation_slots = requested_max_iterations * parallel_mutations
    else:
        if requested_max_iterations != target_iterations:
            raise ValueError(
                "Cannot preflight real resume: requested max_iterations does not match the "
                f"incomplete checkpoint target ({target_iterations})"
            )
        remaining_iterations = target_iterations - completed_iterations
        if phase == "children_inflight":
            if remaining_iterations < 1:
                raise ValueError(
                    "Cannot preflight real resume: children_inflight checkpoint has no remaining iteration"
                )
            assert pending_current_slots is not None
            remaining_mutation_slots = (
                pending_current_slots
                + (remaining_iterations - 1) * parallel_mutations
            )
        else:
            remaining_mutation_slots = remaining_iterations * parallel_mutations

    return _mutation_slot_llm_call_range(remaining_mutation_slots)


def _pre_root_resume_expected_llm_call_range(
    *,
    run_dir: Path,
    benchmark_dir: Path,
    requested_max_iterations: int,
    parallel_mutations: int,
) -> dict[str, int]:
    conditions_payload = _resume_json_object(
        run_dir / "experiment_conditions.json",
        "experiment conditions",
    )
    if conditions_payload.get("schema_version") != 1:
        raise ValueError("Cannot preflight real resume: experiment conditions schema is unsupported")
    conditions = conditions_payload.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("Cannot preflight real resume: experiment conditions payload is invalid")
    try:
        encoded_conditions = json.dumps(
            conditions,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Cannot preflight real resume: experiment conditions are not canonical JSON") from exc
    if conditions_payload.get("conditions_digest") != hashlib.sha256(encoded_conditions).hexdigest():
        raise ValueError("Cannot preflight real resume: experiment conditions digest mismatch")
    if conditions_payload.get("initial_target_iterations") != requested_max_iterations:
        raise ValueError("Cannot preflight real resume: pre-root target iterations mismatch")
    conditions_config = conditions.get("config")
    if not isinstance(conditions_config, dict) or conditions_config.get("use_mock") is not False:
        raise ValueError("Cannot preflight real resume: pre-root conditions are not for a real run")
    if conditions_config.get("benchmark_dir") != str(benchmark_dir.resolve()):
        raise ValueError("Cannot preflight real resume: pre-root benchmark path mismatch")
    conditions_evolution = conditions_config.get("evolution")
    if (
        not isinstance(conditions_evolution, dict)
        or conditions_evolution.get("parallel_mutations") != parallel_mutations
    ):
        raise ValueError("Cannot preflight real resume: pre-root evolution conditions mismatch")
    if not isinstance(conditions.get("llm_runtime"), dict) or not isinstance(
        conditions.get("source_revision"), dict
    ):
        raise ValueError("Cannot preflight real resume: pre-root provenance conditions are invalid")

    try:
        state = inspect_pre_root_resume_state(run_dir, ProblemBundle.load(benchmark_dir))
        validate_pre_root_real_llm_evidence(run_dir, state)
    except Exception as exc:
        raise ValueError("Cannot preflight real resume: invalid pre-root artifacts") from exc

    full_range = estimate_orchestrator_llm_call_range(
        max_iterations=requested_max_iterations,
        parallel_mutations=parallel_mutations,
    )
    return {
        "min": full_range["min"] - state.completed_llm_calls,
        "max": full_range["max"] - state.completed_llm_calls,
    }


def _resume_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot preflight real resume: invalid or missing {label} at {path}: "
            f"{type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Cannot preflight real resume: {label} must be a JSON object")
    return payload


def _resume_non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Cannot preflight real resume: {label} must be a non-negative integer")
    return value


def _resume_pending_inflight_slots(
    payload: object,
    *,
    completed_iterations: int,
    parallel_mutations: int,
) -> int:
    if not isinstance(payload, dict):
        raise ValueError("Cannot preflight real resume: children_inflight requires inflight_batch")
    if payload.get("schema_version") != INFLIGHT_BATCH_SCHEMA_VERSION:
        raise ValueError("Cannot preflight real resume: unsupported inflight_batch schema_version")
    if payload.get("iteration_index") != completed_iterations:
        raise ValueError("Cannot preflight real resume: inflight_batch iteration_index mismatch")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs or len(jobs) > parallel_mutations:
        raise ValueError(
            "Cannot preflight real resume: inflight_batch jobs must contain between one and "
            "parallel_mutations entries"
        )
    job_parents: dict[str, str] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Cannot preflight real resume: inflight_batch job must be an object")
        solution_id = job.get("solution_id")
        parent_id = job.get("parent_id")
        if not isinstance(solution_id, str) or not solution_id:
            raise ValueError("Cannot preflight real resume: inflight_batch solution_id is invalid")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError("Cannot preflight real resume: inflight_batch parent_id is invalid")
        if solution_id in job_parents:
            raise ValueError("Cannot preflight real resume: duplicate inflight_batch solution_id")
        job_parents[solution_id] = parent_id

    completed_children = payload.get("completed_children")
    if not isinstance(completed_children, list):
        raise ValueError("Cannot preflight real resume: completed_children must be a list")
    completed_ids: set[str] = set()
    for entry in completed_children:
        child = entry.get("child") if isinstance(entry, dict) else None
        solution_id = child.get("node_id") if isinstance(child, dict) else None
        if not isinstance(solution_id, str) or solution_id not in job_parents:
            raise ValueError("Cannot preflight real resume: completed child does not match an inflight job")
        if entry.get("parent_id") != job_parents[solution_id]:
            raise ValueError("Cannot preflight real resume: completed child parent mismatch")
        if solution_id in completed_ids:
            raise ValueError("Cannot preflight real resume: duplicate completed inflight child")
        completed_ids.add(solution_id)
    return len(set(job_parents) - completed_ids)


def _mutation_slot_llm_call_range(mutation_slots: int) -> dict[str, int]:
    if mutation_slots == 0:
        return {"min": 0, "max": 0}
    root_range = estimate_orchestrator_llm_call_range(
        max_iterations=0,
        parallel_mutations=1,
    )
    with_root_range = estimate_orchestrator_llm_call_range(
        max_iterations=1,
        parallel_mutations=mutation_slots,
    )
    return {
        "min": with_root_range["min"] - root_range["min"],
        "max": with_root_range["max"] - root_range["max"],
    }


def _load_resume_llm_budget_usage(
    ledger_path: Path,
    budget: LLMBudget,
    *,
    minimum_calls: int,
) -> None:
    if not ledger_path.is_file():
        if minimum_calls == 0:
            return
        raise ValueError(
            f"Cannot preflight real resume: missing LLM call ledger at {ledger_path}"
        )
    load_llm_budget_usage(ledger_path, budget)
    if budget.calls_used < minimum_calls:
        raise ValueError(
            "Cannot preflight real resume: LLM call ledger is stale for the saved progress; "
            f"calls_used={budget.calls_used}, required_at_least={minimum_calls}"
        )


def _resume_minimum_ledger_calls(run_dir: Path, benchmark_dir: Path) -> int:
    if (run_dir / "checkpoint.json").is_file():
        return estimate_orchestrator_llm_call_range(
            max_iterations=0,
            parallel_mutations=1,
        )["min"]
    return inspect_pre_root_resume_state(
        run_dir,
        ProblemBundle.load(benchmark_dir),
    ).completed_llm_calls


def _validate_resume_dry_run_conditions(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    args: argparse.Namespace,
    budget: LLMBudget,
) -> None:
    payload = _resume_json_object(run_dir / "experiment_conditions.json", "experiment conditions")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("conditions"), dict):
        raise ValueError("Cannot preflight real resume: experiment conditions are invalid")
    stored_conditions = payload["conditions"]
    assert isinstance(stored_conditions, dict)
    encoded = json.dumps(
        stored_conditions,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if payload.get("conditions_digest") != hashlib.sha256(encoded).hexdigest():
        raise ValueError("Cannot preflight real resume: experiment conditions digest mismatch")

    config_payload = config.to_dict()
    config_payload.pop("experiment_id", None)
    config_payload.pop("output_dir", None)
    config_payload.pop("resume", None)
    config_payload["benchmark_dir"] = str(config.benchmark_dir.resolve())
    evolution = config_payload.get("evolution")
    if isinstance(evolution, dict):
        evolution = dict(evolution)
        evolution.pop("max_iterations", None)
        config_payload["evolution"] = evolution
    source_revision = read_source_revision(config.benchmark_dir)
    if any(value == "unknown" for value in source_revision.values()):
        raise ValueError("Cannot preflight real resume: Git source provenance is unavailable")
    current_conditions = {
        "config": config_payload,
        "llm_runtime": _dry_run_real_llm_runtime_identity(args, budget),
        "source_revision": source_revision,
    }
    try:
        validate_resume_conditions_compatible(stored_conditions, current_conditions)
    except ValueError as exc:
        raise ValueError(f"Cannot preflight real resume: incompatible conditions: {exc}") from exc


def _dry_run_real_llm_runtime_identity(
    args: argparse.Namespace,
    budget: LLMBudget,
) -> dict[str, object]:
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = _normalize_model_name(os.environ.get("OPENAI_MODEL", "gpt-5-mini"), base_url)
    timeout_s = args.llm_timeout_s if args.llm_timeout_s is not None else _timeout_from_env()
    max_retries = (
        args.llm_max_retries
        if args.llm_max_retries is not None
        else _max_retries_from_env()
    )
    capabilities = capabilities_for_openai_compatible(base_url)
    inner = {
        "adapter_class": f"{OpenAIAdapter.__module__}.{OpenAIAdapter.__qualname__}",
        "model": model,
        "provider": None,
        "provider_name": capabilities.provider,
        "adapter_type": capabilities.adapter_type,
        "timeout_s": timeout_s,
        "max_retries": max_retries,
    }
    return {
        "adapter_class": (
            f"{RecordingLLMClient.__module__}.{RecordingLLMClient.__qualname__}"
        ),
        "model": model,
        "provider": capabilities.provider,
        "provider_name": capabilities.provider,
        "adapter_type": capabilities.adapter_type,
        "timeout_s": None,
        "max_retries": None,
        "inner": inner,
        "budget_limits": {
            "max_prompt_tokens": budget.max_prompt_tokens,
            "max_output_tokens": budget.max_output_tokens,
            "max_total_tokens": budget.max_total_tokens,
            "max_calls": budget.max_calls,
            "max_cost_usd": budget.max_cost_usd,
            "cost_per_1k_tokens_usd": budget.cost_per_1k_tokens_usd,
        },
    }


def _validate_run_arguments(args: argparse.Namespace) -> None:
    if args.max_iterations < 0:
        raise ValueError("--max-iterations must be >= 0")
    if args.parallel_mutations < 1:
        raise ValueError("--parallel-mutations must be >= 1")
    if args.max_children_per_node < 1:
        raise ValueError("--max-children-per-node must be >= 1")
    if args.max_debug_retries < 0:
        raise ValueError("--max-debug-retries must be >= 0")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if args.llm_timeout_s is not None and (
        not math.isfinite(args.llm_timeout_s) or args.llm_timeout_s <= 0
    ):
        raise ValueError("--llm-timeout-s must be a finite number > 0")
    if args.llm_max_retries is not None and args.llm_max_retries < 0:
        raise ValueError("--llm-max-retries must be >= 0")
    if args.selector_vote_count < 1:
        raise ValueError("--selector-vote-count must be >= 1")
    if args.no_kb and args.random_kb:
        raise ValueError("--random-kb cannot be combined with --no-kb")
    if args.resume and not args.experiment_id:
        raise ValueError("--resume requires an explicit --experiment-id")
    reviewer = (args.domain_reviewer or "").strip()
    notes = (args.domain_review_notes or "").strip()
    if args.domain_evaluator_approved and not (reviewer and notes):
        raise ValueError(
            "--domain-evaluator-approved requires --domain-reviewer and --domain-review-notes"
        )
    if not args.domain_evaluator_approved and (reviewer or notes):
        raise ValueError(
            "--domain-reviewer/--domain-review-notes require --domain-evaluator-approved"
        )


def _validated_benchmark_snapshot(benchmark_dir: Path) -> dict[str, object]:
    bundle = ProblemBundle.load(benchmark_dir)
    contract = BenchmarkContractFactory.create_contract(bundle)
    return {
        "name": bundle.benchmark_name,
        "path": str(benchmark_dir),
        "family": bundle.benchmark_spec.family,
        "fidelity_level": bundle.benchmark_spec.fidelity_level,
        "metric": bundle.benchmark_spec.metric,
        "contract_hash": contract.contract_hash,
        "benchmark_source_manifest_digest": contract.benchmark_source_manifest_digest,
    }


def _strategy_seed_ids(args: argparse.Namespace) -> list[str]:
    values = [*args.strategy_seed_id, *_split_csv(args.strategy_seed_ids)]
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def _validate_strategy_seed_ids(strategy_seed_ids: list[str]) -> None:
    known = {algorithm.algorithm_id for algorithm in list_algorithms()}
    unknown = sorted(set(strategy_seed_ids) - known)
    if unknown:
        raise ValueError("Unknown --strategy-seed-id value(s): " + ", ".join(unknown))


def _build_run_plan(
    config: ExperimentConfig,
    benchmark_snapshot: dict[str, object],
    expected_call_range: dict[str, int],
) -> dict[str, object]:
    deterministic_roles = ["data_analyst", "evaluator", "root_engineer", "result_analyst"]
    mutation_roles: list[str] = []
    conditional_roles: list[str] = []
    if config.evolution.max_iterations > 0:
        if config.evolution.use_kb:
            mutation_roles.append("retriever")
        mutation_roles.append("proposer")
        if config.evolution.use_critic:
            mutation_roles.append("critic")
        mutation_roles.append("engineer")
        if config.evolution.use_debugger and config.evolution.max_debug_retries > 0:
            conditional_roles.append("debugger")
        conditional_roles.append("selector")
    return {
        "schema_version": 1,
        "execution_mode": "dry_run",
        "llm_mode": "mock" if config.use_mock else "real",
        "provider_calls": 0 if config.use_mock else "not_executed",
        "benchmark": benchmark_snapshot,
        "experiment_config": config.to_dict(),
        "planned_agent_roles": deterministic_roles + mutation_roles,
        "conditional_agent_roles": conditional_roles,
        "expected_llm_call_range": (
            {"min": 0, "max": 0}
            if config.use_mock
            else dict(expected_call_range)
        ),
        "planned_solution_upper_bound": (
            1 + config.evolution.max_iterations * config.evolution.parallel_mutations
        ),
        "notes": [
            "No provider calls or run artifacts were created.",
            "selector is conditional on the deterministic parent policy requiring an LLM vote.",
            "debugger is conditional on a generated-solution failure and the retry budget.",
        ],
    }


def _completed_run_issues(run_dir: Path) -> list[str]:
    issues: list[str] = []
    try:
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid or missing run_metadata.json: {type(exc).__name__}: {exc}"]
    try:
        tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid or missing tree.json: {type(exc).__name__}: {exc}"]
    try:
        trace_summary = json.loads((run_dir / "trace_summary.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid or missing trace_summary.json: {type(exc).__name__}: {exc}"]

    if trace_summary.get("quality_gate", {}).get("passed") is not True:
        issues.append("trace_summary quality_gate did not pass")
    champion_id = metadata.get("champion")
    if not isinstance(champion_id, str) or not champion_id:
        issues.append("run_metadata champion is missing")
        return issues
    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        issues.append("tree.json nodes must be an array")
        return issues
    champion = next(
        (node for node in nodes if isinstance(node, dict) and node.get("node_id") == champion_id),
        None,
    )
    if champion is None:
        issues.append(f"champion {champion_id} is not present in tree.json")
        return issues
    if champion.get("status") != "evaluated":
        issues.append(f"champion {champion_id} is not evaluated")
    score = champion.get("score")
    value = score.get("value") if isinstance(score, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        issues.append(f"champion {champion_id} has no finite score")
    return issues


def _selector_panel_payloads(args: argparse.Namespace) -> list[dict[str, object]]:
    payload = json.loads(args.selector_panel_json)
    if not isinstance(payload, list):
        raise ValueError("--selector-panel-json must be a JSON array")
    if payload:
        result: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("--selector-panel-json items must be JSON objects")
            result.append(dict(item))
        return result
    return [{"model": model} for model in _split_csv(args.selector_panel_models)]


def _agent_configs_from_role_payloads(payload: dict[str, object]) -> dict[str, AgentConfig]:
    known_roles = set(DEFAULT_AGENT_ROLE_MODEL_SETTINGS)
    unknown_roles = sorted(set(payload) - known_roles)
    if unknown_roles:
        raise ValueError("--agent-models-json unknown role(s): " + ", ".join(unknown_roles))
    result: dict[str, AgentConfig] = {}
    for role, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError("--agent-models-json values must be JSON objects")
        model = str(value.get("model", "")).strip()
        if not model:
            raise ValueError(f"--agent-models-json {role} model is required")
        base_url_value = value.get("base_url")
        base_url = str(base_url_value).strip() if base_url_value is not None else None
        defaults = DEFAULT_AGENT_ROLE_MODEL_SETTINGS[role]
        result[role] = AgentConfig(
            role=role,
            model=model,
            temperature=float(value.get("temperature", defaults["temperature"])),
            reasoning_effort=(
                str(value["reasoning_effort"])
                if value.get("reasoning_effort") is not None
                else None
            ),
            base_url=base_url or None,
        )
    return result


def _agent_config_from_selector_payload(index: int, payload: dict[str, object]) -> AgentConfig:
    model = str(payload.get("model", "")).strip()
    if not model:
        raise ValueError("--selector-panel-json selector member model is required")
    base_url_value = payload.get("base_url")
    base_url = str(base_url_value).strip() if base_url_value is not None else None
    return AgentConfig(
        role=str(payload.get("role") or f"selector_{index:03d}"),
        model=model,
        temperature=float(payload.get("temperature", 0.05)),
        reasoning_effort=(
            str(payload["reasoning_effort"])
            if payload.get("reasoning_effort") is not None
            else "high"
        ),
        base_url=base_url or None,
    )


def _json_object_arg(value: str, label: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _json_object_file_arg(value: str, label: str) -> dict[str, object]:
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _json_array_arg(value: str, label: str) -> list[dict[str, object]]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON array: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON array")
    result: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{label} items must be JSON objects")
        result.append(dict(item))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
