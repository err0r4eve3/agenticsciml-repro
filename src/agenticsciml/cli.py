from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import json
from pathlib import Path

from agenticsciml.ablation_evidence import build_multi_seed_ablation_verified_manifest
from agenticsciml.ablation import DEFAULT_VARIANTS, run_ablation
from agenticsciml.benchmarks import list_benchmarks
from agenticsciml.config import (
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
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.llm_smoke import DEFAULT_SMOKE_VARIANTS, run_llm_smoke, verify_llm_smoke_output
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.paper_workflow_readiness import write_paper_workflow_readiness_bundle
from agenticsciml.reporting import write_sdk_trace_export, write_trace_summary
from agenticsciml.storage import _atomic_write_text


def _default_experiment_id(mock: bool) -> str:
    prefix = "mock" if mock else "real"
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


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
    print(target.resolve())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    benchmark_dir = Path(args.benchmark_dir).resolve()
    if args.dry_run:
        roles = [
            "data_analyst",
            "evaluator",
            "root_engineer",
            "selector",
            "retriever",
            "proposer",
            "critic",
            "engineer",
            "debugger",
            "result_analyst",
        ]
        print("Planned agent calls:")
        for role in roles:
            print(f"- {role}")
        return 0

    evolution = EvolutionConfig(
        max_iterations=args.max_iterations,
        parallel_mutations=args.parallel_mutations,
        timeout_s=args.timeout_s,
        use_kb=not args.no_kb,
        random_kb=args.random_kb,
        random_seed=args.random_seed,
        use_branch_context=not args.no_branch_context,
        selector_vote_count=args.selector_vote_count,
    )
    config = ExperimentConfig(
        experiment_id=args.experiment_id or _default_experiment_id(args.mock),
        benchmark_dir=benchmark_dir,
        output_dir=Path(args.output_dir).resolve(),
        evolution=evolution,
        use_mock=args.mock,
        selector_panel=[
            _agent_config_from_selector_payload(index, payload)
            for index, payload in enumerate(_selector_panel_payloads(args), start=1)
        ],
        visual_audit_mode=args.visual_audit_mode,
        resource_constraints=_json_object_arg(args.resource_constraints_json, "--resource-constraints-json"),
        expert_blueprint_id=args.expert_blueprint_id,
        multi_seed_ablation=_json_object_arg(args.multi_seed_ablation_json, "--multi-seed-ablation-json"),
        auto_approve_evaluation=not args.require_evaluation_approval,
        resume=args.resume,
    )
    llm = MockLLMClient() if args.mock else OpenAIAdapter()
    run_dir = AgenticSciMLOrchestrator(config, llm).run()
    print(run_dir.resolve())
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
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


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
        mock=True,
    )
    print(result.summary_csv.resolve())
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


def cmd_plan_paper_workflow(args: argparse.Namespace) -> int:
    selector_panel = _json_array_arg(args.selector_panel_json, "--selector-panel-json")
    result = write_paper_workflow_readiness_bundle(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        selector_panel=selector_panel,
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


def cmd_plan_iteration_campaign(args: argparse.Namespace) -> int:
    selector_panel = _json_array_arg(args.selector_panel_json, "--selector-panel-json")
    result = write_iteration_campaign(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        rounds=args.rounds,
        batch_size=args.batch_size,
        selector_panel=selector_panel,
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
    result = write_iteration_campaign_verification(Path(args.campaign_json).resolve())
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
    )
    print(result.report_md.resolve())
    return 0


def cmd_verify_smoke_llm(args: argparse.Namespace) -> int:
    result = verify_llm_smoke_output(Path(args.output_dir).resolve())
    print(result.verification_json.resolve())
    return 0 if result.passed else 1


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
    run.add_argument("--mock", action="store_true")
    run.add_argument("--max-iterations", type=int, default=1)
    run.add_argument("--parallel-mutations", type=int, default=2)
    run.add_argument("--timeout-s", type=int, default=60)
    run.add_argument("--output-dir", default="runs")
    run.add_argument("--experiment-id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--no-kb", action="store_true")
    run.add_argument("--random-kb", action="store_true")
    run.add_argument("--random-seed", type=int, default=0)
    run.add_argument("--no-branch-context", action="store_true")
    run.add_argument("--selector-vote-count", type=int, default=3)
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
    ablate.set_defaults(func=cmd_ablate)

    verify_ablation = sub.add_parser("verify-ablation-evidence")
    verify_ablation.add_argument("output_dir")
    verify_ablation.add_argument("--verified-by", required=True)
    verify_ablation.add_argument("--expected-seeds", nargs="+", type=int, default=[])
    verify_ablation.add_argument("--expected-variants", default="")
    verify_ablation.add_argument("--output-json")
    verify_ablation.set_defaults(func=cmd_verify_ablation_evidence)

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
    paper_workflow.add_argument("--domain-approval-json")
    paper_workflow.add_argument("--ablation-output-dir")
    paper_workflow.add_argument("--expected-seeds", nargs="+", type=int, default=[])
    paper_workflow.add_argument("--expected-variants", default="")
    paper_workflow.add_argument("--fail-on-blockers", action="store_true")
    paper_workflow.set_defaults(func=cmd_plan_paper_workflow)

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
    smoke_mode = smoke_llm.add_mutually_exclusive_group()
    smoke_mode.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    smoke_mode.add_argument("--real", dest="dry_run", action="store_false")
    smoke_llm.set_defaults(func=cmd_smoke_llm)

    verify_smoke_llm = sub.add_parser("verify-smoke-llm")
    verify_smoke_llm.add_argument("output_dir")
    verify_smoke_llm.set_defaults(func=cmd_verify_smoke_llm)

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
