from __future__ import annotations

import argparse
import shutil
import sys
import time
import json
from pathlib import Path

from agenticsciml.ablation import DEFAULT_VARIANTS, run_ablation
from agenticsciml.benchmarks import list_benchmarks
from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.llm_smoke import DEFAULT_SMOKE_VARIANTS, run_llm_smoke
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.reporting import write_trace_summary


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
    )
    config = ExperimentConfig(
        experiment_id=args.experiment_id or _default_experiment_id(args.mock),
        benchmark_dir=benchmark_dir,
        output_dir=Path(args.output_dir).resolve(),
        evolution=evolution,
        use_mock=args.mock,
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


def cmd_benchmarks(args: argparse.Namespace) -> int:
    specs = list_benchmarks()
    if args.json:
        print(json.dumps({"benchmarks": [spec.to_dict() for spec in specs]}, indent=2, sort_keys=True))
        return 0
    for spec in specs:
        print(f"{spec.name}\t{spec.paper_section}\t{spec.family}\t{spec.metric}\t{spec.path}")
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

    ablate = sub.add_parser("ablate")
    ablate.add_argument("benchmark_dir")
    ablate.add_argument("--seeds", nargs="+", type=int, default=[0])
    ablate.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ablate.add_argument("--output-dir", default="runs/ablation")
    ablate.set_defaults(func=cmd_ablate)

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

    benchmarks = sub.add_parser("benchmarks")
    benchmarks.add_argument("--json", action="store_true")
    benchmarks.set_defaults(func=cmd_benchmarks)
    return parser


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
