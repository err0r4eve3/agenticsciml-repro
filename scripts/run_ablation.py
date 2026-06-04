from __future__ import annotations

import argparse
from pathlib import Path

from agenticsciml.ablation import DEFAULT_VARIANTS, run_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="examples/function_approx")
    parser.add_argument("--output-dir", default="runs/ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--llm-timeout-s", type=float, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=None)
    parser.add_argument("--llm-fast-mode", action="store_true")
    args = parser.parse_args()

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
    )
    print(result.summary_csv or result.plan_json or result.report_md)


if __name__ == "__main__":
    main()
