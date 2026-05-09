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
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    result = run_ablation(
        benchmark_dir=Path(args.benchmark_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        seeds=args.seeds,
        variants=variants,
    )
    print(result.summary_csv)


if __name__ == "__main__":
    main()
