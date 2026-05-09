from __future__ import annotations

import argparse
import csv
from pathlib import Path

from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


VARIANTS = [
    ("root_only", 0, True, False),
    ("multi_agent_no_kb", 1, False, False),
    ("multi_agent_with_kb", 1, True, False),
    ("multi_agent_random_kb", 1, True, True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="examples/function_approx")
    parser.add_argument("--output-dir", default="runs/ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    rows = []
    for name, iterations, use_kb, random_kb in VARIANTS:
        config = ExperimentConfig(
            experiment_id=name,
            benchmark_dir=Path(args.benchmark_dir).resolve(),
            output_dir=output_dir,
            evolution=EvolutionConfig(
                max_iterations=iterations,
                parallel_mutations=2,
                use_kb=use_kb,
                random_kb=random_kb,
            ),
            use_mock=True,
        )
        run_dir = AgenticSciMLOrchestrator(config, MockLLMClient()).run()
        rows.append([name, run_dir])

    summary = output_dir / "ablation_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "run_dir"])
        writer.writerows(rows)
    print(summary)


if __name__ == "__main__":
    main()
