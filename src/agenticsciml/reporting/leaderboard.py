from __future__ import annotations

import csv
from pathlib import Path

from agenticsciml.state import SolutionNode


def _sort_key(node: SolutionNode) -> float:
    if node.score is None:
        return float("inf")
    return node.score.value if not node.score.higher_is_better else -node.score.value


def write_leaderboard(run_dir: Path, nodes: list[SolutionNode]) -> Path:
    path = run_dir / "leaderboard.csv"
    ordered = sorted(nodes, key=_sort_key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "node_id", "parent_id", "metric", "score", "status"])
        for rank, node in enumerate(ordered, start=1):
            writer.writerow(
                [
                    rank,
                    node.node_id,
                    node.parent_id or "",
                    node.score.metric if node.score else "",
                    node.score.value if node.score else "",
                    node.status,
                ]
            )
    return path
