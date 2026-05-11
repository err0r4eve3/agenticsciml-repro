from __future__ import annotations

import json
from pathlib import Path

from agenticsciml.state import SOLUTION_TREE_SCHEMA_VERSION, SolutionNode


def write_tree_json(run_dir: Path, nodes: list[SolutionNode]) -> Path:
    path = run_dir / "tree.json"
    payload = {
        "schema_version": SOLUTION_TREE_SCHEMA_VERSION,
        "nodes": [node.to_dict() for node in nodes],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def write_tree_mermaid(run_dir: Path, nodes: list[SolutionNode]) -> Path:
    path = run_dir / "tree.mmd"
    lines = ["graph TD"]
    for node in nodes:
        score = "NA" if node.score is None else f"{node.score.value:.6g}"
        label = f"{node.node_id}<br/>{score}"
        lines.append(f'  {node.node_id}["{label}"]')
    for node in nodes:
        for child in node.children:
            lines.append(f"  {node.node_id} --> {child}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
