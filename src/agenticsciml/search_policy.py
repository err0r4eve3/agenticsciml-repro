from __future__ import annotations

import random
from dataclasses import dataclass

from agenticsciml.state import SolutionNode


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    max_children_per_node: int
    random_seed: int = 0
    include_random: bool = False

    def select(self, nodes: list[SolutionNode], max_to_select: int) -> list[SolutionNode]:
        if max_to_select <= 0:
            return []
        available = [
            node
            for node in nodes
            if len(node.children) < self.max_children_per_node
        ]
        if not available:
            return []

        selected: list[SolutionNode] = []
        best = self._best_available(available)
        self._append_unique(selected, best, max_to_select)
        self._append_unique(selected, self._high_improvement(available, selected), max_to_select)
        self._append_unique(selected, self._diverse_underexplored(available, selected), max_to_select)

        remaining = [node for node in available if node.node_id not in {item.node_id for item in selected}]
        if self.include_random:
            rng = random.Random(self.random_seed)
            rng.shuffle(remaining)

        for node in remaining:
            self._append_unique(selected, node, max_to_select)
            if len(selected) >= max_to_select:
                break
        return selected

    def _best_available(self, nodes: list[SolutionNode]) -> SolutionNode:
        scored = [node for node in nodes if node.score is not None and node.status == "evaluated"]
        if not scored:
            return nodes[0]
        best = scored[0]
        for node in scored[1:]:
            if node.score and node.score.better_than(best.score):
                best = node
        return best

    def _high_improvement(
        self,
        nodes: list[SolutionNode],
        selected: list[SolutionNode],
    ) -> SolutionNode | None:
        selected_ids = {node.node_id for node in selected}
        candidates = [
            node
            for node in nodes
            if node.node_id not in selected_ids and node.score_delta_from_parent is not None
        ]
        candidates = [
            node
            for node in candidates
            if self._improvement_value(node) > 0
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda node: (
                self._improvement_value(node),
                -len(node.children),
                node.node_id,
            ),
        )

    def _diverse_underexplored(
        self,
        nodes: list[SolutionNode],
        selected: list[SolutionNode],
    ) -> SolutionNode | None:
        selected_ids = {node.node_id for node in selected}
        selected_tags = {tag for node in selected for tag in node.method_tags}
        candidates = [node for node in nodes if node.node_id not in selected_ids]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda node: (
                len(set(node.method_tags) - selected_tags),
                -len(node.children),
                self._score_rank(node),
                node.node_id,
            ),
        )

    def _improvement_value(self, node: SolutionNode) -> float:
        if node.score_delta_from_parent is None:
            return 0.0
        if node.score and node.score.higher_is_better:
            return node.score_delta_from_parent
        return -node.score_delta_from_parent

    def _score_rank(self, node: SolutionNode) -> float:
        if node.score is None:
            return float("-inf")
        return node.score.value if node.score.higher_is_better else -node.score.value

    def _append_unique(
        self,
        selected: list[SolutionNode],
        node: SolutionNode | None,
        max_to_select: int,
    ) -> None:
        if node is None or len(selected) >= max_to_select:
            return
        if any(item.node_id == node.node_id for item in selected):
            return
        selected.append(node)
