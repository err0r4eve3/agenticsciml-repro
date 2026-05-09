from __future__ import annotations

from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.state import AnalysisReport, SolutionNode


class RetrievalQueryBuilder:
    @staticmethod
    def build(
        problem_bundle: ProblemBundle,
        *,
        parent: SolutionNode,
        parent_analysis: AnalysisReport | None,
        leaderboard: list[SolutionNode],
        top_k: int = 3,
    ) -> str:
        top_nodes = RetrievalQueryBuilder._top_nodes(leaderboard, top_k)
        leaderboard_lines = [
            (
                f"{node.node_id}: score={node.score.value if node.score else 'none'} "
                f"tags={','.join(node.method_tags) or 'none'} "
                f"failure={node.failure_kind or 'none'}"
            )
            for node in top_nodes
        ]
        analysis_summary = parent_analysis.summary if parent_analysis else ""
        return "\n".join(
            [
                f"benchmark_name: {problem_bundle.benchmark_name}",
                f"benchmark_family: {problem_bundle.benchmark_spec.family}",
                f"benchmark_metric: {problem_bundle.benchmark_spec.metric}",
                f"benchmark_description: {problem_bundle.benchmark_spec.description}",
                f"parent_id: {parent.node_id}",
                f"parent_score: {parent.score.value if parent.score else 'none'}",
                f"parent_status: {parent.status}",
                f"parent_failure_kind: {parent.failure_kind or 'none'}",
                f"parent_method_tags: {', '.join(parent.method_tags) or 'none'}",
                f"parent_score_delta: {parent.score_delta_from_parent if parent.score_delta_from_parent is not None else 'none'}",
                f"parent_analysis: {analysis_summary}",
                "leaderboard_top_k:",
                *leaderboard_lines,
            ]
        )

    @staticmethod
    def _top_nodes(nodes: list[SolutionNode], top_k: int) -> list[SolutionNode]:
        scored = [node for node in nodes if node.score is not None]
        if not scored:
            return nodes[:top_k]
        first_score = scored[0].score
        reverse = bool(first_score and first_score.higher_is_better)
        return sorted(scored, key=lambda node: node.score.value if node.score else 0.0, reverse=reverse)[
            :top_k
        ]
