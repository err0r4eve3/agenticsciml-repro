from __future__ import annotations

from agenticsciml.search_policy import SearchPolicy
from agenticsciml.state import SolutionNode, SolutionScore


def _node(
    node_id: str,
    score: float,
    *,
    children: list[str] | None = None,
    tags: list[str] | None = None,
    delta: float | None = None,
    status: str = "evaluated",
) -> SolutionNode:
    return SolutionNode(
        node_id=node_id,
        parent_id=None,
        workspace=f"/tmp/{node_id}",
        score=SolutionScore(metric="validation_mse", value=score, higher_is_better=False),
        children=children or [],
        status=status,
        method_tags=tags or [],
        score_delta_from_parent=delta,
    )


def test_search_policy_includes_best_available_node() -> None:
    nodes = [
        _node("weak", 1.0, tags=["mlp"]),
        _node("best", 0.1, tags=["fourier_features"]),
    ]

    selected = SearchPolicy(max_children_per_node=10, random_seed=0).select(
        nodes,
        max_to_select=1,
    )

    assert [node.node_id for node in selected] == ["best"]


def test_search_policy_skips_nodes_over_child_limit() -> None:
    nodes = [
        _node("full_best", 0.01, children=["a", "b"]),
        _node("available_best", 0.2),
        _node("available_weak", 0.9),
    ]

    selected = SearchPolicy(max_children_per_node=2, random_seed=0).select(
        nodes,
        max_to_select=2,
    )

    selected_ids = [node.node_id for node in selected]
    assert "full_best" not in selected_ids
    assert selected_ids[0] == "available_best"


def test_search_policy_adds_high_improvement_node() -> None:
    nodes = [
        _node("best", 0.2, tags=["baseline"]),
        _node("improved", 0.3, tags=["weighted_loss"], delta=-0.4),
        _node("flat", 0.4, tags=["baseline"], delta=0.0),
    ]

    selected = SearchPolicy(max_children_per_node=10, random_seed=0).select(
        nodes,
        max_to_select=2,
    )

    assert [node.node_id for node in selected] == ["best", "improved"]


def test_search_policy_prefers_underexplored_diverse_node() -> None:
    nodes = [
        _node("best", 0.1, tags=["mlp"], children=["child"]),
        _node("same_family", 0.2, tags=["mlp"]),
        _node("diverse", 0.3, tags=["fourier_features"]),
    ]

    selected = SearchPolicy(max_children_per_node=10, random_seed=0).select(
        nodes,
        max_to_select=2,
    )

    assert [node.node_id for node in selected] == ["best", "diverse"]
