from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage


@dataclass(frozen=True, slots=True)
class SelectorVoteResult:
    selected_parent_ids: list[str]
    votes: list[dict[str, object]]
    vote_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_parent_ids": self.selected_parent_ids,
            "votes": self.votes,
            "vote_counts": self.vote_counts,
        }


class SelectorAgent(AgentBase):
    role = "selector"

    def select(
        self,
        candidates: list[dict[str, object]],
        best_node_id: str,
        max_to_select: int,
    ) -> list[str]:
        return self.select_with_votes(candidates, best_node_id, max_to_select).selected_parent_ids

    def select_with_votes(
        self,
        candidates: list[dict[str, object]],
        best_node_id: str,
        max_to_select: int,
        vote_count: int = 1,
    ) -> SelectorVoteResult:
        self.require_inputs(
            {
                "candidates": candidates,
                "best_node_id": best_node_id,
                "max_to_select": max_to_select,
            }
        )
        vote_count = max(1, vote_count)
        candidate_ids = {str(candidate.get("node_id")) for candidate in candidates}
        messages: list[AgentMessage] = []
        votes: list[dict[str, object]] = []
        vote_counts: dict[str, int] = {}
        for vote_index in range(vote_count):
            prompt = (
                "Select exploration parents for ensemble-guided mutation. "
                "The lowest-loss best solution will be included separately for exploitation. "
                "Vote for candidates with high improvement potential, fixable failures, "
                "or underexplored ideas. Return JSON with selected_parent_ids and rationale. "
                f"Best by loss: {best_node_id}. Max total parents after best inclusion: {max_to_select}. "
                f"Vote index: {vote_index + 1}/{vote_count}. Candidates: {candidates}"
            )
            response = self.complete_json_checked(
                prompt,
                "selector",
                required_fields=("selected_parent_ids", "rationale"),
            )
            raw_selected = [str(item) for item in response.get("selected_parent_ids", [])]
            selected = [
                node_id
                for node_id in raw_selected
                if node_id in candidate_ids and node_id != best_node_id
            ]
            for node_id in selected:
                vote_counts[node_id] = vote_counts.get(node_id, 0) + 1
            vote = {
                "vote_index": vote_index + 1,
                "selected_parent_ids": selected,
                "rationale": str(response.get("rationale", "")),
            }
            votes.append(vote)
            messages.append(AgentMessage(self.role, prompt, str(response)))

        selected_parent_ids = [best_node_id]
        selected_parent_ids.extend(
            _rank_vote_winners(vote_counts, candidates, max(0, max_to_select - 1))
        )
        selected_parent_ids = selected_parent_ids[:max_to_select]
        result = SelectorVoteResult(
            selected_parent_ids=selected_parent_ids,
            votes=votes,
            vote_counts=dict(sorted(vote_counts.items())),
        )
        self.storage.save_json("reports/selector_votes.json", result.to_dict())
        self._save_messages(None, messages)
        return result


def _rank_vote_winners(
    vote_counts: dict[str, int],
    candidates: list[dict[str, object]],
    limit: int,
) -> list[str]:
    if limit <= 0 or not vote_counts:
        return []
    by_id = {str(candidate.get("node_id")): candidate for candidate in candidates}
    return [
        node_id
        for node_id, _count in sorted(
            vote_counts.items(),
            key=lambda item: (
                -item[1],
                -_candidate_score_rank(by_id.get(item[0], {})),
                item[0],
            ),
        )[:limit]
    ]


def _candidate_score_rank(candidate: dict[str, Any]) -> float:
    score = candidate.get("score")
    if not isinstance(score, dict):
        return float("-inf")
    value = score.get("value")
    if not isinstance(value, int | float):
        return float("-inf")
    return float(value) if score.get("higher_is_better") is True else -float(value)
