from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage

SELECTOR_VOTES_SCHEMA_VERSION = 2
SINGLE_SELECTOR_CLAIM_BOUNDARY = (
    "selector_vote_count records repeated votes through one selector path. "
    "It is not heterogeneous selector ensemble evidence."
)
CONFIGURED_PANEL_CLAIM_BOUNDARY = (
    "selector_panel records configured selector member provenance. It is only heterogeneous "
    "provider evidence when the recorded actual_model/provider values differ across members."
)


@dataclass(frozen=True, slots=True)
class SelectorVoteResult:
    selected_parent_ids: list[str]
    votes: list[dict[str, object]]
    vote_counts: dict[str, int]
    ensemble_mode: str = "single_provider_multi_vote"
    panel_members: list[dict[str, object]] = field(default_factory=list)
    claim_boundary: str = SINGLE_SELECTOR_CLAIM_BOUNDARY
    schema_version: int = SELECTOR_VOTES_SCHEMA_VERSION
    diversity: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ensemble_mode": self.ensemble_mode,
            "selector_panel_members": self.panel_members,
            "selector_diversity": self.diversity,
            "selected_parent_ids": self.selected_parent_ids,
            "votes": self.votes,
            "vote_counts": self.vote_counts,
            "claim_boundary": self.claim_boundary,
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
        panel_member: dict[str, object] | None = None,
        ensemble_mode: str = "single_provider_multi_vote",
        panel_members: list[dict[str, object]] | None = None,
        claim_boundary: str = SINGLE_SELECTOR_CLAIM_BOUNDARY,
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
        member = _normalized_panel_member(panel_member or _default_panel_member(self.llm))
        for vote_index in range(vote_count):
            vote, message = self.cast_vote(
                candidates=candidates,
                best_node_id=best_node_id,
                max_to_select=max_to_select,
                vote_index=vote_index + 1,
                vote_count=vote_count,
                panel_member=member,
            )
            votes.append(vote)
            messages.append(message)

        result = build_selector_vote_result(
            candidates=candidates,
            best_node_id=best_node_id,
            max_to_select=max_to_select,
            votes=votes,
            ensemble_mode=ensemble_mode,
            panel_members=panel_members or [member],
            claim_boundary=claim_boundary,
        )
        self.storage.save_json("reports/selector_votes.json", result.to_dict())
        self._save_messages(None, messages)
        return result

    def cast_vote(
        self,
        *,
        candidates: list[dict[str, object]],
        best_node_id: str,
        max_to_select: int,
        vote_index: int,
        vote_count: int,
        panel_member: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], AgentMessage]:
        member = _normalized_panel_member(panel_member or _default_panel_member(self.llm))
        prompt = (
            "Select exploration parents for ensemble-guided mutation. "
            "The lowest-loss best solution will be included separately for exploitation. "
            "Vote for candidates with high improvement potential, fixable failures, "
            "or underexplored ideas. Return JSON with selected_parent_ids and rationale. "
            f"Selector panel member: {member}. "
            f"Best by loss: {best_node_id}. Max total parents after best inclusion: {max_to_select}. "
            f"Vote index: {vote_index}/{vote_count}. Candidates: {candidates}"
        )
        response = self.complete_json_checked(
            prompt,
            "selector",
            required_fields=("selected_parent_ids", "rationale"),
        )
        selected = _unique_valid_selected(
            response.get("selected_parent_ids", []),
            candidates=candidates,
            best_node_id=best_node_id,
        )
        vote = {
            "vote_index": vote_index,
            "member_id": member["member_id"],
            "member_role": member["role"],
            "configured_model": member["configured_model"],
            "actual_model": member["actual_model"],
            "provider": member["provider"],
            "source": member["source"],
            "selected_parent_ids": selected,
            "rationale": str(response.get("rationale", "")),
        }
        return vote, AgentMessage(self.role, prompt, str(response), {"vote_index": vote_index})


def build_selector_vote_result(
    *,
    candidates: list[dict[str, object]],
    best_node_id: str,
    max_to_select: int,
    votes: list[dict[str, object]],
    ensemble_mode: str,
    panel_members: list[dict[str, object]],
    claim_boundary: str,
) -> SelectorVoteResult:
    sanitized_votes: list[dict[str, object]] = []
    vote_counts: dict[str, int] = {}
    for vote in votes:
        sanitized_vote = dict(vote)
        selected = _unique_valid_selected(
            vote.get("selected_parent_ids", []),
            candidates=candidates,
            best_node_id=best_node_id,
        )
        sanitized_vote["selected_parent_ids"] = selected
        sanitized_votes.append(sanitized_vote)
        for node_id in selected:
            vote_counts[node_id] = vote_counts.get(node_id, 0) + 1
    panel_members = [_normalized_panel_member(member) for member in panel_members]
    selected_parent_ids = [best_node_id]
    selected_parent_ids.extend(
        _rank_vote_winners(vote_counts, candidates, max(0, max_to_select - 1))
    )
    selected_parent_ids = selected_parent_ids[:max_to_select]
    return SelectorVoteResult(
        selected_parent_ids=selected_parent_ids,
        votes=sanitized_votes,
        vote_counts=dict(sorted(vote_counts.items())),
        ensemble_mode=ensemble_mode,
        panel_members=panel_members,
        claim_boundary=claim_boundary,
        diversity=_selector_diversity(
            ensemble_mode=ensemble_mode,
            panel_members=panel_members,
            votes=sanitized_votes,
        ),
    )


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


def _unique_valid_selected(
    selected_parent_ids: object,
    *,
    candidates: list[dict[str, object]],
    best_node_id: str,
) -> list[str]:
    if not isinstance(selected_parent_ids, list):
        return []
    candidate_ids = {
        str(candidate.get("node_id"))
        for candidate in candidates
        if candidate.get("node_id") is not None
    }
    selected: list[str] = []
    seen: set[str] = set()
    for item in selected_parent_ids:
        node_id = str(item)
        if node_id == best_node_id or node_id not in candidate_ids or node_id in seen:
            continue
        selected.append(node_id)
        seen.add(node_id)
    return selected


def _selector_diversity(
    *,
    ensemble_mode: str,
    panel_members: list[dict[str, object]],
    votes: list[dict[str, object]],
) -> dict[str, object]:
    actual_models = sorted(
        {
            str(member.get("actual_model"))
            for member in panel_members
            if member.get("actual_model")
        }
    )
    providers = sorted(
        {
            str(member.get("provider"))
            for member in panel_members
            if member.get("provider")
        }
    )
    member_vote_counts: dict[str, int] = {}
    for vote in votes:
        member_id = str(vote.get("member_id", "unknown"))
        member_vote_counts[member_id] = member_vote_counts.get(member_id, 0) + 1
    panel_member_count = len(panel_members)
    actual_vote_count = len(votes)
    mock_evidence = any(
        str(member.get("actual_model")) == "mock" or str(member.get("provider")) == "MockLLMClient"
        for member in panel_members
    )
    actual_model_diversity = len(actual_models) > 1
    provider_diversity = len(providers) > 1
    panel_repeated_members = actual_vote_count > panel_member_count or any(
        count > 1 for count in member_vote_counts.values()
    )
    heterogeneous_selector_evidence = (
        ensemble_mode == "configured_selector_panel"
        and panel_member_count > 1
        and actual_vote_count >= panel_member_count
        and not mock_evidence
        and not panel_repeated_members
        and (actual_model_diversity or provider_diversity)
    )
    return {
        "mock_evidence": mock_evidence,
        "actual_model_diversity": actual_model_diversity,
        "provider_diversity": provider_diversity,
        "unique_actual_models": actual_models,
        "unique_providers": providers,
        "panel_member_count": panel_member_count,
        "actual_vote_count": actual_vote_count,
        "member_vote_counts": dict(sorted(member_vote_counts.items())),
        "panel_repeated_members": panel_repeated_members,
        "heterogeneous_selector_evidence": heterogeneous_selector_evidence,
    }


def _default_panel_member(llm: object) -> dict[str, object]:
    actual_model = getattr(llm, "model", None)
    if not actual_model and llm.__class__.__name__ == "MockLLMClient":
        actual_model = "mock"
    return {
        "member_id": "selector",
        "role": "selector",
        "configured_model": getattr(llm, "model", "default"),
        "actual_model": actual_model or llm.__class__.__name__,
        "provider": llm.__class__.__name__,
        "source": "single_selector",
    }


def _normalized_panel_member(member: dict[str, object]) -> dict[str, object]:
    return {
        "member_id": str(member.get("member_id", "selector")),
        "role": str(member.get("role", "selector")),
        "configured_model": str(member.get("configured_model", "default")),
        "actual_model": str(member.get("actual_model", member.get("configured_model", "default"))),
        "provider": str(member.get("provider", "unknown")),
        "source": str(member.get("source", "unknown")),
    }
