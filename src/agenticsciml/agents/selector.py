from __future__ import annotations

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage


class SelectorAgent(AgentBase):
    role = "selector"

    def select(
        self,
        candidates: list[dict[str, object]],
        best_node_id: str,
        max_to_select: int,
    ) -> list[str]:
        self.require_inputs(
            {
                "candidates": candidates,
                "best_node_id": best_node_id,
                "max_to_select": max_to_select,
            }
        )
        prompt = (
            "Select exploration parents. The current best will be included for exploitation. "
            f"Best: {best_node_id}. Max additional/total: {max_to_select}. "
            f"Candidates: {candidates}"
        )
        response = self.complete_json_checked(
            prompt,
            "selector",
            required_fields=("selected_parent_ids", "rationale"),
        )
        selected = [str(item) for item in response.get("selected_parent_ids", [])]
        if best_node_id not in selected:
            selected.insert(0, best_node_id)
        selected = selected[:max_to_select]
        self._save_messages(None, [AgentMessage(self.role, prompt, str(response))])
        return selected
