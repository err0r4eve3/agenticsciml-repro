from __future__ import annotations

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage


class CriticAgent(AgentBase):
    role = "critic"

    def critique(
        self,
        solution_id: str,
        proposal_summary: str,
        context: str,
        round_index: int,
    ) -> str:
        self.require_inputs(
            {
                "solution_id": solution_id,
                "proposal_summary": proposal_summary,
                "context": context,
                "round_index": round_index,
            }
        )
        prompt = (
            f"Critic round {round_index}: challenge feasibility, gaps, and risks. "
            "Use concise critique, not hidden chain-of-thought.\n\n"
            f"Proposal summary:\n{proposal_summary}\n\n"
            f"Context:\n{context}"
        )
        response = self.complete_text(prompt)
        section = f"## Critic Round {round_index}\n\n{response}\n\n"
        self.storage.append_solution_text(
            solution_id,
            "critic.md",
            section,
            initial_text="# Critic Review\n\n",
        )
        self._save_messages(
            solution_id,
            [AgentMessage(self.role, prompt, response, {"round": round_index})],
            f"critic_round_{round_index}",
        )
        self.require_artifacts(solution_id, ("critic.md",))
        return response
