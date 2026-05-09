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
        workspace = self.storage.create_solution_workspace(solution_id)
        path = workspace / "critic.md"
        section = f"## Critic Round {round_index}\n\n{response}\n\n"
        if path.exists():
            path.write_text(path.read_text(encoding="utf-8") + section, encoding="utf-8")
        else:
            path.write_text(f"# Critic Review\n\n{section}", encoding="utf-8")
        self._save_messages(
            solution_id,
            [AgentMessage(self.role, prompt, response, {"round": round_index})],
            f"critic_round_{round_index}",
        )
        self.require_artifacts(solution_id, ("critic.md",))
        return response
