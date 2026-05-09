from __future__ import annotations

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage


class RootEngineerAgent(AgentBase):
    role = "root_engineer"

    def generate(self, solution_id: str) -> str:
        self.require_inputs({"solution_id": solution_id})
        prompt = (
            "Generate the root single-agent baseline solution.py. "
            "Do not use KB or multi-agent debate. Return JSON with code."
        )
        response = self.complete_json_checked(
            prompt,
            "root_engineer",
            required_fields=("proposal", "code"),
        )
        code = str(response["code"])
        self.storage.save_solution_text(solution_id, "solution.py", code + "\n")
        self.storage.save_solution_text(solution_id, "proposal.md", "# Root Baseline\n\n" + str(response["proposal"]) + "\n")
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, str(response))])
        return code
