from __future__ import annotations

from pathlib import Path

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage, Proposal


class EngineerAgent(AgentBase):
    role = "engineer"

    def mutate(self, solution_id: str, parent_code: str, proposal: Proposal) -> str:
        self.require_inputs(
            {"solution_id": solution_id, "parent_code": parent_code, "proposal": proposal}
        )
        prompt = (
            "Modify the parent solution.py according to the proposal. "
            "Return JSON with summary and code.\n\n"
            f"Proposal:\n{proposal.to_markdown()}\n\nParent code:\n{parent_code[:8000]}"
        )
        response = self.complete_json_checked(
            prompt,
            "engineer",
            required_fields=("summary", "code"),
        )
        code = str(response["code"])
        self.storage.save_solution_text(solution_id, "solution.py", code + "\n")
        self.storage.save_solution_text(solution_id, "engineering_summary.md", "# Engineering Summary\n\n" + str(response["summary"]) + "\n")
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, str(response))])
        return code

    def read_parent_code(self, parent_workspace: Path) -> str:
        return (parent_workspace / "solution.py").read_text(encoding="utf-8")
