from __future__ import annotations

from pathlib import Path

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage


class DebuggerAgent(AgentBase):
    role = "debugger"

    def debug(self, solution_id: str, workspace: Path, error_log: str) -> bool:
        self.require_inputs(
            {"solution_id": solution_id, "workspace": workspace, "error_log": error_log}
        )
        prompt = (
            "Debug the generated solution while preserving the evaluation contract. "
            "Return JSON with optional replacement code.\n\n"
            f"Error log:\n{error_log[-4000:]}"
        )
        response = self.complete_json_checked(
            prompt,
            "debugger",
            required_fields=("summary", "code"),
        )
        code = str(response.get("code", ""))
        if code.strip():
            (workspace / "solution.py").write_text(code + "\n", encoding="utf-8")
            changed = True
        else:
            changed = False
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, str(response))])
        return changed
