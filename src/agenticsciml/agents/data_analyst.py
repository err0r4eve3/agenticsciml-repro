from __future__ import annotations

from pathlib import Path

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage


class DataAnalystAgent(AgentBase):
    role = "data_analyst"

    def analyze(self, benchmark_dir: Path) -> str:
        self.require_inputs({"benchmark_dir": benchmark_dir})
        prompt = (
            "Act as the data analyst for an AgenticSciML run. "
            f"Summarize likely data properties for benchmark at {benchmark_dir}. "
            "Return a concise text-only report."
        )
        response = self.complete_text(prompt)
        self.storage.save_text("reports/data_analysis.md", f"# Data Analysis\n\n{response}\n")
        self._save_messages(None, [AgentMessage(self.role, prompt, response)])
        return response
