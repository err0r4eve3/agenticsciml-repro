from __future__ import annotations

from pathlib import Path

from agenticsciml.agents.base import AgentBase
from agenticsciml.observations import build_data_observation_package, prompt_observation_summary
from agenticsciml.state import AgentMessage


class DataAnalystAgent(AgentBase):
    role = "data_analyst"

    def analyze(self, benchmark_dir: Path) -> str:
        self.require_inputs({"benchmark_dir": benchmark_dir})
        manifest, svg = build_data_observation_package(benchmark_dir)
        self.storage.save_json("reports/data_observations.json", manifest)
        self.storage.save_text("reports/data_overview.svg", svg)
        prompt = (
            "Act as the data analyst for an AgenticSciML run. "
            f"Summarize data properties for benchmark at {benchmark_dir}. "
            "Use only training-data observations and do not infer from private validation labels. "
            "Return a concise text-only report.\n\n"
            "Observation manifest:\n"
            f"{prompt_observation_summary(manifest)}"
        )
        response = self.complete_text(prompt)
        self.storage.save_text("reports/data_analysis.md", f"# Data Analysis\n\n{response}\n")
        self._save_messages(None, [AgentMessage(self.role, prompt, response)])
        return response
