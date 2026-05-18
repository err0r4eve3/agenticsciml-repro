from __future__ import annotations

from pathlib import Path

from agenticsciml.agents.base import AgentBase
from agenticsciml.observations import (
    build_data_eda_package,
    build_data_observation_package,
    prompt_eda_summary,
    prompt_observation_summary,
)
from agenticsciml.state import AgentMessage


class DataAnalystAgent(AgentBase):
    role = "data_analyst"

    def analyze(self, benchmark_dir: Path) -> str:
        self.require_inputs({"benchmark_dir": benchmark_dir})
        manifest, svg = build_data_observation_package(benchmark_dir)
        manifest["multimodal_evidence"] = {
            "plot_artifact_generated": bool(manifest.get("plots")),
            "plot_artifact_paths": [
                str(plot.get("path"))
                for plot in manifest.get("plots", [])
                if isinstance(plot, dict) and plot.get("path")
            ],
            "actual_image_inputs_used": False,
            "analysis_mode": "text_artifact_summary_only",
        }
        eda_output, eda_script = build_data_eda_package(manifest)
        self.storage.save_json("reports/data_observations.json", manifest)
        self.storage.save_text("reports/data_overview.svg", svg)
        self.storage.save_text("reports/data_eda.py", eda_script)
        self.storage.save_json("reports/data_eda.json", eda_output)
        prompt = (
            "Act as the data analyst for an AgenticSciML run. "
            f"Summarize data properties for benchmark at {benchmark_dir}. "
            "Use only training-data observations and do not infer from private validation labels. "
            "Return a concise text-only report.\n\n"
            "Observation manifest:\n"
            f"{prompt_observation_summary(manifest)}\n\n"
            "Replayable EDA summary:\n"
            f"{prompt_eda_summary(eda_output)}"
        )
        response = self.complete_text(prompt)
        self.storage.save_text("reports/data_analysis.md", f"# Data Analysis\n\n{response}\n")
        self._save_messages(None, [AgentMessage(self.role, prompt, response)])
        return response
