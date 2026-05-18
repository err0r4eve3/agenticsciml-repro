from __future__ import annotations

import json
from pathlib import Path

from agenticsciml.agents.base import AgentBase
from agenticsciml.observations import build_solution_observation_package, prompt_observation_summary
from agenticsciml.state import AgentMessage, AnalysisReport


class ResultAnalystAgent(AgentBase):
    role = "result_analyst"

    def analyze(self, solution_id: str, workspace: Path) -> AnalysisReport:
        self.require_inputs({"solution_id": solution_id, "workspace": workspace})
        manifest, svg = build_solution_observation_package(
            solution_id,
            workspace,
            run_dir=self.storage.run_dir,
        )
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
        self.storage.save_json(Path("solutions") / solution_id / "solution_observations.json", manifest)
        self.storage.save_solution_text(solution_id, "prediction_overview.svg", svg)
        eval_payload = {}
        eval_path = workspace / "eval.json"
        if eval_path.exists():
            eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
        log = (workspace / "train.log").read_text(encoding="utf-8") if (workspace / "train.log").exists() else ""
        prompt = (
            "Analyze this solution result using concise report fields. "
            "Use prediction-only observations and do not infer from private validation labels. "
            "Return JSON with summary, strengths, weaknesses, next_steps. "
            "`strengths`, `weaknesses`, and `next_steps` must be JSON arrays of strings, not strings.\n\n"
            f"Eval: {eval_payload}\n\n"
            "Observation manifest:\n"
            f"{prompt_observation_summary(manifest)}\n\n"
            f"Log excerpt:\n{log[-2000:]}"
        )
        response = self.complete_json_checked(
            prompt,
            "analysis",
            required_fields=("summary", "strengths", "weaknesses", "next_steps"),
        )
        report = AnalysisReport.from_dict({"node_id": solution_id, **response})
        self.storage.save_solution_text(solution_id, "analysis.md", report.to_markdown())
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, str(response))])
        return report
