from __future__ import annotations

import json
from pathlib import Path

from agenticsciml.agents.base import AgentBase
from agenticsciml.state import AgentMessage, AnalysisReport


class ResultAnalystAgent(AgentBase):
    role = "result_analyst"

    def analyze(self, solution_id: str, workspace: Path) -> AnalysisReport:
        self.require_inputs({"solution_id": solution_id, "workspace": workspace})
        eval_payload = {}
        eval_path = workspace / "eval.json"
        if eval_path.exists():
            eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
        log = (workspace / "train.log").read_text(encoding="utf-8") if (workspace / "train.log").exists() else ""
        prompt = (
            "Analyze this solution result using concise report fields. "
            "Return JSON with summary, strengths, weaknesses, next_steps.\n\n"
            f"Eval: {eval_payload}\n\nLog excerpt:\n{log[-2000:]}"
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
