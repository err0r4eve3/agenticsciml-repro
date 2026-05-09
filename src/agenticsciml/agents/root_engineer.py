from __future__ import annotations

import json

from agenticsciml.agents.base import AgentBase
from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.config import EvaluationContract
from agenticsciml.state import AgentMessage


class RootEngineerAgent(AgentBase):
    role = "root_engineer"

    def generate(
        self,
        solution_id: str,
        problem_bundle: ProblemBundle,
        contract: EvaluationContract,
        guidelines: str,
        data_report: str | None = None,
    ) -> str:
        self.require_inputs(
            {
                "solution_id": solution_id,
                "problem_bundle": problem_bundle,
                "contract": contract,
                "guidelines": guidelines,
                "data_report": data_report,
            }
        )
        prompt = (
            "Generate the root single-agent baseline solution.py. "
            "Do not use KB or multi-agent debate. Return JSON with proposal and code.\n\n"
            "## ProblemBundle Summary\n\n"
            f"{problem_bundle.summary()}\n\n"
            "## Problem.md\n\n"
            f"{problem_bundle.problem_md[:2500]}\n\n"
            "## Requirements.md\n\n"
            f"{problem_bundle.requirements_md[:2500]}\n\n"
            "## Evaluation.md\n\n"
            f"{problem_bundle.evaluation_md[:2500]}\n\n"
            "## EvaluationContract JSON\n\n"
            f"{json.dumps(contract.to_dict(), indent=2, sort_keys=True)}\n\n"
            "## guidelines.md\n\n"
            f"{guidelines[:2500]}\n\n"
            "## data_analysis.md\n\n"
            f"{data_report or 'No data analysis report available.'}\n\n"
            "Forbidden actions: do not read validation data, do not modify evaluator files, "
            "do not use network or subprocess calls.\n"
            "Contract reminder: solution.py must define class MODEL and support "
            "--mode=validate / --mode=train."
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
