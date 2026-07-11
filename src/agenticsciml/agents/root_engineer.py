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
        problem_intake_context: str | None = None,
    ) -> str:
        self.require_inputs(
            {
                "solution_id": solution_id,
                "problem_bundle": problem_bundle,
                "contract": contract,
                "guidelines": guidelines,
                "data_report": data_report,
                "problem_intake_context": problem_intake_context,
            }
        )
        compact_contract = json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":"))
        prompt = (
            "Generate the root single-agent baseline solution.py. "
            "Do not use KB, strategy seed catalogs, or multi-agent debate. "
            "Return JSON with proposal and code. Keep the root baseline compact: "
            "prefer NumPy-only regression or interpolation code, avoid optional heavy dependencies, "
            "and keep solution.py under about 160 lines unless the contract absolutely requires more.\n\n"
            "## User Problem Intake Context (Non-Contract)\n\n"
            f"{problem_intake_context or 'No user problem-intake context provided.'}\n\n"
            "## ProblemBundle Summary\n\n"
            f"{problem_bundle.summary()}\n\n"
            "## Problem.md\n\n"
            f"{problem_bundle.problem_md[:1800]}\n\n"
            "## Requirements.md\n\n"
            f"{problem_bundle.requirements_md[:1600]}\n\n"
            "## Evaluation.md\n\n"
            f"{problem_bundle.evaluation_md[:1600]}\n\n"
            "## EvaluationContract JSON\n\n"
            f"{compact_contract}\n\n"
            "## guidelines.md\n\n"
            f"{guidelines[:1800]}\n\n"
            "## data_analysis.md\n\n"
            f"{(data_report or 'No data analysis report available.')[:1800]}\n\n"
            "Root baseline isolation: human/planner-selected strategy seeds are intentionally excluded "
            "from this prompt. They may guide later mutations only after the baseline exists.\n\n"
            "Forbidden actions: do not read validation data, do not modify evaluator files, "
            "do not use network or subprocess calls.\n"
            "Contract reminder: solution.py must define class MODEL and support "
            "--mode=validate / --mode=train / --mode=predict. Predict mode receives "
            "only `--input predict_input.npz` with x_val and must write `--output "
            "predictions.npz` containing a `predictions` array. Lifecycle reminder: "
            "`--mode=validate` runs before training and must not require `model.pkl`, "
            "a previous checkpoint, or any other training side effect. Prediction shape "
            "reminder: treat `x_val` as a batch and preserve its leading sample dimension. "
            "When `u_train` exists, predictions should follow its per-sample tail shape. "
            "Do not assign vector outputs into scalar prediction slots."
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
