from __future__ import annotations

from agenticsciml.agents.base import AgentBase
from agenticsciml.config import EvaluationContract
from agenticsciml.state import AgentMessage


class EvaluatorAgent(AgentBase):
    role = "evaluator"

    def create_contract(self, data_report: str | None = None) -> EvaluationContract:
        self.require_inputs({"data_report": data_report})
        prompt = (
            "Create the evaluation contract for function approximation. "
            "Use validation MSE and keep the same evaluator for every solution."
        )
        if data_report:
            prompt += f"\n\nData report:\n{data_report}"
        response = self.complete_json_checked(
            prompt,
            "evaluator",
            required_fields=("metric_name", "higher_is_better", "checkpoint_path"),
        )
        contract = EvaluationContract.default_function_approx()
        self.storage.save_json("evaluation_contract.json", contract.to_dict())
        self.storage.save_text(
            "reports/evaluation_contract.md",
            "# Evaluation Contract\n\n"
            f"- metric: {contract.metric_name}\n"
            f"- higher_is_better: {contract.higher_is_better}\n"
            f"- checkpoint: {contract.checkpoint_path}\n",
        )
        self._save_messages(None, [AgentMessage(self.role, prompt, str(response))])
        return contract
