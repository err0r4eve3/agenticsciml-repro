from __future__ import annotations

from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.agents.base import AgentBase
from agenticsciml.config import EvaluationContract
from agenticsciml.state import AgentMessage


class EvaluatorAgent(AgentBase):
    role = "evaluator"

    def create_contract(
        self,
        problem_bundle: ProblemBundle,
        data_report: str | None = None,
    ) -> EvaluationContract:
        self.require_inputs({"problem_bundle": problem_bundle, "data_report": data_report})
        prompt = (
            "Review the benchmark evaluation contract. The final executable "
            "contract is produced by BenchmarkContractFactory, not by the LLM. "
            "Return exactly one JSON object with these keys: "
            "`metric_name`, `higher_is_better`, and `checkpoint_path`. "
            "Use the benchmark summary metric and `model.pkl` as checkpoint_path. "
            "Do not include alternative key names, markdown, or explanatory text.\n\n"
            f"{problem_bundle.summary()}"
        )
        if data_report:
            prompt += f"\n\nData report:\n{data_report}"
        response = self.complete_json_checked(
            prompt,
            "evaluator",
            required_fields=("metric_name", "higher_is_better", "checkpoint_path"),
        )
        contract = BenchmarkContractFactory.create_contract(problem_bundle)
        self.storage.save_json("evaluation_contract.json", contract.to_dict())
        self.storage.save_text(
            "reports/evaluation_contract.md",
            BenchmarkContractFactory.create_guidelines(problem_bundle, contract),
        )
        self._save_messages(None, [AgentMessage(self.role, prompt, str(response))])
        return contract
