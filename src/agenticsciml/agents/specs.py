from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentSpec:
    role: str
    state_node: str
    purpose: str
    non_role: tuple[str, ...]
    input_schema: tuple[str, ...]
    output_schema: tuple[str, ...]
    visible_context: tuple[str, ...]
    tools: tuple[str, ...]
    budget: dict[str, Any]
    artifacts: tuple[str, ...]
    failure_policy: str
    output_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "state_node": self.state_node,
            "purpose": self.purpose,
            "non_role": list(self.non_role),
            "input_schema": list(self.input_schema),
            "output_schema": list(self.output_schema),
            "visible_context": list(self.visible_context),
            "tools": list(self.tools),
            "budget": self.budget,
            "artifacts": list(self.artifacts),
            "failure_policy": self.failure_policy,
            "output_model": self.output_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSpec":
        return cls(
            role=str(data["role"]),
            state_node=str(data["state_node"]),
            purpose=str(data["purpose"]),
            non_role=tuple(str(item) for item in data.get("non_role", [])),
            input_schema=tuple(str(item) for item in data.get("input_schema", [])),
            output_schema=tuple(str(item) for item in data.get("output_schema", [])),
            visible_context=tuple(str(item) for item in data.get("visible_context", [])),
            tools=tuple(str(item) for item in data.get("tools", [])),
            budget=dict(data.get("budget", {})),
            artifacts=tuple(str(item) for item in data.get("artifacts", [])),
            failure_policy=str(data.get("failure_policy", "")),
            output_model=str(data["output_model"]) if data.get("output_model") is not None else None,
        )


class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def render(self, values: dict[str, Any]) -> str:
        required = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.template)
            if field_name
        }
        missing = sorted(required - set(values))
        if missing:
            raise KeyError(f"Missing prompt values: {', '.join(missing)}")
        return self.template.format(**values)


AGENT_SPECS: dict[str, AgentSpec] = {
    "data_analyst": AgentSpec(
        role="data_analyst",
        state_node="data analysis",
        purpose="Summarize data shape, noise, discontinuities, and modeling risks.",
        non_role=("write solution code", "change evaluator"),
        input_schema=("benchmark_dir",),
        output_schema=("data_report",),
        output_model="data_analyst",
        visible_context=("problem bundle", "data config", "dataset metadata"),
        tools=("llm", "filesystem read"),
        budget={"calls": 1},
        artifacts=("reports/data_analysis.md",),
        failure_policy="fail the run before solution generation",
    ),
    "evaluator": AgentSpec(
        role="evaluator",
        state_node="evaluation contract",
        purpose="Create a fixed metric and validation contract shared by every solution.",
        non_role=("optimize model", "adjust scoring for a candidate solution"),
        input_schema=("problem_bundle", "data_report"),
        output_schema=("metric_name", "higher_is_better", "checkpoint_path"),
        output_model="evaluator",
        visible_context=("problem bundle", "data analysis"),
        tools=("llm", "filesystem write"),
        budget={"calls": 1},
        artifacts=("evaluation_contract.json", "reports/evaluation_contract.md"),
        failure_policy="fail closed if the contract cannot be produced",
    ),
    "root_engineer": AgentSpec(
        role="root_engineer",
        state_node="root solution",
        purpose="Generate the first single-agent baseline solution.",
        non_role=("use KB", "compare multiple parents", "modify evaluator"),
        input_schema=("solution_id", "problem_bundle", "contract", "guidelines", "data_report"),
        output_schema=("code", "proposal"),
        output_model="root_engineer",
        visible_context=("problem bundle", "evaluation contract"),
        tools=("llm", "filesystem write"),
        budget={"calls": 1},
        artifacts=("solution.py", "proposal.md"),
        failure_policy="send runtime failures to debugger",
    ),
    "selector": AgentSpec(
        role="selector",
        state_node="parent selection",
        purpose="Vote on exploration parents while the lowest-loss best node is kept for exploitation.",
        non_role=("generate proposal", "write code", "change scores"),
        input_schema=("candidates", "best_node_id", "max_to_select"),
        output_schema=("selected_parent_ids", "rationale"),
        output_model="selector",
        visible_context=("leaderboard summary", "solution metadata"),
        tools=("llm",),
        budget={"calls": "selector_vote_count"},
        artifacts=("transcripts/selector.json", "reports/selector_votes.json"),
        failure_policy="fall back to current best parent",
    ),
    "retriever": AgentSpec(
        role="retriever",
        state_node="knowledge retrieval",
        purpose="Select zero or one KB entry relevant to the parent weakness.",
        non_role=("return many KB entries", "rewrite KB", "generate proposal"),
        input_schema=("solution_id", "query", "enabled"),
        output_schema=("entry_id",),
        output_model="retriever",
        visible_context=("KB descriptions", "parent analysis"),
        tools=("lexical retrieval", "filesystem read"),
        budget={"calls": 0},
        artifacts=("retrieved_kb.md",),
        failure_policy="continue with no KB entry",
    ),
    "proposer": AgentSpec(
        role="proposer",
        state_node="mutation proposal",
        purpose="Produce an implementation-ready mutation proposal.",
        non_role=("write code", "modify evaluator", "claim unverified score gains"),
        input_schema=("solution_id", "parent_summary", "kb_entry", "related_reports", "branch_context"),
        output_schema=("title", "diagnosis", "mutation_plan", "expected_effect", "risks"),
        output_model="proposal",
        visible_context=("parent analysis", "one KB entry", "local related reports"),
        tools=("llm", "critic feedback"),
        budget={"rounds": 4, "calls": 4},
        artifacts=("proposal.md", "critic.md"),
        failure_policy="retry JSON once in future; currently fail closed",
    ),
    "critic": AgentSpec(
        role="critic",
        state_node="proposal critique",
        purpose="Challenge proposal gaps, feasibility, and risks.",
        non_role=("write code", "change score", "override deterministic evaluator"),
        input_schema=("solution_id", "proposal_summary", "context", "round_index"),
        output_schema=("critique",),
        visible_context=("current proposal", "parent context"),
        tools=("llm",),
        budget={"calls_per_round": 1},
        artifacts=("critic.md",),
        failure_policy="record missing critique and continue proposer loop",
    ),
    "engineer": AgentSpec(
        role="engineer",
        state_node="code mutation",
        purpose="Mutate parent solution code according to the accepted proposal.",
        non_role=("modify evaluator", "change benchmark data", "skip validation"),
        input_schema=(
            "solution_id",
            "parent_code",
            "proposal",
            "problem_bundle",
            "contract",
            "guidelines",
            "parent_analysis",
            "branch_context",
            "parent_digest",
        ),
        output_schema=(
            "mutation_summary",
            "expected_effect",
            "risks",
            "parent_digest",
            "patch",
            "files_changed",
            "full_file_map",
        ),
        output_model="engineer",
        visible_context=(
            "problem bundle",
            "parent code",
            "parent analysis",
            "proposal",
            "evaluation contract",
            "branch context",
        ),
        tools=("llm", "filesystem write"),
        budget={"calls": 1},
        artifacts=("solution.py", "engineering_summary.md"),
        failure_policy="send runtime failures to debugger",
    ),
    "debugger": AgentSpec(
        role="debugger",
        state_node="runtime repair",
        purpose="Patch execution failures without weakening the evaluation contract.",
        non_role=("optimize score", "change evaluator", "expand feature scope"),
        input_schema=(
            "solution_id",
            "workspace",
            "error_log",
            "current_code",
            "current_digest",
            "problem_bundle",
            "contract",
            "guidelines",
            "failure_phase",
        ),
        output_schema=(
            "summary",
            "failure_kind",
            "minimal_fix",
            "parent_digest",
            "patch",
            "files_changed",
            "risks",
            "full_file_map",
        ),
        output_model="debugger",
        visible_context=("error log", "solution code", "evaluation contract"),
        tools=("llm", "filesystem write"),
        budget={"calls": "max_debug_retries"},
        artifacts=("solution.py", "transcripts/debugger.json"),
        failure_policy="stop after debug budget is exhausted",
    ),
    "result_analyst": AgentSpec(
        role="result_analyst",
        state_node="result analysis",
        purpose="Summarize score, logs, strengths, weaknesses, and next steps.",
        non_role=("change score", "edit code", "select champion"),
        input_schema=("solution_id", "workspace"),
        output_schema=("summary", "strengths", "weaknesses", "next_steps"),
        output_model="result_analyst",
        visible_context=("eval.json", "train.log"),
        tools=("llm", "filesystem write"),
        budget={"calls": 1},
        artifacts=("analysis.md",),
        failure_policy="write a minimal failure analysis if LLM output is invalid",
    ),
}
