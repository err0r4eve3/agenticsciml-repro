from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticsciml.agents.base import AgentBase
from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.config import EvaluationContract
from agenticsciml.operator_scheduler import operator_assignment_context
from agenticsciml.patching import PatchApplicationError, apply_unified_patch, solution_digest
from agenticsciml.state import AgentMessage, AnalysisReport, Proposal


class EngineerAgent(AgentBase):
    role = "engineer"

    def mutate(
        self,
        solution_id: str,
        parent_code: str,
        proposal: Proposal,
        problem_bundle: ProblemBundle,
        contract: EvaluationContract,
        guidelines: str,
        parent_analysis: AnalysisReport | None,
        branch_context: dict[str, Any] | None = None,
        problem_intake_context: str | None = None,
        strategy_seed_context: str | None = None,
        operator_assignment: dict[str, Any] | None = None,
    ) -> str:
        parent_digest = solution_digest(parent_code)
        self.require_inputs(
            {
                "solution_id": solution_id,
                "parent_code": parent_code,
                "proposal": proposal,
                "problem_bundle": problem_bundle,
                "contract": contract,
                "guidelines": guidelines,
                "parent_analysis": parent_analysis,
                "branch_context": branch_context,
                "problem_intake_context": problem_intake_context,
                "strategy_seed_context": strategy_seed_context,
                "operator_assignment": operator_assignment,
                "parent_digest": parent_digest,
            }
        )
        prompt = (
            "Modify parent solution.py according to the proposal. "
            "Return JSON with mutation_summary, expected_effect, risks, parent_digest, "
            "patch, files_changed, full_file_map, and optional implemented_kb_points. `risks` must be a JSON array of "
            "strings. Set `files_changed` to exactly [\"solution.py\"]. Always include "
            "`full_file_map` with exactly one key, `solution.py`, containing the complete "
            "mutated file. You may also include a unified diff `patch`, but the complete "
            "file is mandatory because patch context from LLMs can drift.\n\n"
            "## ProblemBundle Summary\n\n"
            f"{problem_bundle.summary()}\n\n"
            "## EvaluationContract JSON\n\n"
            f"{json.dumps(contract.to_dict(), indent=2, sort_keys=True)}\n\n"
            "## guidelines.md\n\n"
            f"{guidelines[:2500]}\n\n"
            "## Parent Analysis\n\n"
            f"{parent_analysis.summary if parent_analysis else 'No parent analysis available.'}\n\n"
            "## Branch Context\n\n"
            f"{json.dumps(branch_context or {}, indent=2, sort_keys=True)}\n\n"
            "## User Problem Intake Context (Non-Contract)\n\n"
            f"{problem_intake_context or 'No user problem-intake context provided.'}\n\n"
            "## Human/Planner Selected Strategy Seeds\n\n"
            f"{strategy_seed_context or 'No strategy seeds selected.'}\n\n"
            "## Assigned Mutation Operator\n\n"
            f"{operator_assignment_context(operator_assignment)}\n\n"
            "## Forbidden Actions\n\n"
            "- Do not read validation data.\n"
            "- Predict mode may read only `predict_input.npz` and must write `predictions.npz` "
            "with a `predictions` array.\n"
            "- Do not modify evaluator files.\n"
            "- Do not use network, subprocess, absolute paths, or home-directory helpers.\n\n"
            "## Contract-Specific Implementation Requirements\n\n"
            "- `solution.py --mode=validate` must run before training and must not require "
            "`model.pkl` or any previous checkpoint.\n"
            "- Use `x_train` and `u_train` directly when they exist in `train_data.npz`, even "
            "if their shapes match.\n"
            "- If training data cannot be parsed, raise an error; do not fabricate synthetic "
            "targets or fall back to synthetic data.\n"
            "- Treat `x_val` as a batch and preserve its leading sample dimension. When "
            "`u_train` exists, predictions should follow its per-sample tail shape. Do not "
            "assign vector outputs into scalar prediction slots.\n\n"
            "## KB Application Checklist\n\n"
            "- If the proposal contains a KB Application section, state which actionable KB points "
            "were implemented in `implemented_kb_points`.\n"
            "- Do not claim implementation of a KB point unless `solution.py` contains an auditable "
            "signal for it, such as explicit sample counts, collocation logic, depth/width settings, "
            "residual weighting, or training schedule values.\n\n"
            f"solution_id: {solution_id}\n"
            f"parent_digest: {parent_digest}\n\n"
            f"Proposal:\n{proposal.to_markdown()}\n\nParent code:\n{parent_code[:8000]}"
        )
        response = self.complete_json_checked(
            prompt,
            "engineer",
            required_fields=(
                "mutation_summary",
                "expected_effect",
                "risks",
                "parent_digest",
                "patch",
                "files_changed",
                "full_file_map",
            ),
        )
        code = self.apply_mutation_output(solution_id, parent_code, response)
        self.storage.save_json(Path("solutions") / solution_id / "engineering_response.json", response)
        implemented_kb_points = response.get("implemented_kb_points")
        implemented_kb_points_md = (
            "\n".join(f"- {point}" for point in implemented_kb_points)
            if isinstance(implemented_kb_points, list) and implemented_kb_points
            else "- None recorded."
        )
        summary = (
            "# Engineering Summary\n\n"
            f"{response.get('mutation_summary', '')}\n\n"
            "## Expected Effect\n\n"
            f"{response.get('expected_effect', '')}\n\n"
            "## Implemented KB Points\n\n"
            f"{implemented_kb_points_md}\n\n"
            "## Risks\n\n"
            + "\n".join(f"- {risk}" for risk in response.get("risks", []))
            + "\n"
        )
        self.storage.save_solution_text(solution_id, "engineering_summary.md", summary)
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, str(response))])
        return code

    def apply_mutation_output(
        self,
        solution_id: str,
        parent_code: str,
        response: dict[str, Any],
    ) -> str:
        expected_digest = solution_digest(parent_code)
        actual_digest = str(response.get("parent_digest", ""))
        if actual_digest != expected_digest:
            raise PatchApplicationError(
                f"Mutation parent_digest mismatch: expected {expected_digest}, got {actual_digest}"
            )
        files_changed = [str(item) for item in response.get("files_changed", [])]
        if files_changed != ["solution.py"]:
            raise PatchApplicationError("Mutation may only change solution.py.")

        patch = str(response.get("patch", ""))
        if patch:
            try:
                code = apply_unified_patch(parent_code, patch)
            except PatchApplicationError:
                full_file_map = response.get("full_file_map")
                if not isinstance(full_file_map, dict) or "solution.py" not in full_file_map:
                    raise
                code = str(full_file_map["solution.py"])
        else:
            full_file_map = response.get("full_file_map")
            if not isinstance(full_file_map, dict) or "solution.py" not in full_file_map:
                raise PatchApplicationError("Mutation response must include patch or full_file_map.solution.py.")
            code = str(full_file_map["solution.py"])

        self.storage.save_solution_text(solution_id, "solution.py", code if code.endswith("\n") else code + "\n")
        return code if code.endswith("\n") else code + "\n"

    def read_parent_code(self, parent_workspace: Path) -> str:
        return (parent_workspace / "solution.py").read_text(encoding="utf-8")
