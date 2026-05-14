from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticsciml.agents.base import AgentBase
from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.config import EvaluationContract
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
                "parent_digest": parent_digest,
            }
        )
        prompt = (
            "Modify parent solution.py according to the proposal. "
            "Return JSON with mutation_summary, expected_effect, risks, parent_digest, "
            "patch, files_changed, and optional full_file_map. `risks` must be a JSON "
            "array of strings. Set `files_changed` to exactly [\"solution.py\"]. Use exactly "
            "one code-change channel: either a unified diff `patch`, or an empty `patch` "
            "plus `full_file_map` containing the complete `solution.py`. Prefer a patch "
            "for small edits, but use `full_file_map.solution.py` when the change is large "
            "or exact patch context may be unreliable.\n\n"
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
            "## Forbidden Actions\n\n"
            "- Do not read validation data.\n"
            "- Predict mode may read only `predict_input.npz` and must write `predictions.npz` "
            "with a `predictions` array.\n"
            "- Do not modify evaluator files.\n"
            "- Do not use network, subprocess, absolute paths, or home-directory helpers.\n\n"
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
            ),
        )
        code = self.apply_mutation_output(solution_id, parent_code, response)
        summary = (
            "# Engineering Summary\n\n"
            f"{response.get('mutation_summary', '')}\n\n"
            "## Expected Effect\n\n"
            f"{response.get('expected_effect', '')}\n\n"
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
