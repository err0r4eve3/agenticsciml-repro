from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticsciml.agents.base import AgentBase
from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.config import EvaluationContract
from agenticsciml.patching import PatchApplicationError, apply_unified_patch, solution_digest
from agenticsciml.state import AgentMessage


class DebuggerAgent(AgentBase):
    role = "debugger"

    def debug(
        self,
        solution_id: str,
        workspace: Path,
        error_log: str,
        problem_bundle: ProblemBundle,
        contract: EvaluationContract,
        guidelines: str,
        failure_phase: str,
    ) -> bool:
        current_code = (workspace / "solution.py").read_text(encoding="utf-8")
        current_digest = solution_digest(current_code)
        self.require_inputs(
            {
                "solution_id": solution_id,
                "workspace": workspace,
                "error_log": error_log,
                "current_code": current_code,
                "current_digest": current_digest,
                "problem_bundle": problem_bundle,
                "contract": contract,
                "guidelines": guidelines,
                "failure_phase": failure_phase,
            }
        )
        prompt = (
            "Patch the generated solution while preserving the evaluation contract. "
            "Return JSON with summary, failure_kind, minimal_fix, parent_digest, "
            "patch, files_changed, risks, and optional full_file_map. `risks` and "
            "`files_changed` must be JSON arrays of strings. Set `files_changed` to "
            "exactly [\"solution.py\"]. Use exactly one code-change channel: either a "
            "unified diff `patch`, or an empty `patch` plus `full_file_map` containing "
            "the complete `solution.py`. Prefer a patch for small edits, but use "
            "`full_file_map.solution.py` when exact current-code context may be "
            "unreliable.\n\n"
            "## ProblemBundle Summary\n\n"
            f"{problem_bundle.summary()}\n\n"
            "## EvaluationContract JSON\n\n"
            f"{json.dumps(contract.to_dict(), indent=2, sort_keys=True)}\n\n"
            "## guidelines.md\n\n"
            f"{guidelines[:2500]}\n\n"
            "## Failure Phase\n\n"
            f"{failure_phase}\n\n"
            "## Forbidden Actions\n\n"
            "- Do not read validation labels or validation file paths.\n"
            "- Predict mode may read only `predict_input.npz` and must write `predictions.npz`.\n"
            "- Do not modify evaluator files.\n"
            "- Do not use network, subprocess, absolute paths, or home-directory helpers.\n\n"
            f"parent_digest: {current_digest}\n\n"
            f"## Error log\n\n{error_log[-4000:]}\n\n"
            f"## Current solution.py\n\n{current_code[:8000]}"
        )
        response = self.complete_json_checked(
            prompt,
            "debugger",
            required_fields=(
                "summary",
                "failure_kind",
                "minimal_fix",
                "parent_digest",
                "patch",
                "files_changed",
                "risks",
            ),
        )
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, str(response))])
        changed = self.apply_debug_output(solution_id, current_code, response)
        return changed

    def apply_debug_output(
        self,
        solution_id: str,
        current_code: str,
        response: dict[str, Any],
    ) -> bool:
        expected_digest = solution_digest(current_code)
        actual_digest = str(response.get("parent_digest", ""))
        if actual_digest != expected_digest:
            raise PatchApplicationError(
                f"Debugger parent_digest mismatch: expected {expected_digest}, got {actual_digest}"
            )
        files_changed = [str(item) for item in response.get("files_changed", [])]
        if files_changed != ["solution.py"]:
            raise PatchApplicationError("Debugger may only change solution.py.")

        patch = str(response.get("patch", ""))
        if not patch.strip():
            full_file_map = response.get("full_file_map")
            if not isinstance(full_file_map, dict) or "solution.py" not in full_file_map:
                return False
            code = str(full_file_map["solution.py"])
            self.storage.save_solution_text(
                solution_id,
                "solution.py",
                code if code.endswith("\n") else code + "\n",
            )
            return True

        try:
            code = apply_unified_patch(current_code, patch)
        except PatchApplicationError:
            full_file_map = response.get("full_file_map")
            if not isinstance(full_file_map, dict) or "solution.py" not in full_file_map:
                raise
            code = str(full_file_map["solution.py"])
        self.storage.save_solution_text(solution_id, "solution.py", code if code.endswith("\n") else code + "\n")
        return True
