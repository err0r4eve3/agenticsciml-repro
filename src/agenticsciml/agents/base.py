from __future__ import annotations

import json
import time
from typing import Any

from agenticsciml.llm.base import LLMClient
from agenticsciml.state import AgentMessage
from agenticsciml.storage import ExperimentStorage
from agenticsciml.agents.specs import AGENT_SPECS, AgentSpec


class ArtifactMissingError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


class InputContractError(RuntimeError):
    pass


class AgentBase:
    role = "agent"

    def __init__(self, llm: LLMClient, storage: ExperimentStorage):
        self.llm = llm
        self.storage = storage
        self.spec: AgentSpec | None = AGENT_SPECS.get(self.role)

    def _spec_metadata(self) -> dict[str, Any]:
        if self.spec is None:
            return {"spec_role": self.role, "state_node": None}
        return {
            "spec_role": self.spec.role,
            "state_node": self.spec.state_node,
            "tools": list(self.spec.tools),
            "budget": self.spec.budget,
        }

    def _estimate_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def require_inputs(self, values: dict[str, Any]) -> None:
        if self.spec is None:
            return
        missing = [field for field in self.spec.input_schema if field not in values]
        metadata = {
            **self._spec_metadata(),
            "passed": not missing,
            "required_fields": list(self.spec.input_schema),
            "missing_fields": missing,
        }
        self.storage.record_trace("guardrail_span", f"{self.role}:input_schema", metadata)
        if missing:
            raise InputContractError(
                f"{self.role} missing required input field(s): {', '.join(missing)}"
            )

    def _save_messages(
        self,
        solution_id: str | None,
        messages: list[AgentMessage],
        agent_name: str | None = None,
    ) -> None:
        self.storage.save_transcript(solution_id, agent_name or self.role, messages)
        self.storage.record_trace(
            "agent_span",
            agent_name or self.role,
            {
                **self._spec_metadata(),
                "solution_id": solution_id,
                "message_count": len(messages),
                "roles": [message.role for message in messages],
            },
        )

    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        started = time.monotonic()
        response = self.llm.complete_text(prompt, system=system, temperature=temperature)
        self.storage.record_trace(
            "generation_span",
            self.role,
            {
                "mode": "text",
                **self._spec_metadata(),
                "prompt_chars": len(prompt),
                "response_chars": len(response),
                "prompt_token_estimate": self._estimate_tokens(prompt),
                "response_token_estimate": self._estimate_tokens(response),
                "duration_s": time.monotonic() - started,
                "temperature": temperature,
                **self._llm_call_metadata(),
            },
        )
        return response

    def complete_json_checked(
        self,
        prompt: str,
        schema_name: str,
        required_fields: tuple[str, ...] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        retries: int = 1,
    ) -> dict[str, Any]:
        if required_fields is None:
            if self.spec is None:
                raise StructuredOutputError(
                    f"{self.role} has no AgentSpec; required_fields must be provided."
                )
            required_fields = self.spec.output_schema
        current_prompt = prompt
        last_error = ""
        for attempt in range(retries + 1):
            started = time.monotonic()
            try:
                data = self.llm.complete_json(
                    current_prompt,
                    schema_name,
                    system=system,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = (
                    f"LLM JSON call failed for {schema_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.storage.record_trace(
                    "generation_span",
                    self.role,
                    {
                        "mode": "json",
                        **self._spec_metadata(),
                        "schema_name": schema_name,
                        "attempt": attempt + 1,
                        "prompt_chars": len(current_prompt),
                        "response_chars": 0,
                        "prompt_token_estimate": self._estimate_tokens(current_prompt),
                        "response_token_estimate": 0,
                        "duration_s": time.monotonic() - started,
                        "field_count": 0,
                        "error_type": type(exc).__name__,
                        **self._llm_call_metadata(),
                    },
                )
                self.storage.record_trace(
                    "guardrail_span",
                    f"{self.role}:{schema_name}:structured_output",
                    {
                        **self._spec_metadata(),
                        "passed": False,
                        "attempt": attempt + 1,
                        "required_fields": list(required_fields),
                        "error": last_error,
                    },
                )
                current_prompt = (
                    f"{prompt}\n\nPrevious output failed schema validation: {last_error}. "
                    "Return corrected JSON only."
                )
                continue
            if not isinstance(data, dict):
                last_error = f"{schema_name} output must be a JSON object."
                self.storage.record_trace(
                    "guardrail_span",
                    f"{self.role}:{schema_name}:structured_output",
                    {
                        **self._spec_metadata(),
                        "passed": False,
                        "attempt": attempt + 1,
                        "required_fields": list(required_fields),
                        "error": last_error,
                    },
                )
                current_prompt = (
                    f"{prompt}\n\nPrevious output failed schema validation: {last_error}. "
                    "Return corrected JSON only."
                )
                continue
            response_text = json.dumps(data, sort_keys=True, default=str)
            self.storage.record_trace(
                "generation_span",
                self.role,
                {
                    "mode": "json",
                    **self._spec_metadata(),
                    "schema_name": schema_name,
                    "attempt": attempt + 1,
                    "prompt_chars": len(current_prompt),
                    "response_chars": len(response_text),
                    "prompt_token_estimate": self._estimate_tokens(current_prompt),
                    "response_token_estimate": self._estimate_tokens(response_text),
                    "duration_s": time.monotonic() - started,
                    "field_count": len(data),
                    **self._llm_call_metadata(),
                },
            )
            missing = [field for field in required_fields if field not in data]
            if not missing:
                self.storage.record_trace(
                    "guardrail_span",
                    f"{self.role}:{schema_name}:structured_output",
                    {
                        **self._spec_metadata(),
                        "passed": True,
                        "attempt": attempt + 1,
                        "required_fields": list(required_fields),
                    },
                )
                return data
            last_error = f"Missing required field(s) for {schema_name}: {', '.join(missing)}"
            self.storage.record_trace(
                "guardrail_span",
                f"{self.role}:{schema_name}:structured_output",
                {
                    **self._spec_metadata(),
                    "passed": False,
                    "attempt": attempt + 1,
                    "required_fields": list(required_fields),
                    "missing_fields": missing,
                },
            )
            current_prompt = (
                f"{prompt}\n\nPrevious output failed schema validation: {last_error}. "
                "Return corrected JSON only."
            )
        raise StructuredOutputError(last_error)

    def require_artifacts(self, solution_id: str | None, artifact_paths: tuple[str, ...]) -> None:
        if solution_id is None:
            base = self.storage.run_dir
        else:
            base = self.storage.solution_workspace(solution_id)
        missing = [path for path in artifact_paths if not (base / path).exists()]
        self.storage.record_trace(
            "guardrail_span",
            f"{self.role}:required_artifacts",
            {
                **self._spec_metadata(),
                "solution_id": solution_id,
                "passed": not missing,
                "required_artifacts": list(artifact_paths),
                "missing_artifacts": missing,
            },
        )
        if missing:
            raise ArtifactMissingError(
                f"{self.role} did not create required artifact(s): {', '.join(missing)}"
            )

    def _llm_call_metadata(self) -> dict[str, Any]:
        metadata = getattr(self.llm, "last_call_metadata", None)
        if not isinstance(metadata, dict):
            return {}
        allowed = {"llm_call_id", "span_kind", "provider", "model", "method", "schema_name"}
        return {key: value for key, value in metadata.items() if key in allowed}
