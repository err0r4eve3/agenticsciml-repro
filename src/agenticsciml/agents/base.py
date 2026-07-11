from __future__ import annotations

import json
import time
from inspect import Parameter, signature
from typing import Any

from agenticsciml.llm.base import LLMClient
from agenticsciml.llm.budget import LLMBudgetExceeded
from agenticsciml.state import AgentMessage
from agenticsciml.storage import ExperimentStorage
from agenticsciml.agents.specs import AGENT_SPECS, AgentSpec
from agenticsciml.agents.output_schemas import validate_output_payload


class ArtifactMissingError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


class InputContractError(RuntimeError):
    pass


class AgentBase:
    role = "agent"

    def __init__(
        self,
        llm: LLMClient,
        storage: ExperimentStorage,
        default_temperature: float = 0.0,
        default_reasoning_effort: str | None = None,
    ):
        self.llm = llm
        self.storage = storage
        self.default_temperature = default_temperature
        self.default_reasoning_effort = default_reasoning_effort
        self.spec: AgentSpec | None = AGENT_SPECS.get(self.role)

    def _spec_metadata(self) -> dict[str, Any]:
        if self.spec is None:
            return {"spec_role": self.role, "state_node": None}
        return {
            "spec_role": self.spec.role,
            "state_node": self.spec.state_node,
            "tools": list(self.spec.tools),
            "budget": self.spec.budget,
            "output_model": self.spec.output_model,
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
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        call_temperature = self.default_temperature if temperature is None else temperature
        call_reasoning_effort = self._call_reasoning_effort(reasoning_effort)
        started = time.monotonic()
        response = self._llm_complete_text(
            prompt,
            system=system,
            temperature=call_temperature,
            reasoning_effort=call_reasoning_effort,
        )
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
                "temperature": call_temperature,
                **self._reasoning_trace_metadata(call_reasoning_effort),
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
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        retries: int = 1,
    ) -> dict[str, Any]:
        call_temperature = self.default_temperature if temperature is None else temperature
        call_reasoning_effort = self._call_reasoning_effort(reasoning_effort)
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
                data = self._llm_complete_json(
                    current_prompt,
                    schema_name,
                    system=system,
                    temperature=call_temperature,
                    reasoning_effort=call_reasoning_effort,
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
                        "temperature": call_temperature,
                        **self._reasoning_trace_metadata(call_reasoning_effort),
                        **self._llm_call_metadata(),
                    },
                )
                if isinstance(exc, LLMBudgetExceeded):
                    raise
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
                if is_non_retryable_llm_api_error(exc):
                    raise StructuredOutputError(last_error) from exc
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
                    "temperature": call_temperature,
                    **self._reasoning_trace_metadata(call_reasoning_effort),
                    **self._llm_call_metadata(),
                },
            )
            try:
                data = validate_output_payload(
                    data,
                    schema_name=schema_name,
                    output_model=self.spec.output_model if self.spec is not None else None,
                )
            except ValueError as exc:
                last_error = f"{schema_name} output failed typed schema validation: {exc}"
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
        allowed = {
            "adapter_type",
            "llm_call_id",
            "method",
            "model",
            "max_retries",
            "provider",
            "provider_capabilities",
            "reasoning_effort",
            "schema_name",
            "span_kind",
            "timeout_s",
            "usage",
        }
        return {key: value for key, value in metadata.items() if key in allowed}

    def _call_reasoning_effort(self, override: str | None) -> str | None:
        return self.default_reasoning_effort if override is None else override

    def _reasoning_trace_metadata(self, reasoning_effort: str | None) -> dict[str, str]:
        return {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}

    def _llm_complete_text(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        reasoning_effort: str | None,
    ) -> str:
        kwargs: dict[str, Any] = {"system": system, "temperature": temperature}
        if reasoning_effort is not None and _accepts_reasoning_effort(self.llm.complete_text):
            kwargs["reasoning_effort"] = reasoning_effort
        return self.llm.complete_text(prompt, **kwargs)

    def _llm_complete_json(
        self,
        prompt: str,
        schema_name: str,
        *,
        system: str | None,
        temperature: float,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"system": system, "temperature": temperature}
        if reasoning_effort is not None and _accepts_reasoning_effort(self.llm.complete_json):
            kwargs["reasoning_effort"] = reasoning_effort
        return self.llm.complete_json(prompt, schema_name, **kwargs)


def _accepts_reasoning_effort(method: Any) -> bool:
    try:
        method_signature = signature(method)
    except (TypeError, ValueError):
        return True
    parameters = method_signature.parameters
    if "reasoning_effort" in parameters:
        return True
    return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


def is_non_retryable_llm_api_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if "Timeout" in name:
        return True
    return name in {
        "APIConnectionError",
        "APIStatusError",
        "AuthenticationError",
        "BadRequestError",
        "PermissionDeniedError",
        "RateLimitError",
    }
