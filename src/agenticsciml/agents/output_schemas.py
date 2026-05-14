from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationReviewOutput(StrictOutputModel):
    metric_name: str
    higher_is_better: bool
    checkpoint_path: str


class RootEngineerOutput(StrictOutputModel):
    proposal: str
    code: str


class SelectorOutput(StrictOutputModel):
    selected_parent_ids: list[str]
    rationale: str


class ProposalOutput(StrictOutputModel):
    title: str
    diagnosis: str
    mutation_plan: list[str]
    expected_effect: str
    risks: list[str]


class EngineerOutput(StrictOutputModel):
    mutation_summary: str
    expected_effect: str
    risks: list[str]
    parent_digest: str
    patch: str = ""
    files_changed: list[str]
    full_file_map: dict[str, str] | None = None


class DebuggerOutput(StrictOutputModel):
    summary: str
    failure_kind: str
    minimal_fix: str
    parent_digest: str
    patch: str
    files_changed: list[str]
    risks: list[str]
    full_file_map: dict[str, str] | None = None


class ResultAnalysisOutput(StrictOutputModel):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    next_steps: list[str]


class DataAnalysisOutput(StrictOutputModel):
    data_report: str


class RetrievalOutput(StrictOutputModel):
    entry_id: str | None = None


OUTPUT_MODELS: dict[str, type[StrictOutputModel]] = {
    "analysis": ResultAnalysisOutput,
    "data_analyst": DataAnalysisOutput,
    "debugger": DebuggerOutput,
    "engineer": EngineerOutput,
    "evaluator": EvaluationReviewOutput,
    "proposal": ProposalOutput,
    "proposer": ProposalOutput,
    "result_analyst": ResultAnalysisOutput,
    "retriever": RetrievalOutput,
    "root_engineer": RootEngineerOutput,
    "selector": SelectorOutput,
}


def output_model_for(schema_name: str, output_model: str | None = None) -> type[StrictOutputModel] | None:
    if output_model:
        return OUTPUT_MODELS.get(output_model)
    return OUTPUT_MODELS.get(schema_name)


def validate_output_payload(
    data: dict[str, Any],
    *,
    schema_name: str,
    output_model: str | None = None,
) -> dict[str, Any]:
    model = output_model_for(schema_name, output_model)
    if model is None:
        return data
    try:
        parsed = model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(exc.errors(include_url=False)) from exc
    return parsed.model_dump(exclude_none=True)


def json_schema_for(schema_name: str, output_model: str | None = None) -> dict[str, Any] | None:
    model = output_model_for(schema_name, output_model)
    if model is None:
        return None
    return model.model_json_schema()
