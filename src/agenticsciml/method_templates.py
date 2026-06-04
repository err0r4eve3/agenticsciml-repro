from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agenticsciml.method_substrate import (
    ExpertBlueprint,
    MethodAction,
    MethodPath,
    ScientificReward,
    validate_repo_relative_artifact,
)


METHOD_TEMPLATE_CLAIM_BOUNDARY = (
    "ATHENA/GRAFT method templates are planning and traceability aids. They do "
    "not change runtime selection, prove GRAFT self-improvement, support "
    "paper-score reproduction, or establish autonomous discovery."
)
WORKFLOW_TRACEABILITY_CLAIM_BOUNDARY = "workflow_traceability_only"


@dataclass(frozen=True, slots=True)
class MethodTemplate:
    template_id: str
    action_id: str
    family: str
    default_parameters: dict[str, Any] = field(default_factory=dict)
    source_scope: str = "athena_graft_method_summary"
    blueprint_ids: tuple[str, ...] = ()
    allowed_override_parameters: tuple[str, ...] = ()
    evaluated_algorithm: bool = False
    runtime_enabled: bool = False
    paper_score_claim: bool = False
    autonomous_discovery_claim: bool = False
    claim_boundary: str = METHOD_TEMPLATE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.template_id.strip():
            raise ValueError("method template requires template_id")
        MethodAction(
            action_id=self.action_id,
            family=self.family,
            parameters=self.default_parameters,
            source_scope=self.source_scope,
        )
        if not self.claim_boundary.strip():
            raise ValueError("method template requires claim_boundary")
        if self.evaluated_algorithm or self.runtime_enabled:
            raise ValueError("method templates must be inert by default")
        if self.paper_score_claim or self.autonomous_discovery_claim:
            raise ValueError("method templates must not carry paper-score or autonomy claims")

    def instantiate(self, overrides: dict[str, Any] | None = None) -> MethodAction:
        allowed = set(self.allowed_override_parameters or tuple(self.default_parameters))
        unknown = sorted(set(overrides or {}) - allowed)
        if unknown:
            raise ValueError(f"unknown override parameters for {self.template_id!r}: {unknown}")
        parameters = {**self.default_parameters, **(overrides or {})}
        return MethodAction(
            action_id=self.action_id,
            family=self.family,
            parameters=parameters,
            source_scope=self.source_scope,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "action_id": self.action_id,
            "family": self.family,
            "default_parameters": self.default_parameters,
            "source_scope": self.source_scope,
            "blueprint_ids": list(self.blueprint_ids),
            "allowed_override_parameters": list(self.allowed_override_parameters or tuple(self.default_parameters)),
            "evaluated_algorithm": self.evaluated_algorithm,
            "runtime_enabled": self.runtime_enabled,
            "paper_score_claim": self.paper_score_claim,
            "autonomous_discovery_claim": self.autonomous_discovery_claim,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class MethodTemplateLibrary:
    templates: tuple[MethodTemplate, ...]
    blueprints: tuple[ExpertBlueprint, ...]
    claim_boundary: str = METHOD_TEMPLATE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        _ensure_unique([template.template_id for template in self.templates], "template_id")
        _ensure_unique([blueprint.blueprint_id for blueprint in self.blueprints], "blueprint_id")
        if not self.claim_boundary.strip():
            raise ValueError("method template library requires claim_boundary")

    def template(self, template_id: str) -> MethodTemplate:
        for template in self.templates:
            if template.template_id == template_id:
                return template
        raise KeyError(f"unknown method template: {template_id}")

    def blueprint(self, blueprint_id: str) -> ExpertBlueprint:
        for blueprint in self.blueprints:
            if blueprint.blueprint_id == blueprint_id:
                return blueprint
        raise KeyError(f"unknown expert blueprint: {blueprint_id}")

    def method_path(
        self,
        template_ids: tuple[str, ...],
        *,
        overrides: dict[str, dict[str, Any]] | None = None,
        source_scope: str = "athena_graft_template_path",
    ) -> MethodPath:
        actions: list[MethodAction] = []
        for template_id in template_ids:
            template = self.template(template_id)
            action = template.instantiate((overrides or {}).get(template_id))
            for blueprint_id in template.blueprint_ids:
                self.blueprint(blueprint_id).validate_action(action)
            actions.append(action)
        return MethodPath(actions=tuple(actions), source_scope=source_scope)

    def to_dict(self) -> dict[str, Any]:
        return {
            "templates": [template.to_dict() for template in self.templates],
            "blueprints": [
                {
                    "blueprint_id": blueprint.blueprint_id,
                    "allowed_families": list(blueprint.allowed_families),
                    "required_parameters": {
                        key: list(value) for key, value in blueprint.required_parameters.items()
                    },
                    "forbidden_parameters": {
                        key: list(value) for key, value in blueprint.forbidden_parameters.items()
                    },
                }
                for blueprint in self.blueprints
            ],
            "claim_boundary": self.claim_boundary,
        }


def athena_graft_method_library() -> MethodTemplateLibrary:
    safe_blueprint = ExpertBlueprint(
        blueprint_id="athena_graft_safe_method_template",
        allowed_families=(
            "workflow_trace",
            "constraint_policy",
            "factored_decision",
            "experience_cache",
        ),
        required_parameters={
            "workflow_trace": ("input_symbol", "output_symbol"),
            "constraint_policy": ("constraint_scope",),
            "factored_decision": ("decision_axis",),
        },
        forbidden_parameters={
            "*": (
                "paper_score_claim",
                "autonomous_discovery_claim",
                "self_improvement_claim",
            ),
        },
    )
    blueprint_ids = (safe_blueprint.blueprint_id,)
    return MethodTemplateLibrary(
        blueprints=(safe_blueprint,),
        templates=(
            MethodTemplate(
                template_id="hena_asr_mapping",
                action_id="hena_structural_action_to_reward",
                family="workflow_trace",
                default_parameters={
                    "input_symbol": "A_n",
                    "output_symbol": "R_n",
                    "intermediate_symbol": "S_n",
                    "purpose": "traceability",
                },
                blueprint_ids=blueprint_ids,
                source_scope="ATHENA arXiv:2512.03476v2 HENA loop summary",
            ),
            MethodTemplate(
                template_id="expert_blueprint_constraint",
                action_id="expert_blueprint_filter",
                family="constraint_policy",
                default_parameters={
                    "constraint_scope": "expert_blueprint",
                    "enforcement": "deterministic_validation",
                    "claim_boundary": "rule_filter_not_physics_understanding",
                },
                blueprint_ids=blueprint_ids,
                source_scope="ATHENA arXiv:2512.03476v2 expert blueprint summary",
            ),
            MethodTemplate(
                template_id="factored_method_path",
                action_id="factored_decision_path",
                family="factored_decision",
                default_parameters={
                    "decision_axis": "method_path",
                    "path_semantics": "ordered_actions",
                    "metric_embedding": "not_implemented",
                },
                blueprint_ids=blueprint_ids,
                source_scope="GRAFT-ATHENA arXiv:2605.11117v1 factored tree summary",
            ),
            MethodTemplate(
                template_id="method_fingerprint_cache",
                action_id="method_fingerprint_cache_key",
                family="experience_cache",
                default_parameters={
                    "cache_key": "method_fingerprint",
                    "lookup_mode": "exact_hash",
                    "metric_embedding": "not_implemented",
                },
                blueprint_ids=blueprint_ids,
                source_scope="GRAFT-ATHENA arXiv:2605.11117v1 fingerprint summary",
            ),
            MethodTemplate(
                template_id="local_experience_record_template",
                action_id="local_knowledge_substrate_record",
                family="experience_cache",
                default_parameters={
                    "record_scope": "local_reward_history",
                    "cross_domain_learning": "not_implemented",
                    "autonomous_action_space_expansion": "not_implemented",
                },
                blueprint_ids=blueprint_ids,
                source_scope="GRAFT-ATHENA arXiv:2605.11117v1 knowledge substrate summary",
            ),
        ),
    )


def builtin_method_template_library() -> MethodTemplateLibrary:
    return athena_graft_method_library()


def method_path_from_templates(
    library: MethodTemplateLibrary,
    template_ids: tuple[str, ...],
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
    source_scope: str = "athena_graft_template_path",
) -> MethodPath:
    return library.method_path(template_ids, overrides=overrides, source_scope=source_scope)


def asr_trace_record(
    method_path: MethodPath,
    *,
    solution_artifact: str,
    reward: ScientificReward,
    claim_boundary: str = WORKFLOW_TRACEABILITY_CLAIM_BOUNDARY,
) -> dict[str, Any]:
    validate_repo_relative_artifact(solution_artifact, label="solution_artifact")
    return {
        "schema_version": 1,
        "method_fingerprint": method_path.fingerprint(),
        "actions": [action.to_dict() for action in method_path.actions],
        "solution_artifact": solution_artifact,
        "reward": reward.to_dict(),
        "claim_boundary": claim_boundary,
    }


def _ensure_unique(values: list[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
