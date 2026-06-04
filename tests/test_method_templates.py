from __future__ import annotations

import pytest

from agenticsciml.method_substrate import ScientificReward
from agenticsciml.method_templates import (
    METHOD_TEMPLATE_CLAIM_BOUNDARY,
    WORKFLOW_TRACEABILITY_CLAIM_BOUNDARY,
    MethodTemplate,
    MethodTemplateLibrary,
    asr_trace_record,
    athena_graft_method_library,
    builtin_method_template_library,
    method_path_from_templates,
)


def test_athena_graft_library_exposes_stable_template_ids_without_score_claims() -> None:
    library = athena_graft_method_library()
    template_ids = [template.template_id for template in library.templates]

    assert template_ids == [
        "hena_asr_mapping",
        "expert_blueprint_constraint",
        "factored_method_path",
        "method_fingerprint_cache",
        "local_experience_record_template",
    ]
    assert "paper-score reproduction" in library.claim_boundary
    assert "autonomous discovery" in library.claim_boundary
    for template in library.templates:
        assert template.claim_boundary == METHOD_TEMPLATE_CLAIM_BOUNDARY
        assert not template.evaluated_algorithm
        assert not template.runtime_enabled
        assert not template.paper_score_claim
        assert not template.autonomous_discovery_claim
        assert "not_implemented" in str(template.default_parameters) or template.template_id in {
            "hena_asr_mapping",
            "expert_blueprint_constraint",
        }


def test_template_instantiation_and_path_fingerprint_are_deterministic() -> None:
    library = athena_graft_method_library()
    template_ids = ("hena_asr_mapping", "factored_method_path", "method_fingerprint_cache")

    first = method_path_from_templates(library, template_ids)
    second = method_path_from_templates(builtin_method_template_library(), template_ids)

    assert first == second
    assert first.fingerprint() == second.fingerprint()
    assert first.method_tags() == (
        "workflow_trace",
        "hena_structural_action_to_reward",
        "factored_decision",
        "factored_decision_path",
        "experience_cache",
        "method_fingerprint_cache_key",
    )


def test_template_overrides_affect_identity_only_when_parameters_change() -> None:
    library = athena_graft_method_library()
    template_ids = ("factored_method_path",)
    default_path = library.method_path(template_ids)
    same_path = library.method_path(
        template_ids,
        overrides={"factored_method_path": {"decision_axis": "method_path"}},
    )
    changed_path = library.method_path(
        template_ids,
        overrides={"factored_method_path": {"decision_axis": "optimizer_schedule"}},
    )

    assert default_path.fingerprint() == same_path.fingerprint()
    assert default_path.fingerprint() != changed_path.fingerprint()


def test_blueprint_validation_runs_when_building_path_from_templates() -> None:
    library = athena_graft_method_library()

    with pytest.raises(ValueError, match="missing required"):
        library.method_path(
            ("factored_method_path",),
            overrides={"factored_method_path": {"decision_axis": None}},
        )
    with pytest.raises(ValueError, match="unknown override"):
        library.method_path(
            ("factored_method_path",),
            overrides={"factored_method_path": {"paper_score_claim": True}},
        )


def test_asr_trace_record_links_actions_solution_artifact_reward_and_boundary() -> None:
    library = athena_graft_method_library()
    method_path = library.method_path(("hena_asr_mapping", "method_fingerprint_cache"))
    reward = ScientificReward(
        metric="validation_mse",
        value=0.05,
        higher_is_better=False,
        source_artifact="solutions/solution_001/eval.json",
    )

    trace = asr_trace_record(
        method_path,
        solution_artifact="solutions/solution_001/solution.py",
        reward=reward,
    )

    assert trace["schema_version"] == 1
    assert trace["method_fingerprint"] == method_path.fingerprint()
    assert trace["actions"] == [action.to_dict() for action in method_path.actions]
    assert trace["solution_artifact"] == "solutions/solution_001/solution.py"
    assert trace["reward"] == reward.to_dict()
    assert trace["claim_boundary"] == WORKFLOW_TRACEABILITY_CLAIM_BOUNDARY

    with pytest.raises(ValueError, match="repo-relative"):
        asr_trace_record(method_path, solution_artifact="/tmp/solution.py", reward=reward)


def test_template_library_rejects_duplicate_template_ids() -> None:
    template = MethodTemplate(
        template_id="duplicate",
        action_id="action",
        family="workflow_trace",
        default_parameters={"input_symbol": "A_n", "output_symbol": "R_n"},
    )

    with pytest.raises(ValueError, match="duplicate template_id"):
        MethodTemplateLibrary(templates=(template, template), blueprints=())


def test_method_template_rejects_runtime_or_paper_claims() -> None:
    with pytest.raises(ValueError, match="inert"):
        MethodTemplate(
            template_id="runtime",
            action_id="action",
            family="workflow_trace",
            default_parameters={"input_symbol": "A_n", "output_symbol": "R_n"},
            runtime_enabled=True,
        )
    with pytest.raises(ValueError, match="paper-score"):
        MethodTemplate(
            template_id="paper_claim",
            action_id="action",
            family="workflow_trace",
            default_parameters={"input_symbol": "A_n", "output_symbol": "R_n"},
            paper_score_claim=True,
        )
