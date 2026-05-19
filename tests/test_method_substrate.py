from __future__ import annotations

import math

import pytest

from agenticsciml.method_substrate import (
    ExperienceRecord,
    ExperienceSubstrate,
    ExpertBlueprint,
    MethodAction,
    MethodPath,
    ScientificReward,
)


def test_method_path_fingerprint_is_stable_for_canonical_parameters() -> None:
    first = MethodPath(
        actions=(
            MethodAction(
                action_id="linear_branch",
                family="operator_architecture",
                parameters={"rank": 8, "bias": False},
            ),
        ),
    )
    second = MethodPath(
        actions=(
            MethodAction(
                action_id="linear_branch",
                family="operator_architecture",
                parameters={"bias": False, "rank": 8},
            ),
        ),
    )

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint().startswith("sha256:")
    assert first.method_tags() == ("operator_architecture", "linear_branch")


def test_method_path_fingerprint_ignores_source_scope() -> None:
    first = MethodPath(
        actions=(
            MethodAction(
                action_id="linear_branch",
                family="operator_architecture",
                parameters={"rank": 8},
                source_scope="paper-a",
            ),
        ),
        source_scope="notebooklm-summary",
    )
    second = MethodPath(
        actions=(
            MethodAction(
                action_id="linear_branch",
                family="operator_architecture",
                parameters={"rank": 8},
                source_scope="paper-b",
            ),
        ),
        source_scope="pro-review",
    )

    assert first.fingerprint() == second.fingerprint()


def test_method_path_fingerprint_preserves_action_order() -> None:
    architecture = MethodAction("spectral_decoder", "architecture", {"modes": 12})
    optimizer = MethodAction("lbfgs_refine", "optimizer", {"max_iter": 20})

    first = MethodPath(actions=(architecture, optimizer))
    second = MethodPath(actions=(optimizer, architecture))

    assert first.fingerprint() != second.fingerprint()


def test_method_path_fingerprint_changes_when_parameter_value_changes() -> None:
    first = MethodPath(
        actions=(MethodAction("spectral_decoder", "architecture", {"modes": 12}),)
    )
    second = MethodPath(
        actions=(MethodAction("spectral_decoder", "architecture", {"modes": 16}),)
    )

    assert first.fingerprint() != second.fingerprint()


def test_expert_blueprint_validates_allowed_required_and_forbidden_parameters() -> None:
    blueprint = ExpertBlueprint(
        blueprint_id="bias_free_operator",
        allowed_families=("operator_architecture",),
        required_parameters={"operator_architecture": ("rank",)},
        forbidden_parameters={"operator_architecture": ("bias",)},
    )

    valid = MethodAction(
        action_id="linear_branch",
        family="operator_architecture",
        parameters={"rank": 4},
    )
    assert blueprint.validate_action(valid) is valid

    with pytest.raises(ValueError, match="missing required"):
        blueprint.validate_action(
            MethodAction("linear_branch", "operator_architecture", parameters={})
        )
    with pytest.raises(ValueError, match="forbidden"):
        blueprint.validate_action(
            MethodAction(
                "linear_branch",
                "operator_architecture",
                parameters={"rank": 4, "bias": True},
            )
        )
    with pytest.raises(ValueError, match="not allowed"):
        blueprint.validate_action(MethodAction("adam", "optimizer", parameters={"lr": 0.01}))


def test_scientific_reward_rejects_non_finite_values() -> None:
    ScientificReward(
        metric="validation_mse",
        value=0.125,
        higher_is_better=False,
        source_artifact="solutions/solution_001/eval.json",
    )

    for bad_value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            ScientificReward(
                metric="validation_mse",
                value=bad_value,
                higher_is_better=False,
                source_artifact="solutions/solution_001/eval.json",
            )


def test_scientific_reward_rejects_absolute_or_escaping_artifact_paths() -> None:
    for bad_path in ("/tmp/eval.json", "../eval.json", "~/eval.json"):
        with pytest.raises(ValueError, match="repo-relative"):
            ScientificReward(
                metric="validation_mse",
                value=0.1,
                higher_is_better=False,
                source_artifact=bad_path,
            )


def test_method_parameters_reject_non_json_and_non_finite_values() -> None:
    with pytest.raises(ValueError):
        MethodAction("bad", "architecture", parameters={"scale": math.nan})
    with pytest.raises(TypeError):
        MethodAction("bad", "architecture", parameters={"callback": object()})


def test_experience_substrate_roundtrips_records_by_method_fingerprint(tmp_path) -> None:
    method_path = MethodPath(
        actions=(
            MethodAction(
                action_id="corner_weighted_residual",
                family="pde_constraint",
                parameters={"corner": [0.0, 0.0], "exponent": 1.2},
            ),
            MethodAction(
                action_id="lbfgs_refine",
                family="optimizer",
                parameters={"max_iter": 30},
            ),
        ),
    )
    record = ExperienceRecord(
        method_path=method_path,
        benchmark_name="poisson_lshape_faithful_small",
        reward=ScientificReward(
            metric="poisson_residual_composite",
            value=0.031,
            higher_is_better=False,
            source_artifact="solutions/solution_003/eval.json",
        ),
        notes="Local workflow evidence only; not a paper-score claim.",
    )
    better_record = ExperienceRecord(
        method_path=method_path,
        benchmark_name="poisson_lshape_faithful_small",
        reward=ScientificReward(
            metric="poisson_residual_composite",
            value=0.024,
            higher_is_better=False,
            source_artifact="solutions/solution_004/eval.json",
        ),
        notes="Second local workflow record for the same fingerprint.",
    )

    substrate = ExperienceSubstrate(tmp_path / "experience_cache.json")
    substrate.save(record)
    substrate.save(better_record)

    loaded = substrate.get_for_path(method_path)
    assert loaded == better_record
    assert substrate.get(method_path.fingerprint()) == better_record
    assert substrate.records_for_fingerprint(method_path.fingerprint()) == [record, better_record]
    assert "corner_weighted_residual" in (tmp_path / "experience_cache.json").read_text()
