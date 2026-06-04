from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticsciml.evidence import CLAIM_LEVEL_WORKFLOW_PROXY
from agenticsciml.method_substrate import (
    ExperienceRecord,
    MethodAction,
    MethodPath,
    ScientificReward,
)
from agenticsciml.state import SolutionNode


def build_method_experience_record(
    *,
    node: SolutionNode,
    run_dir: Path,
    benchmark_name: str,
    benchmark_family: str,
    evidence_mode: str = CLAIM_LEVEL_WORKFLOW_PROXY,
) -> tuple[ExperienceRecord, dict[str, Any]]:
    workspace = Path(node.workspace)
    operator_assignment = _read_json(workspace / "operator_assignment.json")
    mutation_report = _read_json(workspace / "mutation_effect_report.json")
    policy_report = _read_json(workspace / "policy_fidelity_report.json")
    visual_report = _read_json(workspace / "visual_audit_report.json")
    action = _method_action_for_node(
        node=node,
        operator_assignment=operator_assignment,
        mutation_report=mutation_report,
        benchmark_family=benchmark_family,
    )
    method_path = MethodPath(actions=(action,))
    reward = _scientific_reward_for_node(node, run_dir=run_dir, evidence_mode=evidence_mode)
    record = ExperienceRecord(
        method_path=method_path,
        benchmark_name=benchmark_name,
        reward=reward,
        notes=(
            "Local method experience only. Exact-fingerprint and benchmark-family reuse are supported; "
            "metric-space self-improvement and autonomous action-space expansion are not claimed."
        ),
    )
    payload = {
        **record.to_dict(),
        "node_id": node.node_id,
        "benchmark_family": benchmark_family,
        "status": node.status,
        "failure_kind": node.failure_kind,
        "score_delta_from_parent": node.score_delta_from_parent,
        "method_tags": list(node.method_tags),
        "experience_scope": {
            "exact_fingerprint_retrieval": True,
            "benchmark_family_retrieval": True,
            "metric_space_self_improvement_claimed": False,
            "autonomous_action_space_expansion_claimed": False,
        },
        "failure_attribution": _failure_attribution(
            node=node,
            mutation_report=mutation_report,
            policy_report=policy_report,
            visual_report=visual_report,
        ),
        "source_artifacts": _source_artifacts(node, run_dir),
        "claim_boundary": (
            "Method experience records are audit memory for local workflow optimization. They do not support "
            "scientific discovery claims without real LLM, paper-like benchmark, domain approval, and "
            "multi-seed evidence."
        ),
    }
    return record, payload


def method_experience_context(
    *,
    cache_path: Path,
    benchmark_family: str,
    limit: int = 4,
) -> str:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    records_by_fingerprint = payload.get("records")
    if not isinstance(records_by_fingerprint, dict):
        return ""
    rows: list[str] = []
    for fingerprint, records in sorted(records_by_fingerprint.items()):
        if not isinstance(records, list):
            continue
        for record in records[-limit:]:
            if not isinstance(record, dict):
                continue
            if str(record.get("benchmark_family", "")) != benchmark_family:
                continue
            reward = record.get("reward")
            if not isinstance(reward, dict):
                continue
            rows.append(
                "method_experience: "
                f"fingerprint={fingerprint} "
                f"benchmark={record.get('benchmark_name')} "
                f"metric={reward.get('metric')} "
                f"value={reward.get('value')} "
                f"status={record.get('status')} "
                f"failure={record.get('failure_kind') or 'none'}"
            )
    return "\n".join(rows[-limit:])


def _method_action_for_node(
    *,
    node: SolutionNode,
    operator_assignment: dict[str, Any],
    mutation_report: dict[str, Any],
    benchmark_family: str,
) -> MethodAction:
    if operator_assignment:
        action_id = _string_value(operator_assignment.get("operator_id"), "operator_assignment")
        family = _string_value(operator_assignment.get("mutation_axis"), "operator_scheduler")
        parameters = {
            "benchmark_family": benchmark_family,
            "selection_source": operator_assignment.get("selection_source"),
            "selected_algorithm_ids": _jsonable(operator_assignment.get("selected_algorithm_ids", [])),
            "method_tags": list(node.method_tags),
        }
        if operator_assignment.get("policy_fidelity") is not None:
            parameters["policy_fidelity"] = _jsonable(operator_assignment.get("policy_fidelity"))
        return MethodAction(
            action_id=action_id,
            family=family,
            parameters=parameters,
            source_scope="operator_assignment",
        )
    return MethodAction(
        action_id=_string_value(mutation_report.get("status"), "root_or_unassigned"),
        family="baseline" if node.parent_id is None else "unassigned_mutation",
        parameters={
            "benchmark_family": benchmark_family,
            "method_tags": list(node.method_tags),
            "node_status": node.status,
        },
        source_scope="local_workflow_artifact",
    )


def _scientific_reward_for_node(
    node: SolutionNode,
    *,
    run_dir: Path,
    evidence_mode: str,
) -> ScientificReward:
    if node.score is not None:
        return ScientificReward(
            metric=node.score.metric,
            value=node.score.value,
            higher_is_better=node.score.higher_is_better,
            source_artifact=_relative_existing_artifact(Path(node.workspace) / "eval.json", run_dir),
            evidence_mode=evidence_mode,
        )
    return ScientificReward(
        metric=f"failed_{node.failure_kind or 'unknown'}",
        value=1.0,
        higher_is_better=False,
        source_artifact=_relative_existing_artifact(_first_existing_failure_artifact(Path(node.workspace)), run_dir),
        evidence_mode=evidence_mode,
    )


def _failure_attribution(
    *,
    node: SolutionNode,
    mutation_report: dict[str, Any],
    policy_report: dict[str, Any],
    visual_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "node_status": node.status,
        "failure_kind": node.failure_kind,
        "score_delta_from_parent": node.score_delta_from_parent,
        "debug_attempt_count": node.num_debug_attempts,
        "mutation_status": mutation_report.get("status"),
        "policy_fidelity_status": policy_report.get("status"),
        "visual_audit_mode": visual_report.get("visual_audit_mode"),
        "actual_image_inputs_used": visual_report.get("actual_image_inputs_used", False),
        "classification": _experience_classification(node, mutation_report, policy_report),
    }


def _experience_classification(
    node: SolutionNode,
    mutation_report: dict[str, Any],
    policy_report: dict[str, Any],
) -> str:
    if policy_report.get("status") == "blocked":
        return "policy_fidelity_mismatch"
    if node.status != "evaluated":
        return "failure"
    if node.score_delta_from_parent is None:
        return "success"
    if abs(float(node.score_delta_from_parent)) < 1e-12:
        return "plateau"
    higher_is_better = bool(node.score and node.score.higher_is_better)
    improved = node.score_delta_from_parent > 0 if higher_is_better else node.score_delta_from_parent < 0
    if improved:
        return "success"
    if mutation_report.get("status") == "changed_but_score_plateau":
        return "plateau"
    return "regression"


def _source_artifacts(node: SolutionNode, run_dir: Path) -> list[str]:
    workspace = Path(node.workspace)
    candidates = [
        workspace / "operator_assignment.json",
        workspace / "proposal.md",
        workspace / "policy_fidelity_report.json",
        workspace / "mutation_effect_report.json",
        workspace / "visual_audit_report.json",
        workspace / "eval.json",
        workspace / "train.log",
    ]
    return [_relative_existing_artifact(path, run_dir) for path in candidates if path.exists()]


def _first_existing_failure_artifact(workspace: Path) -> Path:
    for name in ("train.log", "engineering_error.md", "policy_fidelity_report.json", "analysis.md"):
        path = workspace / name
        if path.exists():
            return path
    return workspace / "analysis.md"


def _relative_existing_artifact(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return path.name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _string_value(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _jsonable(value: object) -> object:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value
