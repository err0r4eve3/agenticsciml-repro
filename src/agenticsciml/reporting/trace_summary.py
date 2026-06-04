from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agenticsciml.state import (
    SOLUTION_TREE_SCHEMA_VERSION,
    validate_solution_node_artifact_paths,
    validate_solution_node_payload,
    validate_solution_tree_graph_payload,
)
from agenticsciml.trace_contracts import FanoutTraceMetadata, fanout_trace_references
from agenticsciml.evidence import CLAIM_GATE_BLOCKED


REQUIRED_EVENT_TYPES = (
    "workflow_span",
    "agent_span",
    "generation_span",
    "tool_span",
    "guardrail_span",
)

EVIDENCE_METADATA_KEYS = (
    "llm_mode",
    "benchmark_fidelity_level",
    "evidence_mode",
    "scientific_claim",
)

RUN_STATES = {"partial", "completed", "exported", "finalized"}
EXPORTED_RUN_STATES = {"completed", "exported", "finalized"}
TRACE_NODE_REFERENCE_EVENT_NAMES = {
    "agenticsciml.parallel_children.start",
    "agenticsciml.parallel_children.end",
    "agenticsciml.child_mutation.start",
    "agenticsciml.child_mutation.end",
    "child_creation:exception",
    "engineer:patch_application",
    "debugger:patch_application",
    "train_and_evaluate",
    "train_and_evaluate.retry",
}
SELF_TRACE_REFERENCE_KEYS = {
    "node_id",
    "solution_id",
    "child_id",
    "node_ids",
    "solution_ids",
    "child_ids",
    "parent_to_child.child",
    "parent_to_children.child",
    "parent_child_edges.child",
}
TRACE_NODE_LIFECYCLE_STAGE_RULES = {
    ("agent_span", "root_engineer"): "materialized",
    ("agent_span", "engineer"): "materialized",
    ("workflow_span", "agenticsciml.child_mutation.start"): "created",
    ("workflow_span", "agenticsciml.child_mutation.end"): "completed",
    ("workflow_span", "child_creation:exception"): "failed",
    ("tool_span", "train_and_evaluate"): "evaluated",
    ("tool_span", "train_and_evaluate.retry"): "evaluated",
}
def load_trace_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def summarize_trace(run_dir: Path) -> dict[str, Any]:
    events = load_trace_events(run_dir)
    event_counts = Counter(str(event.get("event_type", "")) for event in events)
    missing_event_types = [
        event_type for event_type in REQUIRED_EVENT_TYPES if event_counts.get(event_type, 0) == 0
    ]
    guardrail_failures = [
        {
            "name": str(event.get("name", "")),
            "metadata": event.get("metadata", {}),
        }
        for event in events
        if event.get("event_type") == "guardrail_span"
        and event.get("metadata", {}).get("passed") is False
    ]
    agent_roles = sorted(
        {
            str(event.get("metadata", {}).get("spec_role"))
            for event in events
            if event.get("metadata", {}).get("spec_role")
        }
    )
    artifact_consistency = _check_artifact_consistency(run_dir, events)
    claim_gate = _summary_claim_gate(run_dir, events)
    quality_passed = (
        not missing_event_types
        and not guardrail_failures
        and bool(artifact_consistency.get("passed", True))
    )
    return {
        "event_count": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "agent_roles": agent_roles,
        "guardrail_failures": guardrail_failures,
        "artifact_consistency": artifact_consistency,
        "claim_gate": claim_gate,
        "quality_gate": {
            "passed": quality_passed,
            "required_event_types": list(REQUIRED_EVENT_TYPES),
            "missing_event_types": missing_event_types,
        },
    }


def _summary_claim_gate(run_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            run_metadata = None
        if isinstance(run_metadata, dict) and isinstance(run_metadata.get("claim_gate"), dict):
            return dict(run_metadata["claim_gate"])
    workflow_metadata = _workflow_start_metadata(events)
    if isinstance(workflow_metadata, dict) and isinstance(workflow_metadata.get("claim_gate"), dict):
        return dict(workflow_metadata["claim_gate"])
    return None


def _check_artifact_consistency(run_dir: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    contract_path = run_dir / "evaluation_contract.json"
    metadata_path = run_dir / "run_metadata.json"
    if not contract_path.exists() and not metadata_path.exists():
        return {"checked": False, "passed": True, "issues": []}

    issues: list[str] = []
    contract = _read_json_file(contract_path, issues)
    run_metadata = _read_json_file(metadata_path, issues)
    workflow_metadata = _workflow_start_metadata(events)
    workflow_end_metadata = _workflow_end_metadata(events)
    tree_path = run_dir / "tree.json"
    checkpoint_path = run_dir / "checkpoint.json"
    if _requires_solution_artifacts(run_metadata):
        if not tree_path.exists():
            issues.append("tree.json is required for exported run artifacts")
        if not checkpoint_path.exists():
            issues.append("checkpoint.json is required for exported run artifacts")
    tree = _read_optional_json_file(tree_path, issues)
    checkpoint = _read_optional_json_file(checkpoint_path, issues)

    if contract is not None and run_metadata is not None:
        fidelity = contract.get("benchmark_fidelity")
        if not isinstance(fidelity, dict):
            issues.append("evaluation_contract.json benchmark_fidelity is missing or invalid")
        else:
            _compare_metadata_value(
                issues,
                "benchmark_fidelity_level",
                run_metadata.get("benchmark_fidelity_level"),
                fidelity.get("fidelity_level"),
                "run_metadata.json",
                "evaluation_contract.json",
            )

    if run_metadata is not None and workflow_metadata is not None:
        for key in EVIDENCE_METADATA_KEYS:
            _compare_metadata_value(
                issues,
                key,
                run_metadata.get(key),
                workflow_metadata.get(key),
                "run_metadata.json",
                "trace workflow start",
            )
    elif run_metadata is not None and events:
        issues.append("trace workflow start metadata is missing")

    if run_metadata is not None or contract is not None:
        _check_trace_event_sequence(issues, events)
    _check_workflow_lifecycle_sequence(issues, events)
    _check_run_state_consistency(issues, run_metadata, workflow_metadata, workflow_end_metadata)
    _check_claim_gate_consistency(issues, run_metadata, workflow_metadata)
    scientific_readiness = _check_scientific_discovery_readiness_consistency(issues, run_dir, run_metadata)
    trace_node_reference_counts = _check_solution_artifact_consistency(
        issues,
        run_dir,
        contract,
        run_metadata,
        tree,
        checkpoint,
        events,
    )
    data_analysis_specificity = _check_data_analysis_specificity(run_dir)

    return {
        "checked": True,
        "passed": not issues,
        "issues": issues,
        "scientific_discovery_readiness": scientific_readiness,
        "data_analysis_specificity": data_analysis_specificity,
        "trace_node_reference_events_checked": trace_node_reference_counts["checked"],
        "trace_node_reference_events_skipped": trace_node_reference_counts["skipped"],
        "trace_node_reference_events_checked_by_name": trace_node_reference_counts["checked_by_name"],
        "trace_node_reference_events_skipped_by_name": trace_node_reference_counts["skipped_by_name"],
        "trace_node_references_checked": trace_node_reference_counts["references_checked"],
        "trace_node_reference_events_with_references": trace_node_reference_counts[
            "events_with_references"
        ],
        "trace_node_references_checked_by_name": trace_node_reference_counts["references_checked_by_name"],
        "trace_node_reference_node_coverage": trace_node_reference_counts["node_coverage"],
        "trace_node_lifecycle_stage_coverage": trace_node_reference_counts["lifecycle_stage_coverage"],
    }


def _check_data_analysis_specificity(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "reports" / "data_analysis_structured.json"
    if not path.exists():
        return {
            "checked": False,
            "passed": False,
            "warnings": ["reports/data_analysis_structured.json is missing"],
        }
    warnings: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"checked": True, "passed": False, "warnings": [f"invalid structured data analysis JSON: {exc}"]}
    if not isinstance(payload, dict):
        return {"checked": True, "passed": False, "warnings": ["structured data analysis must be a JSON object"]}
    benchmark_name = payload.get("benchmark_name")
    array_keys = payload.get("training_array_keys")
    task_observations = payload.get("task_specific_observations")
    text_blob = json.dumps(payload, sort_keys=True, default=str).lower()
    if not isinstance(benchmark_name, str) or not benchmark_name:
        warnings.append("benchmark_name is missing")
    elif benchmark_name.lower() not in text_blob:
        warnings.append("benchmark_name is not referenced in structured analysis text")
    if not isinstance(array_keys, list) or not array_keys:
        warnings.append("training_array_keys is missing")
    if not isinstance(task_observations, list) or not task_observations:
        warnings.append("task_specific_observations is missing")
    elif not _task_observations_have_specific_terms(payload, task_observations):
        warnings.append("task_specific_observations lack benchmark-specific terms")
    return {"checked": True, "passed": not warnings, "warnings": warnings}


def _task_observations_have_specific_terms(
    payload: dict[str, Any],
    task_observations: list[Any],
) -> bool:
    observation_text = " ".join(str(item).lower() for item in task_observations)
    terms: set[str] = set()
    for value in (
        payload.get("benchmark_name"),
        payload.get("benchmark_family"),
        payload.get("evaluation_metric"),
    ):
        if isinstance(value, str):
            terms.update(_specificity_tokens(value))
    array_keys = payload.get("training_array_keys")
    if isinstance(array_keys, list):
        for item in array_keys:
            if isinstance(item, str):
                terms.add(item.lower())
                terms.update(_specificity_tokens(item))
    return bool(terms) and any(term in observation_text for term in terms)


def _specificity_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.replace("_", " ").replace("-", " ").lower().split()
        if len(token) >= 3
        and token
        not in {
            "the",
            "and",
            "with",
            "metric",
            "data",
            "fit",
            "model",
            "train",
            "training",
            "validation",
        }
    }


def _check_claim_gate_consistency(
    issues: list[str],
    run_metadata: dict[str, Any] | None,
    workflow_metadata: dict[str, Any] | None,
) -> None:
    if run_metadata is None:
        return
    claim_gate = run_metadata.get("claim_gate")
    if not isinstance(claim_gate, dict):
        if (
            run_metadata.get("paper_level_claim_supported") is True
            or run_metadata.get("scientific_claim_supported") is True
            or (
                isinstance(workflow_metadata, dict)
                and isinstance(workflow_metadata.get("claim_gate"), dict)
            )
        ):
            issues.append("run_metadata.json claim_gate is missing or invalid")
        return
    paper_supported = claim_gate.get("paper_level_claim_supported") is True
    scientific_supported = claim_gate.get("scientific_claim_supported") is True
    if claim_gate.get("status") == CLAIM_GATE_BLOCKED and (paper_supported or scientific_supported):
        issues.append("claim_gate blocked status cannot support paper or scientific claims")
    if run_metadata.get("paper_level_claim_supported") is True and not paper_supported:
        issues.append("run_metadata.json paper_level_claim_supported overclaims claim_gate")
    if run_metadata.get("scientific_claim_supported") is True and not scientific_supported:
        issues.append("run_metadata.json scientific_claim_supported overclaims claim_gate")
    if workflow_metadata is not None:
        workflow_gate = workflow_metadata.get("claim_gate")
        if not isinstance(workflow_gate, dict):
            issues.append("trace workflow start claim_gate is missing or invalid")
        else:
            _compare_metadata_value(
                issues,
                "claim_gate claim_level",
                claim_gate.get("claim_level"),
                workflow_gate.get("claim_level"),
                "run_metadata.json",
                "trace workflow start",
            )
            _compare_metadata_value(
                issues,
                "claim_gate paper_level_claim_supported",
                claim_gate.get("paper_level_claim_supported"),
                workflow_gate.get("paper_level_claim_supported"),
                "run_metadata.json",
                "trace workflow start",
            )


def _check_scientific_discovery_readiness_consistency(
    issues: list[str],
    run_dir: Path,
    run_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    path = run_dir / "reports" / "scientific_discovery_readiness.json"
    if not path.exists():
        return {"checked": False, "passed": True, "path": "reports/scientific_discovery_readiness.json"}
    readiness = _read_optional_json_file(path, issues)
    if not isinstance(readiness, dict):
        issues.append("reports/scientific_discovery_readiness.json is invalid")
        return {"checked": True, "passed": False, "path": "reports/scientific_discovery_readiness.json"}
    supported = readiness.get("scientific_claim_supported") is True
    if run_metadata is not None and run_metadata.get("scientific_claim_supported") is True and not supported:
        issues.append("run_metadata.json scientific_claim_supported overclaims scientific discovery readiness")
    metadata_readiness = run_metadata.get("scientific_discovery_readiness") if run_metadata else None
    if isinstance(metadata_readiness, dict):
        if metadata_readiness.get("scientific_claim_supported") is True and not supported:
            issues.append("run_metadata.json scientific_discovery_readiness overclaims readiness report")
    return {
        "checked": True,
        "passed": supported or not (
            run_metadata is not None and run_metadata.get("scientific_claim_supported") is True
        ),
        "path": "reports/scientific_discovery_readiness.json",
        "status": readiness.get("status"),
        "scientific_claim_supported": supported,
        "blocker_count": len(readiness.get("blockers", []))
        if isinstance(readiness.get("blockers"), list)
        else 0,
    }


def _read_json_file(path: Path, issues: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        issues.append(f"{path.name} is missing")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"{path.name} is not valid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        issues.append(f"{path.name} must contain a JSON object")
        return None
    return payload


def _read_optional_json_file(path: Path, issues: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json_file(path, issues)


def _requires_solution_artifacts(run_metadata: dict[str, Any] | None) -> bool:
    if run_metadata is None:
        return False
    run_state = run_metadata.get("run_state")
    if run_state is not None:
        return run_state in EXPORTED_RUN_STATES
    return "solution_count" in run_metadata


def _check_run_state_consistency(
    issues: list[str],
    run_metadata: dict[str, Any] | None,
    workflow_start_metadata: dict[str, Any] | None,
    workflow_end_metadata: dict[str, Any] | None,
) -> None:
    run_state = run_metadata.get("run_state") if run_metadata else None
    workflow_start_state = workflow_start_metadata.get("run_state") if workflow_start_metadata else None
    workflow_end_state = workflow_end_metadata.get("run_state") if workflow_end_metadata else None
    if run_state is not None and run_state not in RUN_STATES:
        issues.append(f"run_metadata.json run_state is invalid: {run_state!r}")
    if workflow_start_state is not None and workflow_start_state not in RUN_STATES:
        issues.append(f"trace workflow start run_state is invalid: {workflow_start_state!r}")
    if workflow_end_state is not None and workflow_end_state not in RUN_STATES:
        issues.append(f"trace workflow end run_state is invalid: {workflow_end_state!r}")
    if run_state in EXPORTED_RUN_STATES and workflow_end_metadata is None:
        issues.append("trace workflow end run_state is required for exported run_state")
    if run_state is not None and workflow_end_metadata is not None:
        _compare_metadata_value(
            issues,
            "run_state",
            run_state,
            workflow_end_state,
            "run_metadata.json",
            "trace workflow end",
        )


def _check_workflow_lifecycle_sequence(issues: list[str], events: list[dict[str, Any]]) -> None:
    start_indices: list[int] = []
    end_indices: list[int] = []
    end_states: set[str] = set()
    for index, event in enumerate(events):
        if event.get("event_type") != "workflow_span":
            continue
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if event.get("name") == "agenticsciml.run.start":
            start_indices.append(index)
        elif event.get("name") == "agenticsciml.run.end":
            end_indices.append(index)
            state = metadata.get("run_state")
            if isinstance(state, str):
                end_states.add(state)

    if start_indices and end_indices and min(end_indices) < min(start_indices):
        issues.append("workflow end precedes workflow start in trace order")
    if len(end_states) > 1:
        issues.append("conflicting workflow end run_state values: " + ", ".join(sorted(end_states)))


def _check_trace_event_sequence(issues: list[str], events: list[dict[str, Any]]) -> None:
    observed: list[int] = []
    for index, event in enumerate(events, start=1):
        seq = event.get("event_seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            issues.append(f"trace event_seq missing or invalid at trace line {index}")
            return
        observed.append(seq)
    expected = list(range(1, len(events) + 1))
    if observed != expected:
        issues.append(f"trace event_seq must be contiguous and monotonic: observed={observed!r}")


def _check_solution_artifact_consistency(
    issues: list[str],
    run_dir: Path,
    contract: dict[str, Any] | None,
    run_metadata: dict[str, Any] | None,
    tree: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_node_reference_counts: dict[str, Any] = {
        "checked": 0,
        "skipped": 0,
        "checked_by_name": {},
        "skipped_by_name": {},
        "references_checked": 0,
        "events_with_references": 0,
        "references_checked_by_name": {},
        "node_coverage": {"total_nodes": 0, "referenced": [], "unreferenced": [], "nodes": {}},
        "lifecycle_stage_coverage": {"nodes": {}},
    }
    expected_contract_hash = contract.get("contract_hash") if contract else None
    expected_benchmark_name = contract.get("benchmark_name") if contract else None

    tree_nodes = _nodes_by_id("tree.json", tree, issues)
    checkpoint_nodes = _nodes_by_id("checkpoint.json", checkpoint, issues)

    if run_metadata is not None and tree_nodes is not None and "solution_count" in run_metadata:
        _compare_metadata_value(
            issues,
            "solution_count",
            run_metadata.get("solution_count"),
            len(tree_nodes),
            "run_metadata.json",
            "tree.json",
        )

    if checkpoint is not None:
        if expected_contract_hash:
            _compare_metadata_value(
                issues,
                "checkpoint contract_hash",
                checkpoint.get("contract_hash"),
                expected_contract_hash,
                "checkpoint.json",
                "evaluation_contract.json",
            )
        if expected_benchmark_name:
            _compare_metadata_value(
                issues,
                "checkpoint benchmark_name",
                checkpoint.get("benchmark_name"),
                expected_benchmark_name,
                "checkpoint.json",
                "evaluation_contract.json",
            )

    if tree_nodes is not None and checkpoint_nodes is not None:
        tree_ids = sorted(tree_nodes)
        checkpoint_ids = sorted(checkpoint_nodes)
        if tree_ids != checkpoint_ids:
            issues.append(
                "checkpoint.json nodes mismatch tree.json nodes: "
                f"checkpoint={checkpoint_ids!r}, tree={tree_ids!r}"
            )

    for artifact_name, nodes in (("tree.json", tree_nodes), ("checkpoint.json", checkpoint_nodes)):
        if nodes is not None:
            _check_solution_node_schema(artifact_name, nodes, issues)
            _check_solution_tree_graph_invariants(artifact_name, nodes, issues)
            if _requires_solution_artifacts(run_metadata):
                _check_solution_node_artifact_paths(artifact_name, run_dir, nodes, issues)

    for artifact_name, nodes in (("tree.json", tree_nodes), ("checkpoint.json", checkpoint_nodes)):
        if nodes is None:
            continue
        for node_id, node in nodes.items():
            if expected_contract_hash:
                _compare_metadata_value(
                    issues,
                    f"{artifact_name} node {node_id} contract_hash",
                    node.get("contract_hash"),
                    expected_contract_hash,
                    artifact_name,
                    "evaluation_contract.json",
                )
            if expected_benchmark_name:
                _compare_metadata_value(
                    issues,
                    f"{artifact_name} node {node_id} benchmark_name",
                    node.get("benchmark_name"),
                    expected_benchmark_name,
                    artifact_name,
                    "evaluation_contract.json",
                )

    node_ids = set((tree_nodes or checkpoint_nodes or {}).keys())
    if node_ids:
        trace_node_reference_counts = _check_trace_node_references(issues, events, node_ids)
        trace_node_reference_counts["lifecycle_stage_coverage"] = _check_trace_node_lifecycle_stages(
            issues, events, node_ids
        )
        if _requires_solution_artifacts(run_metadata):
            if trace_node_reference_counts["checked"] == 0:
                issues.append("no solution-reference trace events were checked for exported node set")
            elif trace_node_reference_counts["references_checked"] == 0:
                issues.append("no solution node references were checked for exported node set")
            elif trace_node_reference_counts["node_coverage"]["unreferenced"]:
                issues.append(
                    "solution nodes have no allowlisted trace references: "
                    + ", ".join(trace_node_reference_counts["node_coverage"]["unreferenced"])
                )
            else:
                nodes_without_self_references = [
                    node_id
                    for node_id, detail in trace_node_reference_counts["node_coverage"]["nodes"].items()
                    if detail["self_reference_count"] == 0
                ]
                if nodes_without_self_references:
                    issues.append(
                        "solution nodes have no self trace references: "
                        + ", ".join(nodes_without_self_references)
                    )
                nodes_missing_required_stages = _nodes_missing_required_lifecycle_stages(
                    tree_nodes or checkpoint_nodes or {},
                    trace_node_reference_counts["lifecycle_stage_coverage"],
                )
                if nodes_missing_required_stages:
                    issues.append(
                        "solution nodes missing required lifecycle stages: "
                        + "; ".join(
                            f"{node_id}={','.join(missing_stages)}"
                            for node_id, missing_stages in nodes_missing_required_stages.items()
                        )
                    )
                evaluated_nodes_without_evaluated_stage = _evaluated_nodes_without_stage(
                    tree_nodes or checkpoint_nodes or {},
                    trace_node_reference_counts["lifecycle_stage_coverage"],
                    "evaluated",
                )
                if evaluated_nodes_without_evaluated_stage:
                    issues.append(
                        "evaluated solution nodes have no evaluated trace stage: "
                        + ", ".join(evaluated_nodes_without_evaluated_stage)
                    )
                nodes_with_invalid_stage_order = _nodes_with_invalid_lifecycle_stage_order(
                    tree_nodes or checkpoint_nodes or {},
                    trace_node_reference_counts["lifecycle_stage_coverage"],
                )
                if nodes_with_invalid_stage_order:
                    issues.append(
                        "solution nodes have invalid lifecycle stage order: "
                        + "; ".join(
                            f"{node_id}={'>'.join(expected_order)}"
                            for node_id, expected_order in nodes_with_invalid_stage_order.items()
                        )
                    )
    return trace_node_reference_counts


def _nodes_by_id(
    artifact_name: str,
    payload: dict[str, Any] | None,
    issues: list[str],
) -> dict[str, dict[str, Any]] | None:
    if payload is None:
        return None
    schema_version = payload.get("schema_version")
    if schema_version != SOLUTION_TREE_SCHEMA_VERSION:
        issues.append(
            f"{artifact_name} has unsupported schema_version: "
            f"{schema_version!r}; expected {SOLUTION_TREE_SCHEMA_VERSION!r}"
        )
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        issues.append(f"{artifact_name} nodes must be a list")
        return None
    by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            issues.append(f"{artifact_name} node at index {index} must be an object")
            continue
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            issues.append(f"{artifact_name} node at index {index} has invalid node_id")
            continue
        if node_id in by_id:
            issues.append(f"{artifact_name} has duplicate node_id: {node_id}")
            continue
        by_id[node_id] = node
    return by_id


def _check_solution_tree_graph_invariants(
    artifact_name: str,
    nodes: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    issues.extend(validate_solution_tree_graph_payload(nodes, context=artifact_name))


def _check_solution_node_schema(
    artifact_name: str,
    nodes: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    for node_id, node in sorted(nodes.items()):
        issues.extend(validate_solution_node_payload(node, context=f"{artifact_name} node {node_id}"))


def _check_solution_node_artifact_paths(
    artifact_name: str,
    run_dir: Path,
    nodes: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    for node_id, node in sorted(nodes.items()):
        issues.extend(
            validate_solution_node_artifact_paths(
                node,
                run_dir=run_dir,
                context=f"{artifact_name} node {node_id}",
            )
        )


def _check_trace_node_references(
    issues: list[str],
    events: list[dict[str, Any]],
    node_ids: set[str],
) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "checked": 0,
        "skipped": 0,
        "checked_by_name": {},
        "skipped_by_name": {},
        "references_checked": 0,
        "events_with_references": 0,
        "references_checked_by_name": {},
        "node_coverage": {
            "total_nodes": len(node_ids),
            "referenced": [],
            "unreferenced": sorted(node_ids),
            "nodes": {},
        },
        "lifecycle_stage_coverage": {"nodes": {}},
    }
    referenced_node_ids: set[str] = set()
    node_details: dict[str, dict[str, Any]] = {
        node_id: {
            "referenced_by_event_names": set(),
            "reference_keys": set(),
            "self_reference_count": 0,
            "relation_reference_count": 0,
        }
        for node_id in node_ids
    }
    for event in events:
        event_name = str(event.get("name", ""))
        if event_name not in TRACE_NODE_REFERENCE_EVENT_NAMES:
            counts["skipped"] += 1
            counts["skipped_by_name"][event_name] = counts["skipped_by_name"].get(event_name, 0) + 1
            continue
        counts["checked"] += 1
        counts["checked_by_name"][event_name] = counts["checked_by_name"].get(event_name, 0) + 1
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        _validate_parent_child_trace_metadata(event_name, metadata, issues)
        references = _trace_node_references(metadata)
        if references:
            counts["events_with_references"] += 1
            counts["references_checked"] += len(references)
            counts["references_checked_by_name"][event_name] = (
                counts["references_checked_by_name"].get(event_name, 0) + len(references)
            )
        for key, node_id in references:
            if node_id not in node_ids:
                issues.append(
                    f"trace event {event_name} references unknown solution node via {key}: {node_id}"
                )
            else:
                referenced_node_ids.add(node_id)
                detail = node_details[node_id]
                detail["referenced_by_event_names"].add(event_name)
                detail["reference_keys"].add(key)
                if _is_self_trace_reference_key(key):
                    detail["self_reference_count"] += 1
                else:
                    detail["relation_reference_count"] += 1
    node_coverage_details = {
        node_id: {
            "referenced_by_event_names": sorted(detail["referenced_by_event_names"]),
            "reference_keys": sorted(detail["reference_keys"]),
            "self_reference_count": detail["self_reference_count"],
            "relation_reference_count": detail["relation_reference_count"],
        }
        for node_id, detail in sorted(node_details.items())
    }
    counts["node_coverage"] = {
        "total_nodes": len(node_ids),
        "referenced": sorted(referenced_node_ids),
        "unreferenced": sorted(node_ids - referenced_node_ids),
        "nodes": node_coverage_details,
    }
    return counts


def _is_self_trace_reference_key(key: str) -> bool:
    return key in SELF_TRACE_REFERENCE_KEYS


def _check_trace_node_lifecycle_stages(
    issues: list[str],
    events: list[dict[str, Any]],
    node_ids: set[str],
) -> dict[str, Any]:
    stage_events: dict[str, dict[str, dict[str, Any]]] = {node_id: {} for node_id in node_ids}
    for event in events:
        event_type = str(event.get("event_type", ""))
        event_name = str(event.get("name", ""))
        stage = TRACE_NODE_LIFECYCLE_STAGE_RULES.get((event_type, event_name))
        if stage is None:
            continue
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        for key, node_id in _trace_node_references(metadata):
            if not _is_self_trace_reference_key(key):
                continue
            if node_id not in node_ids:
                issues.append(
                    f"trace lifecycle event {event_name} references unknown solution node via {key}: {node_id}"
                )
                continue
            stage_payload = stage_events[node_id].setdefault(
                stage,
                {"event_names": set(), "event_seqs": []},
            )
            stage_payload["event_names"].add(event_name)
            event_seq = event.get("event_seq")
            if isinstance(event_seq, int) and not isinstance(event_seq, bool):
                stage_payload["event_seqs"].append(event_seq)
    nodes_payload: dict[str, dict[str, Any]] = {}
    for node_id, stage_events_for_node in sorted(stage_events.items()):
        nodes_payload[node_id] = {
            "stages": sorted(stage_events_for_node),
            "stage_events": {
                stage: sorted(payload["event_names"]) for stage, payload in sorted(stage_events_for_node.items())
            },
            "stage_event_seqs": {
                stage: sorted(payload["event_seqs"]) for stage, payload in sorted(stage_events_for_node.items())
            },
        }
    return {
        "nodes": nodes_payload
    }


def _evaluated_nodes_without_stage(
    nodes: dict[str, dict[str, Any]],
    lifecycle_stage_coverage: dict[str, Any],
    stage: str,
) -> list[str]:
    coverage_nodes = lifecycle_stage_coverage.get("nodes", {})
    if not isinstance(coverage_nodes, dict):
        return []
    missing: list[str] = []
    for node_id, node in sorted(nodes.items()):
        if node.get("status") != "evaluated":
            continue
        coverage = coverage_nodes.get(node_id, {})
        stages = coverage.get("stages", []) if isinstance(coverage, dict) else []
        if stage not in stages:
            missing.append(node_id)
    return missing


def _nodes_missing_required_lifecycle_stages(
    nodes: dict[str, dict[str, Any]],
    lifecycle_stage_coverage: dict[str, Any],
) -> dict[str, list[str]]:
    coverage_nodes = lifecycle_stage_coverage.get("nodes", {})
    if not isinstance(coverage_nodes, dict):
        return {}
    missing: dict[str, list[str]] = {}
    for node_id, node in sorted(nodes.items()):
        required_stages = _required_lifecycle_stages(node)
        if not required_stages:
            continue
        coverage = coverage_nodes.get(node_id, {})
        stages = set(coverage.get("stages", [])) if isinstance(coverage, dict) else set()
        missing_stages = sorted(required_stages - stages)
        if missing_stages:
            missing[node_id] = missing_stages
    return missing


def _required_lifecycle_stages(node: dict[str, Any]) -> set[str]:
    if node.get("status") != "evaluated":
        return set()
    required = {"evaluated", "materialized"}
    if node.get("parent_id"):
        required.update({"created", "completed"})
    return required


def _nodes_with_invalid_lifecycle_stage_order(
    nodes: dict[str, dict[str, Any]],
    lifecycle_stage_coverage: dict[str, Any],
) -> dict[str, list[str]]:
    coverage_nodes = lifecycle_stage_coverage.get("nodes", {})
    if not isinstance(coverage_nodes, dict):
        return {}
    invalid: dict[str, list[str]] = {}
    for node_id, node in sorted(nodes.items()):
        expected_order = _expected_lifecycle_stage_order(node)
        if not expected_order:
            continue
        coverage = coverage_nodes.get(node_id, {})
        if not isinstance(coverage, dict):
            continue
        stage_event_seqs = coverage.get("stage_event_seqs", {})
        if not isinstance(stage_event_seqs, dict):
            continue
        if not _lifecycle_stage_order_is_valid(stage_event_seqs, expected_order):
            invalid[node_id] = expected_order
    return invalid


def _expected_lifecycle_stage_order(node: dict[str, Any]) -> list[str]:
    if node.get("status") != "evaluated":
        return []
    if node.get("parent_id"):
        return ["created", "materialized", "evaluated", "completed"]
    return ["materialized", "evaluated"]


def _lifecycle_stage_order_is_valid(stage_event_seqs: dict[str, Any], expected_order: list[str]) -> bool:
    previous_seq: int | None = None
    for stage in expected_order:
        seqs = stage_event_seqs.get(stage)
        if not isinstance(seqs, list) or not seqs:
            return True
        int_seqs = [seq for seq in seqs if isinstance(seq, int) and not isinstance(seq, bool)]
        if not int_seqs:
            return True
        current_seq = min(int_seqs)
        if previous_seq is not None and current_seq <= previous_seq:
            return False
        previous_seq = current_seq
    return True


def _trace_node_references(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    for key in ("node_id", "solution_id", "parent_id", "child_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            references.append((key, value))
    for key in ("node_ids", "solution_ids", "parent_ids", "child_ids"):
        value = metadata.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    references.append((key, item))
    references.extend(fanout_trace_references(metadata))
    return references


def _validate_parent_child_trace_metadata(
    event_name: str,
    metadata: dict[str, Any],
    issues: list[str],
) -> None:
    canonical_event = event_name in {
        "agenticsciml.parallel_children.start",
        "agenticsciml.parallel_children.end",
    }
    fanout_keys_present = {
        "parent_to_children",
        "parent_child_edges",
        "parent_to_child",
        "unique_parent_ids",
    }.intersection(metadata)
    if not fanout_keys_present and not canonical_event:
        return
    require_complete = (
        canonical_event
        or "parent_to_children" in metadata
        or "parent_child_edges" in metadata
    )
    issues.extend(
        FanoutTraceMetadata.validate_metadata(
            metadata,
            context=f"trace event {event_name}",
            require_complete=require_complete,
        )
    )


def _workflow_start_metadata(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        if event.get("event_type") == "workflow_span" and event.get("name") == "agenticsciml.run.start":
            metadata = event.get("metadata", {})
            return metadata if isinstance(metadata, dict) else {}
    return None


def _workflow_end_metadata(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event_type") == "workflow_span" and event.get("name") == "agenticsciml.run.end":
            metadata = event.get("metadata", {})
            return metadata if isinstance(metadata, dict) else {}
    return None


def _compare_metadata_value(
    issues: list[str],
    key: str,
    left: Any,
    right: Any,
    left_name: str,
    right_name: str,
) -> None:
    if left != right:
        issues.append(f"{key} mismatch: {left_name}={left!r}, {right_name}={right!r}")


def write_trace_summary(run_dir: Path) -> Path:
    path = run_dir / "trace_summary.json"
    path.write_text(
        json.dumps(summarize_trace(run_dir), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return path
