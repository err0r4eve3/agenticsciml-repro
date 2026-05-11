from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


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
        "quality_gate": {
            "passed": quality_passed,
            "required_event_types": list(REQUIRED_EVENT_TYPES),
            "missing_event_types": missing_event_types,
        },
    }


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
    trace_node_reference_counts = _check_solution_artifact_consistency(
        issues,
        contract,
        run_metadata,
        tree,
        checkpoint,
        events,
    )

    return {
        "checked": True,
        "passed": not issues,
        "issues": issues,
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
            _check_solution_tree_graph_invariants(artifact_name, nodes, issues)

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
    root_ids = [node_id for node_id, node in nodes.items() if node.get("parent_id") is None]
    if len(root_ids) != 1:
        issues.append(f"{artifact_name} must have exactly one root node: roots={sorted(root_ids)!r}")

    for node_id, node in sorted(nodes.items()):
        parent_id = node.get("parent_id")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                issues.append(f"{artifact_name} node {node_id} has invalid parent_id: {parent_id!r}")
            elif parent_id not in nodes:
                issues.append(
                    f"{artifact_name} node {node_id} parent_id references missing node: {parent_id}"
                )

        if "children" not in node:
            issues.append(f"{artifact_name} node {node_id} children is missing")
            children = []
        else:
            children = node.get("children")
        if not isinstance(children, list):
            issues.append(f"{artifact_name} node {node_id} children must be a list")
            continue
        seen_children: set[str] = set()
        for child_id in children:
            if not isinstance(child_id, str) or not child_id:
                issues.append(f"{artifact_name} node {node_id} children has invalid node id: {child_id!r}")
                continue
            if child_id in seen_children:
                issues.append(
                    f"{artifact_name} node {node_id} children contains duplicate node id: {child_id}"
                )
                continue
            seen_children.add(child_id)
            child_node = nodes.get(child_id)
            if child_node is None:
                issues.append(
                    f"{artifact_name} node {node_id} children references missing node: {child_id}"
                )
                continue
            if child_node.get("parent_id") != node_id:
                issues.append(
                    f"{artifact_name} node {node_id} children includes {child_id}, "
                    f"but child parent_id is {child_node.get('parent_id')!r}"
                )

    for node_id, node in sorted(nodes.items()):
        parent_id = node.get("parent_id")
        if not isinstance(parent_id, str) or parent_id not in nodes:
            continue
        parent_children = nodes[parent_id].get("children")
        if not isinstance(parent_children, list):
            continue
        child_count = sum(1 for child_id in parent_children if child_id == node_id)
        if child_count != 1:
            issues.append(
                f"{artifact_name} node {node_id} parent children does not include node exactly once: "
                f"parent_id={parent_id}, count={child_count}"
            )

    cycle_nodes = _solution_tree_cycle_nodes(nodes)
    if cycle_nodes:
        issues.append(f"{artifact_name} parent links contain a cycle: {cycle_nodes!r}")


def _solution_tree_cycle_nodes(nodes: dict[str, dict[str, Any]]) -> list[str]:
    cycle_nodes: set[str] = set()
    for node_id in nodes:
        seen: set[str] = set()
        current_id: str | None = node_id
        while current_id is not None:
            if current_id in seen:
                cycle_nodes.update(seen)
                break
            seen.add(current_id)
            current_node = nodes.get(current_id)
            if current_node is None:
                break
            parent_id = current_node.get("parent_id")
            current_id = parent_id if isinstance(parent_id, str) and parent_id else None
    return sorted(cycle_nodes)


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
    parent_to_child = metadata.get("parent_to_child")
    if isinstance(parent_to_child, dict):
        for parent_id, child_id in parent_to_child.items():
            if isinstance(parent_id, str) and parent_id:
                references.append(("parent_to_child.parent", parent_id))
            if isinstance(child_id, str) and child_id:
                references.append(("parent_to_child.child", child_id))
    return references


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
        json.dumps(summarize_trace(run_dir), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
