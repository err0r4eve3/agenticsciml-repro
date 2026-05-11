from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ParentChildEdge:
    slot_index: int
    parent_id: str
    child_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "parent_id": self.parent_id,
            "child_id": self.child_id,
        }


@dataclass(frozen=True)
class FanoutTraceMetadata:
    parent_ids: list[str]
    child_ids: list[str]
    unique_parent_ids: list[str]
    parent_child_edges: list[ParentChildEdge]
    parent_to_children: dict[str, list[str]]
    parent_to_child: dict[str, str]

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[str, str]]) -> FanoutTraceMetadata:
        if not pairs:
            raise ValueError("FanoutTraceMetadata requires at least one parent/child pair")
        parent_ids: list[str] = []
        child_ids: list[str] = []
        edges: list[ParentChildEdge] = []
        for index, (parent_id, child_id) in enumerate(pairs):
            _require_non_empty_string(parent_id, f"pair[{index}].parent_id")
            _require_non_empty_string(child_id, f"pair[{index}].child_id")
            parent_ids.append(parent_id)
            child_ids.append(child_id)
            edges.append(ParentChildEdge(slot_index=index, parent_id=parent_id, child_id=child_id))
        issues = _validate_canonical_parts(
            parent_ids=parent_ids,
            child_ids=child_ids,
            unique_parent_ids=list(dict.fromkeys(parent_ids)),
            parent_child_edges=edges,
            parent_to_children=_parent_to_children_from_edges(edges),
            parent_to_child={},
            parent_to_child_present=False,
            context="FanoutTraceMetadata.from_pairs",
        )
        if issues:
            raise ValueError("; ".join(issues))
        parent_to_children = _parent_to_children_from_edges(edges)
        return cls(
            parent_ids=parent_ids,
            child_ids=child_ids,
            unique_parent_ids=list(dict.fromkeys(parent_ids)),
            parent_child_edges=edges,
            parent_to_children=parent_to_children,
            parent_to_child={
                parent_id: child_ids_for_parent[-1]
                for parent_id, child_ids_for_parent in parent_to_children.items()
            },
        )

    @classmethod
    def validate_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        context: str,
        require_complete: bool,
    ) -> list[str]:
        required_keys = (
            "parent_ids",
            "child_ids",
            "unique_parent_ids",
            "parent_child_edges",
            "parent_to_children",
        )
        issues: list[str] = []
        if require_complete:
            for key in required_keys:
                if key not in metadata:
                    issues.append(f"{context} fanout metadata requires {key}")
        if not require_complete and not any(key in metadata for key in required_keys + ("parent_to_child",)):
            return []

        parent_ids = _string_list_metadata(metadata, "parent_ids", context, issues)
        child_ids = _string_list_metadata(metadata, "child_ids", context, issues)
        unique_parent_ids = _string_list_metadata(metadata, "unique_parent_ids", context, issues)
        edges = _parent_child_edges_metadata(metadata, context, issues)
        parent_to_children = _parent_to_children_metadata(metadata, context, issues)
        parent_to_child = _parent_to_child_metadata(metadata, context, issues)

        if unique_parent_ids is not None:
            if len(unique_parent_ids) != len(set(unique_parent_ids)):
                issues.append(f"{context} unique_parent_ids contains duplicates")
            if parent_ids is not None:
                expected_unique_parent_ids = list(dict.fromkeys(parent_ids))
                if unique_parent_ids != expected_unique_parent_ids:
                    issues.append(
                        f"{context} unique_parent_ids mismatch: "
                        f"expected {expected_unique_parent_ids!r}, observed {unique_parent_ids!r}"
                    )

        if parent_to_child is not None and parent_to_children is None:
            issues.append(f"{context} parent_to_child requires canonical parent_to_children metadata")

        if parent_ids is None or child_ids is None or unique_parent_ids is None:
            return issues
        if edges is None or parent_to_children is None:
            return issues
        issues.extend(
            _validate_canonical_parts(
                parent_ids=parent_ids,
                child_ids=child_ids,
                unique_parent_ids=unique_parent_ids,
                parent_child_edges=edges,
                parent_to_children=parent_to_children,
                parent_to_child=parent_to_child or {},
                parent_to_child_present=parent_to_child is not None,
                context=context,
            )
        )
        return issues

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], *, context: str) -> FanoutTraceMetadata:
        issues = cls.validate_metadata(metadata, context=context, require_complete=True)
        if issues:
            raise ValueError("; ".join(issues))
        edges = _parent_child_edges_metadata(metadata, context, [])
        parent_ids = _string_list_metadata(metadata, "parent_ids", context, [])
        child_ids = _string_list_metadata(metadata, "child_ids", context, [])
        unique_parent_ids = _string_list_metadata(metadata, "unique_parent_ids", context, [])
        parent_to_children = _parent_to_children_metadata(metadata, context, [])
        parent_to_child = _parent_to_child_metadata(metadata, context, []) or {}
        if edges is None or parent_ids is None or child_ids is None or unique_parent_ids is None:
            raise ValueError(f"{context} missing canonical fanout trace metadata")
        if parent_to_children is None:
            raise ValueError(f"{context} missing parent_to_children")
        return cls(
            parent_ids=parent_ids,
            child_ids=child_ids,
            unique_parent_ids=unique_parent_ids,
            parent_child_edges=edges,
            parent_to_children=parent_to_children,
            parent_to_child=parent_to_child,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_ids": list(self.parent_ids),
            "unique_parent_ids": list(self.unique_parent_ids),
            "child_ids": list(self.child_ids),
            "parent_child_edges": [edge.to_dict() for edge in self.parent_child_edges],
            "parent_to_children": {
                parent_id: list(child_ids)
                for parent_id, child_ids in self.parent_to_children.items()
            },
            "parent_to_child": dict(self.parent_to_child),
        }


def fanout_trace_references(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    parent_to_child = metadata.get("parent_to_child")
    if isinstance(parent_to_child, dict):
        for parent_id, child_id in parent_to_child.items():
            if isinstance(parent_id, str) and parent_id:
                references.append(("parent_to_child.parent", parent_id))
            if isinstance(child_id, str) and child_id:
                references.append(("parent_to_child.child", child_id))
    parent_to_children = metadata.get("parent_to_children")
    if isinstance(parent_to_children, dict):
        for parent_id, child_ids in parent_to_children.items():
            if isinstance(parent_id, str) and parent_id:
                references.append(("parent_to_children.parent", parent_id))
            if isinstance(child_ids, list):
                for child_id in child_ids:
                    if isinstance(child_id, str) and child_id:
                        references.append(("parent_to_children.child", child_id))
    parent_child_edges = metadata.get("parent_child_edges")
    if isinstance(parent_child_edges, list):
        for edge in parent_child_edges:
            if not isinstance(edge, dict):
                continue
            parent_id = edge.get("parent_id")
            child_id = edge.get("child_id")
            if isinstance(parent_id, str) and parent_id:
                references.append(("parent_child_edges.parent", parent_id))
            if isinstance(child_id, str) and child_id:
                references.append(("parent_child_edges.child", child_id))
    return references


def _validate_canonical_parts(
    *,
    parent_ids: list[str],
    child_ids: list[str],
    unique_parent_ids: list[str],
    parent_child_edges: list[ParentChildEdge],
    parent_to_children: dict[str, list[str]],
    parent_to_child: dict[str, str],
    parent_to_child_present: bool,
    context: str,
) -> list[str]:
    issues: list[str] = []
    edge_parent_ids = [edge.parent_id for edge in parent_child_edges]
    edge_child_ids = [edge.child_id for edge in parent_child_edges]
    edge_slot_indexes = [edge.slot_index for edge in parent_child_edges]
    if edge_slot_indexes != list(range(len(parent_child_edges))):
        issues.append(
            f"{context} parent_child_edges slot indexes must be contiguous from zero: "
            f"observed {edge_slot_indexes!r}"
        )
    if len(edge_child_ids) != len(set(edge_child_ids)):
        issues.append(f"{context} parent_child_edges has duplicate child_id")
    if edge_parent_ids != parent_ids:
        issues.append(
            f"{context} parent_child_edges parent order mismatch: "
            f"expected {parent_ids!r}, observed {edge_parent_ids!r}"
        )
    if edge_child_ids != child_ids:
        issues.append(
            f"{context} parent_child_edges child order mismatch: "
            f"expected {child_ids!r}, observed {edge_child_ids!r}"
        )
    expected_unique_parent_ids = list(dict.fromkeys(parent_ids))
    if unique_parent_ids != expected_unique_parent_ids:
        issues.append(
            f"{context} unique_parent_ids mismatch: "
            f"expected {expected_unique_parent_ids!r}, observed {unique_parent_ids!r}"
        )
    derived_parent_to_children = _parent_to_children_from_edges(parent_child_edges)
    if parent_to_children != derived_parent_to_children:
        issues.append(
            f"{context} parent_to_children mismatch: "
            f"expected {derived_parent_to_children!r}, observed {parent_to_children!r}"
        )
    if [child_id for child_ids_for_parent in parent_to_children.values() for child_id in child_ids_for_parent] != child_ids:
        issues.append(
            f"{context} parent_to_children child order mismatch: "
            f"expected {child_ids!r}, observed {parent_to_children!r}"
        )
    if parent_to_child_present:
        for parent_id, child_ids_for_parent in parent_to_children.items():
            expected_child_id = child_ids_for_parent[-1]
            observed_child_id = parent_to_child.get(parent_id)
            if observed_child_id != expected_child_id:
                issues.append(
                    f"{context} parent_to_child legacy mapping mismatch for {parent_id}: "
                    f"expected {expected_child_id!r}, observed {observed_child_id!r}"
                )
    return issues


def _string_list_metadata(
    metadata: dict[str, Any],
    key: str,
    context: str,
    issues: list[str],
) -> list[str] | None:
    if key not in metadata:
        return None
    value = metadata.get(key)
    if not isinstance(value, list):
        issues.append(f"{context} {key} must be a list")
        return None
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            issues.append(f"{context} {key}[{index}] must be a non-empty string")
            continue
        result.append(item)
    return result


def _parent_child_edges_metadata(
    metadata: dict[str, Any],
    context: str,
    issues: list[str],
) -> list[ParentChildEdge] | None:
    if "parent_child_edges" not in metadata:
        return None
    value = metadata.get("parent_child_edges")
    if not isinstance(value, list):
        issues.append(f"{context} parent_child_edges must be a list")
        return None
    edges: list[ParentChildEdge] = []
    slot_indexes: set[int] = set()
    child_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            issues.append(f"{context} parent_child_edges[{index}] must be an object")
            continue
        slot_index = item.get("slot_index")
        parent_id = item.get("parent_id")
        child_id = item.get("child_id")
        if not isinstance(slot_index, int) or isinstance(slot_index, bool) or slot_index < 0:
            issues.append(f"{context} parent_child_edges[{index}].slot_index must be a non-negative integer")
            continue
        if slot_index in slot_indexes:
            issues.append(f"{context} parent_child_edges has duplicate slot_index {slot_index}")
        slot_indexes.add(slot_index)
        if slot_index != index:
            issues.append(f"{context} parent_child_edges[{index}].slot_index must equal its list index {index}")
        if not isinstance(parent_id, str) or not parent_id:
            issues.append(f"{context} parent_child_edges[{index}].parent_id is invalid")
            continue
        if not isinstance(child_id, str) or not child_id:
            issues.append(f"{context} parent_child_edges[{index}].child_id is invalid")
            continue
        if child_id in child_ids:
            issues.append(f"{context} parent_child_edges has duplicate child_id {child_id}")
        child_ids.add(child_id)
        edges.append(ParentChildEdge(slot_index=slot_index, parent_id=parent_id, child_id=child_id))
    return edges


def _parent_to_children_metadata(
    metadata: dict[str, Any],
    context: str,
    issues: list[str],
) -> dict[str, list[str]] | None:
    if "parent_to_children" not in metadata:
        return None
    value = metadata.get("parent_to_children")
    if not isinstance(value, dict):
        issues.append(f"{context} parent_to_children must be an object")
        return None
    result: dict[str, list[str]] = {}
    seen_child_ids: set[str] = set()
    for parent_id, child_ids in value.items():
        if not isinstance(parent_id, str) or not parent_id:
            issues.append(f"{context} parent_to_children contains invalid parent_id")
            continue
        if not isinstance(child_ids, list):
            issues.append(f"{context} parent_to_children[{parent_id!r}] must be a list")
            continue
        parsed_child_ids: list[str] = []
        for index, child_id in enumerate(child_ids):
            if not isinstance(child_id, str) or not child_id:
                issues.append(
                    f"{context} parent_to_children[{parent_id!r}][{index}] "
                    "must be a non-empty string"
                )
                continue
            if child_id in seen_child_ids:
                issues.append(f"{context} parent_to_children has duplicate child_id {child_id}")
            seen_child_ids.add(child_id)
            parsed_child_ids.append(child_id)
        if not parsed_child_ids:
            issues.append(f"{context} parent_to_children[{parent_id!r}] must not be empty")
        result[parent_id] = parsed_child_ids
    return result


def _parent_to_child_metadata(
    metadata: dict[str, Any],
    context: str,
    issues: list[str],
) -> dict[str, str] | None:
    if "parent_to_child" not in metadata:
        return None
    value = metadata.get("parent_to_child")
    if not isinstance(value, dict):
        issues.append(f"{context} parent_to_child must be an object")
        return None
    result: dict[str, str] = {}
    for parent_id, child_id in value.items():
        if not isinstance(parent_id, str) or not parent_id:
            issues.append(f"{context} parent_to_child contains invalid parent_id")
            continue
        if not isinstance(child_id, str) or not child_id:
            issues.append(f"{context} parent_to_child[{parent_id!r}] must be a non-empty string")
            continue
        result[parent_id] = child_id
    return result


def _parent_to_children_from_edges(edges: Sequence[ParentChildEdge]) -> dict[str, list[str]]:
    parent_to_children: dict[str, list[str]] = {}
    for edge in edges:
        parent_to_children.setdefault(edge.parent_id, []).append(edge.child_id)
    return parent_to_children


def _require_non_empty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
