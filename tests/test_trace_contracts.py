import pytest

from agenticsciml.trace_contracts import FanoutTraceMetadata


def test_fanout_trace_metadata_from_pairs_builds_canonical_payload() -> None:
    metadata = FanoutTraceMetadata.from_pairs(
        [
            ("solution_000", "solution_001"),
            ("solution_000", "solution_002"),
        ]
    )

    assert metadata.to_dict() == {
        "parent_ids": ["solution_000", "solution_000"],
        "unique_parent_ids": ["solution_000"],
        "child_ids": ["solution_001", "solution_002"],
        "parent_child_edges": [
            {"slot_index": 0, "parent_id": "solution_000", "child_id": "solution_001"},
            {"slot_index": 1, "parent_id": "solution_000", "child_id": "solution_002"},
        ],
        "parent_to_children": {"solution_000": ["solution_001", "solution_002"]},
        "parent_to_child": {"solution_000": "solution_002"},
    }


def test_fanout_trace_metadata_rejects_duplicate_child_ids() -> None:
    with pytest.raises(ValueError, match="duplicate child_id"):
        FanoutTraceMetadata.from_pairs(
            [
                ("solution_000", "solution_001"),
                ("solution_000", "solution_001"),
            ]
        )


def test_fanout_trace_metadata_validate_metadata_requires_complete_schema() -> None:
    issues = FanoutTraceMetadata.validate_metadata(
        {"parent_ids": ["solution_000"], "child_ids": ["solution_001"]},
        context="trace event agenticsciml.parallel_children.start",
        require_complete=True,
    )

    assert "trace event agenticsciml.parallel_children.start fanout metadata requires unique_parent_ids" in issues
    assert "trace event agenticsciml.parallel_children.start fanout metadata requires parent_child_edges" in issues
    assert "trace event agenticsciml.parallel_children.start fanout metadata requires parent_to_children" in issues


def test_fanout_trace_metadata_validate_metadata_catches_legacy_mismatch() -> None:
    payload = FanoutTraceMetadata.from_pairs(
        [
            ("solution_000", "solution_001"),
            ("solution_000", "solution_002"),
        ]
    ).to_dict()
    payload["parent_to_child"] = {"solution_000": "solution_001"}

    issues = FanoutTraceMetadata.validate_metadata(
        payload,
        context="trace event agenticsciml.parallel_children.start",
        require_complete=True,
    )

    assert any("legacy mapping mismatch" in issue for issue in issues)
