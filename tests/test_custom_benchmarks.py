from pathlib import Path

import pytest

from agenticsciml.custom_benchmarks import create_custom_benchmark_bundle


def _create(root: Path):
    return create_custom_benchmark_bundle(
        root,
        problem_statement="Infer a deterministic operator from sparse field observations.",
        requirements="Keep a private validation split.",
        evaluation_criteria="Use relative L2 error.",
        data_description="Small local arrays for workflow validation.",
    )


def test_repeated_custom_benchmark_intake_preserves_reviewed_evaluator(tmp_path: Path) -> None:
    first = _create(tmp_path)
    evaluator = first.benchmark_dir / "evaluate.py"
    reviewed = "# DOMAIN REVIEWED EVALUATOR\n"
    evaluator.write_text(reviewed, encoding="utf-8")

    second = _create(tmp_path)

    assert second.benchmark_dir == first.benchmark_dir
    assert evaluator.read_text(encoding="utf-8") == reviewed


def test_repeated_custom_benchmark_intake_rejects_incomplete_existing_bundle(tmp_path: Path) -> None:
    first = _create(tmp_path)
    (first.benchmark_dir / "guidelines.md").unlink()

    with pytest.raises(RuntimeError, match="incomplete.*refusing to overwrite reviewed files"):
        _create(tmp_path)
