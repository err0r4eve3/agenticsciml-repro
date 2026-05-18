from __future__ import annotations

from pathlib import Path

from agenticsciml.strategy_inspector import inspect_solution_strategy


def test_strategy_inspector_blocks_missing_required_term_and_forbidden_call(tmp_path: Path) -> None:
    solution = tmp_path / "solution.py"
    solution.write_text(
        """
from __future__ import annotations

import torch


def train(model):
    return torch.optim.Adam(model.parameters())
""",
        encoding="utf-8",
    )

    report = inspect_solution_strategy(
        solution,
        [
            {
                "lock_id": "lock_optimizer",
                "required": True,
                "inspection": {
                    "required_terms": ["L-BFGS"],
                    "forbidden_call_names": ["Adam"],
                },
            }
        ],
    )

    assert report["inspector_version"] == "strategy_fidelity.v1"
    assert report["status"] == "blocked"
    assert report["execution_allowed"] is False
    failed = {check["category"]: check for check in report["checks"] if not check["passed"]}
    assert "required_terms" in failed
    assert "forbidden_call_names" in failed
    assert failed["forbidden_call_names"]["evidence"]["matches"] == ["torch.optim.Adam"]


def test_strategy_inspector_accepts_marker_syntax_from_lock_text(tmp_path: Path) -> None:
    solution = tmp_path / "solution.py"
    solution.write_text(
        """
from scipy.optimize import minimize


def train():
    return "LBFGS compatible strategy"
""",
        encoding="utf-8",
    )

    report = inspect_solution_strategy(
        solution,
        [
            {
                "lock_id": "lock_text",
                "text": "required_terms: L-BFGS\nforbidden_call_names: Adam",
            }
        ],
    )

    assert report["execution_allowed"] is True
    assert report["summary"]["auditable_lock_count"] == 1
    assert {check["passed"] for check in report["checks"]} == {True}


def test_strategy_inspector_keeps_unstructured_locks_nonblocking(tmp_path: Path) -> None:
    solution = tmp_path / "solution.py"
    solution.write_text("class MODEL:\n    pass\n", encoding="utf-8")

    report = inspect_solution_strategy(
        solution,
        [{"lock_id": "lock_note", "text": "Keep the mathematical intuition visible."}],
    )

    assert report["status"] == "passed"
    assert report["execution_allowed"] is True
    assert report["summary"]["auditable_lock_count"] == 0
    assert report["checks"][0]["category"] == "strategy_lock"
