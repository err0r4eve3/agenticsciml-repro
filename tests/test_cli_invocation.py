from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from agenticsciml.cli import build_parser


def test_run_cli_accepts_selector_vote_count() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "examples/function_approx",
            "--mock",
            "--selector-vote-count",
            "3",
        ]
    )

    assert args.selector_vote_count == 3


def test_run_cli_defaults_to_three_selector_votes() -> None:
    parser = build_parser()

    args = parser.parse_args(["run", "examples/function_approx", "--mock"])

    assert args.selector_vote_count == 3


def test_module_cli_smoke_dry_run_is_not_real_evidence(tmp_path: Path, cli_env: dict[str, str]) -> None:
    output_dir = tmp_path / "smoke output with spaces"
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context,no_branch_context",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert Path(smoke.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_report.md"
    assert (output_dir / "real_llm_smoke_plan.json").exists()
    assert (output_dir / "real_llm_smoke_manifest.json").exists()

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-smoke-llm",
            str(output_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=cli_env,
    )
    payload = json.loads((output_dir / "real_llm_smoke_verification.json").read_text(encoding="utf-8"))

    assert verify.returncode == 1
    assert any("dry-run outputs are not real-smoke evidence" in issue for issue in payload["issues"])
    assert any("plan execution_mode must be real" in issue for issue in payload["issues"])


def test_module_cli_smoke_from_checkout_path_with_spaces(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    shutil.copytree(
        source_root / "src",
        checkout / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (checkout / "examples").mkdir()
    shutil.copytree(
        source_root / "examples" / "function_approx",
        checkout / "examples" / "function_approx",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout / "src")
    output_dir = Path("runs") / "real llm smoke"

    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context,no_branch_context",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    bundle = checkout / output_dir
    assert Path(smoke.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_report.md"
    assert (bundle / "real_llm_smoke_plan.json").exists()
    assert (bundle / "real_llm_smoke_manifest.json").exists()

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-smoke-llm",
            str(output_dir),
        ],
        cwd=checkout,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    payload = json.loads((bundle / "real_llm_smoke_verification.json").read_text(encoding="utf-8"))

    assert verify.returncode == 1
    assert any("dry-run outputs are not real-smoke evidence" in issue for issue in payload["issues"])
    assert any("plan execution_mode must be real" in issue for issue in payload["issues"])
