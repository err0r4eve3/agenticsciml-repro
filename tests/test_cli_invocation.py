from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
