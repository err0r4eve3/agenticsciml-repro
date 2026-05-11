import json
import subprocess
import sys
from pathlib import Path

import pytest

from agenticsciml.llm_smoke import run_llm_smoke


def test_llm_smoke_dry_run_writes_plan_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=True,
    )
    plan = json.loads(result.plan_json.read_text(encoding="utf-8"))
    report = result.report_md.read_text(encoding="utf-8")

    assert result.runs_csv is None
    assert plan["evidence_mode"] == "real_llm_smoke"
    assert plan["scientific_claim"] == "not_supported"
    assert [entry["variant"] for entry in plan["runs"]] == ["branch_context", "no_branch_context"]
    assert plan["runs"][0]["use_branch_context"] is True
    assert plan["runs"][1]["use_branch_context"] is False
    assert "API calls: none" in report


def test_llm_smoke_real_mode_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY or --dry-run"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context"],
            dry_run=False,
        )


def test_cli_smoke_llm_dry_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = {**cli_env}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context,no_branch_context",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    report_path = Path(result.stdout.strip().splitlines()[-1])
    assert report_path.name == "real_llm_smoke_report.md"
    assert (tmp_path / "real_llm_smoke_plan.json").exists()
