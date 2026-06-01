# Real Problem Evidence Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed evidence closure layer that answers whether AgenticSciML can be relied on for real-world scientific problem solving without faking missing real assets.

**Architecture:** Add a deterministic closure planner on top of `paper_workflow_readiness`; it maps each required real-world module to concrete proof artifacts, classifies whether the proof can be generated locally or requires external assets, and refuses a real-problem claim until every proof is present. Expose the planner via CLI and documentation, and keep all scientific claims false unless validated completed-run artifacts exist.

**Tech Stack:** Python 3.11, `argparse`, local JSON/Markdown artifacts, `pytest`, existing `agenticsciml.paper_workflow_readiness` and `_atomic_write_text`.

---

### Task 1: Closure Planner Model And Unit Tests

**Files:**
- Create: `tests/test_real_problem_closure.py`
- Create: `src/agenticsciml/real_problem_closure.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
from pathlib import Path

from agenticsciml.real_problem_closure import build_real_problem_closure_plan


def test_real_problem_closure_blocks_missing_real_assets() -> None:
    plan = build_real_problem_closure_plan(
        benchmark_dir=Path("examples/cylinder_wake_reconstruction_faithful_small").resolve(),
        selector_panel=[],
        resource_constraints={},
        expert_blueprint_id=None,
        env={},
    )

    assert plan["status"] == "blocked"
    assert plan["multi_agent_real_problem_claim_supported"] is False
    assert "real_llm_execution" in {item["module_id"] for item in plan["blocked_modules"]}
    assert "paper_like_benchmark" in {item["module_id"] for item in plan["blocked_modules"]}
    assert "domain_approval" in {item["module_id"] for item in plan["blocked_modules"]}
    assert "scientific_claim_supported=true" not in json.dumps(plan)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_real_problem_closure.py -q
```

Expected: FAIL because `agenticsciml.real_problem_closure` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/agenticsciml/real_problem_closure.py` with a `build_real_problem_closure_plan(...)` function that calls `build_paper_workflow_readiness_bundle(...)`, maps failed readiness checks to closure modules, and always leaves `multi_agent_real_problem_claim_supported=false` unless every closure module has proof.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_real_problem_closure.py -q
```

Expected: PASS.

### Task 2: Closure Artifact Writer And CLI

**Files:**
- Modify: `tests/test_real_problem_closure.py`
- Modify: `src/agenticsciml/real_problem_closure.py`
- Modify: `src/agenticsciml/cli.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_write_real_problem_closure_plan_outputs_json_and_markdown(tmp_path: Path) -> None:
    result = write_real_problem_closure_plan(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        env={},
    )
    plan = json.loads(Path(result["paths"]["plan_json"]).read_text(encoding="utf-8"))
    assert plan["status"] == "blocked"
    assert Path(result["paths"]["plan_md"]).exists()


def test_cli_plan_real_problem_closure_can_fail_on_blockers(tmp_path: Path, cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "plan-real-problem-closure",
            "examples/function_approx",
            "--output-dir",
            str(tmp_path),
            "--fail-on-blockers",
        ],
        text=True,
        capture_output=True,
        env=cli_env,
    )
    assert result.returncode == 1
    assert (tmp_path / "real_problem_closure_plan.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_real_problem_closure.py -q
```

Expected: FAIL because writer and CLI command do not exist.

- [ ] **Step 3: Implement writer and CLI**

Add `write_real_problem_closure_plan(...)`, `render_real_problem_closure_markdown(...)`, `cmd_plan_real_problem_closure(...)`, and parser options mirroring `plan-paper-workflow`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_real_problem_closure.py tests/test_cli_invocation.py -q
```

Expected: PASS.

### Task 3: Documentation And Index

**Files:**
- Create: `docs/real_problem_evidence_closure.md`
- Modify: `docs/index.md`
- Modify: `docs/version_notes.md`

- [ ] **Step 1: Document exact claim boundary**

Add `docs/real_problem_evidence_closure.md` explaining that the closure plan is a real-world claim gate, not evidence of discovery, and listing real LLM, multimodal input, heterogeneous selector, paper-like benchmark, domain approval, paper-equivalent KB, multi-seed ablation, resource blueprint, and completed-run audit requirements.

- [ ] **Step 2: Link documentation**

Add the document to `docs/index.md` and record the new CLI/artifacts in `docs/version_notes.md`.

- [ ] **Step 3: Validate docs are reachable**

Run:

```bash
rg -n "Real Problem Evidence Closure|plan-real-problem-closure|real_problem_closure_plan" docs/index.md docs/version_notes.md docs/real_problem_evidence_closure.md
```

Expected: each file appears in the result.

### Task 4: Final Validation And Commit

**Files:**
- All files above.

- [ ] **Step 1: Run targeted checks**

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_real_problem_closure.py tests/test_paper_workflow_readiness.py tests/test_cli_invocation.py -q
PYTHONPATH=src uv run --python 3.11 --extra dev python -m compileall -q src/agenticsciml
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Run full suite**

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest -q
```

Expected: all pass.

- [ ] **Step 3: Commit and push**

```bash
git add src/agenticsciml/real_problem_closure.py src/agenticsciml/cli.py tests/test_real_problem_closure.py docs/real_problem_evidence_closure.md docs/index.md docs/version_notes.md docs/superpowers/plans/2026-06-01-real-problem-evidence-closure.md
git commit -m "Add real problem evidence closure gate"
git push
```

Expected: branch pushes to `origin/codex/chatui-workbench-redesign`.

## Self-Review

- Spec coverage: the plan covers design, local closure artifacts, CLI exposure, docs, validation, and fail-closed claim boundaries. It does not claim to supply external credentials, paper-like data, or domain approval.
- Placeholder scan: no placeholder tasks remain; every step has concrete files and commands.
- Type consistency: functions and artifact names are consistent across tests, implementation, CLI, and docs.
