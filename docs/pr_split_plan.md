# PR Split Plan

[返回文档树](index.md) · 相关文档：[Git 与 Markdown 分层记录方法论](git_markdown_methodology.md)、[Benchmark Fidelity Levels](fidelity_levels.md)、[Paper Gap Report](paper_gap_report.md)

## 目的

`codex/chatui-workbench-redesign` 已经覆盖 evidence/reporting、benchmark、algorithm
catalog/reference primitives 和 ChatUI/web hardening。继续把功能累到同一个 PR 会让
review、回归定位和 claim-boundary 审计变得困难。

拆分目标不是改变功能，而是把当前大 PR 收敛成可独立 review 的 stacked PR。每个切片都必须
保留 fail-closed 科学声明边界，并在合并前跑对应验证命令。

## 建议分支栈

| 顺序 | 分支 | Base | 主题 | 合并条件 |
| --- | --- | --- | --- | --- |
| 1 | `codex/split-evidence-reporting` | `main` | evidence/reporting gates | 证据门控测试和 full pytest 通过 |
| 2 | `codex/split-benchmark-catalog` | `codex/split-evidence-reporting` | benchmark catalog extension | catalog、execution 和 benchmark docs 通过 |
| 3 | `codex/split-algorithm-primitives` | `codex/split-benchmark-catalog` | algorithm catalog/reference primitives | paper algorithms、method substrate、operator policy 测试通过 |
| 4 | `codex/split-chatui-workbench` | `codex/split-algorithm-primitives` | ChatUI / web workspace hardening | web API、frontend build 和 smoke checks 通过 |

使用 stacked PR 而不是四个并行 PR，是因为当前 `orchestrator.py`、`cli.py`、`web/app.py`
中存在跨功能集成点。硬按文件切并行 PR 容易得到不可运行分支。

## Slice 1: Evidence / Reporting Gates

范围：

- `agenticsciml.evidence`、`readiness`、`scientific_readiness`
- `paper_workflow_readiness`、`real_problem_closure`、`paper_gap_report`
- `ablation_evidence`、`selector_evidence`
- `audit_reports` 中的 innovation / scientific result card
- `trace_summary` 的 overclaim / artifact consistency checks
- GitHub Actions CI 的 fast evidence gate 和 full pytest gate

排除：

- frontend UI redesign
- new benchmark task bundles
- paper champion reference primitives
- free-form ChatUI agent behavior changes

验证：

```bash
uv run --frozen --python 3.11 --extra dev pytest -q \
  tests/test_evidence.py \
  tests/test_ablation_evidence.py \
  tests/test_paper_gap_report.py \
  tests/test_paper_workflow_readiness.py \
  tests/test_real_problem_closure.py \
  tests/test_selector_evidence.py \
  tests/test_trace_reporting.py

uv run --frozen --python 3.11 --extra dev pytest -q
```

## Slice 2: Benchmark Catalog Extension

范围：

- `examples/cylinder_wake_reconstruction_faithful_small/`
- `src/agenticsciml/benchmarks.py` catalog registration
- benchmark docs and catalog tests

排除：

- ChatUI pages for selecting the benchmark
- algorithm reference primitives unless needed only as catalog metadata
- scientific claim upgrades

验证：

```bash
uv run --frozen --python 3.11 --extra dev pytest -q \
  tests/test_benchmark_catalog.py \
  tests/test_execution.py

uv run --frozen --python 3.11 --extra dev python -m agenticsciml.cli benchmarks
```

## Slice 3: Algorithm Catalog / Reference Primitives

范围：

- `algorithm_catalog`
- `paper_algorithms`
- `paper_tasks`
- `method_substrate`
- `method_templates`
- `operator_scheduler`
- `strategy_inspector`
- docs that describe reference primitives and method contracts

排除：

- UI layout changes
- real LLM run results
- paper-score claims

验证：

```bash
uv run --frozen --python 3.11 --extra dev pytest -q \
  tests/test_paper_algorithms.py \
  tests/test_method_substrate.py \
  tests/test_method_templates.py \
  tests/test_operator_scheduler.py \
  tests/test_strategy_inspector.py \
  tests/test_search_policy.py
```

## Slice 4: ChatUI / Web Workspace Hardening

范围：

- `frontend/`
- `src/agenticsciml/web/app.py`
- `scripts/web_workflow_smoke.py`
- `scripts/web_ui_dispatch_e2e.mjs`
- `.agents/skills/agenticsciml-chatui-operator/`
- ChatUI docs and README UI sections

排除：

- evaluator contract changes not required by web API
- benchmark score claims
- real LLM ablation conclusions

验证：

```bash
uv run --frozen --python 3.11 --extra web --extra dev pytest -q \
  tests/test_web_api.py \
  tests/test_cli_invocation.py

npm --prefix frontend run build
```

Live smoke stays optional and should not replace tests:

```bash
uv run --python 3.11 --extra web --extra dev python scripts/web_workflow_smoke.py \
  --base-url http://127.0.0.1:8765 \
  --skip-code-server-live
```

## 操作规则

- 不再向 `codex/chatui-workbench-redesign` 追加无关功能。
- 每个 split PR 只提交一个主题，并在 PR body 写清 base branch、included scope、excluded scope 和验证命令。
- 如果某个切片必须带少量 shared plumbing，PR body 必须解释为什么该 plumbing 是当前切片运行所需。
- 原大 PR 只作为集成参考；不要直接合并，除非四个切片都已合并或明确决定放弃拆分。
- real LLM、多 seed ablation、paper-like benchmark 和 OS-level sandbox 后续单独开 PR，不混进这四个收敛 PR。
