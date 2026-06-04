# Paper Gap Report

[返回文档树](index.md) · 相关文档：[Benchmark Fidelity Levels](fidelity_levels.md)、[Paper Workflow Readiness](paper_workflow_readiness.md)、[版本说明](version_notes.md)

## 目的

`paper_gap_report` 把 benchmark catalog 的 fidelity metadata 与已有 run artifacts
合并成一份 fail-closed 差距报告。它用于回答：

- 当前 benchmark 与论文任务还有哪些规模、预算、数据和审批差距；
- supplied run 是否真的有 `run_metadata.json`、`trace_summary.json`、
  `scientific_result_card.json`、readiness 和 ablation evidence；
- 哪些缺口阻止结果进入 paper-score 或 scientific claim 讨论。

该报告只盘点证据，不提升任何 claim 等级。即使某个 run 有 champion 分数，只要
claim gate、readiness、trace quality 或 replication evidence 未满足，对应 gap 仍保持
blocked。

## 命令

生成单个 benchmark 的目录级 gap 报告：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli paper-gap-report \
  --benchmark-dir examples/function_approx_faithful_small \
  --output-dir runs/paper-gap-report
```

附加一个或多个 run 目录作为证据：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli paper-gap-report \
  --benchmark-dir examples/function_approx_faithful_small \
  --run-dir runs/real-20260604-120000 \
  --output-dir runs/paper-gap-report
```

使用 `--fail-on-gaps` 可把 open gaps 转成 non-zero exit code，适合用于 release 或审计门禁：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli paper-gap-report \
  --benchmark-dir examples/function_approx_faithful_small \
  --run-dir runs/real-20260604-120000 \
  --output-dir runs/paper-gap-report \
  --fail-on-gaps
```

## 输出

- `paper_gap_report.json`：机器可读报告，包含 benchmark gap items、run evidence summary、
  unmatched runs 和顶层 claim boundary。
- `paper_gap_report.md`：人类可读摘要。

每个 benchmark 至少检查：

- `benchmark_fidelity`
- `paper_benchmark_equivalence`
- `completed_run_artifacts`
- `trace_quality_gate`
- `real_llm_execution`
- `multi_seed_ablation`
- `scientific_readiness`
- `claim_gate_support`

这些检查都通过之前，报告状态保持 `blocked`。
