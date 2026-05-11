# Ablation Notes

[返回文档树](index.md) · 相关文档：[多 Agent 设计方法](multi_agent_design.md)、[版本说明](version_notes.md)

当前 ablation 是 workflow 与报告链路检查，不是论文分数声明。

## Variants

默认 variants：

1. `root_only`：只生成 root baseline，`max_iterations=0`。
2. `no_kb`：运行多 Agent mutation，但不注入 KB。
3. `kb`：运行 lexical KB retrieval。
4. `random_kb`：运行 seed-controlled random KB retrieval。
5. `no_critic`：跳过 `CriticAgent` 调用，保留 proposer final proposal。
6. `no_debugger`：跳过 debugger loop。
7. `branch_context`：`parallel_mutations=2`，开启 same-parent fanout branch context。
8. `no_branch_context`：`parallel_mutations=2`，关闭 branch context，用于后续真实 LLM smoke 对比。

`no_critic`、`no_debugger` 和 `no_branch_context` 必须映射到真实 workflow
开关，不能只在报表里改标签。

## Commands

CLI：

```bash
uv run --python 3.11 --extra dev agenticsciml ablate examples/function_approx \
  --seeds 0 1 2 \
  --variants root_only,no_kb,kb,random_kb,no_critic,no_debugger,branch_context,no_branch_context \
  --output-dir runs/ablation
```

脚本：

```bash
uv run --python 3.11 --extra dev python scripts/run_ablation.py \
  --benchmark-dir examples/function_approx \
  --seeds 0 1 2 \
  --variants root_only,no_kb,kb,random_kb,no_critic,no_debugger,branch_context,no_branch_context \
  --output-dir runs/ablation
```

## Outputs

- `ablation_runs.csv`：每个 variant/seed 的 run-level 指标。
- `ablation_summary.csv`：按 variant 聚合的 median、mean、std、best、worst、IQR、valid runs。
- `ablation_report.md`：人类可读报告。
- `runs/<variant>-seed-<seed>/`：每个实验的完整 run artifacts。

run-level 指标至少包含：

- evidence mode
- scientific claim boundary
- champion score
- root score
- champion/root improvement
- valid solution rate
- timeout count
- debug success count
- branch context count
- branch intents
- LLM call count
- wall time

## Boundary

Mock-mode ablation 只验证：

- pipeline 能否稳定运行；
- variants 是否真的改变 workflow；
- reporting 是否能聚合多个 seed；
- artifact 是否足够支持后续真实 LLM 复盘。

当前 CSV 和报告会显式标注 `evidence_mode=mock_workflow_shape` 与
`scientific_claim=not_supported`。不要把 mock-mode improvement 当作 SciML
结论，更不能把它解释成 emergent discovery。真实 LLM ablation 至少需要多
seed、固定预算、成本统计和失败样本审查。
