# Paper Workflow Readiness

[返回文档树](index.md) · 相关文档：[Scientific Discovery Evidence Digest](scientific_discovery_evidence.md)、[Real LLM 运行](real_run.md)、[Ablation 说明](ablation.md)

`plan-paper-workflow` 用于把真实科学证据链的剩余缺口变成一个可审计执行包。它不调用模型、
不读取密钥值、不升级任何 claim；只检查当前环境和本地证据是否已经满足进入
`paper_workflow` 运行的最低条件。

## 命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-paper-workflow \
  examples/cylinder_wake_reconstruction_faithful_small \
  --output-dir runs/paper-workflow-readiness \
  --selector-panel-json '[{"model":"gpt-5-mini"},{"model":"deepseek-v4-pro","base_url":"https://api.deepseek.com"}]' \
  --resource-constraints-json '{"cpu":"local","gpu":false,"timeout_s":120,"dependency_limits":["numpy"],"data_limits":"faithful-small"}' \
  --expert-blueprint-id fluid_pde
```

可选输入：

- `--domain-approval-json <path>`：领域审核 packet。
- `--ablation-output-dir <path>`：已有 ablation 输出目录。
- `--expected-seeds 0 1 2`：期望 seed 覆盖。
- `--expected-variants root_only,kb,random_kb`：期望 variant 覆盖。
- `--fail-on-blockers`：生成 artifact 后，如果仍 blocked 则返回非零。

## 输出

- `paper_workflow_readiness.json`
- `paper_workflow_readiness.md`
- `domain_approval_template.json`
- `paper_benchmark_manifest_template.json`
- `paper_workflow_commands.md`

## Gate

当前 gate 同时检查：

- `OPENAI_API_KEY` 是否存在，但不记录值。
- LLM 预算环境变量是否至少配置一项。
- provider capability 是否支持真实 image input。
- selector panel 是否至少有两个真实异构 provider/model candidate。
- benchmark 是否是 `paper-like` 且存在完整 `paper_benchmark_manifest.json`。
- KB 是否标记 `paper_kb_equivalent=true`。
- domain approval packet 是否通过固定 checklist。
- ablation 输出是否经 `ablation_runs.csv` / `ablation_summary.csv` 验证。
- `resource_constraints` 和 `expert_blueprint_id` 是否完整。

## 边界

该命令是预执行门禁，不是运行证据。即使所有预执行 gate 通过，也必须完成真实 run，
并由 `reports/scientific_discovery_readiness.json`、`trace_summary.json`、selector votes、
visual audit、domain approval、ablation evidence 和 paper-like benchmark dossier 共同
通过，才允许讨论 paper workflow 级别 claim。默认仍保持
`scientific_claim_supported=false`。
