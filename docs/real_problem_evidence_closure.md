# Real Problem Evidence Closure

[返回文档树](index.md) · 相关文档：[Paper Workflow Readiness](paper_workflow_readiness.md)、[Scientific Discovery Evidence Digest](scientific_discovery_evidence.md)

`plan-real-problem-closure` 用于回答一个更高层的问题：当前项目是否已经能依靠多智能体解决真实科学或现实问题。该命令不会调用模型、不会读取密钥值、不会把 readiness artifact 写成结果；它只把真实问题声明所需的证据模块列成 fail-closed closure plan。

## 命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-real-problem-closure \
  examples/cylinder_wake_reconstruction_faithful_small \
  --output-dir runs/real-problem-closure \
  --selector-panel-json '[{"model":"gpt-5-mini"},{"model":"deepseek-v4-pro","base_url":"https://api.deepseek.com"}]' \
  --resource-constraints-json '{"cpu":"local","gpu":false,"timeout_s":120,"dependency_limits":["numpy"],"data_limits":"faithful-small"}' \
  --expert-blueprint-id fluid_pde
```

CI 或最终验收可加：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-real-problem-closure \
  examples/cylinder_wake_reconstruction_faithful_small \
  --output-dir runs/real-problem-closure \
  --fail-on-blockers
```

## 输出

- `real_problem_closure_plan.json`
- `real_problem_closure_plan.md`
- `paper_workflow_readiness.json`
- `paper_workflow_readiness.md`
- `domain_approval_template.json`
- `paper_benchmark_manifest_template.json`
- `paper_workflow_commands.md`

## Closure Modules

真实问题声明需要以下模块全部有 proof artifact：

- `real_llm_execution`：真实 provider、预算和 completed real run metadata。
- `real_multimodal_input`：真实 image-capable provider 实际接收图像输入。
- `heterogeneous_selector`：至少两个真实异构 provider/model 的 selector runtime votes。
- `paper_like_benchmark`：paper-equivalent 数据、evaluator 和 manifest。
- `paper_equivalent_kb`：paper/code/data provenance 与 source digest。
- `domain_approval`：领域 reviewer、review notes、checklist 和失败样本复核。
- `multi_seed_ablation`：多 seed 和 non-baseline ablation 的 verified manifest。
- `resource_blueprint`：专家蓝图和 CPU/GPU/timeout/dependency/data limits。
- `completed_run_audit`：`reports/scientific_discovery_readiness.json`、`trace_summary.json`、`run_metadata.json` 和 claim gate 共同证明 completed run。

## 边界

当前 closure plan 是“真实问题 claim gate”，不是科学结果。即使某些配置项已经填写，只要缺 real LLM、真实多模态输入、异构 real selector、paper-like benchmark、领域审批、多 seed/ablation 或 completed run audit，`multi_agent_real_problem_claim_supported` 必须保持 `false`。

能本地补齐的是计划、模板、门禁、artifact schema 和 verifier；不能本地伪造的是真实 provider 运行、真实数据等价性、领域专家审批和真实实验闭环。
