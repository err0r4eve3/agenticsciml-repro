# LLM Problem Context Pack

[返回文档树](index.md) · 相关文档：[Reference Capability Matrix](reference_capability_matrix.md)、[Paper Workflow Readiness](paper_workflow_readiness.md)、[Real LLM 运行](real_run.md)

`build-llm-problem-context` 是面向真实 LLM 执行链的离线上下文准备工具。它不替代 LLM，
也不模拟 LLM 结果；它把 problem intake、专家蓝图、资源限制和 reference capability matrix
整理成未来 real provider 可直接消费的角色任务包。

## 命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli build-llm-problem-context \
  --problem-intake-json planning/problem_intake.json \
  --resource-constraints-json '{"cpu":"local","gpu":false,"timeout_s":120,"dependency_limits":["numpy"],"data_limits":"faithful-small"}' \
  --expert-blueprint-id fluid_pde \
  --output-dir runs/llm-problem-context
```

输出：

- `llm_problem_context_pack.json`
- `llm_problem_context_pack.md`

`plan-paper-workflow`、`plan-real-problem-closure` 和 `plan-iteration-campaign` 也会把同一份
`llm_problem_context_pack` 嵌入各自 artifact。

## Pack 内容

pack 固定记录：

- `problem_decomposition`：hypothesis、observable、metric、failure modes、physical constraints、
  domain review checklist 和 data source。
- `role_task_plan`：`data_analyst`、`root_engineer`、`proposer`、`critic`、`engineer`、
  `debugger`、`selector`、`result_analyst` 和 `visual_audit` 的输入、输出 schema、禁止动作、
  停止条件、证据 artifact 和默认 `reasoning_effort`。
- `orchestrator_owned_decisions`：evaluation scoring、champion selection、selector eligibility、
  sandbox execution、artifact writes、trace summary、claim gate 和 budget enforcement。
- `llm_owned_judgments`：方法假设映射、受控代码生成、批判/修复诊断、真实图像输入可用时的视觉解释、
  artifact-grounded rationale summary。
- `blockers`：缺失 problem intake 字段、专家蓝图或 CPU/GPU/timeout/dependency/data limits 时保持 blocked。

## 边界

该 pack 的用途是提升未来 real LLM 调用质量：让模型拿到明确角色、上下文、输出 schema 和禁区。
它不是无 LLM 的替代求解器，也不证明多智能体已经能解决真实问题。

`scientific_claim_supported` 固定为 `false`。只有真实 provider 调用、真实多模态输入、
paper-like benchmark、领域审批、多 seed/ablation、trace summary 和 completed run audit
全部通过后，才允许进入科学 claim 讨论。
