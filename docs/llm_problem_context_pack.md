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

循环审计十个 Wiki 论文-现实问题案例：

```bash
PYTHONPATH=src uv run --python 3.11 --extra web --extra dev python -m agenticsciml.cli paper-problem-loop-audit \
  --output-dir runs/paper-problem-loop \
  --fail-on-issues
```

该命令复用 `/api/problem-intake/plan` 的本地 planner 和 `build_llm_problem_context_pack`，
写出 `agenticsciml_paper_problem_loop_audit.json` 与 `summary.md`。默认不联网、不调用真实 LLM、
不写 evaluator 证据。

需要刷新真实论文候选源时，先写本地 source collection cache：

```bash
PYTHONPATH=src uv run --python 3.11 --extra web --extra dev python -m agenticsciml.cli collect-paper-sources \
  --output-dir runs/paper-problem-loop
```

持续循环时加 `--repeat`，用 Ctrl-C 手动暂停；每轮写入一个 `round-*` 子目录：

```bash
PYTHONPATH=src uv run --python 3.11 --extra web --extra dev python -m agenticsciml.cli paper-problem-loop-audit \
  --output-dir runs/paper-problem-loop \
  --refresh-source-collection \
  --source-candidate-limit 3 \
  --repeat \
  --interval-s 300 \
  --fail-on-issues
```

循环会维护顶层 `paper_problem_loop_index.json` 和 `paper_problem_loop_index.md`，记录每轮 audit
路径、通过状态、issue 数、source candidate 数、prompt gate 计数和 Wiki audit 状态；JSON 给工具读取，
Markdown 给人工快速审计最新轮次。

生产巡检可直接检查最新轮次、索引摘要和 Wiki artifact 一致性：

```bash
PYTHONPATH=src uv run --python 3.11 --extra web --extra dev python -m agenticsciml.cli verify-paper-problem-loop \
  runs/paper-problem-loop \
  --max-age-s 900
```

刷新后的 source candidates 会写入每轮 `source_candidates/*/{source_review,planner,reference_matrix,llm_context_pack}.json`，
并在总审计中标记 `wiki_promotion_status=manual_review_required`。它们不会自动改写 LLM Wiki。
每轮还会写入 `llm_wiki/llm_wiki_okf.json` 与 `llm_wiki/llm_wiki_audit.json`，验证 OKF
根字段、节点字段、双语论文问题字段、边引用和手动编辑持久化边界。
审计还会写 `llm_wiki/manual_edit_roundtrip.json`，用隔离副本验证手动编辑 payload 能通过同一
Wiki 保存校验；该 roundtrip 不修改真实 account Wiki。
`prompt_quality_controls` 也是循环 fail gate：curated cases 和 source candidates 都必须保留完整
control ids，否则 `paper-problem-loop-audit --fail-on-issues` 会失败。

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
- `prompt_quality_controls`：论文上下文非权威、PDE/operator 诊断匹配、失败/风险字段显式化、
  模型路由和预算可审计、双语 Wiki 保留英文 schema/benchmark/algorithm/path 标识。
- `blockers`：缺失 problem intake 字段、专家蓝图或 CPU/GPU/timeout/dependency/data limits 时保持 blocked。

## 边界

该 pack 的用途是提升未来 real LLM 调用质量：让模型拿到明确角色、上下文、输出 schema 和禁区。
它不是无 LLM 的替代求解器，也不证明多智能体已经能解决真实问题。

`scientific_claim_supported` 固定为 `false`。只有真实 provider 调用、真实多模态输入、
paper-like benchmark、领域审批、多 seed/ablation、trace summary 和 completed run audit
全部通过后，才允许进入科学 claim 讨论。
