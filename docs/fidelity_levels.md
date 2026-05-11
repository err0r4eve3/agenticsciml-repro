# Benchmark Fidelity Levels

[返回文档树](index.md) · 相关文档：[Benchmark 与实验设计](benchmark_plan.md)、[版本说明](version_notes.md)

## 目的

本项目复刻的是 AgenticSciML 的多 Agent workflow，不默认声明复现论文分数。
因此每个 benchmark 都必须显式标注 fidelity level，并把该信息写入
`EvaluationContract.benchmark_fidelity` 与 `contract_hash`。

这样单个 artifact 被单独查看时，也能知道它支持什么结论、不能支持什么结论。

## Levels

| Level | 含义 | 允许声明 |
| --- | --- | --- |
| `proxy` | NumPy / 小规模代理任务，用于验证 workflow、evaluator、artifact 和 guardrail shape。 | workflow engineering evidence |
| `faithful-small` | 使用同类 SciML 模型、损失、数据结构或 PDE/operator 目标，但缩小数据规模和训练预算。 | low-budget scientific smoke evidence |
| `paper-like` | 尽量贴近论文数据、训练预算、模型类别和指标，可用于论文趋势对比。 | paper-comparison candidate evidence |

## Claim Rules

- `proxy` 不得被描述为论文全量 SciML 复现。
- mock LLM run 的 `scientific_claim` 必须是 `not_supported`。
- real LLM + `proxy` benchmark 也只能是 `proxy_workflow_only`。
- 只有 real LLM + `faithful-small` 或 `paper-like` 才能进入科学结果讨论，但仍需要多 seed、ablation 和失败样本审查。
- 分数必须来自固定 evaluator，不允许 LLM judge 生成科学分数。

## OpenAI Agents SDK Alignment

这些规则对应 code-orchestrated workflow 的边界控制：

- Python orchestrator 固定状态转移、budget、evaluation contract 和 champion selection。
- LLM agents 只作为 bounded specialist tools，输出结构化 proposal、patch、analysis 或 concise rationale。
- `EvaluationContract` 是 tool boundary，不由 LLM 自由改写。
- fidelity metadata、evidence mode 和 scientific claim 是 run artifact 的 guardrail，不是 README 文案。

## 升级准入

从 `proxy` 升级到 `faithful-small` 前，至少需要：

- benchmark 使用同类 SciML 任务结构，例如 PINN residual、operator input/output、或 sparse reconstruction；
- evaluator 指标与论文任务同方向；
- `requires_torch`、`requires_gpu`、expected runtime 和 paper-gap notes 更新；
- `tests/test_benchmark_catalog.py` 覆盖新 fidelity metadata；
- `docs/benchmark_plan.md` 记录 runtime、依赖和仍然缺失的 paper gap；
- mock run 仍只验证 workflow shape，不提升 scientific claim。

从 `faithful-small` 升级到 `paper-like` 前，至少需要：

- 明确论文版本、数据生成方式、训练预算和指标口径；
- 多 seed 真实 LLM ablation；
- 成本、失败率、debug 成功率和代表性失败案例；
- 外部审计确认没有把 proxy 结果混入 paper-like 结论。
