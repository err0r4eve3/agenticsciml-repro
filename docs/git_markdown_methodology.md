# Git 与 Markdown 分层记录方法论

[返回文档树](index.md) · 相关文档：[论文机制笔记](paper_notes.md)、[多 Agent 设计方法](multi_agent_design.md)、[OpenAI Agents SDK 升级复盘](openai_agents_sdk_upgrade_review.md)、[版本说明](version_notes.md)

## 目标

本项目在 AgenticSciML 论文 workflow 和 OpenAI Agents SDK 思路之上，采用一套可审计的记录方法：

- Git 记录事实变更的时间轴。
- Markdown 文档树记录人能理解的知识结构。
- Run artifacts 记录每次实验的机器证据。
- 代码和测试定义真实行为，文档解释行为边界和演化原因。

这套方法不是额外报告层，而是项目 workflow 的一部分。每次功能、benchmark、实验边界或 SDK 对齐变化，都应能从 Git commit、文档树入口和运行 artifact 三处互相追溯。

## 三个依据

### 论文依据

AgenticSciML 的核心不是单次 prompt，而是可迭代的 solution tree、固定 evaluator、parent selection、mutation、debug 和 result analysis。对应到本项目，论文机制应记录在：

- [论文机制笔记](paper_notes.md)：论文中的角色、阶段和 task family。
- [Benchmark 与实验设计](benchmark_plan.md)：本地 benchmark 如何映射论文任务。
- [Benchmark Fidelity Levels](fidelity_levels.md)：哪些结果只能说明 workflow shape，哪些才接近科学实验。

论文层文档只能声明 source-grounded 事实和明确 gap，不记录未验证的性能结论。

### SDK 依据

OpenAI Agents SDK 官方文档把 code-first orchestration、agent definitions、models/providers、guardrails、sandbox、tracing 和 evals 分成不同入口。项目采用同样边界，但保持 Python orchestrator 控制全局状态：

- SDK 对齐策略见 [OpenAI Agents SDK 升级复盘](openai_agents_sdk_upgrade_review.md)。
- Agent 设计与 bounded worker 约束见 [多 Agent 设计方法](multi_agent_design.md)。
- Real LLM provider、budget、trace 和 smoke gate 见 [Real LLM 运行](real_run.md)。

SDK 层文档只记录可落到代码、artifact 或验证命令的设计，不把术语相似当作架构符合。

### 用户方法论依据

项目知识采用 Git + Markdown 树双轨记录：

- Git commit 是不可变时间点：说明一次变更解决了什么问题，并绑定验证结果。
- Markdown 树是可导航知识图谱：说明当前系统如何理解论文、SDK、benchmark、实验和风险。
- 文档必须互链，不能形成孤立页面。
- 长期决策沉淀到稳定文档；短期运行证据留在 run directory；版本变化汇总到 [版本说明](version_notes.md)。

## 分层记录模型

| 层级 | 记录对象 | 主要位置 | 进入条件 | 不应放入 |
| --- | --- | --- | --- | --- |
| L0 代码事实 | 当前实现、测试、CLI、schema | `src/`、`tests/`、`examples/` | 行为会被执行或验证 | 设计愿望、未验证结论 |
| L1 运行证据 | prompt、response、score、trace、checkpoint、leaderboard | `runs/<experiment_id>/` | 单次 run 产生的机器证据 | 需要长期维护的架构说明 |
| L2 论文与 SDK 依据 | paper mechanism、SDK alignment、fidelity、provider boundary | `docs/paper_notes.md`、`docs/openai_agents_sdk_upgrade_review.md`、`docs/fidelity_levels.md` | 外部依据影响实现边界 | 私有凭据、长篇原文复制 |
| L3 项目方法 | 多 Agent 设计、Git/Markdown 记录、benchmark 方法 | `docs/multi_agent_design.md`、本文件、`docs/benchmark_plan.md` | 影响后续开发方式 | 一次性日志 |
| L4 版本叙事 | 已实现能力、验证命令、边界、下一步 | `docs/version_notes.md` | milestone、workflow contract、benchmark 或 CLI 改动 | 细碎 diff 复述 |
| L5 对外汇报 | 导师进度、可展示图、结论边界 | 专门的汇报文档或导出的报告 | 需要给外部读者快速理解 | 不能追溯的夸大结论 |

## Git 记录规则

- 一个 commit 应对应一个可解释的概念变化：功能、benchmark、文档治理、实验边界或修复。
- 大变更应同时更新相关文档入口，避免代码已经改变但文档树仍停留在旧架构。
- Commit message 要说明变更性质，不要只写 "update docs" 这类无法追溯的描述。
- 提交前检查 `git status --short`，不要把 run artifacts、缓存、密钥、私有数据或无关改动混入。
- 如果工作树里存在其他人的未提交改动，只暂存本次任务相关文件。
- 推送后以 commit URL 作为汇报锚点；版本叙事以 [版本说明](version_notes.md) 为入口。

## Markdown 树规则

- 所有长期文档都必须能从 [docs/index.md](index.md) 到达。
- 每个文档顶部保留返回文档树和相关文档链接。
- 新增文档前先判断它属于 L2、L3、L4 还是 L5；不要把方法、运行日志和对外汇报混在一个页面。
- 文档使用中文为主，代码符号、CLI、路径、API 名称保留英文。
- 文档只描述当前已验证能力、明确假设和剩余 gap；不把 mock 结果写成科学发现。
- 外部论文、SDK、代码仓库只摘录短摘要、版本、链接和影响边界，不复制大段原文或源码。

## 与 Agent 工作流的结合

每次 AgenticSciML run 形成三类可追溯对象：

```text
Git commit
  -> docs/index.md reachable explanation
  -> docs/version_notes.md milestone summary
  -> runs/<experiment_id>/ machine evidence
```

Agent 只消费最小充分上下文：

- 论文和 SDK 依据通过短文档摘要进入 prompt。
- KB 条目通过 Retriever 进入单个 mutation。
- run artifact 作为事实来源，不把聊天历史当事实来源。
- ResultAnalyst 只能总结 `eval.json`、`train.log`、`analysis.md`、trace 和 score，不能自行改变 benchmark claim boundary。

## 变更流程

1. 先确定变更属于哪一层。
2. 修改代码或文档时保持层级清楚。
3. 更新 `docs/index.md`，让新文档可达。
4. 如果改变 workflow contract、benchmark、CLI、SDK 对齐或 claim boundary，同步更新 [版本说明](version_notes.md)。
5. 运行最小相关验证。
6. 提交并推送；最终回复给出 commit、验证命令和未验证风险。

## 当前适用边界

- 本方法论约束项目记录方式，不改变 evaluator、score、champion selection 的 trusted Python path。
- Git 与 Markdown 树是知识治理层，不替代测试、trace summary、real LLM smoke gate 或 benchmark fidelity gate。
- 对外汇报必须从 L0-L4 取证，不能单独在 L5 写无法追溯的结论。

## 外部参考

- [OpenAI Agents SDK starting points](https://developers.openai.com/api/docs/guides/agents#choose-your-starting-point)：官方把 code-first app、agent definitions、models/providers、running agents、sandbox、orchestration/handoffs、guardrails、tools/tracing 和 evals 分成独立入口。
- [OpenAI SDKs and Agents SDK](https://developers.openai.com/api/docs/libraries#use-the-agents-sdk)：官方说明 Agents SDK 适用于 code-first orchestration、tools、handoffs、guardrails、tracing 和 sandbox execution。
