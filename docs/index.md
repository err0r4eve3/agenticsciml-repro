# AgenticSciML Reproduction 文档树

本文档树是项目内容的入口。新增文档应优先挂到这里，避免形成孤立页面。

## 项目入口

- [项目概览](../README.md)：MVP 目标、快速运行方式和 artifact 输出。
- [项目 Agent 规则](../AGENTS.md)：仓库工作边界、验证命令和多 Agent 设计规则。
- [版本说明](version_notes.md)：当前版本能力、验证状态和后续路线。

## 设计与依据

- [论文机制笔记](paper_notes.md)：AgenticSciML 论文中的阶段、角色、solution tree 和 evaluation contract。
- [多 Agent 设计方法](multi_agent_design.md)：什么时候使用多 Agent、状态机优先原则、通信方式、可靠性和新增 Agent 检查清单。
- [Git 与 Markdown 分层记录方法论](git_markdown_methodology.md)：把论文依据、SDK 边界、Git 时间轴、Markdown 文档树和 run artifacts 统一为可追溯记录协议。
- [OpenAI Agents SDK 升级复盘](openai_agents_sdk_upgrade_review.md)：基于 2026 年官方 SDK 文档的已对齐项、差距和版本路线。
- [Benchmark 与实验设计](benchmark_plan.md)：6 类论文任务家族、本地代理、六项 faithful-small 升级、验证矩阵和真实 LLM 实验顺序。
- [论文算法 Reference Primitives](paper_algorithm_primitives.md)：论文结果区列出的 6 个 champion strategy 的本地 NumPy reference primitives 和 claim 边界。
- [Method Substrate 合约](method_substrate.md)：ATHENA / GRAFT-ATHENA 启发的本地 action、blueprint、method fingerprint、reward 和 experience cache 合约。
- [ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md)：两篇新增参考文献的方法机制、当前项目落点和不做项。
- [Benchmark Fidelity Levels](fidelity_levels.md)：`proxy`、`faithful-small`、`paper-like` 的准入标准和 claim 边界。
- [Paper Gap Report](paper_gap_report.md)：把 benchmark fidelity metadata 与已有 run artifact 合并成 fail-closed 论文差距报告。
- [PR Split Plan](pr_split_plan.md)：把当前大 PR 收敛成 evidence、benchmark、algorithm 和 ChatUI 四个 stacked PR 的拆分边界。
- [SciML 论文与代码知识库](sciml_knowledge_base.md)：为 `burgers_pinn` 选取 PINNs 论文和公开代码，并记录本地 KB 写入规则。

## 实施与运行

- [Codex 实施任务](codex_tasks.md)：MVP 的分步实现任务和验收标准。
- [Real LLM 运行](real_run.md)：真实模型模式、环境变量、dry-run 和 trace summary 检查。
- [Ablation 说明](ablation.md)：root-only、无 KB、有 KB、随机 KB 的比较边界。
- [ChatUI 实验操作台](chatui_console.md)：本地 Web 控制台、算法 tool API、code-server sidecar 和 repo-local skill 工作流。

## 推荐阅读路径

1. 先读 [项目概览](../README.md)，确认本项目只复刻 workflow，不承诺复现论文分数。
2. 再读 [论文机制笔记](paper_notes.md)、[多 Agent 设计方法](multi_agent_design.md)、[Git 与 Markdown 分层记录方法论](git_markdown_methodology.md) 和 [OpenAI Agents SDK 升级复盘](openai_agents_sdk_upgrade_review.md)，理解为什么采用代码控制的多 Agent 进化搜索、Git/Markdown 分层记录以及后续 SDK 对齐路线。
3. 开发前读 [项目 Agent 规则](../AGENTS.md) 和 [Codex 实施任务](codex_tasks.md)。
4. 做实验时读 [Benchmark 与实验设计](benchmark_plan.md)、[论文算法 Reference Primitives](paper_algorithm_primitives.md)、[Method Substrate 合约](method_substrate.md)、[ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md)、[Benchmark Fidelity Levels](fidelity_levels.md)、[Paper Gap Report](paper_gap_report.md)、[PR Split Plan](pr_split_plan.md)、[SciML 论文与代码知识库](sciml_knowledge_base.md)、[Real LLM 运行](real_run.md)、[Ablation 说明](ablation.md)、[ChatUI 实验操作台](chatui_console.md) 和 [版本说明](version_notes.md)。
