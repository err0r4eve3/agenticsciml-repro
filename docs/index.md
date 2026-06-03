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
- [AgenticSciML Assistant 规范](agenticsciml_assistant.md)：项目 AI 助手定位、repo-local skill 边界、内部 algorithm tool 和未来 MCP/tool 合约。
- [学长审计 Issue 留档](senior_review_issues.md)：4 个审计问题的证据、根因、修复设计、验证命令和对外答复。
- [Benchmark 与实验设计](benchmark_plan.md)：6 类论文任务家族、本地代理、六项 faithful-small 升级、验证矩阵和真实 LLM 实验顺序。
- [论文算法 Reference Primitives](paper_algorithm_primitives.md)：论文结果区列出的 6 个 champion strategy 的本地 NumPy reference primitives 和 claim 边界。
- [Method Substrate 合约](method_substrate.md)：ATHENA / GRAFT-ATHENA 启发的本地 action、blueprint、method fingerprint、reward 和 experience cache 合约。
- [ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md)：两篇新增参考文献的方法机制、当前项目落点和不做项。
- [Scientific Discovery Evidence Digest](scientific_discovery_evidence.md)：NotebookLM 论文来源、流体/PDE 证据链映射、readiness gate 和明确不做项。
- [Reference Capability Matrix](reference_capability_matrix.md)：把参考文献机制映射成本地 artifact、problem intake rubric 和 no-key implementation path。
- [LLM Problem Context Pack](llm_problem_context_pack.md)：在不调用 real LLM 的测试条件下，把 problem intake、资源边界和参考机制整理成未来真实 LLM agent 可消费的角色任务包。
- [Real Problem Evidence Closure](real_problem_evidence_closure.md)：把“能否依靠多智能体解决真实问题”的声明拆成 real LLM、真实多模态、异构 selector、paper-like benchmark、领域审批、ablation 和 completed-run audit 的 fail-closed closure plan。
- [Benchmark Fidelity Levels](fidelity_levels.md)：`proxy`、`faithful-small`、`paper-like` 的准入标准和 claim 边界。
- [SciML 论文与代码知识库](sciml_knowledge_base.md)：为 `burgers_pinn` 选取 PINNs 论文和公开代码，并记录本地 KB 写入规则。

## 实施与运行

- [Codex 实施任务](codex_tasks.md)：MVP 的分步实现任务和验收标准。
- [Real LLM 运行](real_run.md)：真实模型模式、环境变量、dry-run 和 trace summary 检查。
- [Ablation 说明](ablation.md)：root-only、无 KB、有 KB、随机 KB 的比较边界。
- [Paper Workflow Readiness](paper_workflow_readiness.md)：真实多模态、异构 selector、paper-like benchmark、领域审批、ablation 和 60 轮 campaign 的预执行门禁包。
- [ChatUI 实验操作台](chatui_console.md)：本地 Web 控制台、算法 tool API、code-server sidecar 和 repo-local skill 工作流。
- [AgenticSciML Problem Intake Prompt](agenticsciml_problem_intake_prompt.md)：把科学机器学习问题转成可审计 workflow-proxy / faithful-small 求解计划的提示模板。

## 推荐阅读路径

1. 先读 [项目概览](../README.md)，确认本项目只复刻 workflow，不承诺复现论文分数。
2. 再读 [论文机制笔记](paper_notes.md)、[多 Agent 设计方法](multi_agent_design.md)、[Git 与 Markdown 分层记录方法论](git_markdown_methodology.md)、[OpenAI Agents SDK 升级复盘](openai_agents_sdk_upgrade_review.md)、[AgenticSciML Assistant 规范](agenticsciml_assistant.md)、[学长审计 Issue 留档](senior_review_issues.md)、[Method Substrate 合约](method_substrate.md)、[ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md) 和 [Scientific Discovery Evidence Digest](scientific_discovery_evidence.md)，理解为什么采用代码控制的多 Agent 进化搜索、Git/Markdown 分层记录、Assistant 边界、审计答复、方法路径合约、科学证据门槛以及后续 SDK/tool 对齐路线。
3. 开发前读 [项目 Agent 规则](../AGENTS.md) 和 [Codex 实施任务](codex_tasks.md)。
4. 做实验时读 [Benchmark 与实验设计](benchmark_plan.md)、[论文算法 Reference Primitives](paper_algorithm_primitives.md)、[Method Substrate 合约](method_substrate.md)、[ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md)、[Scientific Discovery Evidence Digest](scientific_discovery_evidence.md)、[Reference Capability Matrix](reference_capability_matrix.md)、[LLM Problem Context Pack](llm_problem_context_pack.md)、[Real Problem Evidence Closure](real_problem_evidence_closure.md)、[Benchmark Fidelity Levels](fidelity_levels.md)、[SciML 论文与代码知识库](sciml_knowledge_base.md)、[Real LLM 运行](real_run.md)、[Ablation 说明](ablation.md)、[Paper Workflow Readiness](paper_workflow_readiness.md)、[ChatUI 实验操作台](chatui_console.md)、[AgenticSciML Assistant 规范](agenticsciml_assistant.md)、[学长审计 Issue 留档](senior_review_issues.md) 和 [版本说明](version_notes.md)。
