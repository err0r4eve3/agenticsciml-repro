# 多 Agent 设计方法

[返回文档树](index.md) · 相关文档：[论文机制笔记](paper_notes.md)、[Codex 实施任务](codex_tasks.md)

## 核心原则

多 Agent 工作流不是先堆角色，而是按这个顺序设计：

1. 先设计任务状态机。
2. 再决定哪些状态节点需要 LLM。
3. 最后才决定是否拆成多个 Agent。

对 AgenticSciML 复现，推荐架构是代码控制的多 Agent 进化搜索系统：

```text
Python orchestrator
+ fixed evaluator
+ solution tree
+ artifact-based memory
+ proposer-critic loop
+ engineer-debugger loop
+ sandboxed evaluation
```

不要实现自由群聊式 group chat。Python 负责状态、评估、排序、失败处理和文件系统；LLM 只负责判断、生成和总结。

## 什么时候使用多 Agent

适合使用多 Agent 的情况：

- 任务可以自然分解为研究检索、代码生成、实验评估、结果分析等不同职责。
- 单个 Agent 工具太多或上下文太长，容易发生工具选择错误或信息污染。
- 有统一 evaluator、测试或数值指标，可以做可验证的迭代改进。
- 需要上下文隔离，让不同角色只看到最小充分信息。

不适合使用多 Agent 的情况：

- 路径固定、输入输出简单、并行性低。
- 评估标准不清晰，只能依赖主观判断。
- 每一步都强依赖同一份完整上下文。
- 成本和延迟不能接受。

## Workflow 与 Autonomous Agent

本项目优先使用 workflow，而不是 autonomous agent。

- `workflow`：由代码预定义路径、状态转移和 gate。适合统一评估、实验复现、artifact 追踪。
- `agent`：由 LLM 动态决定流程和工具使用。适合开放式探索，但更难保证复现性。

AgenticSciML 的主流程需要固定 evaluator、solution tree、sandbox 和日志追踪，因此应由 Python orchestrator 控制。Agent 只作为局部节点处理开放式判断。

## 核心工作流模式

- `prompt chaining`：用于早期固定流水线，例如从 `Problem.md`、`Requirements.md`、`Evaluation.md` 形成 evaluation contract。
- `routing`：用于把失败或需求交给合适角色，例如训练失败交给 Debugger，策略改进交给 Proposer。
- `parallelization`：用于并行生成或评估多个 child solution；必须受 `parallel_mutations`、`timeout_s` 和预算约束。
- `orchestrator-workers`：Python orchestrator 维护全局状态，DataAnalyst、Evaluator、Retriever、Proposer、Critic、Engineer、Debugger、ResultAnalyst 作为 worker。
- `handoff`：不作为主流程模式。SciML 评估需要强控制，不能让某个 Agent 自由接管全流程。
- `evaluator-optimizer`：Engineer 生成代码，Runner 执行 validate/train/evaluate，Debugger 修复失败，ResultAnalyst 汇总结果，下一轮基于报告继续优化。
- `SOP pipeline`：每轮 mutation 固化为 parent selection、KB retrieval、proposal、critique、implementation、validation、training、evaluation、analysis、tree update。

## Agent 通信方式

本项目优先使用 artifact filesystem，辅以结构化 shared state。

- `conversation transcript`：保存每次 prompt/response，便于审计；不作为主要上下文传递方式。
- `shared state`：保存 config、solution node、score、analysis summary 和 tree metadata。
- `artifact filesystem`：每个 solution workspace 保存 `solution.py`、`proposal.md`、`train.log`、`eval.json`、`analysis.md`、transcripts 和 checkpoint。
- `agents-as-tools`：可作为未来真实 LLM 编排方式，但 manager 必须保持控制权。

每次 proposal 只应看到最小上下文：

```text
Problem 摘要
Evaluation contract
parent solution 摘要
parent analysis
最多 1 个 KB entry
少量 sibling/uncle analysis
leaderboard top-k
```

不要把完整 solution tree、所有日志或全部 KB 塞进 prompt。

## 状态与记忆

记忆分为五类：

- `global_config`：实验配置、预算、benchmark、模型配置。
- `problem_state`：`Problem.md`、`Requirements.md`、`Evaluation.md`、`Data_config.json`。
- `solution_state`：每个 solution 的代码、score、日志、分析和子节点关系。
- `knowledge_base`：可检索的论文方法、技巧和历史经验。
- `run_trace`：agent 调用、prompt、response、stdout、stderr、时间和成本。

记忆不是越多越好。每个 Agent 只能看到完成当前职责所需的最小充分信息。

## 可靠性要求

最低可靠性边界：

- `sandbox`：生成代码只在 per-solution workspace 中运行。
- `timeout`：agent call、训练和评估都要有最大时间。
- `retry`：LLM JSON 解析失败和 runtime failure 应有明确重试预算。
- `checkpoint`：每个阶段写入磁盘，避免中断后丢失上下文。
- `tracing`：保存 prompt、response、stdout、stderr、score 和 artifact。
- `guardrails`：禁止修改 evaluator、禁止删除工作区外文件、测试禁止联网、限制无限循环和资源消耗。
- `static sandbox checks`：运行 generated `solution.py` 前先阻断网络模块、子进程、危险删除操作和明显绝对路径写入。

SciML 的 evaluator 应尽量是 deterministic code，而不是 LLM judge。

## AgenticSciML 推荐架构

```text
Phase 0: Load inputs
  Problem.md / Requirements.md / Evaluation.md / Data_config.json

Phase 1: Evaluation contract
  EvaluatorAgent creates evaluate.py / guidelines.md
  Python freezes the evaluator contract

Phase 2: Root solution
  RootEngineerAgent writes solution.py
  Runner validates, trains, evaluates
  DebuggerAgent fixes runtime failures within budget
  ResultAnalystAgent writes analysis.md

Phase 3: Evolution loop
  SelectorAgent chooses parents
  RetrieverAgent selects 0 or 1 KB entry
  ProposerAgent drafts mutation
  CriticAgent challenges proposal
  EngineerAgent mutates parent solution.py
  Runner evaluates child
  DebuggerAgent fixes failures
  ResultAnalystAgent writes report
  Tree updater stores node

Phase 4: Champion export
  leaderboard.csv / tree.json / champion/solution.py / final report
```

## 新增 Agent 检查清单

新增 Agent 前必须能回答：

- 它解决的是哪个状态机节点？
- 这个节点是否真的需要开放式 LLM 判断？
- 它的 `role` 和 `non_role` 是什么？
- 输入 schema、输出 schema 和失败策略是什么？
- 它能看到哪些上下文，哪些上下文明确不能看到？
- 它是否会修改代码、文件或配置？如果会，runner 如何隔离和验证？
- 它的最大调用次数、token 预算和停止条件是什么？
- 它的输出是否可被 deterministic evaluator 或测试验证？

如果这些问题回答不清楚，不要新增 Agent；优先用普通 Python 函数或现有 Agent。

## 当前实现状态

当前代码已把设计落到基础合同层：

- `AgentSpec` 定义每个 Agent 的职责、非职责、输入输出、可见上下文、工具、预算、artifact 和失败策略。
- `PromptTemplate` 负责显式渲染 prompt，缺失字段会直接失败。
- `AgentBase.require_artifacts()` 用于检查 Agent 是否写出了约定 artifact。
- `AgentBase.require_inputs()` 将 `AgentSpec.input_schema` 接入运行时，缺少输入字段会 fail closed 并写入 guardrail trace。
- `AgentBase.complete_json_checked()` 默认使用 `AgentSpec.output_schema` 校验代码消费的 LLM 输出。
- `CriticAgent` 已从 `ProposerAgent` 中拆出，单独写入 `critic.md` 和 critic transcript。
- `ProposerAgent` 仍负责 4-round proposal flow，但 critic 调用通过独立 Agent 完成。
- `trace_summary.json` 汇总 `trace.jsonl`，用 required span types 和 guardrail failures 形成最小 trace quality gate。
- `run_metadata.json` 汇总 wall time、champion、solution count 和按 role 聚合的 LLM 调用统计。

后续增强应优先补更强 sandbox、成本统计、trace viewer 和真实 LLM ablation，而不是继续增加新 Agent。

## OpenAI Agents SDK 对齐要求

本项目不默认把主流程迁移到 OpenAI Agents SDK runtime，但每个实现步骤必须对齐 SDK 的核心设计：

- 使用 code orchestration 作为默认多 Agent 编排方式，只有明确需要专家接管用户回合时才考虑 handoff。
- Specialist agent 作为 bounded tool 使用，不能自行改变全局状态转移、score 或 champion。
- 代码消费的 LLM 输出必须使用 structured outputs：校验字段、有限 retry、失败后 fail closed。
- 每个 tool-like 边界必须有 guardrail：LLM JSON 输出、artifact 写入、生成代码执行、evaluator contract 完整性。
- generated solution 在进入 validate/train/evaluate 前必须通过静态 sandbox 检查。
- 每次运行必须产生 trace：`workflow_span`、`agent_span`、`generation_span`、`tool_span`、`guardrail_span`。
- Trace 和 artifacts 是后续 trace grading / evals 的输入，不把未验证的 LLM 总结当作实验事实。
