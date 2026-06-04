# 多 Agent 设计方法

[返回文档树](index.md) · 相关文档：[论文机制笔记](paper_notes.md)、[Git 与 Markdown 分层记录方法论](git_markdown_methodology.md)、[Codex 实施任务](codex_tasks.md)

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
- `evaluator-optimizer`：Engineer 生成代码，Runner 执行 validate/train/predict/evaluate，Debugger 修复失败，ResultAnalyst 汇总结果，下一轮基于报告继续优化。
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

## Git 与 Markdown 记录层

多 Agent 工作流的长期记忆不靠聊天历史，而靠三层记录互相校验：

- Git commit：记录一次 scoped 变更和验证结果，是项目演化时间轴。
- Markdown 文档树：记录论文依据、SDK 边界、benchmark fidelity、架构方法和版本叙事，是人可导航的知识图谱。
- Run artifacts：记录 prompt、response、score、trace、checkpoint 和 leaderboard，是单次实验机器证据。

AgenticSciML 的每次架构或实验边界变化，都应能从 `docs/index.md` 找到解释，从 Git commit 找到变更点，从 run directory 或测试命令找到证据。具体规则见 [Git 与 Markdown 分层记录方法论](git_markdown_methodology.md)。

## 可靠性要求

最低可靠性边界：

- `sandbox`：生成代码只在 per-solution workspace 中运行。
- `timeout`：agent call、训练和评估都要有最大时间。
- `retry`：LLM JSON 解析失败和 runtime failure 应有明确重试预算。
- `checkpoint`：每个阶段写入磁盘，避免中断后丢失上下文。
- `tracing`：保存 prompt、response、stdout、stderr、score 和 artifact。
- `guardrails`：禁止修改 evaluator、禁止删除工作区外文件、测试禁止联网、限制无限循环和资源消耗。
- `static sandbox checks`：运行 generated `solution.py` 前先阻断网络模块、子进程、危险删除操作和明显绝对路径写入。
- `strategy fidelity inspector`：若 run 带有可机器审计的 manual strategy locks，
  在执行 generated `solution.py` 前检查 required / forbidden terms、imports 和 calls，
  并把结果写入 `solutions/<id>/policy_fidelity_report.json` 与 guardrail trace。

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
- `DataAnalystAgent` 会生成 `reports/data_observations.json` 和 `reports/data_overview.svg`，
  以及可复跑的 `reports/data_eda.py` / `reports/data_eda.json`。prompt 只消费训练数据观察
  和 replayable EDA 摘要，不接触 private validation labels。
- `ResultAnalystAgent` 会生成 `solution_observations.json` 和
  `prediction_overview.svg`，prompt 只消费 `predict_input.npz`、`predictions.npz`、
  `eval.json` 和日志摘要，保持 prediction-only 边界。
- `CriticAgent` 已从 `ProposerAgent` 中拆出，单独写入 `critic.md` 和 critic transcript。
- `ProposerAgent` 仍负责 4-round proposal flow，但 critic 调用通过独立 Agent 完成。
- `RootEngineerAgent` 和 `EngineerAgent` 的 prompt 已包含 benchmark、contract、guidelines 和分析上下文，避免脱离评估契约生成代码。
- `EngineerAgent` 使用 `parent_digest` 校验和 unified diff patch / file map 应用，Python 端负责落盘，避免无校验整文件替换。
- `DebuggerAgent` 使用 contract-aware minimal repair：prompt 包含当前代码、失败阶段和错误日志，输出必须带 `parent_digest` 和 unified diff patch，Python 端拒绝 wrong digest、malformed patch 和非 `solution.py` 改动。
- Parent selection 分两层：最低 loss 的 best available node 总是进入下一轮作为 exploitation；
  mature stage 的额外 parent 由 selector vote 产生并落盘到 `reports/selector_votes.json`，
  不足名额再由 deterministic `SearchPolicy` 用 recent improvement、diverse underexplored
  和 `max_children_per_node` 约束补齐。默认 `selector_vote_count=3`，这是同一
  provider 的多票 evidence；除非后续显式配置多 provider，不声明论文级异构
  selector ensemble。若配置 `selector_panel`，默认每个 member 一票，并记录 member、
  configured model、actual model、provider、source 和 deterministic diversity flags；
  mock run 中多个 configured member 仍不等同真实异构 provider evidence。
- Selector policy 是 checkpoint 的一部分：`checkpoint.json` 保存
  `selector_policy_digest`，resume 时禁止静默切换 selector role config 或 panel
  config。Selector 真正投票时会保留 latest view 与
  `reports/selector_votes/selection_*.json` 历史；早期节点数不足、尚未进入 mature
  selection 阶段时，metadata 用 `selector_voting_exercised=false` 明确表示 panel 只是
  configured，没有实际投票证据。
- Analysis Base 会在每个 child mutation 前写出
  `solutions/<solution_id>/analysis_context.json`，把 mutation parent、已有 sibling
  children、uncle nodes 和缺失报告分别结构化记录。Proposer prompt 使用
  parent/sibling/uncle 关系标签，而不是无类型 related report 拼接。
- `RetrievalQueryBuilder` 用 benchmark metadata、parent analysis、failure kind、method tags、score trend 和 leaderboard top-k 构造 KB query；`use_kb` 与 `random_kb` 可用于 ablation。
- `SolutionNode` 持久化 selector/retriever 需要的结构化元数据，包括 `method_tags`、`failure_kind`、`score_delta_from_parent`、`num_debug_attempts`、`benchmark_name` 和 `contract_hash`。
- `EmergenceAudit` 会为每个 solution 写入 `emergence_report.json`，只给出
  `candidate_emergent` 等保守标签；它检查 KB/catalog overlap、prior-result 证据、
  score improvement 和 policy fidelity，不输出论文级 proved emergent discovery。
- `trace_summary.json` 汇总 `trace.jsonl`，用 required span types 和 guardrail failures 形成最小 trace quality gate。
- `run_metadata.json` 汇总 wall time、champion、solution count 和按 role 聚合的 LLM 调用统计。

后续增强应优先补更强 sandbox、成本统计、trace viewer 和真实 LLM ablation，而不是继续增加新 Agent。

## OpenAI Agents SDK 对齐要求

本项目不默认把主流程迁移到 OpenAI Agents SDK runtime，但每个实现步骤必须对齐 SDK 的核心设计：

- 使用 code orchestration 作为默认多 Agent 编排方式，只有明确需要专家接管用户回合时才考虑 handoff。
- Specialist agent 作为 bounded tool 使用，不能自行改变全局状态转移、score 或 champion。
- 代码消费的 LLM 输出必须使用 structured outputs：校验字段、有限 retry、失败后 fail closed。
- 每个 tool-like 边界必须有 guardrail：LLM JSON 输出、artifact 写入、生成代码执行、evaluator contract 完整性。
- generated solution 在进入 validate/train/predict/evaluate 前必须通过静态 sandbox 检查。
- evaluation 默认使用 prediction-only protocol：generated solution 只接收
  `predict_input.npz` 中的 `x_val`，可信 evaluator 用私有标签计算分数，
  不在持有验证标签的进程中 import `solution.py`。
- 每次运行必须产生 trace：`workflow_span`、`agent_span`、`generation_span`、`tool_span`、`guardrail_span`。
- Trace 和 artifacts 是后续 trace grading / evals 的输入，不把未验证的 LLM 总结当作实验事实。
