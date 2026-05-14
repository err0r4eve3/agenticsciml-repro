# OpenAI Agents SDK 升级复盘

[返回文档树](index.md) · 相关文档：[多 Agent 设计方法](multi_agent_design.md)、[Real LLM 运行](real_run.md)、[版本说明](version_notes.md)

## 结论

截至 2026-05-14，当前项目不应把主流程改成自由 handoff 或 SDK agent loop 全托管。AgenticSciML 的核心仍是固定 evaluator、solution tree、artifact persistence、checkpoint/resume 和 deterministic champion selection，这些应该继续由 Python orchestrator 控制。

更合适的升级方向是：保留代码编排，把 OpenAI Agents SDK 已经成熟的 typed output、tool guardrail、trace/eval、sandbox/session 和 model-provider contract 思路逐步落到本项目的边界层。这样能升级真实 LLM 可审计性和安全性，同时不破坏论文 workflow 复现目标。

## 参考依据

本轮参考了官方文档中的这些稳定设计点：

- [Agents SDK](https://developers.openai.com/api/docs/guides/agents)：SDK 适合应用自己拥有 orchestration、tool execution、state 和 approvals 的 code-first agent app。
- [Agent definitions](https://developers.openai.com/api/docs/guides/agents/define-agents)：agent 应封装 model、instructions、tools、guardrails、handoffs 和 structured outputs；需要代码消费时使用 typed output。
- [Orchestration and handoffs](https://developers.openai.com/api/docs/guides/agents/orchestration)：多 agent 有两种模式，handoffs 让 specialist 接管回复，agents-as-tools 让 manager 保持控制。
- [Running agents](https://developers.openai.com/api/docs/guides/agents/running-agents)：一次 run 是应用层 turn；sessions、conversation ID、previous response ID 和 serialized state 需要选一种一致策略。
- [Guardrails and human review](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals)：input/output/tool guardrails 与 human-in-the-loop approvals 应放在不同风险边界。
- [Sandbox Agents](https://developers.openai.com/api/docs/guides/agents/sandboxes)：sandbox 把 harness 控制面和 compute 执行面分离；适合文件、命令、包、artifact、snapshot 和 resumable workspace。
- [Integrations and observability](https://developers.openai.com/api/docs/guides/agents/integrations-observability)：tracing 应覆盖 model calls、tool calls、handoffs、guardrails 和 custom spans，并用于后续 eval。
- [Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)：trace grading 适合从单次 debug 进入可重复 workflow eval。
- [Models and providers](https://developers.openai.com/api/docs/guides/agents/models)：生产应显式选择 agent/model/run-level model，不依赖 SDK 默认值；非 OpenAI provider 要单独处理 transport/capability。
- [Using reasoning models](https://developers.openai.com/api/docs/guides/latest-model#using-reasoning-models)：Responses API、reasoning effort、verbosity、Structured Outputs、prompt caching、hosted tools 和 state management 是当前 reasoning model 工作流的主要优化面。
- [Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)：Structured Outputs 比 JSON mode 更适合代码消费的 schema adherence；Pydantic/Zod 类型应尽量避免 schema/type drift。

## 当前已经对齐的部分

- 代码编排优先：`src/agenticsciml/orchestrator.py` 保持全局状态、score、checkpoint、tree update 和 champion export 的所有权，符合 manager-style workflow。
- Specialist contract：`src/agenticsciml/agents/specs.py` 已有 `AgentSpec`，覆盖 role、non-role、input/output schema、visible context、tools、budget、artifacts 和 failure policy。
- 结构化输出基础：`AgentBase.complete_json_checked()` 会校验 required fields、有限 retry，并把失败记录为 `guardrail_span`。
- Guardrails：已有 evaluator contract hash、benchmark source manifest、prediction-only evaluation、sanitized subprocess env、static sandbox checks、artifact path semantics 和 solution tree schema gates。
- Trace：已有 `trace.jsonl`、`trace_summary.json`、`generation_span`、`agent_span`、`tool_span`、`guardrail_span`、workflow start/end、event sequence 和 node lifecycle coverage。
- Agents-as-tools 思路：DataAnalyst、Evaluator、Selector、Retriever、Proposer、Critic、Engineer、Debugger、ResultAnalyst 都是 bounded worker，不直接改变全局 champion。
- Real LLM 边界：`smoke-llm --real` 必须显式开启，`verify-smoke-llm` 会检查正数真实 LLM calls、branch-context contrast 和 trace/ledger evidence。

## 已修复的前五项

2026-05-14 已把前五个不符合项落成代码、测试和文档：

- 原生 Structured Outputs adapter：`OpenAIAdapter` 在 OpenAI-native 路径使用 Responses `parse` + typed output model；`OPENAI_BASE_URL` 路径保留兼容 fallback。
- provider capability matrix：manifest、ledger、trace metadata 和 run metadata 记录 provider、adapter type 与 capability flags。
- cost/token budget gate：新增 `AGENTICSCIML_MAX_LLM_CALLS`、`AGENTICSCIML_MAX_PROMPT_TOKENS`、`AGENTICSCIML_MAX_OUTPUT_TOKENS`、`AGENTICSCIML_MAX_TOTAL_TOKENS`、`AGENTICSCIML_MAX_COST_USD` 和 `AGENTICSCIML_COST_PER_1K_TOKENS_USD`。
- Pydantic 化 `AgentSpec.output_schema`：代码消费的 agent JSON 输出现在通过 closed Pydantic schema 校验，unknown fields 和错误类型会 fail closed。
- SDK trace/export bridge：完成运行会写 `openai_sdk_trace.json`，把本地 `trace.jsonl` 映射为 sanitized SDK-style span bundle。

## 仍值得升级的地方

| 优先级 | 升级项 | 当前状态 | 建议实现 | 验证方式 |
| --- | --- | --- | --- | --- |
| P1 | trace grading eval set | 当前 `trace_summary` 主要验证结构完整性，不评价 agent 决策质量 | 建立 `docs/eval_rubrics.md` 或 `evals/trace_graders/`：检查是否选对 parent、是否泄漏验证集、是否按 proposal patch、是否触发 debugger、是否错误宣称科学结果 | mock trace fixtures + grader expected labels；真实 LLM smoke 附带 trace sample export |
| P1 | container/sandbox backend | 当前是 per-solution workspace + static AST + sanitized env，不是 OS/container 级隔离 | 抽象 `ExecutionBackend`，先加 Docker backend：non-root user、read-only benchmark mount、private eval mount、no-network 默认、CPU/memory/time limits；保留 local backend | targeted execution tests；恶意 solution fixture 验证网络、绝对路径、symlink、resource limit |
| P1 | human review pause/resume | 配置里有 `auto_approve_evaluation`，但缺少 SDK-style interruption/state 审批流 | 对高风险动作建 `approval_request.json`：真实 LLM 花费超阈值、开启网络、修改 benchmark source、导出外部 trace、长时间训练；审批后从 checkpoint resume | CLI tests 覆盖 pause、approve、reject、resume；checkpoint 记录 pending approval |
| P2 | prompt/version/cache contract | prompt 是 repo markdown，run metadata 未系统记录 prompt digests/caching key | 记录每个 prompt file digest、stable prefix digest、dynamic context digest；OpenAI-native adapter 支持 `prompt_cache_key` | run metadata 与 trace ledger 对账；prompt 修改会改变 digest |
| P2 | session/compaction strategy | 现在用 filesystem checkpoint，不保存 SDK session 或 previous response state | 对长 real LLM run 增加 compaction artifact：completed actions、assumptions、solution IDs、tool outcomes、blockers、next goal；如切 SDK runtime，再选择一种 session/state 策略 | compaction artifact tests；resume 后 prompt 不重复注入历史噪声 |
| P2 | MCP/hosted tools | 当前 retrieval 是本地 lexical KB；没有 MCP 或 hosted tools surface | 先不把 evaluator/runner 做成远程工具。可把外部 paper/reference retrieval 做成 local MCP 或 file-search-like tool，但必须保持 benchmark-aware 和 no-network tests | offline tests；network disabled tests；tool approval policy |

## 不建议现在做的事

- 不建议把整个 evolution loop 迁移成 LLM 自主 handoff。论文复现需要 deterministic evaluator、tree invariants、checkpoint 和 score-driven selection，handoff 会削弱复现性。
- 不建议为了 SDK 名义去删除当前本地 trace 和 artifact system。官方 tracing 适合 observability/evals，本项目的 `trace.jsonl` 仍是可离线验证的证据 bundle。
- 不建议把 DeepSeek/OpenAI-compatible 路径强行塞进 OpenAI-native Structured Outputs 假设。应通过 provider capability matrix 明确哪些 gate 是 native、哪些是 compatibility fallback。
- 不建议把 sandbox beta API 直接作为唯一执行后端。应先抽象 backend，再加入 Docker/SDK sandbox provider；local backend 继续服务测试和 mock mode。

## 推荐版本路线

### v0.2.0 - Real LLM contract hardening

- 已完成 OpenAI-native Responses/Structured Outputs adapter。
- 已完成 Pydantic 化 code-consumed agent outputs。
- 已完成 provider capability matrix 和 real-run budget gate。
- 已完成 `smoke-llm` manifest capability、预算和 adapter 类型记录。
- 下一步补 usage/cost 的 provider-specific pricing policy 和 trace grading dataset。

### v0.3.0 - Execution isolation

- 增加 `ExecutionBackend` 抽象和 Docker backend。
- 默认 no-network、non-root、resource limits。
- private evaluator 和 benchmark source 通过 read-only / private mounts 隔离。

### v0.4.0 - Trace eval flywheel

- 基于已导出的 `openai_sdk_trace.json` 建立 trace grading rubrics。
- 增加小型 regression trace dataset。
- 将 trace quality gate 分成结构完整性、artifact 一致性、decision quality 三层。

### v0.5.0 - Approval and long-run state

- 增加 human-review interruption artifact。
- 支持 approval pause/resume。
- 增加 compaction artifact，面向长 real LLM run 和多轮真实 benchmark。

### v0.6.0 - Tool ecosystem

- 评估 local MCP/file-search 式参考检索。
- 只把非评分、非执行、非敏感 sidecar 能力放进 MCP 或 hosted tools。
- 保持 evaluator、score、champion selection 在 Python trusted path 内。

## 下一步最小任务

1. 先做 trace grading eval set，让 `openai_sdk_trace.json` 不只是导出格式，而能进入可重复评分。
2. 再做 container/sandbox backend，补 OS/container 级隔离。
3. 然后补 human review pause/resume，把真实 LLM 花费、导出外部 trace、长训练等高风险动作变成可审批状态。
4. 最后评估 MCP/hosted tools，只放非评分、非执行、非敏感 sidecar 能力。

## 验收口径

完成上述升级后，不应只看 CLI exit code。每次真实 LLM benchmark 至少要同时满足：

- `run_metadata.json` 记录真实 provider/model/adapter/capability/budget/usage。
- `llm_call_ledger.jsonl` 与 `generation_span` 一一对应。
- `trace_summary.json` 的 structural quality gate 通过。
- trace/eval report 能解释 parent selection、branch context、debugger use、guardrail status 和 evidence boundary。
- benchmark fidelity、mock/real LLM mode、scientific claim boundary 在 run-level artifact 中可独立读取。

## 外部参考

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI Agents SDK starting points](https://developers.openai.com/api/docs/guides/agents#choose-your-starting-point)
