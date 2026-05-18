# 版本说明

[返回文档树](index.md) · 相关文档：[项目概览](../README.md)、[Ablation 说明](ablation.md)

## 2026-05-18 Custom Proxy Evaluator Generation

本次把非 catalog 问题从 scaffold-only 推进到可运行的本地 proxy evaluator 生成。

已实现：

- `POST /api/problem-intake/plan` 在 `allow_custom_benchmark=true` 时会在当前账号
  namespace 下生成 `.agenticsciml/accounts/<account_id>/benchmarks/<custom_id>/`
  benchmark bundle。
- 生成的 bundle 包含 `Problem.md`、`Requirements.md`、`Evaluation.md`、
  `Data_config.json`、`Benchmark_spec.json`、`generate_data.py`、`evaluate.py` 和
  `guidelines.md`。
- `Benchmark_spec.json` 可被 `ProblemBundle.load()` 动态识别，不需要把每个临时问题
  写入 checked-in `BENCHMARKS` catalog。
- `generate_data.py` 生成确定性的训练/验证代理数据；`evaluate.py` 使用私有验证标签
  计算 `custom_proxy_relative_l2`，继续走 prediction-only evaluation boundary。
- Planner 返回的 `start_run` action 会指向生成的 custom benchmark，并保留
  `problem_intake`、`planner_snapshot`、algorithm strategy seeds、account namespace 和
  run budget。

验证：

- `tests/test_web_api.py::test_problem_intake_custom_benchmark_generates_runnable_evaluator`
  覆盖 custom problem 生成 evaluator bundle 后可立即启动 mock run，并写出
  `evaluation_contract.json` 与 `solutions/solution_000/eval.json`。

边界：

- 自动 evaluator 是 deterministic workflow proxy，只能证明 AgenticSciML loop、artifact
  plumbing、private-label evaluation boundary 可运行。
- 它不是 paper-like benchmark，不支持科学结论、paper score reproduction 或真实有限元
  指标声明；需要人工替换/审查 domain evaluator 后才能提升证据等级。

## 2026-05-18 Selector Panel Provenance

本次补齐 selector ensemble 的配置与证据层，但仍保持 claim boundary：默认
`selector_vote_count` 只是同一 selector 路径多票，不自动声明论文级异构模型 panel。

已实现：

- `ExperimentConfig` 新增 `selector_panel`，可配置多个 selector member 的 model、
  temperature 和 reasoning effort。
- CLI 新增 `--selector-panel-models`，Web API `POST /api/runs` /
  `/api/problem-intake/plan` 支持 `selector_panel`。
- 有 `selector_panel` 时，orchestrator 默认按 panel member 一成员一票 cast vote，避免
  因 `selector_vote_count` 与 panel 长度不一致造成成员权重不均；没有配置时保留现有
  single-selector multi-vote 行为。
- `reports/selector_votes.json` 使用 `schema_version=2`，新增 `ensemble_mode`、
  `selector_panel_members`、per-vote `member_id`、`configured_model`、`actual_model`、
  `provider`、`source`、`selector_diversity` 和 `claim_boundary`。其中
  `selector_diversity` 明确记录 `mock_evidence`、actual model/provider diversity、
  panel member/vote count、repeated-member 状态和 `heterogeneous_selector_evidence`。
- 每次 selector 真正参与 parent selection 时，除了覆盖 `reports/selector_votes.json`
  作为 latest view，还会写入
  `reports/selector_votes/selection_000001.json` 这类 per-selection artifact，避免
  多轮 run 丢失早期 vote rationale 和 panel provenance。
- `checkpoint.json` 和 `run_metadata.json` 记录 `selector_policy` 与
  `selector_policy_digest`。Resume 时若 selector vote schema、single/panel mode、
  selector role config 或 panel config 改变，会 fail closed，且在拒绝前不覆盖旧
  `config.json`。
- `run_metadata.json` 新增 `selector_panel` runtime metadata，记录实际 selector panel
  证据来源，并用 `selector_voting_exercised` / `selector_vote_events` 区分“配置了
  panel”和“selector 实际参与了本 run 的选择”。

验证：

- `tests/test_llm_and_agents.py` 覆盖默认 selector votes 仍标注为
  `single_provider_multi_vote`，并覆盖单张 ballot 内重复/非法 ID 不会重复计票。
- `tests/test_orchestrator_cli.py` 覆盖 configured selector panel 的 per-member vote
  provenance，以及两成员 panel 不因默认 `selector_vote_count=3` 产生第三张偏置票。
- `tests/test_orchestrator_cli.py` 还覆盖 per-selection vote artifact retention、
  early-stage no-vote metadata，以及 resume selector policy mismatch fail-closed。
- `tests/test_web_api.py` 覆盖 Web run request 会持久化 selector panel config 和 metadata。

边界：

- Mock run 中 `actual_model=mock`，即使配置了多个 member，也只能说明 panel 配置和
  provenance 管线可用；只有真实 provider/model provenance 不同时，才能作为异构
  selector evidence。

## 2026-05-18 Data Analyst Replayable EDA

本次继续补论文 workflow 缺口，把 Data Analyst 的训练集观察从单次 summary 扩展为
可复跑 EDA artifact。

已实现：

- `DataAnalystAgent` 继续只读取 training data，不接触 private validation labels。
- 新增 `reports/data_eda.py`：一个可复跑的训练 `.npz` EDA 脚本，会拒绝明显的私有
  label 路径，并输出数组统计。
- 新增 `reports/data_eda.json`：保存训练集-only array checks、plot checks、
  modeling notes、replay command 和 claim boundary。
- Data Analyst prompt 增加 `Replayable EDA summary`，让后续 root/evaluator 看到的是
  可审计 EDA 摘要，而不是只依赖自然语言观察。

验证：

- `tests/test_llm_and_agents.py` 覆盖 EDA script/json 写入、prompt 注入和脚本 replay。
- `tests/test_orchestrator_cli.py` 覆盖完整 mock run 生成 EDA artifacts。

边界：

- 这仍是 deterministic local EDA，不是论文级自动生成任意 EDA 程序或任意 evaluator
  synthesis；它只加强 workflow traceability 和训练数据观察可复验性。

## 2026-05-18 Typed Analysis Base Context

本次根据 Pro 复审建议，补齐论文 workflow 中 Analysis Base 的可审计上下文边界。

已实现：

- 每个 child mutation 在 proposal 前写入
  `solutions/<solution_id>/analysis_context.json`。
- artifact 明确区分 `parent_report`、`sibling_reports`、`uncle_reports` 和
  `omitted_reports`，不再只把若干 analysis summary 作为无类型字符串塞进 prompt。
- sibling 定义为 mutation parent 已存在的 children；uncle 定义为 mutation parent
  的 parent 的其他 children。缺失 node 或 analysis report 会写入 `omitted_reports`
  和原因，避免静默丢上下文。
- Proposer prompt 的上下文标题改为
  `Analysis Base context (parent/sibling/uncle reports)`，并保留关系标签。

验证：

- `tests/test_orchestrator_cli.py` 覆盖 parent/sibling/uncle 抽取、缺失报告记录、
  child mutation artifact 落盘。
- `tests/test_llm_and_agents.py` 覆盖 Proposer prompt 使用关系标签。

边界：

- Analysis Base context 只是 workflow traceability，不证明论文级 emergent discovery、
  异构 selector ensemble 或 paper-score reproduction。

## 2026-05-18 Paper Workflow P0 Alignment

本次修复论文工作流对齐审查中的三个 P0 边界：root baseline 隔离、evaluation
approval pause、以及非库内问题不能被错误当作已有 benchmark 运行。

已实现：

- Root Engineer prompt 不再包含 human/planner selected strategy seeds；这些策略种子只允许
  进入后续 proposer/engineer mutation 上下文，避免污染单 agent root baseline。
- Orchestrator 新增 `evaluation_approval.json` 审批门。默认 mock 流程仍可
  `auto_approved` 兼容现有测试和 CLI；当 `auto_approve_evaluation=False` 或 CLI 使用
  `--require-evaluation-approval` 时，会在 root 生成前暂停，要求人工审查
  `evaluation_contract.json`、`reports/evaluation_contract.md`、data analysis 和 benchmark
  文件后把状态改为 `approved`，再用 `--resume` 继续。
- Resume 支持 evaluation-pending pre-root run：该状态不写空 solution-tree checkpoint，
  避免把“尚未生成 root”的审批暂停误报为合法 solution tree。
- `POST /api/problem-intake/plan` 支持 `allow_custom_benchmark=true` 的安全路径：早期
  版本只返回 scaffold；当前版本会生成本地 deterministic proxy evaluator bundle，并
  返回可运行的 `start_run` action，但仍把 evidence 标记为 proxy / not supported。

验证：

- `tests/test_llm_and_agents.py` 覆盖 root prompt 不包含策略种子段。
- `tests/test_orchestrator_cli.py` 覆盖 evaluation approval pending、approval 后 resume
  生成 root。
- `tests/test_web_api.py` 覆盖 custom benchmark scaffold-only response。

边界：

- 当前自动 evaluator 仍是 workflow proxy，不是 paper-style Data Analyst 自动合成真实
  领域 evaluator；非库内问题可以跑通流程，但不能据此声称科学有效。
- Selector ensemble 仍是当前单 selector 多票机制，不是 GPT/Grok/Gemini 多 provider
  panel。

## 2026-05-18 Role Model Policy Defaults

本次把角色级模型设置从“全员默认 0.0 temperature + 无 reasoning effort”改为按角色
职责分配的默认 policy。

已实现：

- 确定性/执行类 role 使用低温：`evaluator=0.00/high`、
  `root_engineer=0.10/xhigh`、`engineer=0.10/xhigh`、`debugger=0.05/xhigh`、
  `selector=0.05/high`、`retriever=0.00/medium`。
- 创造/分析类 role 使用更高温度：`proposer=0.55/xhigh`、
  `critic=0.35/high`、`data_analyst=0.35/high`、`result_analyst=0.30/high`。
- Orchestrator 在没有用户 override 时直接使用这些 defaults；用户传入 role override
  时仍优先生效。
- `GET /api/agent-roles` 和 `GET /api/solver/settings` 会返回 role default policy，
  前端 Layered model routing 面板也会展示对应默认值。
- `run_metadata.json` 的 `agent_models` 现在记录所有 role 的有效 policy，并用
  `source=role_default | request_override` 标识来源。

边界：

- role default 不改变模型 ID；没有 role model override 时仍使用后端当前 adapter/model。
- Mock run 继续只验证工作流形状，不产生科学结论。

## 2026-05-18 Reasoning Effort Routing

本次把 role-level `reasoning_effort` 从审计字段升级为真实 provider 参数。

已实现：

- ChatUI/API 允许 `low | medium | high | xhigh`，`GET /api/solver/settings`
  返回同一组选项。
- Orchestrator 会把每个 role 的 `reasoning_effort` 传入对应 agent；Proposer 内部
  Critic 调用也使用 Critic role 配置。
- `OpenAIAdapter` 在 OpenAI-native Responses 路径传 `reasoning={"effort": ...}`，
  在 OpenAI-compatible Chat Completions 路径传 `reasoning_effort=...`。
- `generation_span` 和 real LLM smoke ledger 会记录实际请求的 `reasoning_effort`。

边界：

- 默认模式仍保持 `ask=medium`、`plan=high`、`agent=high`；`xhigh` 只在显式覆盖时使用。
- `none` 和 `minimal` 是官方可见枚举，但本轮不暴露在默认 UI，避免部分模型或兼容
  endpoint 不支持时误配置。

## 2026-05-18 Candidate Emergence Audit

本次根据 NotebookLM 对 AgenticSciML emergent discovery 定义的证据表，新增保守的
候选涌现审计层。ChatGPT Pro 复核已准备非敏感 prompt，但本轮被 Chrome 登录态阻塞，
页面显示会话过期；因此代码实现依据当前仓库事实和 NotebookLM 的来源综合，不把 Pro
作为已完成复核证据。

已实现：

- 新增 `agenticsciml.emergence_audit`。它对每个 solution 的 proposal、analysis、
  engineering summary、method tags、KB entries、algorithm catalog、parent/root score
  和 `policy_fidelity_report.json` 做确定性审计。
- Orchestrator 在每个 evaluated/failed node 形成后写入
  `solutions/<id>/emergence_report.json`，并记录 `tool_span=emergence_audit`。
- 审计报告只允许保守标签：`not_assessed`、`kb_direct`、`catalog_seeded`、
  `prior_result_adapted`、`implementation_unfaithful` 和 `candidate_emergent`。
  当前不会输出 `proved_emergent_discovery`。
- `candidate_emergent` 需要同时满足：低 KB/catalog 直接重叠、有 prior-result 适配证据、
  相对 parent 的 evaluator score 改善、且 policy fidelity artifact 存在并通过。
- `GET /api/runs/{id}/solutions` 返回 `emergence_audit` 摘要；solution artifact index
  增加 `emergence_report.json`。
- 前端 Solution loss/tree 表新增 emergence claim 列，显示保守标签和 blocking gap 数量。
- readiness artifact capture requirements 增加
  `solutions/*/emergence_report.json`。

验证：

- `tests/test_emergence_audit.py` 覆盖 missing artifacts、direct KB match 和
  candidate-emergent 三种判定。
- `tests/test_orchestrator_cli.py` 覆盖 mock run 会生成 per-solution
  `emergence_report.json`，且不会产生强 discovery claim。
- `tests/test_web_api.py` 覆盖 solution summary 和 readiness artifact requirement。

边界：

- 该审计只提供候选标签，不是论文级涌现发现证明。
- Mock run 仍必须保持 `scientific_claim=not_supported`。
- 真正接近论文声明还需要 real LLM、多 seed、KB/no-KB/random-KB ablation、失败样本审查
  、物理/可视一致性检查和外部复核。

## 2026-05-18 Strategy Fidelity Inspector

本次根据 NotebookLM 对 ATHENA Inspector / AgenticSciML KB symptom mapping 的建议，
把上一轮 manual strategy locks 继续推进到执行前 guardrail。

已实现：

- 新增 `agenticsciml.strategy_inspector`。它对 `solution.py` 做确定性静态检查，
  支持 `required_terms`、`forbidden_terms`、`required_imports`、`forbidden_imports`、
  `required_call_names` 和 `forbidden_call_names`。
- Orchestrator 在 `train_and_evaluate` 前运行 inspector。若 required lock 的可机器
  检查条件失败，会写入 `solutions/<id>/policy_fidelity_report.json`，记录
  `guardrail_span=strategy_fidelity_inspector`，并以 `guardrail_error` 标记 failed node。
- readiness normalization 会保留 `manual_strategy_locks.inspection` 和同名顶层条件字段，
  避免 Web/API 启动路径丢失用户提供的审计约束。
- `GET /api/runs/{id}/solutions` 现在会返回每个 solution 的 `policy_fidelity` 摘要；
  solution artifact index 也会列出 `policy_fidelity_report.json`。
- readiness artifact capture requirements 增加
  `solutions/*/policy_fidelity_report.json`。

验证：

- `tests/test_strategy_inspector.py` 覆盖 required / forbidden 条件、lock text 标记语法和
  非结构化 lock 的非阻断行为。
- `tests/test_orchestrator_cli.py` 覆盖不符合 required lock 的 generated solution 会在执行前
  fail closed。

边界：

- Inspector 只检查明确写出的机器可审计条件；普通自然语言 lock 仍只作为 planning context。
- 该检查不是科学验证，不证明策略有效，也不替代 evaluator score 或真实 run artifact。

## 2026-05-17 算法库双语说明

本次增强第三页算法库的信息密度和中英双语兼容。

已实现：

- `/api/algorithms` 的每个算法条目新增中文说明、English / 中文 `features`、
  English / 中文 `problem_fit` 和中文边界提醒。
- 前端算法库新增 `中文` / `EN` 切换；每张算法卡片都明确展示“特点”和“对应问题”，
  同时保留 family、status、compatible benchmark families、implementation path 和
  选择按钮。
- 浏览器 E2E 脚本新增算法库语言切换检查，避免未来 UI 回退成单语纯描述。

边界：

- 双语目录仍只是人工选择和 Problem Intake 的 strategy seed，不是已评估实现。
- 分数、champion 和 scientific claim 仍只能来自 evaluator、trace 和 run artifacts。

## 2026-05-17 Run Readiness / Strategy Audit

本次根据 NotebookLM 现有研究和一次性 Pro 复核，把下一轮迭代落在启动前的
Inspector-style pre-run audit，而不是继续增加 benchmark 或自动生成 evaluator。

已实现：

- 新增纯确定性 `agenticsciml.readiness` 检查层：输入 benchmark catalog、
  algorithm catalog、selected algorithm ids、manual strategy locks、branch context、
  run budget 和 real-mode gate，输出 `run_readiness.v1` 报告。
- 新增 `POST /api/run-readiness/preview`。该端点不调用 LLM、不联网、不执行代码，
  只返回 `ready` / `ready_with_warnings` / `blocked`、checks、benchmark fidelity
  preview、algorithm seed preview、strategy lock preview、real-mode gates 和 artifact
  capture requirements。
- `POST /api/runs` 现在会为非 dry-run 生成并持久化
  `planning/readiness_report.json`；`config.json` 保存完整 readiness report，
  `run_metadata.json` 保存 `readiness_summary`。
- 前端 Paper Run Lab 的 `Run Config` 区域新增 `检查 run readiness` 操作，展示
  blocker/warning、benchmark fidelity、strategy seed 数量和 pre-run audit 边界。
- readiness 把 algorithm catalog 条目继续标注为 `strategy_seed`，不会把它们提升为
  已评估实现；proxy / faithful-small claim boundary 仍必须由 run artifacts 和
  evaluator 结果支撑。
- real mode 缺少二次确认或服务端开关时，readiness preview 会返回 blocker；实际
  run 启动仍沿用既有 fail-closed Web API gate。

验证：

- `tests/test_web_api.py` 覆盖 readiness preview、unknown algorithm / real-mode blocker、
  mock run 持久化 readiness report 和无 hidden chain-of-thought 字段。
- 前端类型和打包用 `npm run build` 验证。

边界：

- readiness 是启动前一致性审计，不是科学验证、不是 evaluator 合成、不是论文分数复现。
- manual strategy locks 是用户假设/约束，不是事实；当前只记录并检查继承期望。
- 后续可在 run evidence/report endpoint 中复用同一套 claim-boundary 语言。

## 2026-05-17 Manual Strategy Locks UI

本次继续把 readiness 审计往实际工作台推进：前端 Paper Run Lab 新增 `Strategy Locks`
面板，允许用户把高阶数学直觉或约束作为人工锁定输入。

已实现：

- 前端新增 strategy lock 状态、增删改 UI、kind / scope 选择和非空过滤。
- `检查 run readiness` 会携带 `manual_strategy_locks` 与
  `branch_context.expected_inherited_lock_ids`，让 readiness preview 可以显示锁定策略和
  branch inheritance 期望。
- `启动配置 run` 和 Agent action 启动路径都会把当前非空 strategy locks 传给后端；
  后端既有 readiness artifact 会随 run 保存。
- 文档明确 strategy locks 是用户假设/约束，不是科学事实，也不改变 evaluator。

验证：

- `npm run build`
- 浏览器 smoke：打开远端/本地 Paper Run Lab，添加 lock 后刷新 readiness，确认
  `strategy seeds` / warnings 区域正常渲染。

## 2026-05-17 ChatUI workflow live smoke

本次新增可重复运行的 Web 控制面 smoke，用于在本地或部署环境中检查 ChatUI 依赖的
核心 API contract 与 code-server sidecar 边界。

已实现：

- 新增 `scripts/web_workflow_smoke.py`，只通过 HTTP 调用 live FastAPI，不 import
  后端内部函数。
- smoke 覆盖 `ask` / `plan` / `agent` 三种模式：Ask 不返回 actions，Plan 只返回
  `open_code_server` 建议 action，Agent 可从高上下文 cylinder wake 问题自动选择
  benchmark、algorithm seed 和 mock run budget。
- smoke 增加负向边界检查：Agent 模式拒绝 shared repo workspace；real mode 未带
  `real_confirmed=true` 时拒绝启动；第二个账号 namespace 不能读取第一个账号的 run
  detail、selector votes、solution summaries、run artifact 或 code-server run
  workspace。
- smoke 会同步创建账号 namespace 下的 mock run，并检查 passing `trace_summary`、
  selector votes、solution loss/tree、champion workspace 和 code-server workspace 列表。
- code-server 检查确认 API payload 不携带 password/token，命令提示使用
  `code-server --auth none --bind-addr 127.0.0.1:8080 ...`；默认还会探测 live
  sidecar HTML 是否仍出现 password login。
- smoke 默认拒绝非 loopback code-server URL。远端部署如果通过
  `AGENTICSCIML_CODE_SERVER_URL` 返回 HTTPS 上游鉴权入口，需要显式加
  `--allow-non-loopback-code-server-url`。
- smoke 支持 `--expect-repo-root-contains <release-sha>`，用于远端发布后检查 Web API
  是否仍停留在旧 release。
- 新增 `scripts/web_ui_dispatch_e2e.mjs`，使用 `playwright-core` 和本机 Chrome
  channel 做真实浏览器 E2E，覆盖第一页 ChatUI Ask/Plan/Agent 分发、Agent 打开第二页
  VS Code Web、第二页侧栏 ChatUI、第三页算法库边界，以及可选 code-server WebSocket
  路径检查。WebSocket 检查现在会等待连接稳定、捕获浏览器控制台中的 `403` / `1006`
  错误，并检查 iframe 内没有 workbench connection failure。
- 远端 nginx 反代 code-server 时必须保留端口化 Host。VS Code server 块应传
  `proxy_set_header Host $http_host` 和
  `proxy_set_header X-Forwarded-Host $http_host`，否则 code-server origin guard 会把浏览器
  `Origin: https://host:port` 与上游 `Host: host` 判为不匹配，WebSocket 握手返回
  `403`，前端表现为 `WebSocket close with status code 1006`。
- 第二页 IDE 的右侧 ChatUI 侧栏支持拖动调整宽度和折叠，侧栏与 VS Code Web 之间改为
  细弱分隔线；浏览器 E2E 会验证 resize、collapse 和展开路径。

验证：

- `PYTHONPATH=src uv run --python 3.11 --extra web --extra dev python scripts/web_workflow_smoke.py --base-url http://127.0.0.1:8765`
- `node scripts/web_ui_dispatch_e2e.mjs --base-url http://127.0.0.1:8765 --expect-code-server-websocket`

边界：

- 该 smoke 产生的是 mock workflow shape evidence，不是 scientific evidence。
- 本地账号 namespace 仍不是认证/授权系统；cross-account negative checks 只能证明
  目录解析不会混用账号 run。公网部署仍需上游账号鉴权和反代路径校验。
- 脚本默认假设 FastAPI 和 code-server 已经由部署或本地 session 启动；没有 sidecar
  时可加 `--skip-code-server-live` 只检查 API payload。

## 2026-05-17 Problem Intake 与人工算法选择

本次补齐 Paper Run Lab 中更接近论文入口的使用方式：高阶用户先完整描述问题，再由
agent 控制面自动评选 benchmark、算法 seed 和 run budget；人工算法库仍可手动选择。

已实现：

- 新增 `POST /api/problem-intake/plan`，输入完整问题描述、requirements、
  evaluation、data description、run budget 和人工选择的 algorithm ids。
- planner 在当前本地 benchmark catalog 内返回推荐 benchmark、候选 benchmark 排序、
  algorithm rankings、selected strategy seeds、run config 和可展示的 `start_run`
  action。
- 前端 Paper Run Lab 新增 `Problem Intake` 表单，支持自动评选 benchmark 与解法 seed。
- 算法库卡片支持人工选择；已选 algorithm ids 会进入 `POST /api/runs`。
- `ExperimentConfig` 新增 `strategy_seed_ids`、`problem_intake` 和
  `planner_snapshot`，并写入 `config.json`。
- `run_metadata.json` 记录 `strategy_seed_ids`、数量、完整 problem intake 和 planner
  snapshot；run 目录同时写入 `planning/problem_intake.json` 作为可审计规划工件。
- Root Engineer、Proposer 和 Engineer prompt 新增 strategy seed context，使人工或
  planner 选出的算法真正参与候选解法构思。
- Root Engineer、Proposer 和 Engineer prompt 同时接收 problem-intake context，但固定
  标注为非权威上下文：只能指导生成策略，不能覆盖 `ProblemBundle`、
  `EvaluationContract`、guidelines、sandbox rules 或 evaluator contract。
- Web resume 会在未显式覆盖时读取既有 `config.json`，保留 benchmark、
  `strategy_seed_ids`、`agent_models`、`problem_intake` 和 `planner_snapshot`，避免
  前端的轻量 resume payload 清空原 run 的策略种子和审计上下文。
- `/api/solver/chat` 的 `agent` / `plan` 模式现在可以识别高上下文求解请求，复用
  Problem Intake planner 自动评选 benchmark、algorithm seeds 和 run budget，并返回
  带 `problem_intake` / `planner_snapshot` 的 `start_run` action。
- 前端 Agent action dispatcher 现在按 action payload 启动 run，而不是退回当前 UI
  state，避免 Agent 自动评选出的 benchmark 或 seed 被陈旧选择覆盖。

边界：

- planner 只在当前本地 catalog 中做受控映射，不创建新 evaluator、不改写 benchmark
  contract，也不宣称选中算法已经有效。
- problem intake 不是 evaluator 合成机制；当前只能作为生成类 agent 的非权威上下文，
  具体评分仍完全来自既有 benchmark contract。
- 真正的全流程仍由 Python orchestrator 生成 root、选择 parent、展开 solution tree、
  执行 evaluator、写 trace/leaderboard/champion。

## 2026-05-17 Paper Run Lab

本次把第三页“算法库”升级为 AgenticSciML Paper Run Lab，第一页纯 ChatUI 和第二页
VS Code Web + 侧栏 ChatUI 保持轻量，不展示 benchmark/run 工作台。

已实现：

- 新增 `src/agenticsciml/paper_tasks.py` 和 `GET /api/paper-tasks`，按 `S1.1` 到
  `S1.6` 返回本地 benchmark、paper reference primitive、figure label、local SVG
  artifact 约定和 claim boundary。
- 新增 `GET /api/runs/{id}/selector-votes`，只读读取
  `reports/selector_votes.json`，缺失时返回空状态。
- 新增 `GET /api/runs/{id}/solutions`，从 `tree.json`、`leaderboard.csv` 与
  `solutions/*/eval.json` 汇总 node、parent、status、score/loss、delta、method tags
  和本地 figure artifact。
- `POST /api/runs` 新增 `target_solution_count`、`max_children_per_node` 和
  `agent_models`；`target_solution_count` 会被转换为确定性的
  `EvolutionConfig.max_iterations + parallel_mutations` 预算。
- 新增 `GET /api/agent-roles`，前端可为 Data Analyst、Evaluator、Root Engineer、
  Retriever、Proposer、Critic、Engineer、Debugger、Result Analyst 和 Selector 配置
  role-level `model`、`temperature` 与 provider-routed `reasoning_effort`。
- orchestrator 支持 role-based LLM routing：默认继续使用单一 adapter；只有配置 role
  override 时才为该 role 选择模型。`config.json` 和 `run_metadata.json` 记录实际
  `agent_models` map。
- 第三页新增 Paper Tasks、Run Config、Layered model routing、Selector votes、
  Solution loss/tree 和 Local figures 区域。

验证：

- `uv run --python 3.11 --extra web --extra dev pytest tests/test_web_api.py -q`
- `cd frontend && npm run build`

边界：

- Paper Run Lab 只展示本地 evidence 和 paper-section 对齐信息，不复制论文图片本体。
- `faithful-small` / `proxy` benchmark 和 mock run 仍不是 paper-score evidence。
- 前端不接管 evaluator、selector、champion selection、solution tree schema 或
  artifact schema。

## 2026-05-17 S1.6 Cylinder Wake Faithful-Small Benchmark

本轮按 Pro 复核建议补齐论文 S1.6 的 faithful-small benchmark。Pro 的关键约束是：
可以做 SHRED-style sparse temporal sensor reconstruction，但不能写成复现
SHRED-ROM / PySHRED / Nature Communications 结果；如果没有真正实现 LSTM，就必须
写成 lagged sensor-history encoder + shallow decoder benchmark。

已实现：

- 新增 `examples/cylinder_wake_reconstruction_faithful_small`，使用确定性
  synthetic cylinder-wake-like vorticity generator。
- 输入为五个时间步的八个固定 sparse sensor readings 加当前 normalized time，
  输出为 flattened `24 x 24` vorticity field。
- evaluator 使用 mean per-sample relative L2 作为官方分数，mean/max absolute
  error 只作为 diagnostics。
- KB 新增 SHRED-ROM / PySHRED source-grounded note，记录 temporal sensor
  history、shallow decoder、POD/low-rank basis 的启发和边界。
- benchmark catalog 新增
  `cylinder_wake_reconstruction_faithful_small`，并更新相关算法目录示例。
- 测试新增 S1.6 faithful-small 的 shape/window/no-future-leakage/evaluator
  断言。

边界：

- 这是 low-budget faithful-small benchmark，不是 `paper-like`。
- 不运行 SHRED-ROM 原始代码，不使用原始数据，不训练 LSTM/RNN，不声明论文分数。
- 分数和 scientific claim 仍只来自本仓库 evaluator 与 run artifacts。

## 2026-05-17 Web API 边界加固

本次收紧 ChatUI / VS Code Web 远端部署前最关键的控制面边界：

- `POST /api/runs` 的 real mode 现在需要请求体 `real_confirmed=true`，且服务端必须设置
  `AGENTICSCIML_ENABLE_REAL_WEB_RUNS=1`。
- account-scoped Web 请求强制使用 `.agenticsciml/accounts/<account_id>/runs/`，并拒绝
  自定义 `output_dir`、`benchmark_dir` 或 path-like benchmark。
- shared repo code-server workspace 默认禁用；本地开发要显式设置
  `AGENTICSCIML_ALLOW_REPO_WORKSPACE=1`。
- 前端 real action 面板新增显式确认按钮，确认后才会向后端发送
  `real_confirmed=true`。

边界：

- 账号 namespace 仍不是认证/授权系统；公开部署仍需要反向代理/账号 session、系统
  用户/容器隔离和 secret 管理。
- 这些改动不改变 orchestrator、evaluator、selector、champion selection 或 artifact schema。

## 2026-05-17 论文算法 Reference Primitives

新增文档：[论文算法 Reference Primitives](paper_algorithm_primitives.md)。

本次把 AgenticSciML 论文结果区列出的 6 个 champion strategy 落成本地
dependency-light reference primitives，并通过算法目录暴露。

已实现：

- 新增 `src/agenticsciml/paper_algorithms.py`，覆盖 sigmoid-gated MoE、Poisson
  particular-plus-residual + corner-biased sampling、Burgers staged PINN schedule
  / self-adaptive weights / RAR helpers、linear bias-free DeepONet branch、
  reaction-diffusion derivative-enhanced loss / hard BC-IC / spectral smoothing
  helpers，以及 cylinder wake U-FNO/CNO-style bandlimited filter。
- `src/agenticsciml/algorithm_catalog.py` 新增 6 个
  `status=reference_implementation` 的 paper champion strategy entries，并记录
  `source_scope` 与 `implementation_path`。
- 新增 `tests/test_paper_algorithms.py`，验证数学性质、shape、determinism、
  catalog exposure 和 no-overclaim 边界。
- Web API 算法目录测试现在检查 reference implementation entries 会正确暴露。
- repo-local skill `.agents/skills/agenticsciml-chatui-operator/SKILL.md` 记录
  算法目录现在可包含 reference primitives，但仍不代表 benchmark score 或
  scientific claim。

边界：

- 这些 primitives 是本地构件和 prompt-seeding aids，不是完整论文训练管线。
- 不改变 evaluator、benchmark contract、champion selection 或 paper-score
  reproduction claim。
- 分数和 scientific claim 仍只能来自 benchmark evaluator 和 run artifacts。

## 2026-05-17 ChatUI ask/plan/agent 模式

本次把 ChatUI 的交互授权拆成 `ask` / `plan` / `agent` 三种模式，并默认使用
`ask`。

已实现：

- `/api/solver/chat` 新增 `assistant_mode` 输入和输出回显。
- `/api/solver/chat` 新增 `model_settings` 回显，并允许请求覆盖
  `reasoning_effort` 和 `temperature`。
- `GET /api/solver/settings` 作为前端读取模式默认模型设置的只读来源，ChatUI 和
  IDE 侧边栏会在模式切换控件旁显示当前 `thinking/temp`。
- 三种模式的默认设置分别为：`ask` 使用 `reasoning_effort=medium`、
  `temperature=0.2`；`plan` 使用 `reasoning_effort=high`、`temperature=0.35`；
  `agent` 使用 `reasoning_effort=high`、`temperature=0.1`。
- `ask` 模式现在直接回答身份、能力、项目、benchmark、算法、run、trace、
  artifact 和边界问题；只有动作型请求才提示切换到 `plan` 或 `agent`，且不返回
  可执行 actions。
- `plan` 模式返回结构化建议 actions，但前端不会自动分发。
- `agent` 模式才调用既有安全分发器执行 `start_run`、`resume_run`、
  `open_code_server` 或 `summarize_artifact`。
- `agent` 模式要求当前 `account_id`，并拒绝 shared repo workspace；前端默认
  `workspace_scope=account`。
- ChatUI 首页 composer 和 VS Code Web 右侧 Agent 面板都提供三段切换控件。
- 删除 ChatUI 首页里“像 ChatGPT 一样输入问题或任务”的文案。
- repo-local skill 升级到 `version: 0.3.4`，记录三种模式边界、Ask 普通问答语义
  和模型设置。

边界：

- `agent` 模式仍不能绕过 real LLM 显式确认、evaluator、selector、champion
  selection 或 artifact schema。
- `agent` 模式只能操作当前账号创建的 workspace/run/solution，不能跨账号操作。
- `plan` 模式只展示动作计划，不代表用户已经授权执行。

## 2026-05-17 账号隔离工作空间与算法目录

本次继续迭代 ChatUI / VS Code Web / 算法库三页结构，新增本地账号 namespace 和
算法策略目录。

已实现：

- 新增 `GET /api/accounts` / `POST /api/accounts`，创建
  `.agenticsciml/accounts/<account_id>/workspace/` 与独立 `runs/` 目录。
- run、artifact、SSE、code-server workspace 和 `/api/solver/chat` 支持可选
  `account_id`；未传时保持旧 shared `runs/` 兼容，前端默认使用 `local` 账号。
- `GET /api/code-server/workspaces?account_id=<id>` 只返回该账号下的独立
  code-server 目录，不混入 shared repo 根目录。
- 新增 `GET /api/algorithms` 和 `src/agenticsciml/algorithm_catalog.py`，提供
  MLP、Fourier feature、PINN、weak-form PINN、XPINN、DeepONet、FNO-lite、
  kernel surrogate、low-rank operator、sparse sensor reconstruction、SINDy
  sparse discovery 等策略条目。它们是 planning / prompt-seeding aids，不是
  已验证科学结果。
- 前端左侧增加本地账号切换/创建；`VS Code` 工作空间选择页展示账号隔离目录；
  进入编辑态后仍只保留 VS Code Web iframe 和右侧 ChatUI。
- `算法库` 页新增算法策略卡片，同时继续承载 benchmark/run/trace/artifact 工作台。
- repo-local skill `.agents/skills/agenticsciml-chatui-operator/SKILL.md` 升级到
  `version: 0.3.1`，记录本地账号 namespace、算法目录和 code-server workspace
  边界。

边界：

- 本地账号 namespace 不是公网认证、ACL 或多租户安全模型。
- 算法目录不绕过 evaluator、selector、champion selection 或 artifact schema。
- `.agenticsciml/` 属于本地生成状态，不提交到 Git。

## 2026-05-17 AgenticSciML Assistant 规范

新增文档：[AgenticSciML Assistant 规范](agenticsciml_assistant.md)。

本次是一次性 Pro 复审辅助后的治理和 tool-contract 文档更新，未修改运行时代码。已
落地：

- `AGENTS.md` 新增 AgenticSciML Assistant 行为边界、scientific claim policy、
  ChatUI/tool policy、code-server sidecar policy 和 prompt-injection boundary。
- repo-local skill `.agents/skills/agenticsciml-chatui-operator/SKILL.md` 升级到
  `version: 0.2.0`，明确 `/api/solver/chat` 是 internal algorithm-tool endpoint，
  不是 MCP server。
- 新增未来 MCP wrapper 合约草案：tool listing、JSON Schema input/output、
  structured content、`readOnlyHint` / `destructiveHint` / `openWorldHint`、
  approval policy 和 non-goals。
- 更新 ChatUI 文档，记录 hardened public code-server sidecar 条件和
  `/api/solver/chat` / future MCP wrapper 的边界。

边界：

- 当前仍不发布 OpenAI App/MCP server。
- 不把 evaluator、selector、champion selection、artifact schema 或 score rewrite
  暴露为 tool。
- Pro 输出只作为外部建议；最终规则以仓库源码、测试、官方文档和本次 checked-in
  文档为准。

同日后续 UI 调整：

- `VS Code` 页改为工作空间选择入口，参考 UnitaryLab workspace 页的信息结构。
- 新增 `GET /api/code-server/workspaces`，列出 repo、run、champion 和
  `solutions/solution_*` 的独立代码目录及对应 code-server URL。
- 选择 workspace 后，AI IDE 编辑态只显示 VS Code Web iframe 和右侧 ChatUI
  侧边栏，不再显示实验工作台、workspace 说明块、启动命令或 sidecar 边界卡。

## 2026-05-16 ChatUI 实验操作台

新增本地优先 Web 控制面：[ChatUI 实验操作台](chatui_console.md)。

已实现能力：

- `agenticsciml web` 启动 FastAPI 后端，默认监听 `127.0.0.1:8765`。
- Web API 覆盖 benchmark catalog、run 启动/恢复、run status、SSE trace events、
  read-only artifact browsing、code-server URL 生成和 `/api/solver/chat` 算法 tool。
- `/api/solver/chat` 只做意图解析和结构化 action 返回，不接管 Python
  orchestrator 的状态机、evaluator、solution tree 或 champion selection。
- React/Vite ChatUI 前端提供 UnitaryLab 风格的分页式本地工作台：左侧功能栏
  切换 `ChatUI`、`VS Code Web` 和 `算法库`。
- `ChatUI` 页保持纯对话界面；`VS Code Web` 页只显示 code-server workspace
  和右侧可收起 ChatUI Agent 侧边栏；benchmark/run/gate、
  dashboard、trace 和 artifact 工作台集中放到 `算法库` 页。
- ChatUI 主入口用于自然语言交互和切换到算法库 / VS Code Web；实验运行与证据
  浏览从算法库页进入。
- 前端视觉调整为 Claude Code 风格的暖米色工作台，保留本地实验控制台的高密度
  信息结构。
- Agent 面板会展示 `/api/solver/chat` 返回的结构化 actions、warnings、
  artifact refs 和 trace refs；`real` mode action 默认拦截为显式确认状态。
- repo-local skill `.agents/skills/agenticsciml-chatui-operator/SKILL.md` 记录
  ChatUI 操作顺序、artifact 检查顺序、code-server 安全边界和禁止事项。

边界：

- 第一版只面向本地 loopback，不是公网 SaaS。
- code-server 由用户单独启动，默认使用 `--auth none`，但只能置于 loopback 或上游账号
  鉴权之后；ChatUI 只生成 workspace 链接。
- mock run 仍只支持 workflow-shape evidence，不支持科学复现结论。

## 2026-05-17 code-server 上游账号鉴权边界

code-server sidecar 的建议启动方式从本机 password 改为：

```bash
code-server --auth none --bind-addr 127.0.0.1:8080 <workspace>
```

API 返回的 `command_hint` 不再包含 `PASSWORD=`，并新增 `auth_mode=upstream_account`
标识。公开部署时必须由 ChatUI session、Cloudflare Access、nginx `auth_request` 或等价
账号鉴权保护 code-server 入口；不得把 `--auth none` 的 sidecar 直接暴露到公网。

## 2026-05-14 OpenAI Agents SDK 升级复盘

新增文档：[OpenAI Agents SDK 升级复盘](openai_agents_sdk_upgrade_review.md)。

本次是文档和路线复盘，没有修改运行时代码。结论是继续保持 Python
orchestrator 控制 evaluator、solution tree、checkpoint/resume 和 champion
selection；后续升级优先做 OpenAI-native Structured Outputs adapter、
provider capability matrix、真实 LLM budget gate、Pydantic agent output
schemas、container/sandbox backend、trace export/eval flywheel 和 human review
pause/resume。

同日后续实现已修复前五项：OpenAI-native Responses structured outputs、
provider capability matrix、真实 LLM token/cost budget gate、Pydantic closed
agent output schemas、以及 sanitized `openai_sdk_trace.json` trace export
bridge。剩余主要升级项转为 trace grading eval set、container/sandbox
backend、human review pause/resume 和 MCP/hosted tools sidecar。

同日临时禁用项目内 ChatGPT Pro 外部审计流程：`AGENTS.md` 现在要求不再
自动调用 `codex-chatgpt-pro-research`、ChatGPT Pro 或外部审计 brief；
只有用户明确恢复该策略或点名一次性 Pro review 时才允许重新启用。

同日新增 [Git 与 Markdown 分层记录方法论](git_markdown_methodology.md)：
把论文机制、OpenAI Agents SDK 边界、Git commit 时间轴、互链 Markdown
文档树和 run artifacts 统一为项目记录协议。后续架构、benchmark、SDK
对齐、claim boundary 和导师汇报，都应能从 `docs/index.md`、Git commit
和运行/测试证据互相追溯。

## v0.1.0 MVP

当前版本目标是复刻 AgenticSciML 的多 Agent workflow，而不是复现论文分数。

已实现能力：

- Python 3.11 package scaffold。
- Observation artifact layer：`DataAnalystAgent` 生成
  `reports/data_observations.json` 和 `reports/data_overview.svg`，把公开训练数据
  的 shape/statistics/overview plot 写入 run evidence；`ResultAnalystAgent`
  生成 `solution_observations.json` 和 `prediction_overview.svg`，只基于
  `predict_input.npz`、`predictions.npz`、`eval.json` 和日志摘要分析结果，保持
  private validation labels 不进入 downstream prompt。
- deterministic mock LLM adapter。
- optional OpenAI adapter。
- OpenAI-native structured output path：未设置 `OPENAI_BASE_URL` 时，
  `OpenAIAdapter` 使用 Responses parse + typed Pydantic output model；
  设置 `OPENAI_BASE_URL` 时保留 OpenAI-compatible chat/JSON fallback，并继续
  使用本地 Pydantic schema fail closed。
- OpenAI-compatible provider support：`OpenAIAdapter` 可通过
  `OPENAI_BASE_URL` 指向 DeepSeek 等兼容 endpoint，并通过
  `OPENAI_TIMEOUT_S` 设置 provider 请求超时；当 role 配置了
  `reasoning_effort` 时，兼容 Chat Completions 请求会携带同名参数。
- DeepSeek real-run hardening：`OpenAIAdapter` 会把 `deepseekv4pro` /
  `deepseekv4flash` 规范为 DeepSeek API 接受的 `deepseek-v4-pro` /
  `deepseek-v4-flash`；OpenAI-compatible chat JSON fallback 会把 Pydantic
  JSON Schema、array 字段规则和 no-extra-field 规则注入 prompt，并能从带说明
  或 markdown fence 的响应中提取首个 JSON object 后再做 closed schema 校验。
- Debugger patch hardening：`DebuggerOutput` 支持可选
  `full_file_map.solution.py`，当真实 LLM 返回的 unified diff context 与当前
  `solution.py` 不匹配时，可在 `parent_digest` 和 `files_changed=["solution.py"]`
  约束下用完整文件兜底，降低真实 provider patch 漂移导致的无效 repair。
- Debugger full-file repair contract：真实 DeepSeek run 证明可选
  `full_file_map` 仍可能被模型省略；Debugger 现在要求每次 repair 都返回完整
  `full_file_map.solution.py`，并在 prompt 中明确 `--mode=validate` 不得依赖
  `model.pkl` 或训练前置状态。
- Engineer full-file mutation contract：真实 DeepSeek resume 进一步暴露 Engineer
  patch context 漂移；Engineer 现在同样要求每次 mutation 返回完整
  `full_file_map.solution.py`，并显式提示 validate/checkpoint 与训练数据解析约束。
- training-data integrity guard：`validate` / `train` 阶段若返回 0 但日志显示
  训练数据加载失败或 synthetic fallback，runner 会 fail closed，避免真实 LLM
  solution 用自造数据通过 benchmark 流程。
- provider capability matrix：real-smoke manifest、ledger、trace metadata 和
  run metadata 记录 provider、adapter type、Responses/Structured Outputs/
  usage/trace export/prompt-cache capability。
- real LLM budget gate：`AGENTICSCIML_MAX_LLM_CALLS`、
  `AGENTICSCIML_MAX_PROMPT_TOKENS`、`AGENTICSCIML_MAX_OUTPUT_TOKENS`、
  `AGENTICSCIML_MAX_TOTAL_TOKENS`、`AGENTICSCIML_MAX_COST_USD` 和
  `AGENTICSCIML_COST_PER_1K_TOKENS_USD` 可让真实 LLM run 超预算 fail closed。
- `AgentSpec` / `PromptTemplate` / artifact guard 基础合同层。
- runtime `AgentSpec` enforcement：agent 方法入口校验 `input_schema`，JSON 输出默认校验 `output_schema`。
- Pydantic closed output schemas：Proposal、Evaluator review、RootEngineer、
  Selector、Engineer、Debugger 和 ResultAnalyst 等代码消费 JSON 输出拒绝
  unknown fields 与错误类型。
- OpenAI Agents SDK 对齐的结构化输出校验、JSON retry、trace span 和 guardrail 事件。
- structured output failure handling：LLM JSON parse/API/schema 失败会在 agent 边界转换为 `StructuredOutputError`，写入 guardrail trace，并在预算内重试。
- evaluator contract guardrail：检测 generated solution 是否篡改 `evaluate.py` 等受保护评估文件。
- benchmark-aware contract：`ProblemBundle` 和 `BenchmarkContractFactory` 为每个 benchmark 生成 hash-stable `evaluation_contract.json`，不再对所有任务静默使用 function approximation 默认值。
- contract-bound fidelity metadata：`EvaluationContract` 持久化 `benchmark_fidelity` 并纳入 `contract_hash`，包括 paper task name、paper section、`fidelity_level`、expected runtime、dependency flags 和 paper-gap notes。
- strict fidelity contract schema：`EvaluationContract.from_dict()` 对新 contract 强制要求 `benchmark_fidelity`，并校验 `schema_version`、未知字段、非空字符串字段和 proxy paper-gap notes。
- source-bound contract hash：`evaluation_contract.json` hash 覆盖 `evaluate.py`、`Data_config.json` 和 problem bundle digest；resume/load 会从磁盘重读 benchmark source，并拒绝 stale、tampered、缺失 contract、checkpoint/node contract mismatch 或 node contract metadata 缺失。
- benchmark source manifest：contract 记录 `BenchmarkSourceManifest`，覆盖 `Problem.md`、`Requirements.md`、`Evaluation.md`、`Data_config.json`、`evaluate.py`、`generate_data.py`、`guidelines.md`，以及 repo 中已有或 seed 0 生成的 `train_data.npz` / `val_data.npz` digest。
- manifest integrity guardrail：`EvaluationContract.from_dict()` 会校验 manifest 内容与 `benchmark_source_manifest_digest` 一致；生成数据 digest 时复用 sanitized subprocess env，不继承宿主 secrets。
- manifest schema governance：`BenchmarkSourceManifest` 记录 `schema_version`、`digest_algorithm`、`data_source_mode` 和 normalized `generator_command`；partial train/validation data artifact 会 fail closed。
- manifest semantic validation：`EvaluationContract.from_dict()` 会验证 manifest schema version、digest algorithm、data source mode、generated flag、generator command 和必需 artifact digests。
- atomic storage writes：`ExperimentStorage` 对 JSON、transcript、report 和 solution text artifact 使用同目录临时文件 + `os.replace()` 原子写入。
- thread-safe artifact writes：`ExperimentStorage` 在并行 child jobs 下用进程内锁保护 workspace 创建、trace append、JSON、transcript、report 和 solution artifact 写入。
- validation leak guardrail：`val_data.npz` 放在 run-private `private_eval/solution_*/` 目录，generated `solution.py` 的 validate/train cwd 下不存在 evaluator-private 目录或验证集。
- benchmark generator privacy：generated solution workspace 不再复制 `generate_data.py`；缺少数据文件时由 trusted runner 从 benchmark 目录调用 generator，避免 faithful-small 任务把闭式目标函数源码暴露给 candidate solution。
- prediction-only evaluation：可信代码只把 `x_val` 写入 `predict_input.npz`，generated `solution.py --mode=predict` 写 `predictions.npz`，`evaluate.py` 使用私有 `u_val` 计算分数且不 import `solution.py`。
- clean subprocess env：generated solution validate/train/predict/evaluate 使用最小安全环境，不继承宿主 API key、代理、SSH agent、真实 `HOME` 等变量。
- agent context hardening：RootEngineer / Engineer prompt 显式包含 `ProblemBundle`、`EvaluationContract` JSON、`guidelines.md` 和可用分析上下文。
- patch-based mutation：Engineer 输出包含 `parent_digest` 和 patch/file map；Python 端校验 parent digest 后才写入 `solution.py`。
- patch fallback hardening：Engineer 仍优先使用 digest-checked unified diff，
  但当 provider 同时返回显式 `full_file_map.solution.py` 且 patch context
  不匹配时，可用 file map 兜底，减少真实兼容模型长 patch 的上下文漂移失败。
- failed mutation containment：Engineer patch/schema 失败会生成 failed child node、`engineering_error.md` 和 guardrail trace，不再中断整次 run。
- contract-aware debugger：Debugger prompt 显式包含当前代码、`parent_digest`、`ProblemBundle`、`EvaluationContract` JSON、`guidelines.md`、失败阶段和错误日志；修复只能通过 digest-checked unified diff patch 修改 `solution.py`。
- search policy metadata：`SolutionNode` 持久化 `method_tags`、`failure_kind`、`score_delta_from_parent`、`num_debug_attempts`、`benchmark_name` 和 `contract_hash`，供 selector、retriever 和 ablation 使用。
- vote-aware parent selection：mature stage 先由 loss 选择 best available node 做 exploitation，
  再由 selector vote 选择额外 exploration parents，并把 vote counts 写入
  `reports/selector_votes.json`；不足名额才由 deterministic `SearchPolicy`
  用 recent improvement、diverse underexplored node 和 `max_children_per_node`
  约束补齐。当前默认是单 LLM provider 的 `selector_vote_count=3`，不是论文里的
  GPT/Grok/Gemini 三模型 ensemble；CLI 可用 `--selector-vote-count` 调整重复投票次数。
- parallel child mutation jobs：当 `parallel_mutations > 1` 时，orchestrator 用 bounded `ThreadPoolExecutor` 并行创建 child solution，并写入 `agenticsciml.parallel_children.*` trace。
- early-stage mutation fanout：早期只有 root 或 selector 返回 parent 数不足时，orchestrator 会按 mutation budget 对可用 parent 做 deterministic fanout，从同一 parent 生成多个 child 分支，同时遵守 `max_children_per_node`。
- mutation budget semantics：`parallel_mutations` 同时限制每轮 child job 数和 worker 数；并行 trace 记录 canonical `parent_child_edges`、slot-level `parent_ids`、`unique_parent_ids`、legacy parent-to-child / canonical parent-to-children 映射、duration、child-level start/end、status 和 failure kind。
- fanout trace schema gate：`trace_summary.json` 要求 `parallel_children.start/end` 携带完整 canonical fanout schema，并校验 `parent_child_edges`、`parent_to_children`、`unique_parent_ids`、top-level `child_ids` / `parent_ids` 和 legacy `parent_to_child` 之间的一致性；缺字段或 malformed fanout trace 会 fail closed。
- shared fanout trace contract：`FanoutTraceMetadata` 是 fanout trace 的共享 typed contract，orchestrator 写 trace 前和 trace summary 读 trace 时复用同一套 schema 约束。
- resume-safe solution ID allocation：新 child ID 从已加载 node 和现有 `solutions/solution_*` workspace 的最大 numeric suffix 后继续分配，不再依赖 `len(nodes)`；遇到 malformed existing node ID 会 fail closed。
- branch context for fanout：同一 parent 的多个 fanout child 会写入 `branch_context.json`、child mutation trace metadata 和 proposer/engineer prompt，并在 `method_tags` 里记录 `branch:<intent>`，用于审计 sibling branch diversity；这仍是 diversity intent/evidence，不等同于真实论文级 emergent discovery。
- branch-context ablation switch：`EvolutionConfig.use_branch_context`、CLI `--no-branch-context` 和 ablation variants `branch_context` / `no_branch_context` 支持后续真实 LLM smoke 对比；mock ablation 只验证开关与 artifact。
- branch-context evidence flag：`run_metadata.json`、workflow-start trace 和 child mutation trace 显式记录 `branch_context_enabled`，避免把存在但为空的 `branch_context` 字段误读为启用了分支上下文。
- real LLM smoke scaffold：`agenticsciml smoke-llm` 默认 dry-run，无 key 生成 `real_llm_smoke_plan.json` / `real_llm_smoke_report.md`；真实模式必须显式 `--real`，会要求 `OPENAI_API_KEY` 并运行 branch_context/no_branch_context 最小对比。
- real LLM smoke gate：真实 smoke 是 exact paired contrast，必须且只能包含 `branch_context` 和 `no_branch_context`；会先写 `real_llm_smoke_manifest.json`，读取 `trace_summary.json`、要求两侧正数 LLM call count，验证 request-side branch-context prompt-delivery 证据，并检查 no-branch request prompt 不泄漏分支字段。gate 失败时 CLI non-zero，而不是只输出正常报告。
- real LLM smoke verifier：新增 `agenticsciml verify-smoke-llm <output_dir>`，可在真实 smoke 后离线校验 plan/manifest/runs CSV、重算 paired gate，并检查 `parallel_mutations > 1` 时的 parallel-child trace evidence。
- smoke verifier provenance：`verify-smoke-llm` 会锚定 `plan.output_dir` / `manifest.output_dir` 到当前 evidence bundle，并把 CSV/manifest 数字字段类型错误记录为 fail-closed issue，而不是让 verifier crash。
- smoke verifier schema：`verify-smoke-llm` 对 seed、`parallel_mutations`、`expected_llm_call_range.min/max` 使用 strict integer schema，拒绝 bool、float、空缺 bounds 和不完整 call-range。
- smoke verifier diagnostics：manifest schema 会在 run rows 损坏时仍独立报告，parallel-child trace 的 `max_workers` 也使用 strict integer schema，避免坏 evidence 让 verifier crash。
- smoke verifier LLM-call schema：`run_metadata.json` 的 `llm_calls.total` 使用 strict positive integer schema，manifest `provider` / `model` 必须是非空字符串。
- smoke LLM-call ledger：`smoke-llm` 为每个 real/mock smoke run 写入 hash-only `llm_call_ledger.jsonl`，`verify-smoke-llm` 会对账 ledger、`run_metadata.llm_calls.total`、generation spans 与 manifest provider/model。
- smoke ledger contract：`verify-smoke-llm` 校验每条 ledger entry 的连续 `call_id`、provider/model/method/schema、SHA-256 hashes、success/duration/timestamp，并拒绝 raw prompt/response 字段。
- smoke ledger closed schema：ledger entry 采用 exact-key allowlist，记录 `span_kind="generation_span"`，并校验 `temperature` 为 finite number，防止未知 raw payload 字段混入公开 evidence bundle。
- smoke ledger trace bijection：smoke-only LLM wrapper 会把 `llm_call_id`、provider/model/method/schema 元数据注入 `generation_span` trace，verifier 要求 ledger call 序列与 trace span 序列逐项一致。
- smoke trace keyed bijection：ledger/trace 对账按 `llm_call_id` canonical fingerprint 比较，不依赖并发完成顺序；缺失、重复或字段不一致的 trace metadata 会 fail-closed。
- CLI editable-install note：`pyproject.toml` 明确 `src` package-dir；README/AGENTS 记录路径含空格时可用 `PYTHONPATH=src python -m agenticsciml.cli` 作为本地 CLI fallback。
- CLI smoke invocation boundary：`python -m agenticsciml.cli smoke-llm` / `verify-smoke-llm` 覆盖 checkout 路径和 output-dir 同时含空格的 dry-run bundle，并验证 dry-run artifact 会被 verifier 明确拒绝为真实 LLM evidence。
- benchmark claim boundaries：`agenticsciml benchmarks --json` 输出每个 benchmark 的 `claim_boundaries`，普通列表也显示 `fidelity_level` 和 real-LLM `scientific_claim`，避免把 proxy catalog 误读成论文全量 SciML 复现。
- claim-boundary digest separation：`BenchmarkSpec.contract_digest_metadata()` 将 evaluator contract hash 绑定到任务/指标/fidelity 元数据，而不把公开 catalog 的 `claim_boundaries` 文案变化纳入 contract hash。
- faithful-small Burgers / antiderivative expansion：新增
  `examples/burgers_pinn_faithful_small` 和
  `examples/antiderivative_operator_faithful_small`，分别加入
  IC/BC/collocation 结构与 100 点函数到反导数的 operator-learning 结构；
  二者仍是 low-budget faithful-small，不声明论文分数复现。
- faithful-small reaction-diffusion expansion：新增
  `examples/reaction_diffusion_operator_faithful_small`，把 S1.5 升级为
  diffusion/source/initial 三通道输入到 40x50 时空响应的 multiple-input
  operator-learning 任务；官方分数仍是 mean per-sample relative L2，时间和
  梯度指标只作为 diagnostics。
- faithful-small benchmark seed：新增 `examples/poisson_lshape_faithful_small`，在 L-shaped Poisson 任务中加入 boundary/residual collocation 数据和 finite-difference residual composite score，用于缩小 proxy 与论文 PINN 任务结构的差距；仍不声明 paper-like 分数。
- faithful-small function approximation：新增 `examples/function_approx_faithful_small`，
  使用论文 S1.1 描述的分段振荡函数、200 个训练样本和 500 个验证样本；
  problem prompt 不暴露闭式函数，仍不声明论文分数复现。
- benchmark-aware retrieval query：`RetrievalQueryBuilder` 使用 benchmark family/metric/description、parent analysis、failure kind、method tags 和 leaderboard top-k 生成检索 query。
- KB ablation switches：`use_kb=False` 不注入 KB，`random_kb=True` 使用 seed-controlled random KB retrieval。
- ablation runner：`agenticsciml ablate` 和 `scripts/run_ablation.py` 生成 `ablation_runs.csv`、`ablation_summary.csv`、`ablation_report.md`，支持 `root_only`、`no_kb`、`kb`、`random_kb`、`no_critic`、`no_debugger`。
- ablation metrics：每个 variant 汇总 champion score、root score、champion/root improvement、valid solution rate、timeout count、debug success count、LLM call count 和 wall time。
- direction-aware ablation summary：ablation run rows 记录 `higher_is_better`，summary 中的 champion best/worst 会按 metric direction 聚合，避免未来 accuracy/R2 类 benchmark 仍按 lower-is-better 解释。
- mock evidence boundary：ablation run 和 summary 输出显式记录 `evidence_mode=mock_workflow_shape`、`scientific_claim=not_supported`，避免把 mock 分数误解为 emergent discovery。
- run evidence boundary：`run_metadata.json` 和 workflow-start trace 记录 `llm_mode`、`benchmark_fidelity_level`、`evidence_mode` 和 `scientific_claim`，避免单个 run artifact 被过度解读。
- centralized evidence constants：`agenticsciml.evidence` 集中定义 `llm_mode`、`evidence_mode` 和 `scientific_claim`，减少字符串漂移。
- static sandbox guardrail：运行前阻断网络模块、子进程、危险文件操作和明显绝对路径写入。
- checkpoint/resume：每轮关键阶段写 `checkpoint.json`，CLI 支持 `--resume` 继续已有 run。
- trace summary：每次 orchestrator 完成后写入 `trace_summary.json`，并提供 `agenticsciml trace-summary <run_dir>` 重新生成和检查 trace quality gate。
- SDK-style trace export：每次 orchestrator 完成后写入 sanitized
  `openai_sdk_trace.json`，CLI 提供 `agenticsciml export-sdk-trace <run_dir>`。
- trace artifact consistency gate：`trace_summary.json` 会检查 `evaluation_contract.json`、`run_metadata.json`、workflow-start trace、`tree.json` 和 `checkpoint.json` 中的 fidelity/evidence、node set、contract hash、benchmark name 与 solution count 一致性；不一致时 quality gate fail closed。
- completed-run artifact requiredness：当 `run_metadata.run_state` 为 `completed`、`exported` 或 `finalized` 时，`tree.json` 和 `checkpoint.json` 被视为必需 artifact；缺失会使 trace quality gate fail closed。旧 metadata 无 `run_state` 时才回退到 `solution_count`。
- run metadata：`run_metadata.json` 和 workflow-end trace 记录 `run_state=exported`、wall time、champion、solution count，以及按 role 汇总的 LLM 调用次数和 prompt/response token 估算占位；workflow-start trace 记录 `run_state=partial`。
- run state schema：trace summary 校验 `run_state` 只能是 `partial`、`completed`、`exported` 或 `finalized`，要求 exported run state 必须有 workflow-end trace，并要求 `run_metadata.run_state` 与 workflow-end trace `run_state` 一致。
- existing-run guard：非 `--resume` run 遇到已存在且非空的 run directory 会 fail closed，避免复用 `--experiment-id` 时把旧 trace 和新 tree 混在一起。
- lifecycle trace ordering：trace summary 会拒绝 workflow-end 早于 workflow-start 的 trace，也会拒绝同一 run 中互相冲突的 workflow-end `run_state`。
- trace event sequence：`ExperimentStorage.record_trace()` 为新 trace event 写入连续递增的 `event_seq`；trace summary 对 exported run artifact 校验 `event_seq` 必须完整且单调。
- trace node reference integrity：trace summary 会在 allowlisted solution lifecycle events 上检查 trace metadata 中的 `solution_id`、`parent_id`、`child_id`、ID 列表和 `parent_to_child` 映射是否都能在最终 `tree.json` / `checkpoint.json` node set 中找到，避免误伤非 solution 语义 metadata。
- trace node reference audit counters：`trace_summary.json` 输出 allowlist 中被检查的 solution-reference event 数、被跳过的非 solution event 数、携带实际 node reference 的 event 数、实际 checked node reference 数，以及按 event name 聚合的分布；exported/completed/finalized run 若有最终 node set 但没有任何 checked solution-reference event 或实际 checked node reference，会使 quality gate fail closed，partial/resume run 不因 zero checked 被误伤。
- trace node coverage gate：`trace_summary.json` 输出最终 node set 的 referenced/unreferenced 覆盖情况和每个 node 的 `referenced_by_event_names`、`reference_keys`、`self_reference_count`、`relation_reference_count`；exported/completed/finalized run 中任何 final node 没有 allowlisted trace reference，或只作为 relation endpoint 出现而没有 self trace reference，都会使 quality gate fail closed。
- trace lifecycle stage coverage：`trace_summary.json` 输出每个 node 的 lifecycle stages 和对应 event names；exported/completed/finalized run 使用 status-aware required stage matrix，`status=evaluated` 的 root node 需要 `materialized` 和 `evaluated`，child node 还需要 `created` 和 `completed`。
- trace lifecycle order gate：`trace_summary.json` 输出每个 lifecycle stage 的 `event_seq`，并校验完成态 root `materialized < evaluated`、child `created < materialized < evaluated < completed`。
- solution tree graph invariants：`trace_summary.json` 校验 `tree.json` 和 `checkpoint.json` 都有唯一 root、非 root parent 存在、parent links 无环、`children` 必填且无重复、`children` 只指向存在节点，并要求 parent `children` 与 child `parent_id` 双向一致。
- solution node schema gate：`trace_summary.json` 在 graph invariant 前校验 final node required fields、status enum、nullable string fields、score schema、method tags、debug count 和 score delta 类型。
- shared solution node load schema：`trace_summary.json` 与 `SolutionNode.from_dict()` 复用同一套 schema validator；resume/load 遇到坏 `checkpoint.json` node payload 会在加载边界 fail closed，而不是只在事后 trace 报告中暴露。
- shared solution tree load schema：resume/load 与 `trace_summary.json` 复用 solution-tree graph invariant validator；坏 `checkpoint.json` 拓扑（缺失 parent、环、重复 child、children/parent_id 不一致等）会在进入进化搜索前 fail closed。
- finite score schema：solution node `score.value` 和 `score_delta_from_parent` 必须是 finite number；`NaN`、`Infinity`、`-Infinity` 会在 node schema、trace summary 或 JSON artifact 写入时 fail closed。空 solution tree 也会在 resume/load 边界被拒绝。
- status-aware node semantics：solution node schema 现在校验 lifecycle status 语义，`evaluated` 必须有 score 且无 error/failure kind，`failed` 必须无 score 且有 error 或 failure kind，`created` 不能携带 score/error/failure kind。
- closed node schema：solution node 与 score payload 默认拒绝 unknown fields；`from_dict()`、resume/load 和 trace summary 会对 schema drift fail closed。`tree.json` 和 `trace_summary.json` 写入也使用 `allow_nan=False`。
- solution tree schema version：`tree.json` 和 `checkpoint.json` 顶层写入 `schema_version=solution_tree.v1`；resume/load 和 trace summary 会拒绝缺失或 unsupported schema version，后续迁移必须显式版本化。
- artifact path semantics：resume/load 与 trace summary 都会校验 node `workspace` 位于当前 run 的 `solutions/` 下且 basename 匹配 `node_id`，并要求 `proposal_path` / `analysis_path` 位于 node workspace 内且存在，防止 checkpoint/tree 指向外部 artifact。
- benchmark catalog：6 类论文任务家族都有本地 deterministic engineering proxy，包括 `function_approx`、`poisson_lshape`、`burgers_pinn`、`antiderivative_operator`、`reaction_diffusion_operator`、`cylinder_wake_reconstruction`；另有 `function_approx_faithful_small`、`poisson_lshape_faithful_small`、`burgers_pinn_faithful_small`、`antiderivative_operator_faithful_small`、`reaction_diffusion_operator_faithful_small` 和 `cylinder_wake_reconstruction_faithful_small` 作为 faithful-small 升级。每个 catalog entry 记录 `fidelity_level`、expected runtime、dependency flags 和 paper-gap notes。
- benchmark metadata validation：`BenchmarkSpec` 构造时校验 `fidelity_level`、expected runtime、dependency flags、paper task name 和 proxy paper-gap notes。
- fidelity levels document：`docs/fidelity_levels.md` 定义 `proxy`、`faithful-small`、`paper-like` 的准入标准和 claim rules。
- evaluation contract：`solution.py` 定义 `MODEL`，支持 validate/train/predict，predict 写 `predictions.npz`，`evaluate.py` 输出 `eval.json`。
- per-solution workspace 和 artifact persistence。
- solution tree、leaderboard、Mermaid tree 和 champion export。
- DataAnalyst、Evaluator、RootEngineer、Retriever、Proposer、Critic、Engineer、Debugger、ResultAnalyst、Selector 等角色 wrapper。
- 0-1 KB retrieval。
- proposer/critic 4-round proposal flow。
- 独立 `CriticAgent` artifact：`critic.md` 和 per-round transcript。
- root-only / no-KB / KB / random-KB ablation script。

已验证命令：

```bash
uv run --python 3.11 --extra dev pytest -q
uv run --python 3.11 --extra dev pytest tests/test_benchmark_catalog.py -q
uv run --python 3.11 --extra dev pytest tests/test_patch_mutation.py tests/test_orchestrator_cli.py -q
uv run --python 3.11 --extra dev pytest tests/test_ablation.py -q
uv run --python 3.11 --extra dev pytest tests/test_ablation.py::test_ablation_aggregate_respects_score_direction -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_reports_trace_node_reference_check_counts tests/test_trace_reporting.py::test_trace_summary_fails_when_exported_run_checks_zero_solution_reference_events -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_exported_run_checks_no_actual_solution_node_references -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_exported_node_is_not_referenced_by_trace -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_exported_node_only_has_relation_reference -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_evaluated_node_has_no_evaluated_stage -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_evaluated_root_has_no_materialized_stage -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_lifecycle_stage_order_is_invalid -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_on_missing_solution_tree_parent tests/test_trace_reporting.py::test_trace_summary_fails_on_invalid_solution_tree_child_link tests/test_trace_reporting.py::test_trace_summary_fails_on_solution_tree_parent_cycle -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_children_field_is_missing tests/test_trace_reporting.py::test_trace_summary_fails_on_duplicate_solution_tree_child_link tests/test_trace_reporting.py::test_trace_summary_fails_when_child_is_missing_from_parent_children -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_solution_node_required_field_is_missing tests/test_trace_reporting.py::test_trace_summary_fails_when_solution_node_status_is_invalid tests/test_trace_reporting.py::test_trace_summary_fails_when_solution_node_score_shape_is_invalid -q
uv run --python 3.11 --extra dev pytest tests/test_state_storage.py::test_solution_node_from_dict_rejects_missing_required_field tests/test_state_storage.py::test_solution_node_from_dict_rejects_invalid_status tests/test_state_storage.py::test_solution_node_from_dict_rejects_invalid_score_shape tests/test_orchestrator_cli.py::test_resume_rejects_invalid_solution_node_schema -q
uv run --python 3.11 --extra dev pytest tests/test_orchestrator_cli.py::test_resume_rejects_invalid_solution_tree_graph -q
uv run --python 3.11 --extra dev pytest tests/test_state_storage.py::test_solution_node_from_dict_rejects_non_finite_score_values tests/test_state_storage.py::test_solution_tree_payload_rejects_empty_node_list tests/test_state_storage.py::test_storage_rejects_non_finite_json_artifacts tests/test_orchestrator_cli.py::test_resume_rejects_empty_solution_tree tests/test_trace_reporting.py::test_trace_summary_fails_when_solution_node_score_is_not_finite -q
uv run --python 3.11 --extra dev pytest tests/test_state_storage.py::test_solution_node_from_dict_rejects_invalid_status_semantics tests/test_orchestrator_cli.py::test_resume_rejects_invalid_solution_node_status_semantics tests/test_trace_reporting.py::test_trace_summary_fails_when_solution_node_status_semantics_are_invalid -q
uv run --python 3.11 --extra dev pytest tests/test_orchestrator_cli.py::test_resume_rejects_solution_node_artifact_path_escape tests/test_orchestrator_cli.py::test_resume_rejects_solution_node_workspace_path_escape -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_fails_when_solution_node_artifact_path_escapes_run -q
uv run --python 3.11 --extra dev pytest tests/test_state_storage.py::test_solution_node_from_dict_rejects_unknown_fields tests/test_orchestrator_cli.py::test_resume_rejects_solution_node_unknown_fields tests/test_trace_reporting.py::test_trace_summary_fails_when_solution_node_has_unknown_fields -q
uv run --python 3.11 --extra dev pytest tests/test_state_storage.py::test_solution_tree_artifact_payload_rejects_unsupported_schema_version tests/test_orchestrator_cli.py::test_resume_rejects_unsupported_solution_tree_schema_version tests/test_trace_reporting.py::test_trace_summary_fails_on_unsupported_solution_tree_schema_version -q
uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_allows_partial_run_with_zero_solution_reference_events -q
uv run --python 3.11 --extra dev agenticsciml benchmarks
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 1
uv run --python 3.11 --extra dev agenticsciml trace-summary runs/<experiment_id>
uv run --python 3.11 --extra dev agenticsciml ablate examples/function_approx --seeds 0 --variants root_only,kb --output-dir runs/ablation-smoke
uv run --python 3.11 --extra dev python scripts/run_ablation.py --output-dir runs/ablation-smoke
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --dry-run
```

边界：

- 不保证论文报告的 improvement factor。
- 不假设官方完整代码、完整 prompt 或模型配置可用。
- mock mode 只验证 workflow shape，不代表真实 SciML 表现。
- 当前 benchmark 主要仍是 `fidelity_level=proxy` 的小规模工程代理；faithful-small 任务只缩小对应任务结构差距，不是论文原始全预算实验。
- `parallel_mutations` 已并行执行多个 child mutation job，并支持早期单 parent fanout；但训练仍在本机 CPU/subprocess 资源上竞争，且不是论文完整分布式 ensemble search。
- 当前 sandbox 是本地 workspace 隔离，不是强安全容器。
- prediction-only protocol 降低验证标签泄漏风险，但还不是 OS/container 级强隔离；同用户进程仍不能视为恶意代码安全沙箱。

后续版本建议：

- `v0.2.0`：支持 run resume、阶段 checkpoint 恢复和更完整的 run metadata。
- `v0.3.0`：增强 sandbox，限制网络、文件系统和资源消耗。
- `v0.4.0`：把 Poisson PINN、Burgers PINN 和 operator benchmarks 升级到更接近论文设置的数据规模与训练预算。
- `v0.5.0`：真实 LLM ablation，比较 root-only、no-KB、KB、no-critic、no-debugger。
- `v0.6.0`：加入成本统计、token 估算、trace viewer、trace grading 报表和失败案例报告。
