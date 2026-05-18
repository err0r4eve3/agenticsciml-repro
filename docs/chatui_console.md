# ChatUI 实验操作台

[返回文档树](index.md) · 相关文档：[项目概览](../README.md)、[多 Agent 设计方法](multi_agent_design.md)、[AgenticSciML Assistant 规范](agenticsciml_assistant.md)、[版本说明](version_notes.md)

本地 ChatUI 控制台是现有 Python orchestrator 的外层操作面，不替代
`AgenticSciMLOrchestrator`、evaluation contract、solution tree、checkpoint/resume
或 champion selection。

## 启动

```bash
uv run --python 3.11 --extra web agenticsciml web --host 127.0.0.1 --port 8765
cd frontend
npm install
npm run dev
```

如果 checkout 路径包含空格并导致 editable console script 无法 import
`agenticsciml`，后端改用 module form：

```bash
PYTHONPATH=src uv run --python 3.11 --extra web python -m agenticsciml.cli web --host 127.0.0.1 --port 8765
```

打开 Vite 输出的本地地址。生产预览可先 `npm run build`，然后让 FastAPI 挂载
`frontend/dist`。

## 工作台布局

ChatUI 前端是单页本地工作台，不引入路由层，但用左侧功能栏做页内切换：

- 左侧功能栏选择 `ChatUI`、`VS Code` 或 `算法库`。
- 左侧底部的本地账号选择器只切换 workspace namespace，不是登录系统。默认
  `local` 账号会使用 `.agenticsciml/accounts/local/` 下的独立代码目录和 runs
  目录；新建账号只创建本地目录和 `account.json` 元数据，不写入密钥。
- `ChatUI` 页是纯对话界面：顶部只做页面标识，中间是对话流，底部是主输入框；
  不显示 benchmark、run、quality gate、trace 或 artifact 工作台。
- ChatUI 和右侧 Agent 面板都有 `ask` / `plan` / `agent` 三种交互模式，默认
  `ask`。`ask` 直接回答身份、能力、项目、benchmark、算法、run、trace、artifact
  和边界问题，动作型请求只解释边界且不返回可执行动作；`plan` 返回结构化建议动作
  但前端不自动分发；`agent` 才会自动执行受控 action。
- `VS Code` 页先显示工作空间选择列表，风格参考 UnitaryLab 的 workspace
  页：用户选择一个账号隔离的独立代码目录后进入 AI IDE。编辑态只保留 VS Code Web
  iframe 和右侧可拖动调整宽度、可收起的 ChatUI 侧边栏，不显示 benchmark、run、
  quality gate、artifact、启动命令或 sidecar 说明块。VS Code Web 和侧边栏之间只保留
  弱分隔线，避免形成明显的双栏卡片边框。
- `算法库` 页承载 Paper Run Lab：benchmark、运行模式、run/gate 状态、S1 论文任务
  映射、run budget、分层模型配置、selector votes、solution loss/tree、leaderboard、
  trace preview 和 artifact 浏览。
- ChatUI 自然语言输入仍调用 `/api/solver/chat`；`open_code_server` action 会切到
  `VS Code Web` 页，`summarize_artifact` action 会切到 `算法库` 页。
- Agent 侧边栏负责自然语言意图、结构化 actions、warnings、artifacts 和 trace
  refs 展示。
- 视觉语言采用 Claude Code 风格的暖米色工作台：纸面背景、深棕文字、陶土色主
  action、低饱和边框和暗色 trace/code 区域。

Agent 面板只分发受控动作。`mock` 和 `dry_run` actions 可由前端调用现有 API
执行；`real` mode action 默认拦截为待确认状态，不会隐式触发真实 LLM 调用。

## Paper Run Lab

第三页把原算法库升级为论文对齐实验页，但仍是控制面和证据浏览器，不是新的
evaluator、selector 或 champion selection 实现。

页面结构：

- `Paper Tasks`：按 `S1.1` 到 `S1.6` 展示本地 benchmark 映射、paper reference
  primitive、figure label 和 claim boundary。页面只引用论文小标题/标签，不嵌入未授权
  论文原图。
- `Problem Intake`：高阶用户可输入完整问题描述、requirements、evaluation 和 data
  description。`POST /api/problem-intake/plan` 会在当前本地 benchmark catalog 内自动评选
  benchmark 候选、算法 seed 候选、run budget 和 start_run action；这一步是受控规划层，
  不会生成新 evaluator 或绕过 contract。
- `Run Config`：可设置 `target_solution_count`、`max_iterations`、
  `parallel_mutations`、`selector_vote_count`、`max_children_per_node` 和 `mode`。
  `target_solution_count` 是 UI 便捷输入；后端会转换成确定性的
  `EvolutionConfig.max_iterations + parallel_mutations` 预算，并在预览区显示
  `root + children`。同一区域可手动刷新 `Run Readiness`，在启动前查看
  claim-boundary warning、strategy seed 状态、real-mode blocker 和 artifact capture
  要求。
- `Strategy Locks`：记录用户想强制保留的约束、数学直觉、invariant、modeling choice
  或 assumption。当前前端会把非空 locks 送入 readiness 和 run request；后端只把它们
  当作用户假设/约束保存和审计，不当作科学事实。若请求体中提供了明确的机器可审计
  条件（例如 `required_terms`、`forbidden_call_names` 或 `inspection.required_imports`），
  orchestrator 会在执行 `solution.py` 前写入 `policy_fidelity_report.json` 并 fail closed
  于 required 条件失败的候选解。
- `Layered model routing`：列出 Data Analyst、Evaluator、Root Engineer、
  Retriever、Proposer、Critic、Engineer、Debugger、Result Analyst 和 Selector。
  默认仍使用后端单一 adapter；只有填写 role override 时，后端才按 role 创建模型配置。
  `reasoning_effort` 当前允许 `low | medium | high | xhigh`，并会写入配置、
  metadata 和传给支持该字段的 OpenAI-native Responses / OpenAI-compatible Chat
  Completions provider。未填写 override 时，后端按 role 默认 policy 生效：
  `evaluator=0.00/high`、`root_engineer=0.10/xhigh`、`engineer=0.10/xhigh`、
  `debugger=0.05/xhigh`、`proposer=0.55/xhigh`、`critic=0.35/high`、
  `data_analyst=0.35/high`、`result_analyst=0.30/high`、`selector=0.05/high`、
  `retriever=0.00/medium`。
- `Evidence`：只读展示 `reports/selector_votes.json`、`tree.json`、
  `leaderboard.csv`、各 `solutions/solution_*/eval.json` 汇总出的 votes、loss/score、
  parent/tree summary、method tags、`policy_fidelity`、`emergence_audit` 和本地 SVG artifact。
- `Local figures`：优先列出 `reports/data_overview.svg` 与
  `solutions/*/prediction_overview.svg`。未来 loss curve artifact 可按同样路径规则接入。

边界文案统一使用 `faithful-small`、`proxy`、`not paper-score evidence`。mock run
只表示 workflow shape；即使 UI 显示 loss、votes 或 figure，也不能写成论文分数或科学结论。

算法库支持人工选择。选中的 algorithm id 会作为 `selected_algorithm_ids` 进入
`POST /api/runs`，写入 `config.json` 的 `strategy_seed_ids` 和 `run_metadata.json`，
并作为 strategy seed context 注入 Root Engineer、Proposer 和 Engineer prompt。它们
只影响候选解法构思，不替代 evaluator 或真实分数。

Problem Intake 的完整输入会作为非权威 run context 一起传入 `POST /api/runs`：
后端写入 `config.json`、`run_metadata.json` 和 `planning/problem_intake.json`，
并用固定边界文案注入 Root Engineer、Proposer 和 Engineer prompt。该上下文只能
指导生成策略；`ProblemBundle`、`EvaluationContract`、benchmark guidelines、sandbox
规则和 evaluator contract 仍是评分与安全边界的上位事实来源。

右侧 Agent 面板在 `agent` 模式下也可处理高上下文求解请求：当用户要求自动选择
benchmark / 解法并启动 mock/dry-run/real run 时，`/api/solver/chat` 会复用
Problem Intake planner 生成 `start_run` action。前端执行该 action 时必须使用 action
payload 中的 benchmark、run budget、`selected_algorithm_ids`、`problem_intake` 和
`planner_snapshot`，不能退回当前 UI 里的陈旧选择。

## code-server sidecar

code-server 不由 ChatUI 自动启动。默认取消 code-server 自身 password，由 ChatUI /
反向代理的账号鉴权保护入口；sidecar 本身只绑定 loopback 或受限内网地址：

```bash
code-server --auth none --bind-addr 127.0.0.1:8080 /path/to/workspace
```

默认 API 生成 `http://127.0.0.1:8080/?folder=<workspace>` 链接。可用
`AGENTICSCIML_CODE_SERVER_URL` 覆盖 base URL。shared repo workspace 默认不通过
Web API 暴露；本地开发需要打开仓库根目录时，必须显式设置
`AGENTICSCIML_ALLOW_REPO_WORKSPACE=1`。

公网部署只能作为显式配置的 hardened sidecar：必须有 TLS、上游账号鉴权
（例如 ChatUI session、Cloudflare Access 或 nginx auth_request）、受限 workspace、
最小权限和 secret scanning。不要把 `--auth none` 的 code-server 直接暴露到公网。
即使公网部署，ChatUI 也不得把 token、password、API key、cookie、真实 `HOME`、
浏览器 profile 或私有数据集路径放进 URL 或消息体。

`VS Code` 页通过 `GET /api/code-server/workspaces` 列出可打开的独立代码目录。
未传 `account_id` 时只列出 shared run 目录、champion 目录和各
`solutions/solution_*` 目录；只有 `AGENTICSCIML_ALLOW_REPO_WORKSPACE=1` 时才会额外
列出 repo 根目录。传入 `account_id` 时只列出该账号 namespace 下的 `workspace/`、
`runs/<run_id>/` 和 `runs/<run_id>/solutions/solution_*` 目录；不混入 shared repo
根目录。选择后 iframe 使用对应 `?folder=<workspace>` 打开该
目录。ChatUI 不携带 code-server token，不自动启动 sidecar，iframe 入口必须先经过
整体账号鉴权。

账号隔离是本地目录隔离，不是多用户认证或权限边界。公开部署时还必须由反向代理/
账号 session、系统用户/容器权限和 secret scanning 提供真实访问控制。

## API 边界

- `GET /api/benchmarks`：读取当前 benchmark catalog。
- `GET /api/algorithms`：读取算法策略目录。目录项只用于规划和 prompt seed，不
  代表已经通过 evaluator 的实现。目录项同时暴露 English / 中文说明、`features`
  / `features_zh`、`problem_fit` / `problem_fit_zh` 和中文边界提醒，前端第三页可
  在 `中文` / `EN` 间切换。
- `GET /api/paper-tasks`：读取 S1 小标题、本地 benchmark 映射、reference primitive、
  local figure artifact 约定和 claim boundary。
- `GET /api/agent-roles`：读取可配置的 agent role 列表、分层模型配置说明和
  每个 role 的默认 `temperature` / `reasoning_effort` policy。
- `POST /api/problem-intake/plan`：从完整问题描述生成本地 benchmark 推荐、
  algorithm rankings、selected strategy seeds、run budget、`problem_intake`、
  `planner_snapshot` 和可展示的 start_run action。
- `POST /api/run-readiness/preview`：启动前生成确定性 pre-run audit。它只检查本地
  benchmark fidelity、claim boundary、selected strategy seeds、人工 strategy locks、
  branch inheritance expectations、run budget、real-mode gates 和 artifact capture
  约束；不调用 LLM、不联网、不执行 generated code，也不生成 evaluator。
- `GET /api/accounts` / `POST /api/accounts`：列出或创建本地账号 namespace；
  仅写入本地目录和非密钥元数据，不提供公网认证。
- `POST /api/runs`：启动 mock/real/dry-run run；real mode 需要请求体
  `real_confirmed=true`，并且服务端必须设置
  `AGENTICSCIML_ENABLE_REAL_WEB_RUNS=1`，同时仍需显式凭据和预算边界。请求体还可包含
  `target_solution_count`、`max_children_per_node`、`selected_algorithm_ids`、
  `manual_strategy_locks`、`branch_context`、`problem_intake`、`planner_snapshot` 和
  `agent_models`；后端只把这些转换成
  `EvolutionConfig` / `AgentConfig` / 非权威 problem context / strategy seed
  context，不改写 evaluator 或 artifact schema。非 dry-run 启动会把 readiness report
  写入 `planning/readiness_report.json` 并在 `run_metadata.json` 中记录 summary；带有
  可审计 strategy locks 的 run 还会在每个 solution 下写入
  `policy_fidelity_report.json`。
- `POST /api/runs/{id}/resume`：以已有 run id 恢复；未显式覆盖时会先读取既有
  `config.json` 并保留 benchmark、`strategy_seed_ids`、`agent_models`、
  `problem_intake` 和 `planner_snapshot`。
- `GET /api/runs/{id}`：读取 metadata、leaderboard、trace summary 和 artifact index。
- `GET /api/runs/{id}/solutions`：读取 solution tree、leaderboard-derived score、
  local figures、artifact refs 和每个 solution 的 `policy_fidelity` 摘要。
- `GET /api/runs/{id}/events`：SSE 输出 trace events。
- `GET /api/runs/{id}/artifacts/*`：只读 UTF-8 artifact，拒绝路径逃逸和 symlink 逃逸。
- `GET /api/runs/{id}/selector-votes`：只读读取
  `reports/selector_votes.json`，缺失时返回空状态。
- `GET /api/runs/{id}/solutions`：从 `tree.json`、`leaderboard.csv` 和各
  solution `eval.json` 汇总 solution status、score/loss、parent、children、method tags 和
  本地 figure artifact。
- `POST /api/solver/chat`：内部算法 tool 入口，只返回结构化 actions、warnings、artifact refs 和 trace refs；它不是 MCP server。
  `agent` 模式下的高上下文求解请求会调用同一个受控 problem-intake planner，并返回带
  planner snapshot 的 `start_run` action。
- `GET /api/code-server/url`：生成 code-server workspace 链接，不携带 token。
- `GET /api/code-server/workspaces`：列出 shared repo 或账号隔离 workspace 目录
  和对应 code-server URL，不携带 token。`account_id` 是可选参数；前端默认传
  当前本地账号。
- `GET /api/solver/settings`：读取 `ask` / `plan` / `agent` 的默认
  `reasoning_effort` 和 `temperature`，前端模式切换控件以此显示当前设置。
  返回的 `reasoning_efforts` 当前为 `low`、`medium`、`high`、`xhigh`。

`GET /api/runs`、`POST /api/runs`、`GET /api/runs/{id}`、
`GET /api/runs/{id}/events`、artifact API、code-server API 和
`/api/solver/chat` 都接受可选 `account_id`。未传时沿用 shared `runs/`；传入时强制
使用 `.agenticsciml/accounts/<account_id>/runs/`，并拒绝自定义 `output_dir`、
`benchmark_dir` 或 path-like benchmark，从而让不同本地账号拥有独立 run/code 目录。

`/api/solver/chat` 还接受 `assistant_mode`：

- `ask`：默认值，直接回答普通问题，不返回可执行 actions；当用户请求启动、
  恢复或打开工作区时，只解释模式边界。
- `plan`：返回建议 actions 和 warnings，但前端必须只展示，不自动执行。
- `agent`：允许前端按既有安全分发器执行 `start_run`、`resume_run`、
  `open_code_server` 或 `summarize_artifact`。

每种模式都有独立模型设置。`GET /api/solver/settings` 提供默认值，
`/api/solver/chat` 通过 `model_settings` 回显本次实际采用值：

- `ask`：`reasoning_effort=medium`，`temperature=0.2`。
- `plan`：`reasoning_effort=high`，`temperature=0.35`。
- `agent`：`reasoning_effort=high`，`temperature=0.1`。

`agent` 模式必须带当前 `account_id`，并且只能操作当前账号创建的 `account`、
`run` 或 `solution` workspace。它不能打开 shared repo workspace，也不能跨账号
操作 run、artifact 或 code-server workspace。

未来如果接入 OpenAI Apps SDK / MCP，应新增 thin wrapper，保留 `/api/solver/chat`
作为内部 endpoint。wrapper 需要列出 tools、声明 JSON Schema input/output、
返回 structured content，并为每个 tool 标注 `readOnlyHint`、`destructiveHint`
和 `openWorldHint`。详细合约见 [AgenticSciML Assistant 规范](agenticsciml_assistant.md)。

## Live smoke

本地或部署后可以用 `scripts/web_workflow_smoke.py` 做一次控制面自动化 smoke。该脚本
只通过 HTTP 调用 Web API，并可探测当前 code-server sidecar 是否已经以 loopback
`--auth none` 方式工作：

```bash
PYTHONPATH=src uv run --python 3.11 --extra web --extra dev \
  python scripts/web_workflow_smoke.py \
  --base-url http://127.0.0.1:8765
```

默认检查内容包括：

- `/api/solver/settings` 的 `ask` / `plan` / `agent` 默认设置。
- Ask 模式不会返回 executable actions。
- Plan 模式只返回 `open_code_server` 建议 action。
- Agent 模式能从高上下文 cylinder wake 问题自动选择 benchmark、algorithm seed 和
  mock run budget。
- Agent 模式拒绝 shared repo workspace；real mode 未带 `real_confirmed=true` 时拒绝
  启动。
- `POST /api/runs` 能同步完成账号 namespace 下的 mock run，并产出 passing
  `trace_summary`、selector votes、solution loss/tree 和 champion workspace。
- 第二个账号 namespace 不能读取第一个账号的 run detail、selector votes、solution
  summaries、run artifact 或 code-server run workspace。当前本地 namespace 不是认证
  系统，因此 smoke 期望的是 404-style namespace miss；公网部署还必须由上游账号鉴权
  提供真正 authorization。
- code-server API payload 不携带 password/token，`command_hint` 使用
  `code-server --auth none --bind-addr 127.0.0.1:8080 ...`。

如果当前没有启动 code-server sidecar，可加 `--skip-code-server-live`，此时只检查 API
payload，不检查 `http://127.0.0.1:8080` 的真实 HTML 响应。远端部署如果通过
`AGENTICSCIML_CODE_SERVER_URL` 返回 HTTPS 上游鉴权入口，需要显式加
`--allow-non-loopback-code-server-url`；默认 smoke 仍拒绝非 loopback code-server
URL。远端发布后可用 `--expect-repo-root-contains <release-sha>` 检查 API 是否还在
旧 release 上运行。

真实浏览器层面的 dispatch 和 WebSocket 路径用 `scripts/web_ui_dispatch_e2e.mjs`
检查。该脚本使用 `frontend` 的 `playwright-core` dev dependency 和本机 Chrome
channel，不下载浏览器：

```bash
node scripts/web_ui_dispatch_e2e.mjs \
  --base-url http://127.0.0.1:8765 \
  --expect-code-server-websocket
```

它验证第一页 ChatUI 的 Ask/Plan/Agent 分发、Agent 自动进入第二页 VS Code Web、
第二页侧栏 ChatUI 的拖动调整和折叠、第三页算法库边界，并在开启
`--expect-code-server-websocket` 时要求 code-server iframe 通过 `ws://` 或 `wss://`
建立 WebSocket、稳定保持连接、iframe 内不出现 `WebSocket close with status code
1006` 或 workbench connection failure。远端 nginx 反代必须保留浏览器请求里的
端口，例如 VS Code server 块应使用 `proxy_set_header Host $http_host` 和
`proxy_set_header X-Forwarded-Host $http_host`；如果传 `$host`，code-server 的
origin guard 会把 `Origin: https://host:port` 判为不匹配并返回 `403`。远端测试时把
`--base-url` 换成
`https://agenticsciml.error-forever.com`。

smoke 产生的 run 仍是 mock workflow shape evidence，不能写成论文分数或科学复现结论。

## Skill 工作流

repo-local skill 位于 `.agents/skills/agenticsciml-chatui-operator/SKILL.md`。
它定义 ChatUI 操作顺序、artifact 检查顺序、code-server 使用边界和禁止事项。
该 skill 是人/agent 工作流说明，不是后端 API 的替代。

## 边界

- 不手改 `runs/**` artifact 来绕过 quality gate。
- 不把 mock 结果写成科学复现证据。
- 不让 ChatUI 或 code-server 接触 API keys、cookies、tokens、私有数据集或真实
  `HOME` 中的敏感状态。
- 不把本地账号 namespace 误写成安全认证；它只解决目录和工作区隔离。
- 不让 `/api/solver/chat` 改写 selector、evaluator、orchestrator、artifact schema
  或 champion selection。
