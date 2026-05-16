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
  `ask`。`ask` 只解释不返回可执行动作；`plan` 返回结构化建议动作但前端不自动
  分发；`agent` 才会自动执行受控 action。
- `VS Code` 页先显示工作空间选择列表，风格参考 UnitaryLab 的 workspace
  页：用户选择一个账号隔离的独立代码目录后进入 AI IDE。编辑态只保留 VS Code Web
  iframe 和右侧可收起 ChatUI 侧边栏，不显示 benchmark、run、quality gate、
  artifact、启动命令或 sidecar 说明块。
- `算法库` 页承载实验工作台：benchmark、运行模式、run/gate 状态、run
  dashboard、算法策略目录、run 列表、leaderboard、trace preview 和 artifact 浏览。
- ChatUI 自然语言输入仍调用 `/api/solver/chat`；`open_code_server` action 会切到
  `VS Code Web` 页，`summarize_artifact` action 会切到 `算法库` 页。
- Agent 侧边栏负责自然语言意图、结构化 actions、warnings、artifacts 和 trace
  refs 展示。
- 视觉语言采用 Claude Code 风格的暖米色工作台：纸面背景、深棕文字、陶土色主
  action、低饱和边框和暗色 trace/code 区域。

Agent 面板只分发受控动作。`mock` 和 `dry_run` actions 可由前端调用现有 API
执行；`real` mode action 默认拦截为待确认状态，不会隐式触发真实 LLM 调用。

## code-server sidecar

code-server 不由 ChatUI 自动启动。先用本地 token/password 和 loopback 绑定启动：

```bash
PASSWORD=<local-token> code-server --bind-addr 127.0.0.1:8080 /path/to/workspace
```

默认 API 生成 `http://127.0.0.1:8080/?folder=<workspace>` 链接。可用
`AGENTICSCIML_CODE_SERVER_URL` 覆盖 base URL。

公网部署只能作为显式配置的 hardened sidecar：必须有 TLS、password auth、受限
workspace、最小权限和 secret scanning。即使公网部署，ChatUI 也不得把 token、
password、API key、cookie、真实 `HOME`、浏览器 profile 或私有数据集路径放进 URL
或消息体。

`VS Code` 页通过 `GET /api/code-server/workspaces` 列出可打开的独立代码目录。
未传 `account_id` 时保留旧兼容行为：列出 repo 根目录、run 目录、champion 目录和
各 `solutions/solution_*` 目录。传入 `account_id` 时只列出该账号 namespace 下的
`workspace/`、`runs/<run_id>/` 和 `runs/<run_id>/solutions/solution_*` 目录；
不混入 shared repo 根目录。选择后 iframe 使用对应 `?folder=<workspace>` 打开该
目录。ChatUI 不携带 code-server token，不自动启动 sidecar，iframe 也必须经过
code-server 自身鉴权。

账号隔离是本地目录隔离，不是多用户认证或权限边界。公开部署时还必须由反向代理、
code-server auth、系统用户/容器权限和 secret scanning 提供真实访问控制。

## API 边界

- `GET /api/benchmarks`：读取当前 benchmark catalog。
- `GET /api/algorithms`：读取算法策略目录。目录项只用于规划和 prompt seed，不
  代表已经通过 evaluator 的实现。
- `GET /api/accounts` / `POST /api/accounts`：列出或创建本地账号 namespace；
  仅写入本地目录和非密钥元数据，不提供公网认证。
- `POST /api/runs`：启动 mock/real/dry-run run；real mode 仍需显式凭据和预算边界。
- `POST /api/runs/{id}/resume`：以已有 run id 恢复。
- `GET /api/runs/{id}`：读取 metadata、leaderboard、trace summary 和 artifact index。
- `GET /api/runs/{id}/events`：SSE 输出 trace events。
- `GET /api/runs/{id}/artifacts/*`：只读 UTF-8 artifact，拒绝路径逃逸和 symlink 逃逸。
- `POST /api/solver/chat`：内部算法 tool 入口，只返回结构化 actions、warnings、artifact refs 和 trace refs；它不是 MCP server。
- `GET /api/code-server/url`：生成 code-server workspace 链接，不携带 token。
- `GET /api/code-server/workspaces`：列出 shared repo 或账号隔离 workspace 目录
  和对应 code-server URL，不携带 token。`account_id` 是可选参数；前端默认传
  当前本地账号。

`GET /api/runs`、`POST /api/runs`、`GET /api/runs/{id}`、
`GET /api/runs/{id}/events`、artifact API、code-server API 和
`/api/solver/chat` 都接受可选 `account_id`。未传时沿用旧的 shared `runs/`；
传入时默认使用 `.agenticsciml/accounts/<account_id>/runs/`，从而让不同本地账号
拥有独立 run/code 目录。

`/api/solver/chat` 还接受 `assistant_mode`：

- `ask`：默认值，只回答和解释，不返回可执行 actions。
- `plan`：返回建议 actions 和 warnings，但前端必须只展示，不自动执行。
- `agent`：允许前端按既有安全分发器执行 `start_run`、`resume_run`、
  `open_code_server` 或 `summarize_artifact`。

`agent` 模式必须带当前 `account_id`，并且只能操作当前账号创建的 `account`、
`run` 或 `solution` workspace。它不能打开 shared repo workspace，也不能跨账号
操作 run、artifact 或 code-server workspace。

未来如果接入 OpenAI Apps SDK / MCP，应新增 thin wrapper，保留 `/api/solver/chat`
作为内部 endpoint。wrapper 需要列出 tools、声明 JSON Schema input/output、
返回 structured content，并为每个 tool 标注 `readOnlyHint`、`destructiveHint`
和 `openWorldHint`。详细合约见 [AgenticSciML Assistant 规范](agenticsciml_assistant.md)。

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
