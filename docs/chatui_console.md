# ChatUI 实验操作台

[返回文档树](index.md) · 相关文档：[项目概览](../README.md)、[多 Agent 设计方法](multi_agent_design.md)、[版本说明](version_notes.md)

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

- 左侧功能栏选择 `ChatUI` 或 `AI IDE`。
- `ChatUI` 页是纯对话界面，接近 ChatGPT 网页版：顶部只做页面标识，中间是
  对话流，底部是主输入框；不显示 benchmark、run、quality gate、trace 或
  artifact 工作台。
- `VS Code Web` 页采用 AI IDE 布局：左侧为 VS Code Web / code-server
  workspace，右侧为可收起 ChatUI Agent 侧边栏；不显示实验工作台控件。
- `算法库` 页承载实验工作台：benchmark、运行模式、run/gate 状态、run
  dashboard、run 列表、leaderboard、trace preview 和 artifact 浏览。
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
`AGENTICSCIML_CODE_SERVER_URL` 覆盖 base URL，但第一版不支持公网、多用户鉴权
或代理层权限模型。

`VS Code Web` 页显示 `repo`、`run` 或 `solution` workspace 路径、启动命令、VS
Code Web iframe 和新标签页打开链接。ChatUI 不携带 code-server token，不自动
启动 sidecar，iframe 也必须经过 code-server 自身鉴权。

## API 边界

- `GET /api/benchmarks`：读取当前 benchmark catalog。
- `POST /api/runs`：启动 mock/real/dry-run run；real mode 仍需显式凭据和预算边界。
- `POST /api/runs/{id}/resume`：以已有 run id 恢复。
- `GET /api/runs/{id}`：读取 metadata、leaderboard、trace summary 和 artifact index。
- `GET /api/runs/{id}/events`：SSE 输出 trace events。
- `GET /api/runs/{id}/artifacts/*`：只读 UTF-8 artifact，拒绝路径逃逸和 symlink 逃逸。
- `POST /api/solver/chat`：算法 tool 入口，只返回结构化 actions、warnings、artifact refs 和 trace refs。
- `GET /api/code-server/url`：生成 code-server workspace 链接，不携带 token。

## Skill 工作流

repo-local skill 位于 `.agents/skills/agenticsciml-chatui-operator/SKILL.md`。
它定义 ChatUI 操作顺序、artifact 检查顺序、code-server 使用边界和禁止事项。
该 skill 是人/agent 工作流说明，不是后端 API 的替代。

## 边界

- 不手改 `runs/**` artifact 来绕过 quality gate。
- 不把 mock 结果写成科学复现证据。
- 不让 ChatUI 或 code-server 接触 API keys、cookies、tokens、私有数据集或真实
  `HOME` 中的敏感状态。
- 不让 `/api/solver/chat` 改写 selector、evaluator、orchestrator、artifact schema
  或 champion selection。
