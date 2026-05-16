---
name: agenticsciml-chatui-operator
description: Operate the AgenticSciML local ChatUI console, algorithm tool API, run artifacts, and code-server sidecar without bypassing evaluation contracts or claim boundaries.
---

# AgenticSciML ChatUI Operator

Use this skill when operating the local AgenticSciML Web console or when an agent needs to drive the ChatUI + algorithm tool workflow.

## Operating Model

- Treat the Python orchestrator as the source of truth for workflow state, solution tree updates, evaluator contracts, champion selection, and run artifacts.
- Use ChatUI for user intent capture, run status, artifact inspection, and controlled calls to the `/api/solver/chat` algorithm tool.
- Use code-server only as an editor for the repository or selected run workspace. Do not use it to bypass validation, mutate generated run artifacts by hand, or expose host secrets.
- Keep mock evidence boundaries visible. Mock runs validate workflow shape only and do not support scientific reproduction claims.

## Standard Workflow

1. Start the API: `uv run --python 3.11 --extra web agenticsciml web --host 127.0.0.1 --port 8765`.
2. Start the frontend from `frontend/`: `npm run dev`.
3. Start code-server only on loopback with auth, for example: `PASSWORD=<local-token> code-server --bind-addr 127.0.0.1:8080 <workspace>`.
4. In ChatUI, select a benchmark and start a mock run before any real LLM run.
5. Inspect `trace_summary.json`, `leaderboard.csv`, `tree.json`, and champion artifacts through the UI or artifact API.
6. If code changes are needed, open the repo or solution workspace through the code-server link, then re-run or resume through the API/CLI.
7. Validate repository changes with the smallest relevant checks, and keep generated `runs/` artifacts out of commits.

## Algorithm Tool Boundary

Use `/api/solver/chat` for intent parsing and structured action suggestions only. Valid output categories are:

- `start_run`
- `resume_run`
- `open_code_server`
- `summarize_artifact`

The tool must not rewrite selector policy, evaluator logic, solution-tree schema, artifact schema, or champion selection. Those remain owned by Python code and tests.

## Prohibited

- Do not manually edit `runs/**` artifacts to make quality gates pass.
- Do not continue a run if `evaluation_contract.json` or checkpoint validation fails.
- Do not call real LLM mode implicitly. Real mode requires explicit user intent, credentials, and budget/claim-boundary checks.
- Do not expose code-server outside `127.0.0.1` unless a future task explicitly adds a hardened multi-user deployment.
- Do not store API keys, cookies, passwords, tokens, or private datasets in ChatUI messages, skill docs, run artifacts, or commits.
