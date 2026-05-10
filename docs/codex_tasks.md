# Codex Implementation Tasks

[返回文档树](index.md) · 相关文档：[多 Agent 设计方法](multi_agent_design.md)、[版本说明](version_notes.md)

## Task 0: Documentation And Rules

Objective: define the reproduction scope and repository rules.

Files: `README.md`, `docs/paper_notes.md`, `AGENTS.md`.

Acceptance: files exist and state that the MVP reproduces workflow mechanics,
not exact paper scores.

## Task 1: Project Scaffold And Types

Objective: create Python 3.11 package metadata, config types, state types, and
filesystem storage.

Files: `pyproject.toml`, `src/agenticsciml/config.py`,
`src/agenticsciml/state.py`, `src/agenticsciml/storage.py`.

Acceptance: serialization tests pass and a solution workspace can be created.

## Task 2: LLM Adapters

Objective: define an LLM abstraction, deterministic mock client, and optional
OpenAI client.

Files: `src/agenticsciml/llm/`.

Acceptance: mock JSON output is stable and tests do not require network calls.

## Task 3: Function Approximation Benchmark

Objective: create the first benchmark with deterministic data generation.

Files: `examples/function_approx/`.

Acceptance: generated train and validation arrays have expected shapes and are
deterministic by seed.

## Task 4: Execution And Evaluation

Objective: run generated solutions in isolated workspaces with timeouts.

Files: `src/agenticsciml/execution/`, `examples/function_approx/evaluate.py`.

Acceptance: a trivial handwritten solution validates, trains, writes
`predictions.npz` from `predict_input.npz`, and produces `eval.json` through
prediction-only evaluation.

## Task 5: Agent Wrappers

Objective: implement role-specific wrappers around `LLMClient`.

Files: `src/agenticsciml/agents/`, `src/agenticsciml/prompts/`.

Acceptance: mock mode writes proposal, analysis, selector transcripts.

## Task 6: Knowledge Base

Objective: provide a minimal KB and deterministic lexical retrieval.

Files: `examples/function_approx/kb/`, `src/agenticsciml/retrieval/`.

Acceptance: relevant discontinuity/oscillation queries retrieve a related KB
entry.

## Task 7: Orchestrator

Objective: coordinate the full mock workflow and persist artifacts.

Files: `src/agenticsciml/orchestrator.py`.

Acceptance: one mock iteration generates root plus child, `tree.json`, and
`leaderboard.csv`.

## Task 8: CLI And Reporting

Objective: expose run, leaderboard, and tree export commands.

Files: `src/agenticsciml/cli.py`, `src/agenticsciml/reporting/`.

Acceptance: one CLI command runs mock mode and exports champion artifacts.

## Task 9: Real LLM Dry Run

Objective: make real mode explicit and fail clearly without credentials.

Files: `src/agenticsciml/llm/openai_adapter.py`, `docs/real_run.md`,
`examples/function_approx/config.yaml`.

Acceptance: mock tests pass without credentials; real mode reports missing API
configuration clearly.

## Task 10: Ablation Script

Objective: compare root-only, no-KB, KB, and random-KB workflow variants.

Files: `scripts/run_ablation.py`, `docs/ablation.md`.

Acceptance: script writes `ablation_summary.csv`; tests validate pipeline shape
instead of performance gains.
