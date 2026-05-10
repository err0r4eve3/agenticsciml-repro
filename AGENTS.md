# AgenticSciML Reproduction Instructions

## Scope

- Reproduce the AgenticSciML workflow, not exact paper scores.
- Python 3.11 only.
- Linux is the target runtime; do not add Windows compatibility layers.
- Use local filesystem persistence first.
- Use `pytest` for validation.

## Development Rules

- No network calls in tests.
- Keep mock LLM mode deterministic.
- Generated solution code must run in isolated per-solution workspaces with
  timeouts.
- Store every prompt, model response, score, log, and artifact under the run
  directory.
- Prefer structured JSON outputs where possible.
- Do not ask models to reveal hidden chain-of-thought. Request concise rationale
  summaries, diagnoses, implementation plans, expected effects, and risks.

## Project Commands

```bash
uv run --python 3.11 --extra dev pytest -q
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 1
```

## Source Of Truth

- Current source code and tests define implementation behavior.
- `docs/paper_notes.md` records the paper-derived workflow constraints.
- `docs/index.md` is the documentation tree root.
- `docs/multi_agent_design.md` records the multi-agent workflow design method.
- `docs/version_notes.md` records current MVP capabilities and release notes.
- `docs/benchmark_plan.md` records benchmark catalog, validation matrix, and
  real LLM experiment order.
- `examples/` contains the runnable benchmark catalog.

## Documentation Goals

- New project documentation should be linked from `docs/index.md`.
- Use Simplified Chinese for new repository governance docs unless a public API,
  command, path, package, or code symbol is clearer in English.
- Keep docs tree-shaped and cross-linked. Avoid adding standalone Markdown pages
  that cannot be reached from `docs/index.md`.
- Version notes should describe verified commands, known boundaries, and next
  version suggestions without implying paper-score reproduction.

## GitHub And Versioning Policy

- Every completed repository change must be committed and pushed to the GitHub
  remote before the final response, unless the user explicitly asks not to push.
- Before pushing, run the smallest relevant validation for the change. For
  cross-cutting runtime or orchestration changes, run
  `uv run --python 3.11 --extra dev pytest -q`.
- Do not commit generated run artifacts, secrets, credentials, local caches, or
  private datasets. Verify `git status --short` before staging.
- If normal `git push` fails because of local network/TLS issues, use an
  authenticated `gh` / GitHub API fallback only after confirming the update is a
  fast-forward of the configured remote branch.
- Large milestones, release-like changes, benchmark additions, security
  boundary changes, public CLI changes, or workflow contract changes must update
  `docs/version_notes.md` in the same commit.
- Version notes for large milestones should include: implemented capabilities,
  verified commands, known boundaries, remaining risks, and suggested next
  version work.
- Do not create Git tags, GitHub releases, or release artifacts unless the user
  explicitly asks for them.
- Final responses after a push should include the commit SHA, GitHub URL,
  validation commands/results, and any unverified risk.

## External ChatGPT Pro Audit Policy

- After completing implementation, validation, and GitHub push for this
  project, consult the user's ChatGPT Pro web session before the final response
  whenever the browser session is available.
- Send ChatGPT Pro a concise audit brief with: repository URL or commit URL,
  goal, changed files summary, validation commands/results, known boundaries,
  and specific risks where review is requested.
- Ask ChatGPT Pro to provide an audit conclusion, concrete issues, and
  prioritized improvement guidance. Prefer direct, actionable findings over
  general advice.
- Do not paste secrets, credentials, private datasets, raw `.env` values,
  private tokens, or excessive generated artifacts into ChatGPT Pro.
- If ChatGPT Pro returns actionable findings that are in scope and low-risk,
  address them before the final response, then validate and push again. If the
  findings are larger follow-up work, summarize them as next steps.
- If the ChatGPT Pro web session is unavailable, blocked, or not authenticated,
  state that explicitly in the final response and include the local validation
  results instead of inventing an external audit.

## Multi-Agent Design Rules

- Design the task state machine first, then choose which nodes need LLM calls,
  and only then decide whether to split roles into multiple agents.
- Keep deterministic control in Python: state transitions, evaluation, sorting,
  filesystem writes, retries, timeouts, and champion selection.
- Use LLM agents only for judgment, generation, critique, debugging suggestions,
  and concise result summaries.
- Do not implement free-form group chat for the AgenticSciML workflow.
- Agent outputs should be structured JSON when consumed by code, or concise
  rationale summaries when saved for humans.
- Do not ask models for hidden chain-of-thought. Request diagnoses,
  implementation plans, expected effects, risks, and short rationale summaries.
- Every new agent must have a unique role, explicit non-role, input/output
  schema, minimum context, budget, stop condition, and failure policy.

## OpenAI Agents SDK Alignment

- Every agent step must follow the OpenAI Agents SDK distinction between
  code-orchestrated workflows and LLM-orchestrated handoffs. This project uses
  code orchestration by default; do not introduce handoff-style control transfer
  unless a future task explicitly requires a specialist to own the next turn.
- Treat specialist agents as bounded tools under manager/orchestrator control:
  they may return structured outputs or artifacts, but they do not choose global
  state transitions.
- Use structured outputs for code-consumed LLM responses. Validate required
  fields, retry within a small budget, and fail closed when validation still
  fails.
- Treat `AgentSpec` as a runtime contract, not only documentation. New agent
  methods must call input-schema validation before LLM/tool work and use the
  spec output schema for code-consumed JSON.
- Add guardrails around each tool-like boundary: generated code execution,
  artifact writes, evaluator contract integrity, and JSON output validation.
- Generated solution execution must pass static sandbox checks before any
  command runs. Block network/client modules, subprocess execution, destructive
  file operations, and obvious absolute-path writes unless an explicit future
  task expands the allowlist with tests.
- Record tracing events for workflow, agent, generation, tool execution, and
  guardrail checks so traces can later be graded or inspected.
- Generation trace events must preserve `duration_s`, `prompt_token_estimate`,
  and `response_token_estimate` placeholders. `run_metadata.json` must keep an
  `llm_calls` aggregate for total calls and calls by role.
- Preserve trace summary semantics. Completed orchestrator runs should write
  `trace_summary.json`; `agenticsciml trace-summary <run_dir>` should report a
  passing `quality_gate` only when required span types exist and no guardrail
  failure is present.
- Preserve resume semantics. When changing orchestration, keep `checkpoint.json`
  current after root creation, child creation, and final report export.
- Keep evaluator and benchmark contracts deterministic. LLM judges may summarize
  results, but scores must come from code.
- New benchmarks must be added to `agenticsciml.benchmarks.BENCHMARKS`, linked
  from `docs/benchmark_plan.md`, and validated by `tests/test_benchmark_catalog.py`.
- Validation data is evaluator-only. Generated `solution.py` must not see
  `val_data.npz` during validate/train; keep validation data outside
  `solutions/solution_*/` under run-private evaluator directories and preserve
  static guardrails against obvious validation-data reads.
- Generated solution subprocesses must use a sanitized environment. Do not let
  validate/train inherit host secrets or arbitrary variables such as
  `OPENAI_API_KEY`, `GITHUB_TOKEN`, proxy settings, `SSH_AUTH_SOCK`, or the
  user's real `HOME`.
- Evaluation contracts must be benchmark-aware and hash-stable. Do not reintroduce
  silent `function_approx` defaults for non-`function_approx` benchmarks.
- RootEngineer and Engineer prompts must include the relevant `ProblemBundle`,
  `EvaluationContract` JSON, `guidelines.md`, and available analysis context.
- Engineer mutations must verify the parent solution digest and apply a
  structured patch or explicit file map through Python. Do not blindly replace
  `solution.py` with unverified raw LLM text.
- Parent selection must preserve a deterministic policy layer before any LLM
  selector output: include the best available valid node, prefer recent
  improving nodes, preserve underexplored/diverse method tags, and never select
  nodes at `max_children_per_node`.
- Retrieval queries must be benchmark-aware. Build them from `ProblemBundle`,
  parent analysis, failure kind, method tags, score trend, and top leaderboard
  context rather than fixed benchmark-specific keywords.
- `use_kb=False` must mean no KB entry is injected into proposal context.
  `random_kb=True` must use deterministic seed-controlled random retrieval for
  ablation, not lexical retrieval disguised as random.
- Ablation variants must map to real workflow switches. `no_critic` must skip
  `CriticAgent` calls; `no_debugger` must skip the debugger loop instead of only
  changing report labels.
- Ablation outputs must include per-run rows and aggregate rows with champion
  score, root score, champion/root improvement, valid solution rate, timeout
  count, debug success count, LLM call count, wall time, and example run dirs.
- Ablation tests use mock mode only. Do not claim scientific improvement from
  mock ablation results.
- `SolutionNode` metadata used by selector/retriever/ablation must stay
  persisted in `tree.json` and `checkpoint.json`: `benchmark_name`,
  `contract_hash`, `method_tags`, `failure_kind`, `score_delta_from_parent`,
  and `num_debug_attempts`.

## Boundaries

- Do not commit API keys, model credentials, cookies, generated run outputs, or
  private datasets.
- Do not edit generated run artifacts by hand except for debugging a local run.
- Do not add new runtime dependencies unless the repository has a clear need and
  the reason is documented.

## Final Response Expectations

For repository changes, report changed files, validation commands and results,
assumptions, and remaining unverified risks.
