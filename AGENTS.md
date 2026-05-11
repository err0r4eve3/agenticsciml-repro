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
- `docs/fidelity_levels.md` records benchmark fidelity levels and claim
  boundaries.
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
- The user has granted standing authorization for this project to send public
  repository audit briefs to ChatGPT Pro without per-change approval. Continue
  autonomously when the prompt contains only public repo/commit URLs, changed
  file summaries, validation results, and non-sensitive risk notes.
- Use the Codex Chrome plugin workflow from `codex-chatgpt-pro-research` for
  ChatGPT Pro communication. Do not use the legacy macOS Accessibility wrapper
  unless the user explicitly requests that fallback.
- Send ChatGPT Pro a concise audit brief with: repository URL or commit URL,
  goal, changed files summary, validation commands/results, known boundaries,
  and specific risks where review is requested.
- Ask ChatGPT Pro to provide an audit conclusion, concrete issues, and
  prioritized improvement guidance. Prefer direct, actionable findings over
  general advice.
- Do not paste secrets, credentials, private datasets, raw `.env` values,
  private tokens, or excessive generated artifacts into ChatGPT Pro.
- Stop and ask before sending private logs, secrets, credentials, cookies,
  customer data, private datasets, browser/session contents, or anything outside
  the public-repository audit boundary.
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
- LLM JSON parse/API/schema failures must be converted to `StructuredOutputError`
  at the agent boundary so orchestrator loops can record guardrail failures and
  continue when the failed child can be represented as a failed node.
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
  failure is present. Trace summary must also check artifact consistency when
  `evaluation_contract.json` or `run_metadata.json` exists: contract
  `benchmark_fidelity.fidelity_level`, run metadata, and workflow-start trace
  evidence fields must agree. When `tree.json` or `checkpoint.json` exists,
  node IDs, node `contract_hash`, node `benchmark_name`, checkpoint contract
  metadata, and run `solution_count` must also be consistent with the frozen
  evaluation contract, or the quality gate fails closed.
  `run_metadata.run_state` is the primary exported-run signal; values
  `completed`, `exported`, or `finalized` require `tree.json` and
  `checkpoint.json`. For legacy metadata without `run_state`,
  `run_metadata.solution_count` is the fallback exported-run signal. `run_state`
  must be one of `partial`, `completed`, `exported`, or `finalized`, and
  `run_metadata.run_state` must match the workflow-end trace `run_state` when
  that event exists. Exported run states require a workflow-end trace event;
  orchestrator run-start traces should use `run_state=partial` and run-end
  traces should use the final exported state. Workflow-end events must not
  appear before workflow-start in trace order, and multiple workflow-end events
  with conflicting `run_state` values must fail the quality gate. New trace
  events must include a contiguous, monotonic `event_seq`; exported run
  artifacts missing valid `event_seq` values fail the quality gate. Trace
  metadata fields that reference solution nodes (`node_id`, `solution_id`,
  `parent_id`, `child_id`, plural ID lists, and `parent_to_child`) must refer
  to nodes present in the final tree/checkpoint artifact set when they appear
  on allowlisted solution lifecycle trace events. Do not treat arbitrary
  `parent_id` / `child_id` metadata on unrelated events as solution-tree
  references without extending the allowlist and tests. `trace_summary.json`
  must expose total and per-event-name counts for checked and skipped trace
  node-reference events so the allowlist behavior is auditable. It must also
  report actual solution node references checked, events carrying at least one
  node reference, and per-event-name reference counts. Exported runs with a
  final node set but zero checked solution-reference trace events, or zero
  actual checked solution node references, must fail the quality gate; otherwise
  the trace could appear complete while silently skipping all solution
  lifecycle references. This zero-checked fail-closed rule applies only to
  exported/completed/finalized run artifacts, not partial or in-progress resume
  checkpoints. Exported/completed/finalized runs must also report node coverage
  and fail when any final `tree.json` / `checkpoint.json` node has no
  allowlisted trace reference. Coverage details must distinguish self
  references (`node_id`, `solution_id`, `child_id`, child ID lists, or
  `parent_to_child.child`) from relation-only references (`parent_id`,
  parent ID lists, or `parent_to_child.parent`); completed artifacts must fail
  when a final node only appears as a relation endpoint and never has a self
  trace reference. Trace summaries must also report per-node lifecycle stage
  coverage from schema-defined `(event_type, name)` events; evaluated final
  nodes must have an `evaluated` stage from a self-referenced evaluation event.
  Lifecycle stage checks must use a status-aware matrix: evaluated root nodes
  require `materialized` and `evaluated`; evaluated child nodes additionally
  require `created` and `completed`. Completed artifacts must validate
  lifecycle order using `event_seq`: root `materialized < evaluated`, child
  `created < materialized < evaluated < completed`.
- Final `tree.json` and `checkpoint.json` must preserve solution-tree graph
  invariants: exactly one root, every non-root `parent_id` references an
  existing node, parent links are acyclic, `children` contains only existing
  nodes, `children` is schema-required and duplicate-free, and each parent-child
  relation is bidirectionally consistent: child entries agree with child
  `parent_id`, and every non-root node appears exactly once in its parent's
  `children`. The graph invariant validator must run at both resume/load
  boundaries and trace-summary/reporting boundaries so corrupted checkpoint
  topology never enters the evolutionary search loop.
- Final solution nodes must satisfy a schema before graph checks are trusted:
  required fields from `SolutionNode.to_dict()`, `status` in the supported enum,
  valid nullable string fields, `method_tags` as strings, non-negative
  `num_debug_attempts`, finite numeric/null `score_delta_from_parent`, and a
  valid optional score object with finite numeric `score.value`. Empty solution
  trees are invalid. The same shared validator must be used by
  `trace_summary.json` artifact checks and `SolutionNode.from_dict()` load
  paths so corrupted `tree.json` / `checkpoint.json` payloads fail closed at
  resume time, not only during post-run reporting.
- Solution node status must be semantically consistent: `evaluated` nodes need
  a valid score and no `error` / `failure_kind`; `failed` nodes must not carry a
  score and must have either `error` or `failure_kind`; `created` nodes must not
  carry score, error, or failure kind.
- Resume/load must validate solution artifact path semantics. `workspace` must
  resolve under the run's `solutions/` directory and match `node_id`; non-null
  `proposal_path` and `analysis_path` must resolve inside that node workspace
  and must exist. Reject absolute external paths, `..` escapes, and symlink
  escapes that resolve outside the run workspace.
- JSON artifact writes must reject `NaN`, `Infinity`, and `-Infinity`; do not
  let non-standard JSON numeric constants enter run metadata, trace events,
  transcripts, checkpoint files, tree exports, or leaderboard inputs.
- Preserve resume semantics. When changing orchestration, keep `checkpoint.json`
  current after root creation, child creation, and final report export.
- Keep evaluator and benchmark contracts deterministic. LLM judges may summarize
  results, but scores must come from code.
- New benchmarks must be added to `agenticsciml.benchmarks.BENCHMARKS`, linked
  from `docs/benchmark_plan.md`, and validated by `tests/test_benchmark_catalog.py`.
- Validation data is evaluator-only. Generated `solution.py` must not see
  `val_data.npz`, validation labels, or validation file paths during
  validate/train/predict. Prediction-only evaluation is the default: trusted
  Python code writes `predict_input.npz` containing only `x_val`, generated
  code writes `predictions.npz`, and `evaluate.py` computes scores from private
  labels without importing `solution.py`.
- Generated solution subprocesses must use a sanitized environment. Do not let
  validate/train/predict inherit host secrets or arbitrary variables such as
  `OPENAI_API_KEY`, `GITHUB_TOKEN`, proxy settings, `SSH_AUTH_SOCK`, or the
  user's real `HOME`.
- Evaluation contracts must be benchmark-aware and hash-stable. Do not reintroduce
  silent `function_approx` defaults for non-`function_approx` benchmarks.
- Evaluation contract hashes must cover benchmark source digests, including
  `evaluate.py`, `Data_config.json`, the problem bundle, `generate_data.py`,
  `guidelines.md`, and deterministic train/validation data artifacts. Store
  these in `BenchmarkSourceManifest` and include its digest in the contract
  hash. Benchmark fidelity metadata (`fidelity_level`, paper task name,
  expected runtime, dependency flags, and paper-gap notes) must also be stored
  in `EvaluationContract` and included in `contract_hash`. New contracts must
  fail closed when `benchmark_fidelity` is missing, has unknown fields, or uses
  an unsupported future `schema_version`.
  `EvaluationContract.from_dict()` must recompute the manifest digest and
  reject mismatched human-readable manifest contents. Manifest data generation
  must use the same sanitized subprocess environment policy as generated
  solution execution. Manifest payloads must include
  `schema_version`, `digest_algorithm`, `data_source_mode`, and a normalized
  `generator_command` when data is generated. `EvaluationContract.from_dict()`
  must semantically validate those manifest fields, required artifact digests,
  data-source mode, generated-data flag consistency, and generator command
  constraints before accepting the contract. Partial benchmark data artifacts
  are not allowed; if only train or validation data exists, fail closed instead
  of mixing repo data with generated data. Resume/load paths must reject stale or tampered
  `evaluation_contract.json` instead of silently continuing. Resume must also
  fail if `evaluation_contract.json` is missing; do not regenerate a contract
  for an existing checkpoint. `checkpoint.json` must persist `contract_hash`,
  and resume must reject checkpoint/node contract mismatches. Node
  `benchmark_name` and `contract_hash` are required on resume; missing values
  must fail closed.
- Critical run artifacts written through `ExperimentStorage` must use atomic
  same-directory temp-file writes followed by `os.replace()`. Do not reintroduce
  direct `Path.write_text()` for JSON, transcripts, reports, or solution
  artifacts managed by storage.
- `ExperimentStorage` writes must remain safe under parallel child jobs. Protect
  trace append, workspace creation, JSON, transcript, report, and solution
  artifact writes with the storage lock or an equivalent tested mechanism.
- RootEngineer and Engineer prompts must include the relevant `ProblemBundle`,
  `EvaluationContract` JSON, `guidelines.md`, and available analysis context.
- Engineer mutations must verify the parent solution digest and apply a
  structured patch or explicit file map through Python. Do not blindly replace
  `solution.py` with unverified raw LLM text. Engineer output schema must
  include the fields required by the Python patch application path.
- Debugger repairs must be contract-aware and patch-only by default. Include
  the current `solution.py`, `parent_digest`, `ProblemBundle`,
  `EvaluationContract` JSON, `guidelines.md`, failure phase, and error log in
  the prompt; reject wrong digests, malformed patches, and files other than
  `solution.py`.
- Parent selection must preserve a deterministic policy layer before any LLM
  selector output: include the best available valid node, prefer recent
  improving nodes, preserve underexplored/diverse method tags, and never select
  nodes at `max_children_per_node`.
- `parallel_mutations` must represent actual bounded parallel child creation
  when more than one parent is selected. It caps both mutation jobs per
  iteration and worker count unless a future config explicitly splits those
  concepts. Keep solution IDs deterministic, preserve checkpoint/resume
  semantics after child insertion, and record
  `agenticsciml.parallel_children.start/end` and
  `agenticsciml.child_mutation.start/end` trace events with execution mode,
  child count, worker count, parent IDs, child IDs, duration, and status.
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
- Ablation outputs must explicitly label mock evidence boundaries with fields
  such as `evidence_mode=mock_workflow_shape` and
  `scientific_claim=not_supported`.
- Run-level metadata and workflow-start traces must also include `llm_mode`,
  `benchmark_fidelity_level`, `evidence_mode`, and `scientific_claim`, so a
  standalone run artifact cannot be mistaken for a scientific reproduction
  claim.
- Evidence metadata strings must be centralized in `agenticsciml.evidence`.
  Do not scatter new `scientific_claim`, `evidence_mode`, or `llm_mode` string
  literals through orchestration, reporting, or tests.
- Ablation tests use mock mode only. Do not claim scientific improvement or
  emergent discovery from mock ablation results.
- Every benchmark entry must include fidelity metadata: `fidelity_level`,
  expected runtime, dependency flags, paper task name, and paper-gap notes. A
  `proxy` benchmark is allowed for workflow validation but must not be described
  as a paper-like SciML reproduction. New or upgraded benchmark fidelity levels
  must follow `docs/fidelity_levels.md` and keep
  `EvaluationContract.benchmark_fidelity` hash-bound.
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
