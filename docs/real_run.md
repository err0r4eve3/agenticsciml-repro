# Real LLM Run

[返回文档树](index.md) · 相关文档：[版本说明](version_notes.md)、[Ablation 说明](ablation.md)

Mock mode is recommended first:

```bash
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 1
uv run --python 3.11 --extra dev agenticsciml run examples/poisson_lshape --mock --max-iterations 1
```

List available benchmarks:

```bash
uv run --python 3.11 --extra dev agenticsciml benchmarks
```

Run the full local suite through the Python module entry point. The `dev` extra
includes FastAPI and httpx, so Web API tests are collected in a clean test
environment:

```bash
uv run --python 3.11 --extra dev python -m pytest -q
```

After moving a checkout, an existing `.venv/bin/pytest` or console-script
shebang may still point at the old absolute path. Recreate the environment;
do not edit the generated shebang in place:

```bash
uv venv --clear --python 3.11
uv sync --python 3.11 --extra dev
uv run --python 3.11 --extra dev python -m pytest -q
```

Real mode requires credentials and the optional adapter dependency:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --real --max-iterations 1
```

The main `run` command requires the explicit `--real` confirmation before it
constructs a provider client. Omitting both `--mock` and `--real` keeps the
deterministic mock default; installing the `real-llm` extra alone never
authorizes provider calls.

If an editable console script cannot import the checkout, use the module form
without changing the scientific mode or confirmation gate:

```bash
PYTHONPATH=src uv run --python 3.11 --extra real-llm python -m agenticsciml.cli run examples/function_approx --real --max-iterations 1
```

For OpenAI-compatible providers, set `OPENAI_BASE_URL` as well. For example,
DeepSeek-style endpoints can be used without changing the CLI surface:

```bash
export OPENAI_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY=...
export OPENAI_MODEL=deepseek-v4-pro
export OPENAI_TIMEOUT_S=120
export OPENAI_MAX_RETRIES=0
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --real --max-iterations 0
```

Use the provider's exact model id. For example, if the provider rejects
`deepseekv4pro` and reports `deepseek-v4-pro`, use the hyphenated id.

When `OPENAI_BASE_URL` is unset, `OpenAIAdapter` treats the provider as an
OpenAI-native Responses path and uses typed Structured Outputs for agent JSON.
When `OPENAI_BASE_URL` is set, the adapter records an OpenAI-compatible
capability profile and falls back to chat completions plus local Pydantic
schema validation. `https://api.gatexflow.com/v1` and
`https://api.error-forever.com/v1` are treated as OpenAI-compatible multimodal
chat providers: structured text output still uses local schema validation,
while `--visual-audit-mode real` may send diagnostic images through chat image
content when the configured model supports it.

`OPENAI_TIMEOUT_S` is the per-request provider HTTP timeout. `OPENAI_MAX_RETRIES`
defaults to `0` so a configured agent-call budget is not silently multiplied by
SDK retries. The same provider-call budget can be supplied per run:

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx \
  --real \
  --llm-timeout-s 45 \
  --llm-max-retries 0
```

Provider timeout/retry fields are written into generation trace metadata and
`run_metadata.json`. Generated solution execution still uses `--timeout-s`;
that is the sandbox/evaluator subprocess budget, not the provider HTTP budget.

Real `run` and `smoke-llm` commands emit secret-free `llm-progress` JSON lines
to stderr when each provider call starts and finishes. Progress events contain
call/provider/model/schema status, duration, error type, and budget counters;
paired smoke events also include the current variant. Events never include
prompts, system messages, responses, or content hashes.

Latency-sensitive runs can enable run-level fast mode:

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/poisson_lshape \
  --real \
  --llm-fast-mode \
  --llm-timeout-s 75 \
  --llm-max-retries 0
```

Fast mode routes any agent role without an explicit `reasoning_effort` override
to `low`. Explicit per-role settings in `--agent-models-json` still win. This
is a runtime budget choice for getting real provider evidence; it does not
change benchmark fidelity, selector heterogeneity, or scientific claim gates.

Optional fail-closed budget gates:

```bash
export AGENTICSCIML_MAX_LLM_CALLS=80
export AGENTICSCIML_MAX_PROMPT_TOKENS=200000
export AGENTICSCIML_MAX_OUTPUT_TOKENS=80000
export AGENTICSCIML_MAX_TOTAL_TOKENS=280000
export AGENTICSCIML_MAX_COST_USD=5
export AGENTICSCIML_COST_PER_1K_TOKENS_USD=0.01
```

`AGENTICSCIML_MAX_COST_USD` requires
`AGENTICSCIML_COST_PER_1K_TOKENS_USD`, because provider pricing is not inferred
from the model name.

The main real run applies two distinct gates:

- before constructing the provider client, call-count preflight compares the
  planned `expected_llm_call_range.max` with `AGENTICSCIML_MAX_LLM_CALLS` and
  fails before provider access if the configured ceiling is too small;
- during execution, one run-scoped `llm_call_ledger.jsonl` reserves calls and
  prompt-token estimates before each request, then records response usage and
  enforces output-token, total-token, and estimated-cost ceilings.

Output-token and cost limits are not provider-side `max_output_tokens`
settings. A provider response can cross one of those accounting limits; the
ledger then raises a budget error and the run fails closed. Token counts use
provider usage when available and otherwise remain estimates. Review the
ledger plus `run_metadata.json` `llm_budget` before reporting cost or run
completeness.

Role-level model policy can be set from CLI with `--agent-models-json`. The
payload is a JSON object keyed by role; each role value supports `model`,
`base_url`, `temperature`, and `reasoning_effort`. If `temperature` or
`reasoning_effort` is omitted, the role default is used:

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx \
  --real \
  --agent-models-json '{"root_engineer":{"model":"gpt-5.5","reasoning_effort":"xhigh"},"retriever":{"model":"gpt-5.4-mini","reasoning_effort":"medium"}}'
```

Selector ensembles still use `--selector-panel-json`, because each selector
member is an independent voter rather than a normal workflow role override.
For a cost-bounded GatexFlow-style tiering run, keep evidence and contract roles
on a mini model, route synthesis roles to the strongest model with `xhigh`, and
use a two-member selector panel only when `max_iterations >= 2` so selector
voting is actually exercised:

```json
{
  "data_analyst": {"model": "gpt-5.4-mini", "reasoning_effort": "high"},
  "evaluator": {"model": "gpt-5.4-mini", "reasoning_effort": "high"},
  "root_engineer": {"model": "gpt-5.5", "reasoning_effort": "xhigh"},
  "retriever": {"model": "gpt-5.4-mini", "reasoning_effort": "medium"},
  "proposer": {"model": "gpt-5.5", "reasoning_effort": "xhigh"},
  "critic": {"model": "gpt-5.4", "reasoning_effort": "high"},
  "engineer": {"model": "gpt-5.5", "reasoning_effort": "xhigh"},
  "debugger": {"model": "gpt-5.5", "reasoning_effort": "xhigh"},
  "result_analyst": {"model": "gpt-5.4-mini", "reasoning_effort": "high"},
  "visual_audit": {"model": "gpt-5.4-mini", "reasoning_effort": "high"},
  "selector": {"model": "gpt-5.4-mini", "reasoning_effort": "high"}
}
```

For first real runs, prefer root-only smoke before mutation:

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/burgers_pinn --real --max-iterations 0 --experiment-id burgers-root
```

Use `--dry-run` to print the planned role calls without making API requests.

Branch-context smoke can be prepared before credentials are available:

```bash
uv run --python 3.11 --extra dev agenticsciml smoke-llm examples/function_approx \
  --variants branch_context,no_branch_context \
  --dry-run \
  --max-iterations 1 \
  --parallel-mutations 2 \
  --output-dir runs/real-llm-smoke
```

The dry run writes `real_llm_smoke_plan.json` and
`real_llm_smoke_report.md`; it does not call an API and does not require
`OPENAI_API_KEY`.

If a local checkout path contains spaces and the editable console script cannot
import `agenticsciml`, use the module form. This is especially useful for smoke
artifact paths that also contain spaces:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli smoke-llm examples/function_approx \
  --variants branch_context,no_branch_context \
  --dry-run \
  --max-iterations 1 \
  --parallel-mutations 2 \
  --output-dir "runs/real llm smoke"
```

With credentials, the same command without `--dry-run` runs a minimal real LLM
smoke only when `--real` is explicit. Treat the output as prompt-delivery /
behavioral-difference evidence only, not as paper-scale SciML reproduction:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
uv run --python 3.11 --extra real-llm agenticsciml smoke-llm examples/function_approx \
  --variants branch_context,no_branch_context \
  --real \
  --llm-fast-mode \
  --max-iterations 1 \
  --parallel-mutations 2 \
  --output-dir runs/real-llm-smoke
```

The real smoke gate is a paired contrast: non-dry-run smoke must include the
exact pair `branch_context,no_branch_context`. It writes
`real_llm_smoke_manifest.json` before provider calls, reads
`trace_summary.json`, requires positive LLM call counts on both sides, checks
request-side prompt-delivery evidence for branch context, and verifies that
`no_branch_context` request prompts do not leak branch fields. If the gate
fails, the CLI returns non-zero and points at the report.

The run CSV and Markdown report also expose within-run score diagnostics:
metric direction, evaluated/failed solution counts, root score, best child
score, direction-normalized improvement, whether any child mutation beat its
root, failed-solution kinds, and concrete smoke-gate issues. These diagnostics
make regressions and provider/trace failures visible but do not affect the smoke
gate. Each variant generates its own root independently, so comparing the two
root or child scores does not isolate a causal branch-context effect and does
not support a performance claim.

The same rows distinguish debugger activity from recovery: attempted node
count, total attempts, recovered nodes, nodes still failed after debugging, and
the node-level recovery rate. A successful debugger API call is therefore not
misreported as a successful repair when the final solution node still fails.
Rows with no debugger attempts keep the CSV rate empty and render
`not_applicable` in Markdown rather than implying unknown evidence.

Budget, adapter, provider/orchestrator, post-run artifact, and paired-gate
failures finalize the manifest with `status=failed`, `report_status=failed`,
`failure_kind`, `error_type`, the failure-report digest, and the final
token-budget ledger. `verify-smoke-llm` must still reject that bundle as
incomplete real-smoke evidence, and it validates manifest/plan schema versions
plus the final call/token/cost ledger and configured limits against recomputed
run evidence.

Before any real provider call, smoke tooling writes the manifest and runs a
shared LLM call-budget preflight. If `expected_llm_call_range.max` exceeds
`AGENTICSCIML_MAX_LLM_CALLS`, the command exits before provider calls with
`blocked_by_budget` and writes a blocked report.

Real ablation manifests also include `budget_batch_plan`. When a full planned
matrix is over the configured call budget, run a single explicit batch with
`agenticsciml ablate --real --budget-batch-index N`; the manifest keeps the
full-stage run count, expected call range, and `full_stage_plan_hash` so partial
batches cannot be mistaken for complete Stage A evidence. Use
`agenticsciml collect-ablation-batches` to aggregate batches; it reorders rows by
the canonical stage plan, rejects duplicate/extra run IDs, and recomputes
`ablation_summary.csv` instead of concatenating stale summaries.

After a real or mock run completes, inspect the trace quality gate:

```bash
uv run --python 3.11 --extra dev agenticsciml trace-summary runs/<experiment_id>
```

After a real `smoke-llm --real` run completes, verify the whole smoke output
directory:

```bash
uv run --python 3.11 --extra dev agenticsciml verify-smoke-llm runs/real-llm-smoke
```

The same verifier can be invoked through the module form:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli verify-smoke-llm "runs/real llm smoke"
```

When pointed at a dry-run bundle, `verify-smoke-llm` is expected to exit
non-zero. That negative result confirms the bundle is not being accepted as
real LLM evidence.

The verifier rejects dry-run-only artifacts, recomputes the paired
branch/no-branch gates from the run directories, checks manifest/plan
consistency, and requires parallel-child trace evidence when
`parallel_mutations > 1`.

`run_metadata.json` also records aggregate LLM call counts by role plus prompt
and response text token estimates. When the adapter supplies numeric usage,
`llm_calls.provider_usage` separately records provider prompt/completion/total
tokens, the number of calls carrying usage, and whether coverage is complete.
Before a request, `llm_budget` reserves the local prompt estimate so limits can
fail before a provider call. After a response, it replaces that reservation
with provider prompt usage when available, then checks prompt/output/total/cost
limits again. Each ledger row preserves `prompt_token_estimate` separately from
`prompt_tokens_accounted`, plus `prompt_token_source` and
`response_token_source`; post-response failures remain billable and are written
before the error propagates. This includes provider responses that are billed
before JSON parsing or schema validation fails; if reconciled usage crosses a
budget, `LLMBudgetExceeded` stops further retries and the controlled underlying
error type remains in the ledger. Adapter metadata is thread-local so parallel
mutations cannot exchange per-call usage. Resume rejects partial or inconsistent
new accounting fields, while legacy ledgers with all new fields absent still
resume from local estimates. Real-smoke verification also binds each provider
accounted value to the matching generation trace by `llm_call_id`; a legacy
ledger row is accepted only when its matching legacy trace also has no provider
usage, so deleting the new fields cannot downgrade current evidence. Local
estimates and provider usage must not be presented as interchangeable billing
records.

The recording wrapper covers `complete_text`, `complete_json`, and
`complete_json_with_images`. A real visual audit therefore shares the same call,
prompt, output, total-token, and estimated-cost budget as every other agent role,
and its ledger row remains linked to the visual generation trace. The
`visual_audit` role uses its configured model/base URL through the same shared
recording state. `LLMBudgetExceeded` is never treated as a schema retry, visual
warning, or recoverable failed child; it propagates to the CLI/smoke boundary so
the run cannot be exported as completed after crossing its hard budget.
If a request is rejected before provider access, its failure trace intentionally
has no `llm_call_id` or provider usage. The wrapper clears the previous
thread-local call metadata before reserving budget so stale evidence cannot be
attached to the blocked attempt. It also resets the inner adapter's per-call
metadata before every request; a local image-read or request-construction error
therefore cannot reuse and rebill the preceding multimodal usage.
Any uncaught orchestrator failure also closes the current
`invocation_history.json` row as `status=failed`, with a controlled
`error_type`, completion timestamp, and invocation wall time. Deliberate
evaluation-approval pauses keep their existing paused/rejected statuses. A
failed CLI process must not leave an invocation looking indefinitely `running`.
For real runs, `run_metadata.llm_calls.total`, role counts, and call token
estimates count only generation events bound to an `llm_call_id`. A request
blocked before provider access remains visible in `generation_attempt_count`,
`unbound_generation_attempt_count`, and `pre_provider_rejection_count`, but is
not mislabeled as a provider call. This keeps ledger totals and provider usage
coverage interpretable after a failed invocation is successfully resumed.
If mandatory post-run finalization fails after an exported marker was written,
`run_metadata.run_state` is downgraded to `partial`, finalization failure fields
are recorded, a `run.failed` trace is appended, and `trace_summary.json` is
regenerated to fail the artifact-consistency gate. An existing
`openai_sdk_trace.json` is regenerated from the same final trace; if that rebuild
fails, the stale SDK export is removed. Cleanup failures are attached as a
controlled exception note and never replace the original run error.

Real visual audit retry is limited to output/schema correction. Budget excess is
recorded as non-retryable `budget_exceeded` and aborts the run; provider boundary
errors such as timeout/auth/rate-limit are recorded once as `provider_error` and
produce the visual guardrail failure without another provider request. The
visual report and request both use the role-specific provider capability record,
so a text-only base model plus image-capable visual override cannot produce
contradictory provenance.

After a real run, scan run artifacts for accidental secret leakage before
sharing or attaching them:

```bash
uv run --python 3.11 --extra dev agenticsciml secret-hygiene runs/<experiment_id> --fail-on-findings
```

The scanner reports file paths, rule IDs, locations, redaction metadata, and
non-secret-derived finding IDs only. It does not print matched secret values or
store secret-value hashes.
Older reports may contain legacy secret-value hashes; regenerate the hygiene
report before sharing a run artifact bundle.

Real-smoke manifest and run metadata also record provider capability, adapter
type, and budget state:

- `provider_capabilities`
- `adapter_type`
- `token_budget`
- `llm_provider_capabilities`
- `llm_budget`

Completed orchestrator runs export `openai_sdk_trace.json`, a sanitized
SDK-style span bundle derived from local `trace.jsonl`. The local trace remains
the source of truth.

Resume uses the existing run directory and checkpoint:

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --real --max-iterations 0 --experiment-id first-real-run
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --real --resume --max-iterations 1 --experiment-id first-real-run
```

Checkpoint-free failures before root creation are also resumable when their
artifacts are internally consistent. The resume preflight classifies the run as
`initialized`, `data_ready`, or `contract_ready`. A `data_ready` run reuses the
completed Data Analyst artifacts and resumes at evaluator creation, so an
already billed analysis call is not repeated. A `contract_ready` run reuses the
verified evaluation contract and continues at the approval/root boundary.
Partial or contradictory artifact sets fail closed instead of guessing which
work completed.

Consistency includes exact reconstruction of `data_analysis.md` from its
structured JSON, complete stage transcripts, and—for real runs—a successful
role/schema ledger row linked to a generation trace by the same `llm_call_id`.
A missing checkpoint cannot downgrade a directory that already contains a
tree, solution, champion, or run metadata into a pre-root run. An exclusive
`.invocation.lock` also covers the full orchestrator invocation; a concurrent
resume fails before it can append invocation, ledger, or trace records.
After acquiring the lock, a recording client reloads its shared call counter
and usage from the latest ledger. A client object constructed from an older
snapshot therefore cannot reuse a call ID after another invocation completes.

Run-level LLM ceilings may be increased for recovery, including changing a
finite ceiling to unlimited. They may not be decreased, and the cost rate,
provider/model identity, experiment configuration, benchmark, and source
revision remain frozen. `invocation_history.json` records the secret-free LLM
runtime and effective budget limits for every attempt; the checkpoint remains
bound to the original experiment-conditions digest. This recovery behavior is
operational evidence only and does not upgrade benchmark fidelity or support a
scientific claim.

Real resume dry-run reconstructs the current secret-free runtime identity and
applies the same compatibility rule without requiring an API key or making a
provider call. Model/config/source drift, a tighter ceiling, or a changed cost
rate therefore fails during dry-run preflight.

Source provenance includes a content digest of the effective local runtime
tree under `src/` plus `pyproject.toml` and `uv.lock`, including untracked files
that would affect an editable checkout. It excludes `.env`, run directories,
and unrelated local outputs. Legacy clean runs remain compatible when their
commit matches; legacy dirty runs without this digest fail closed because a
boolean `dirty=true` cannot identify the code that actually ran.

The real adapter is intentionally thin. It asks for structured JSON when an
agent expects structured output, but the orchestrator still validates generated
artifacts through the same local evaluation contract.
