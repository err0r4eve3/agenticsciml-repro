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

Real mode requires credentials and the optional adapter dependency:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --max-iterations 1
```

For OpenAI-compatible providers, set `OPENAI_BASE_URL` as well. For example,
DeepSeek-style endpoints can be used without changing the CLI surface:

```bash
export OPENAI_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY=...
export OPENAI_MODEL=deepseek-v4-pro
export OPENAI_TIMEOUT_S=120
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --max-iterations 0
```

Use the provider's exact model id. For example, if the provider rejects
`deepseekv4pro` and reports `deepseek-v4-pro`, use the hyphenated id.

When `OPENAI_BASE_URL` is unset, `OpenAIAdapter` treats the provider as an
OpenAI-native Responses path and uses typed Structured Outputs for agent JSON.
When `OPENAI_BASE_URL` is set, the adapter records an OpenAI-compatible
capability profile and falls back to chat completions plus local Pydantic
schema validation. `https://api.gatexflow.com/v1` is treated as an
OpenAI-compatible multimodal chat provider: structured text output still uses
local schema validation, while `--visual-audit-mode real` may send diagnostic
images through chat image content when the configured model supports it.

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

Role-level model policy can be set from CLI with `--agent-models-json`. The
payload is a JSON object keyed by role; each role value supports `model`,
`base_url`, `temperature`, and `reasoning_effort`. If `temperature` or
`reasoning_effort` is omitted, the role default is used:

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx \
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
uv run --python 3.11 --extra real-llm agenticsciml run examples/burgers_pinn --max-iterations 0 --experiment-id burgers-root
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
and response token estimates. These are accounting placeholders, not provider
billing records.

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
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --max-iterations 0 --experiment-id first-real-run
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --resume --max-iterations 1 --experiment-id first-real-run
```

The real adapter is intentionally thin. It asks for structured JSON when an
agent expects structured output, but the orchestrator still validates generated
artifacts through the same local evaluation contract.
