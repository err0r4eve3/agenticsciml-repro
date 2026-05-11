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

The real smoke gate reads `trace_summary.json`, requires positive LLM call
counts, and checks that `branch_context` / `no_branch_context` artifacts match
their variant. If the gate fails, the CLI returns non-zero and points at the
report.

After a real or mock run completes, inspect the trace quality gate:

```bash
uv run --python 3.11 --extra dev agenticsciml trace-summary runs/<experiment_id>
```

`run_metadata.json` also records aggregate LLM call counts by role plus prompt
and response token estimates. These are accounting placeholders, not provider
billing records.

Resume uses the existing run directory and checkpoint:

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --max-iterations 0 --experiment-id first-real-run
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --resume --max-iterations 1 --experiment-id first-real-run
```

The real adapter is intentionally thin. It asks for structured JSON when an
agent expects structured output, but the orchestrator still validates generated
artifacts through the same local evaluation contract.
