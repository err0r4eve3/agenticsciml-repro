# AgenticSciML Workflow Reproduction

This repository is an engineering MVP that recreates the workflow described in
`AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in
Scientific Machine Learning` (`arXiv:2511.07262`).

The goal is to reproduce the multi-agent process, not the paper's exact scores.
The official implementation details and full prompts are not assumed to be
available, so this project implements a source-grounded approximation:

- structured `Problem.md`, `Requirements.md`, `Evaluation.md`, and optional
  `Data_config.json` inputs
- data analysis and evaluation-contract artifacts
- benchmark-aware evaluation contracts with deterministic content hashes
- root single-agent solution generation
- solution tree with deterministic exploitation/exploration parent selection,
  parallel child mutation jobs, evaluation, and analysis
- benchmark-aware 0-1 knowledge-base retrieval per mutation, with deterministic
  `random_kb` mode for ablation
- proposer/critic debate with concise rationale summaries
- typed Pydantic schemas for code-consumed agent JSON outputs
- engineer and debugger roles around generated code
- digest-checked patch mutation for generated `solution.py`
- contract-aware debugger repairs using digest-checked unified diff patches
- per-solution workspaces, logs, scores, prompts, responses, and reports
- prediction-only evaluation: generated code sees validation features only,
  while trusted evaluator code keeps labels and validation paths private
- sanitized subprocess environments for generated solution execution
- champion export

## Quick Start

```bash
uv run --python 3.11 --extra dev pytest -q
uv run --python 3.11 --extra dev agenticsciml benchmarks
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 1
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 1 --random-kb --random-seed 11
```

If an editable install is being used from a checkout path with spaces and the
console script cannot import `agenticsciml`, use the module form:

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli run examples/function_approx --mock --max-iterations 1
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli smoke-llm examples/function_approx --variants branch_context,no_branch_context --dry-run --max-iterations 1 --parallel-mutations 2 --output-dir "runs/real llm smoke"
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli verify-smoke-llm "runs/real llm smoke"
```

The dry-run `verify-smoke-llm` command is an expected negative check and should
exit non-zero. Use `smoke-llm --real` before expecting verification to pass.

Inspect the trace quality gate for a completed run:

```bash
uv run --python 3.11 --extra dev agenticsciml trace-summary runs/<experiment_id>
```

Run a mock ablation suite:

```bash
uv run --python 3.11 --extra dev agenticsciml ablate examples/function_approx --seeds 0 1 2 --variants root_only,no_kb,kb,random_kb,no_critic,no_debugger
```

Resume an interrupted or staged run by reusing the same `--output-dir` and
`--experiment-id` with `--resume`:

```bash
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 0 --experiment-id demo
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --resume --max-iterations 1 --experiment-id demo
```

The mock run writes a directory under `runs/` with:

- `solutions/solution_*/solution.py`
- `solutions/solution_*/proposal.md`
- `solutions/solution_*/train.log`
- `solutions/solution_*/eval.json`
- `solutions/solution_*/analysis.md`
- `solutions/solution_*/solution_observations.json`
- `solutions/solution_*/prediction_overview.svg`
- `solutions/solution_*/retrieval_query.txt`
- `reports/data_analysis.md`
- `reports/data_observations.json`
- `reports/data_overview.svg`
- `leaderboard.csv`
- `tree.json`
- `tree.mmd`
- `checkpoint.json`
- `trace.jsonl`
- `trace_summary.json`
- `openai_sdk_trace.json`
- `run_metadata.json` with wall time, champion, solution count, and LLM call
  count/token-estimate placeholders, plus `llm_mode`,
  `benchmark_fidelity_level`, `evidence_mode`, and `scientific_claim`
- `champion/solution.py`
- `champion/analysis.md`

The ablation command writes `ablation_runs.csv`, `ablation_summary.csv`, and
`ablation_report.md`. Mock ablation rows are labeled with
`evidence_mode=mock_workflow_shape` and `scientific_claim=not_supported`; they
validate workflow shape and reporting only, not emergent discovery or
paper-score reproduction.

## Documentation Tree

中文文档入口在 [docs/index.md](docs/index.md)。新增项目说明、设计笔记、
运行说明或版本说明时，先挂到这个文档树，避免孤立页面。
项目采用 [Git 与 Markdown 分层记录方法论](docs/git_markdown_methodology.md)：
Git 记录变更时间轴，互链 Markdown 树记录长期知识结构，run artifacts
记录单次实验机器证据。

## Real LLM Mode

Mock mode is deterministic and is the default for tests. Real LLM mode is
optional and requires an adapter dependency and API credentials:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --max-iterations 1
```

OpenAI-native runs use Responses structured outputs when `OPENAI_BASE_URL` is
unset. OpenAI-compatible providers keep the chat/JSON fallback and record their
capability matrix in run artifacts. Optional real-LLM budget gates:

```bash
export AGENTICSCIML_MAX_LLM_CALLS=80
export AGENTICSCIML_MAX_PROMPT_TOKENS=200000
export AGENTICSCIML_MAX_OUTPUT_TOKENS=80000
export AGENTICSCIML_MAX_COST_USD=5
export AGENTICSCIML_COST_PER_1K_TOKENS_USD=0.01
```

The benchmark catalog now includes all six paper task families as lightweight
offline examples, plus five `faithful-small` upgrades for S1.1 function
approximation, S1.2 L-shaped Poisson, S1.3 Burgers, S1.4 antiderivative
operator learning, and S1.5 reaction-diffusion operator learning. Each catalog entry records
`fidelity_level`, expected runtime, dependency flags, and paper-gap notes so
proxy tasks are not mistaken for full paper experiments:

- `examples/function_approx`
- `examples/function_approx_faithful_small`
- `examples/poisson_lshape`
- `examples/poisson_lshape_faithful_small`
- `examples/burgers_pinn`
- `examples/burgers_pinn_faithful_small`
- `examples/antiderivative_operator`
- `examples/antiderivative_operator_faithful_small`
- `examples/reaction_diffusion_operator`
- `examples/reaction_diffusion_operator_faithful_small`
- `examples/cylinder_wake_reconstruction`

These examples keep PyTorch as an optional `sciml` extra for real generated
SciML solutions. Most checked-in benchmark fixtures use NumPy to keep local
verification light; `function_approx_faithful_small` follows the paper's
piecewise S1.1 data shape, `poisson_lshape_faithful_small` adds boundary
and PDE residual collocation arrays, `burgers_pinn_faithful_small` adds
IC/BC/collocation structure, and `antiderivative_operator_faithful_small`
adds 100-point function-to-function operator scoring, and
`reaction_diffusion_operator_faithful_small` adds a multiple-input
function-to-spatiotemporal-field task without claiming paper-like score parity.

## Scope

This is not a claim that the paper's reported improvement factors are reproduced.
The MVP validates orchestration, persistence, evaluation contracts, and
deterministic mock behavior first. The benchmark catalog now mixes engineering
proxies with five faithful-small tasks; none of these are full paper-score
reproductions.
