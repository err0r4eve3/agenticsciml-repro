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
- `solutions/solution_*/retrieval_query.txt`
- `leaderboard.csv`
- `tree.json`
- `tree.mmd`
- `checkpoint.json`
- `trace.jsonl`
- `trace_summary.json`
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

## Real LLM Mode

Mock mode is deterministic and is the default for tests. Real LLM mode is
optional and requires an adapter dependency and API credentials:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-mini
uv run --python 3.11 --extra real-llm agenticsciml run examples/function_approx --max-iterations 1
```

The benchmark catalog now includes all six paper task families as lightweight
offline examples. Each catalog entry records `fidelity_level`, expected runtime,
dependency flags, and paper-gap notes so proxy tasks are not mistaken for full
paper experiments:

- `examples/function_approx`
- `examples/poisson_lshape`
- `examples/burgers_pinn`
- `examples/antiderivative_operator`
- `examples/reaction_diffusion_operator`
- `examples/cylinder_wake_reconstruction`

These examples keep PyTorch as an optional `sciml` extra for real generated
SciML solutions; the checked-in benchmark fixtures use NumPy to keep local
verification light.

## Scope

This is not a claim that the paper's reported improvement factors are reproduced.
The MVP validates orchestration, persistence, evaluation contracts, and
deterministic mock behavior first. The added benchmarks are engineering proxies
for the paper tasks, not full paper-score reproductions.
