# Paper Notes: AgenticSciML

[返回文档树](index.md) · 相关文档：[多 Agent 设计方法](multi_agent_design.md)、[版本说明](version_notes.md)

Sources checked:

- arXiv abstract page for `2511.07262`, version 2 revised on 2026-02-15.
- arXiv HTML v2 methods and supplementary table of contents.
- npj Artificial Intelligence article page, published 2026-04-30 as an early
  unedited manuscript.

## Workflow Phases

1. User input: `Problem.md`, `Requirements.md`, `Evaluation.md`, and optional
   `Data_config.json`.
2. Data analysis and evaluation criteria: a data analyst writes a text-only
   report, then an evaluator formalizes a testing contract.
3. Solution evolution: a root single-agent baseline is generated, then a
   solution tree is expanded through retrieval, debate, mutation, debugging,
   evaluation, and analysis.

## Agent Roles

- Data Analyst: inspects data and writes `data_analysis.md`.
- Evaluator: creates `evaluate.py`, `guidelines.md`, and the scoring contract.
- Root Engineer: creates the first baseline solution without KB or debate.
- Retriever: selects 0-1 relevant KB entries for a parent solution.
- Proposer: develops a concise diagnosis and concrete mutation plan.
- Critic: challenges gaps, risks, and feasibility.
- Engineer: modifies parent code according to the final proposal.
- Debugger: fixes execution failures within retry limits.
- Result Analyst: writes the solution report used by later mutations.
- Selector: votes for exploration parents while the best solution is always
  included for exploitation.

## Solution Tree

Every node stores code, logs, score, proposal, and analysis. Parent selection
always includes the current best node, adds exploration parents through selector
votes, and enforces a maximum child count per node.

## Evaluation Contract

Every solution must be evaluated by the same script and metric. This MVP uses:

- `solution.py` defines class `MODEL`
- `python solution.py --mode=validate`
- `python solution.py --mode=train`
- `python solution.py --mode=predict --input predict_input.npz --output predictions.npz`
- training writes `model.pkl`
- `python <private_eval>/evaluate.py` writes `eval.json`
- metric: benchmark-specific scalar loss, lower is better

The local implementation uses prediction-only evaluation. `solution.py` sees
training data during validate/train and only `x_val` during predict. It writes
`predictions.npz`; trusted evaluator code reads private `u_val` labels and
computes the metric without importing `solution.py`.

## Benchmark Coverage

The local benchmark catalog mirrors the six paper task families with lightweight
offline proxies:

- `S1.1` `function_approx`: discontinuous oscillatory function approximation.
- `S1.2` `poisson_lshape`: L-shaped Poisson / re-entrant corner proxy.
- `S1.3` `burgers_pinn`: time-dependent Burgers-style PINN proxy.
- `S1.4` `antiderivative_operator`: input functions to antiderivatives.
- `S1.5` `reaction_diffusion_operator`: multiple-input reaction-diffusion operator proxy.
- `S1.6` `cylinder_wake_reconstruction`: sparse sensor to vorticity-field reconstruction proxy.

These examples preserve workflow pressure points from the paper but use small
deterministic NumPy datasets for local validation.

## Knowledge Base And Analysis Base

The retriever may return zero or one KB entry per mutation to avoid context
pollution. The analysis base supplies parent, sibling, and uncle reports so new
children can build on nearby successes and failures.

## MVP Interpretation

The repository recreates the workflow mechanics. It does not attempt to match
the paper's exact benchmark scores, agent model mix, token counts, or private
prompt implementation.
