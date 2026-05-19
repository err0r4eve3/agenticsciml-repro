# Paper Notes: AgenticSciML

[返回文档树](index.md) · 相关文档：[多 Agent 设计方法](multi_agent_design.md)、[论文算法 Reference Primitives](paper_algorithm_primitives.md)、[版本说明](version_notes.md)

Sources checked:

- arXiv abstract page for `2511.07262`, version 2 revised on 2026-02-15.
- arXiv HTML v2 methods and supplementary table of contents.
- npj Artificial Intelligence article page, published 2026-04-30 as an early
  unedited manuscript.
- arXiv abstract page for `2605.11117v1`, submitted on 2026-05-11:
  GRAFT-ATHENA.
- arXiv abstract page for `2512.03476v2`, submitted on 2025-12-03 and revised
  on 2026-05-08: ATHENA.

## Workflow Phases

1. User input: `Problem.md`, `Requirements.md`, `Evaluation.md`, and optional
   `Data_config.json`.
2. Data analysis and evaluation criteria: a data analyst writes observation
   artifacts and a text report, then an evaluator formalizes a testing
   contract.
3. Solution evolution: a root single-agent baseline is generated, then a
   solution tree is expanded through retrieval, debate, mutation, debugging,
   evaluation, and analysis.

## Agent Roles

- Data Analyst: inspects public training data and writes `data_analysis.md`,
  `data_observations.json`, `data_overview.svg`, and replayable EDA artifacts
  `data_eda.py` / `data_eda.json`.
- Evaluator: creates `evaluate.py`, `guidelines.md`, and the scoring contract.
- Root Engineer: creates the first baseline solution without KB or debate.
- Retriever: selects 0-1 relevant KB entries for a parent solution.
- Proposer: develops a concise diagnosis and concrete mutation plan.
- Critic: challenges gaps, risks, and feasibility.
- Engineer: modifies parent code according to the final proposal.
- Debugger: fixes execution failures within retry limits.
- Result Analyst: writes the solution report and prediction-only observation
  artifacts used by later mutations.
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

## ATHENA / GRAFT-ATHENA Update

ATHENA describes a HENA loop where structural actions `A_n` are selected from
expert-blueprint-guided combinatorial spaces, translated into executable code
`S_n`, and evaluated into scientific rewards `R_n`. GRAFT-ATHENA adds a
factored-tree framing: a method is an ordered path through a reduced action
space, and the path can be treated as a stable fingerprint for later retrieval
or experience accumulation.

The local implementation now captures only the safe deterministic subset in
[Method Substrate 合约](method_substrate.md) and
[ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md): blueprint-bounded
`MethodAction` objects, ordered `MethodPath` fingerprints, finite
`ScientificReward` records, local JSON `ExperienceSubstrate` records,
source-grounded method templates, and explicit `A_n -> S_n -> R_n` trace
payloads. This is a workflow traceability contract, not a claim that the
project implements GRAFT's probabilistic policy factorization, cross-domain
self-improvement, autonomous action-space expansion, or the ATHENA paper's
reported performance.

## Benchmark Coverage

The local benchmark catalog mirrors the six paper task families with lightweight
offline proxies and six faithful-small upgrades:

- `S1.1` `function_approx`: discontinuous oscillatory function approximation proxy.
- `S1.1` `function_approx_faithful_small`: paper-described piecewise function
  approximation with 200 training samples and 500 validation samples.
- `S1.2` `poisson_lshape`: L-shaped Poisson / re-entrant corner proxy.
- `S1.2` `poisson_lshape_faithful_small`: L-shaped Poisson with boundary and
  PDE residual collocation data, still low-budget and not paper-like.
- `S1.3` `burgers_pinn`: time-dependent Burgers-style PINN proxy.
- `S1.3` `burgers_pinn_faithful_small`: time-dependent Burgers-style task with
  IC/BC anchors and collocation coordinates, still low-budget and not paper-like.
- `S1.4` `antiderivative_operator`: input functions to antiderivatives.
- `S1.4` `antiderivative_operator_faithful_small`: 100-point
  function-to-antiderivative operator learning with per-sample relative L2.
- `S1.5` `reaction_diffusion_operator`: multiple-input reaction-diffusion operator proxy.
- `S1.5` `reaction_diffusion_operator_faithful_small`: diffusion/source/initial
  fields to a 40x50 spatiotemporal response, still low-budget and not paper-like.
- `S1.6` `cylinder_wake_reconstruction`: sparse sensor to vorticity-field reconstruction proxy.
- `S1.6` `cylinder_wake_reconstruction_faithful_small`: SHRED-style lagged
  sparse sensor-history reconstruction on deterministic synthetic
  cylinder-wake-like vorticity fields, still low-budget and not paper-like.

These examples preserve workflow pressure points from the paper but use small
deterministic local datasets for validation. The faithful-small entries narrow
the task-structure gap, but they still do not reproduce the paper's training
budget, prompt set, model mix, or reported scores.

## Knowledge Base And Analysis Base

The retriever may return zero or one KB entry per mutation to avoid context
pollution. The analysis base supplies parent, sibling, and uncle reports so new
children can build on nearby successes and failures.

## Paper-Listed Algorithm Primitives

The local algorithm layer now includes reference primitives for the six champion
strategy summaries reported in the paper results section: sigmoid-gated MoE,
Poisson particular-plus-residual decomposition with corner-biased sampling,
Burgers staged PINN schedule helpers, linear bias-free DeepONet branch mapping,
reaction-diffusion FNO-style derivative/constraint helpers, and cylinder
U-FNO/CNO-style bandlimited decoder filtering. Details and claim boundaries are
recorded in [论文算法 Reference Primitives](paper_algorithm_primitives.md).

## MVP Interpretation

The repository recreates the workflow mechanics. It does not attempt to match
the paper's exact benchmark scores, agent model mix, token counts, or private
prompt implementation.
