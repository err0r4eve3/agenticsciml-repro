# Benchmark 与实验设计

[返回文档树](index.md) · 相关文档：[论文机制笔记](paper_notes.md)、[论文算法 Reference Primitives](paper_algorithm_primitives.md)、[Benchmark Fidelity Levels](fidelity_levels.md)、[Real LLM 运行](real_run.md)、[Ablation 说明](ablation.md)

## 目标

当前目标是补齐论文 6 类 benchmark 的可运行工程代理，而不是声明复现论文分数。

每个 benchmark 必须满足同一个最小契约：

- `Problem.md`
- `Requirements.md`
- `Evaluation.md`
- `Data_config.json`
- `generate_data.py`
- `evaluate.py`
- `guidelines.md`
- optional `kb/`
- train-time data: `train_data.npz`
- evaluator-only data: `private_eval/solution_*/val_data.npz` outside solution
  workspaces
- `x_train` / `u_train` / `x_val` / `u_val`
- `solution.py --mode=validate`
- `solution.py --mode=train`
- `solution.py --mode=predict --input predict_input.npz --output predictions.npz`
- training writes `model.pkl`
- predict writes `predictions.npz` with a `predictions` array
- evaluator writes `eval.json`
- `evaluation_contract.json` records `benchmark_name`, `contract_hash`,
  `allowed_train_files`, and `evaluator_only_files`
- `evaluation_contract.json` records benchmark fidelity metadata and binds it
  into `contract_hash`
- mutation output records `parent_digest` and applies a Python-verified patch or
  explicit `solution.py` file map
- solution tree nodes record `method_tags`, `failure_kind`,
  `score_delta_from_parent`, `num_debug_attempts`, `benchmark_name`, and
  `contract_hash`
- retrieval query is built from benchmark metadata, parent analysis, failure
  kind, method tags, score trend, and leaderboard top-k context
- catalog metadata records `fidelity_level`, `expected_runtime_s`,
  `requires_torch`, `requires_gpu`, and `paper_gap_notes`
- `agenticsciml benchmarks --json` exposes per-benchmark `claim_boundaries`,
  including mock claim, real-LLM claim, and explicit
  `paper_score_reproduction=not_supported`

## Benchmark Catalog

| Name | Paper section | Fidelity | Local path | Metric | Purpose |
| --- | --- | --- | --- | --- | --- |
| `function_approx` | `S1.1` | `proxy` | `examples/function_approx` | `validation_mse` | 不连续振荡函数拟合 |
| `function_approx_faithful_small` | `S1.1` | `faithful-small` | `examples/function_approx_faithful_small` | `validation_mse` | Paper-described piecewise oscillatory fitting with 200 train / 500 validation samples |
| `poisson_lshape` | `S1.2` | `proxy` | `examples/poisson_lshape` | `relative_l2` | L-shaped Poisson / corner singularity proxy |
| `poisson_lshape_faithful_small` | `S1.2` | `faithful-small` | `examples/poisson_lshape_faithful_small` | `poisson_residual_composite` | L-shaped Poisson with boundary/residual collocation scoring |
| `burgers_pinn` | `S1.3` | `proxy` | `examples/burgers_pinn` | `relative_l2` | Burgers-style time-dependent PINN proxy |
| `burgers_pinn_faithful_small` | `S1.3` | `faithful-small` | `examples/burgers_pinn_faithful_small` | `burgers_pinn_composite` | Time-dependent Burgers task with IC/BC anchors and collocation coordinates |
| `antiderivative_operator` | `S1.4` | `proxy` | `examples/antiderivative_operator` | `relative_l2` | 函数到反导数的 operator learning |
| `antiderivative_operator_faithful_small` | `S1.4` | `faithful-small` | `examples/antiderivative_operator_faithful_small` | `relative_l2` | 100-point function-to-antiderivative operator learning with per-sample relative L2 |
| `reaction_diffusion_operator` | `S1.5` | `proxy` | `examples/reaction_diffusion_operator` | `relative_l2` | 多输入 reaction-diffusion operator proxy |
| `reaction_diffusion_operator_faithful_small` | `S1.5` | `faithful-small` | `examples/reaction_diffusion_operator_faithful_small` | `relative_l2` | 多输入函数到 40x50 时空响应的 reaction-diffusion operator learning |
| `cylinder_wake_reconstruction` | `S1.6` | `proxy` | `examples/cylinder_wake_reconstruction` | `relative_l2` | 稀疏传感器到 2D 涡量场重建 proxy |
| `cylinder_wake_reconstruction_faithful_small` | `S1.6` | `faithful-small` | `examples/cylinder_wake_reconstruction_faithful_small` | `relative_l2` | SHRED-style lagged sparse sensor-history reconstruction on 24x24 synthetic cylinder-wake-like fields |

`poisson_lshape_faithful_small` is residual-scored but still prediction-only:
the trusted evaluator asks generated code for function values at solution,
boundary, residual-center, and finite-difference stencil points, then computes
a private composite score without importing `solution.py`.

`burgers_pinn_faithful_small` adds initial-condition, periodic-boundary, and
unlabeled collocation arrays to the training interface. The evaluator remains
prediction-only and scores dense solution, initial, and boundary blocks without
asking generated code to reveal or access private labels.

`antiderivative_operator_faithful_small` upgrades the operator-learning task to
100-point input/output functions and mean per-sample relative L2 scoring. It is
intended to pressure DeepONet-style or linear-operator strategies while keeping
all data local and deterministic.

`reaction_diffusion_operator_faithful_small` upgrades the multiple-input
operator task to diffusion/source/initial-condition channels over a 50-point
grid and a flattened 40x50 spatiotemporal response. The official score remains
mean per-sample relative L2; early/late-time and gradient diagnostics are
reported as secondary evidence only.

`cylinder_wake_reconstruction_faithful_small` upgrades the sparse reconstruction
task to a SHRED-style lagged sensor-history interface: five time steps of eight
fixed sparse sensors plus current time are mapped to a flattened 24x24 synthetic
cylinder-wake-like vorticity field. It is grounded by SHRED-ROM / PySHRED
architecture patterns and public code, but it does not run their original code,
use their datasets, train an LSTM, or claim their scores.

`burgers_pinn` now has a source-grounded local KB seeded from Raissi,
Perdikaris, and Karniadakis's PINNs paper plus the public `maziarraissi/PINNs`
Burgers example. The KB is documented in
[SciML 论文与代码知识库](sciml_knowledge_base.md) and remains a retrieval hint,
not a paper-score reproduction claim.

`fidelity_level` 的含义：

- `proxy`：本地 NumPy 小规模代理任务，只验证 workflow / evaluator / artifact shape。
- `faithful-small`：后续目标，使用更接近论文的 SciML/PyTorch 训练目标，但缩小预算。
- `paper-like`：后续目标，尽量贴近论文数据、训练预算和指标，不默认承诺复现论文分数。

详细准入标准见 [Benchmark Fidelity Levels](fidelity_levels.md)。

用 CLI 查看当前 catalog：

```bash
uv run --python 3.11 --extra dev agenticsciml benchmarks
uv run --python 3.11 --extra dev agenticsciml benchmarks --json
```

The plain table includes `fidelity_level` and the maximum real-LLM
`scientific_claim`; the JSON output includes explicit mock/real claim
boundaries so proxy tasks cannot be mistaken for paper-score reproduction.

## Paper-Listed Algorithm Primitives

`GET /api/algorithms` exposes six `status=reference_implementation` entries
mapped to the paper's champion strategy summaries. They live in
`src/agenticsciml/paper_algorithms.py` and remain local primitives for prompt
seeding and unit-level verification. They do not change benchmark scores,
evaluator contracts, champion selection, or paper-score claim boundaries.

## 离线验证矩阵

真实 LLM 接入前，必须先验证 benchmark shape：

```bash
uv run --python 3.11 --extra dev python -m pytest -q \
  tests/test_benchmark_catalog.py \
  tests/test_scientific_benchmarks.py
```

这个测试覆盖：

- catalog 是否包含论文 6 类任务；
- catalog 是否为每个任务记录 fidelity metadata 和 paper-gap notes；
- 每个 benchmark 是否有必需 artifact；
- `ProblemBundle` 是否能按 benchmark 加载；
- `BenchmarkContractFactory` 是否生成 hash-stable 的 benchmark-aware contract；
- 数据生成是否 deterministic；
- 每个 evaluator 是否接受 generic baseline；
- 全部 12 个内置 evaluator 是否拒绝 broadcast/reshape、non-finite prediction、
  non-standard JSON score 反例；
- L-shape notch-edge 连续性、Poisson forcing 和 faithful-small Burgers PDE/residual
  是否与文档口径一致；
- solution train/validate workspace 是否不暴露验证集；
- 全部 12 个 benchmark 是否都能运行 deterministic root-only mock，以及 1 轮
  `max_iterations=1, parallel_mutations=2` 的 mock evolution。

当前 catalog-wide 断言覆盖全部 12 个 benchmark 的 artifact、bundle、contract、
deterministic generator 和 generic-baseline evaluator。checked-in orchestrator smoke
回归矩阵也覆盖全部 12 个 benchmark：每项运行一个 root-only deterministic mock 和一个
`max_iterations=1, parallel_mutations=2` deterministic mock，共 24 个 run 场景。每个场景
都要求节点成功评估、`trace_summary.json` quality gate 通过，同时
`scientific_discovery_readiness` 保持 `status=blocked`、
`scientific_claim_supported=false`。该 `12/12` 只表示本地 mock workflow coverage；它不支持
真实 LLM、paper parity、SOTA 或科学发现声明。真实实验仍须按下文顺序逐项运行并保留
artifact。

验证集泄漏是 P0 约束：`solution.py` 训练/验证阶段的 cwd 下不能存在
`val_data.npz` 或 evaluator-private 目录；predict 阶段只能读取
`predict_input.npz` 中的 `x_val`，不能看到 `u_val`、validation path 或
evaluator env。`evaluate.py` 只读取 `predictions.npz` 和私有标签计算分数，
不得 import generated `solution.py`。明显的 `val_data.npz`、parent
traversal、`.evaluator` / `private_eval` 字符串引用仍会被 static guardrail
拦截。

## 真实 LLM 实验顺序

接入真实 LLM 后按风险从低到高执行：

1. 每个 benchmark 先跑 `--max-iterations 0`，只验证 root baseline。
2. 每个 benchmark 跑 `--max-iterations 1 --parallel-mutations 1`，验证 proposal/critic/engineer/debugger 链路。
3. 对 `function_approx`、`function_approx_faithful_small`、`poisson_lshape`、`poisson_lshape_faithful_small`、`burgers_pinn`、`burgers_pinn_faithful_small`、`antiderivative_operator_faithful_small`、`reaction_diffusion_operator_faithful_small`、`cylinder_wake_reconstruction_faithful_small` 跑
   `root_only` / `no_kb` / `kb` / `random_kb` / `no_critic` / `no_debugger`
   ablation。
4. 对 KB ablation 同时比较 `use_kb=False`、lexical KB 和 deterministic
   `random_kb`，并用 `ablation_summary.csv` 的 median/IQR 判断稳定性。
5. 对 operator benchmarks 增加 multi-seed 重复。
6. 最后再跑 `cylinder_wake_reconstruction` 和 `cylinder_wake_reconstruction_faithful_small`，因为输出维度更高，debug 成本更大。

示例：

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/poisson_lshape --real --max-iterations 0 --experiment-id poisson-root
uv run --python 3.11 --extra real-llm agenticsciml run examples/poisson_lshape --real --max-iterations 1 --parallel-mutations 1 --experiment-id poisson-one-iter
uv run --python 3.11 --extra real-llm agenticsciml run examples/poisson_lshape_faithful_small --real --max-iterations 1 --parallel-mutations 1 --experiment-id poisson-faithful-small
```

## 边界

- 这些 benchmark 是论文任务家族的 deterministic engineering proxies。
- 当前数据规模刻意较小，优先验证 workflow 和 evaluator 稳定性。
- 当前 `fidelity_level=proxy` 明确表示不是论文全量 SciML 实验。
- `fidelity_level` 必须进入 `EvaluationContract` 和 `contract_hash`，不能只写在 README 或 catalog 展示层。
- 不保证论文 improvement factor。
- 不把 mock-mode 分数当作 SciML 结论。
- 后续如要追论文数值，应替换为更完整的 PDE/operator 数据和训练预算。
