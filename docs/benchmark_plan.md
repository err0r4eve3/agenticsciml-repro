# Benchmark 与实验设计

[返回文档树](index.md) · 相关文档：[论文机制笔记](paper_notes.md)、[Real LLM 运行](real_run.md)、[Ablation 说明](ablation.md)

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
- `train_data.npz` / `val_data.npz`
- `x_train` / `u_train` / `x_val` / `u_val`
- `solution.py --mode=validate`
- `solution.py --mode=train`
- training writes `model.pkl`
- evaluator writes `eval.json`

## Benchmark Catalog

| Name | Paper section | Local path | Metric | Purpose |
| --- | --- | --- | --- | --- |
| `function_approx` | `S1.1` | `examples/function_approx` | `validation_mse` | 不连续振荡函数拟合 |
| `poisson_lshape` | `S1.2` | `examples/poisson_lshape` | `relative_l2` | L-shaped Poisson / corner singularity proxy |
| `burgers_pinn` | `S1.3` | `examples/burgers_pinn` | `relative_l2` | Burgers-style time-dependent PINN proxy |
| `antiderivative_operator` | `S1.4` | `examples/antiderivative_operator` | `relative_l2` | 函数到反导数的 operator learning |
| `reaction_diffusion_operator` | `S1.5` | `examples/reaction_diffusion_operator` | `relative_l2` | 多输入 reaction-diffusion operator proxy |
| `cylinder_wake_reconstruction` | `S1.6` | `examples/cylinder_wake_reconstruction` | `relative_l2` | 稀疏传感器到 2D 涡量场重建 proxy |

用 CLI 查看当前 catalog：

```bash
uv run --python 3.11 --extra dev agenticsciml benchmarks
uv run --python 3.11 --extra dev agenticsciml benchmarks --json
```

## 离线验证矩阵

真实 LLM 接入前，必须先验证 benchmark shape：

```bash
uv run --python 3.11 --extra dev pytest tests/test_benchmark_catalog.py -q
```

这个测试覆盖：

- catalog 是否包含论文 6 类任务；
- 每个 benchmark 是否有必需 artifact；
- 数据生成是否 deterministic；
- 每个 evaluator 是否接受 generic baseline；
- 每个 benchmark 是否能跑 root-only mock；
- 每个 benchmark 是否能跑 1 轮 mock evolution。

## 真实 LLM 实验顺序

接入真实 LLM 后按风险从低到高执行：

1. 每个 benchmark 先跑 `--max-iterations 0`，只验证 root baseline。
2. 每个 benchmark 跑 `--max-iterations 1 --parallel-mutations 1`，验证 proposal/critic/engineer/debugger 链路。
3. 对 `function_approx`、`poisson_lshape`、`burgers_pinn` 跑 root-only / no-KB / KB ablation。
4. 对 operator benchmarks 增加 multi-seed 重复。
5. 最后再跑 `cylinder_wake_reconstruction`，因为输出维度更高，debug 成本更大。

示例：

```bash
uv run --python 3.11 --extra real-llm agenticsciml run examples/poisson_lshape --max-iterations 0 --experiment-id poisson-root
uv run --python 3.11 --extra real-llm agenticsciml run examples/poisson_lshape --max-iterations 1 --parallel-mutations 1 --experiment-id poisson-one-iter
```

## 边界

- 这些 benchmark 是论文任务家族的 deterministic engineering proxies。
- 当前数据规模刻意较小，优先验证 workflow 和 evaluator 稳定性。
- 不保证论文 improvement factor。
- 不把 mock-mode 分数当作 SciML 结论。
- 后续如要追论文数值，应替换为更完整的 PDE/operator 数据和训练预算。
