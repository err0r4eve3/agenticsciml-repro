# 论文算法 Reference Primitives

[返回文档树](index.md) · 相关文档：[论文机制笔记](paper_notes.md)、[Benchmark 与实验设计](benchmark_plan.md)

## 范围

`src/agenticsciml/paper_algorithms.py` 实现的是 AgenticSciML 论文结果区列出的
6 个 champion strategy 的 dependency-light reference primitives。它们用于
本地算法目录、prompt seeding、单元测试和后续生成方案的构件复核，不是论文全量
训练管线，也不支持 paper-score reproduction 声明。

这些 primitives 只依赖 NumPy，并保持 deterministic / offline testing。

## 已实现 Primitive

| Paper task | Catalog id | Implementation entrypoint | Local guarantee |
| --- | --- | --- | --- |
| S1.1 discontinuous function approximation | `paper_sigmoid_moe_gate` | `sigmoid_moe_prediction` | sigmoid gate 有界、单调，按 gate 组合左右 experts |
| S1.2 L-shaped Poisson PINN | `paper_poisson_decomposition_sampler` | `particular_plus_residual` | particular + residual 组合、corner-biased sampling weights 可复现 |
| S1.3 Burgers PINN | `paper_burgers_staged_pinn_schedule` | `burgers_three_phase_schedule` | IC/BC pretrain、gPINN/self-adaptive residual、RAR/L-BFGS phase intent 可审计 |
| S1.4 antiderivative operator learning | `paper_linear_bias_free_deeponet` | `linear_bias_free_deeponet_prediction` | branch map 无 bias，局部验证线性性 |
| S1.5 reaction-diffusion multiple-input operator | `paper_reaction_diffusion_fno_helpers` | `derivative_enhanced_loss` | derivative-enhanced loss、hard BC/IC enforcement、spectral truncation helper |
| S1.6 cylinder wake sparse reconstruction | `paper_cylinder_bandlimited_filter` | `bandlimit_preserving_activation` | Gaussian low-pass / bandlimit-preserving activation 降低高频能量 |

## Claim Boundary

- `status=reference_implementation` 只表示有可调用、可测试的本地 primitive。
- 分数、champion、scientific claim 仍只能来自 benchmark evaluator 和 run artifacts。
- 这些实现不复制论文私有 prompt、训练预算、模型 ensemble、GPU 训练管线或 reported
  improvement factors。
- 如果后续要升级为 `paper-like`，需要新增完整数据规模、训练预算、多 seed real LLM
  run、ablation、trace summary 和外部审计证据。

## 验证

新增测试：

```bash
uv run --python 3.11 --extra dev pytest tests/test_paper_algorithms.py -q
```

测试覆盖数学性质、shape、determinism、catalog exposure 和 no-overclaim 边界。
