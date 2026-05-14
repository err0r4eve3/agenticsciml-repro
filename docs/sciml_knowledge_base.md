# SciML 论文与代码知识库

[返回文档树](index.md) · 相关文档：[Benchmark 与实验设计](benchmark_plan.md)、[OpenAI Agents SDK 升级复盘](openai_agents_sdk_upgrade_review.md)

## 选取结果

本轮选取 Raissi, Perdikaris, and Karniadakis 的 PINNs 论文作为
`burgers_pinn` benchmark 的第一组外部知识源：

- 论文：Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019), 686-707, DOI `10.1016/j.jcp.2018.10.045`。
- 公开代码：`maziarraissi/PINNs`，MIT License；仓库 README 标注该实现不再积极维护，因此本项目只抽取方法约束和实验提示，不复制运行时代码。
- 相关代码路径：`appendix/continuous_time_inference (Burgers)/Burgers.py` 和 `appendix/continuous_time_inference (Burgers)/Burgers_systematic.py`。
- SDK 依据：OpenAI Agents SDK 官方文档中的 code-first orchestration、typed output、guardrails、tracing、agents-as-tools 设计。

## 选择理由

这个来源与当前项目匹配度最高：

- 当前 catalog 已有 `examples/burgers_pinn`，论文和公开代码都包含 Burgers continuous-time PINN 示例。
- 论文方法能直接转成 Retriever 可注入的短知识条目：物理残差项、初边值数据项、collocation sampling、输入归一化、预算化结构搜索。
- 公开代码是 TensorFlow 1 风格且已声明维护风险，不适合作为新依赖；但其实验结构适合作为本项目的 source-grounded mutation hint。
- OpenAI Agents SDK 约束用于控制知识库消费方式：manager/orchestrator 保持状态所有权，agent 只返回结构化计划、预期影响和风险，不接管 evaluator、runner 或 champion selection。

## 已落地的本地 KB

新增条目位于 `examples/burgers_pinn/kb/`：

- `pinn_residual_objective.md`：把 PINN 目标拆成 data term 与 residual term，并要求候选实现保持 prediction-only evaluation 边界。
- `collocation_and_scaling.md`：记录 collocation point、边界/初值覆盖、输入归一化和 steep-gradient 区域采样提示。
- `budgeted_pinn_mutation.md`：把公开代码中的系统性预算扫描转成 AgenticSciML mutation 规则，并绑定 SDK-style typed output、guardrail 和 trace 约束。

这些条目是 retrieval hint，不是 benchmark source of truth。实现行为仍由
`Problem.md`、`Requirements.md`、`Evaluation.md`、`Data_config.json`、
`generate_data.py`、`evaluate.py` 和 `guidelines.md` 决定。

## 知识库写入规则

- 每个条目必须说明适用 benchmark、来源、可尝试的 mutation idea、边界和风险。
- 不粘贴论文或公开仓库的大段原文/源码；只保留短摘要、路径和可验证 URL。
- 不引入新 runtime dependency，除非后续任务明确把 benchmark 升级到更高 fidelity level。
- 不把 KB 条目写成隐藏推理提示；要求 agent 输出 concise rationale summary、implementation plan、expected effect 和 risks。
- KB 不得覆盖 evaluator、private validation data、contract hash、sandbox、trace summary 或 run metadata 的 fail-closed 行为。

## 后续候选

如果要继续扩展，可优先为这些现有 benchmark 建 KB：

- `antiderivative_operator` / `reaction_diffusion_operator`：选择 DeepONet 或 Fourier Neural Operator 公开实现，提炼 operator learning 的 basis、branch/trunk、spectral 和 rollout 稳定性提示。
- `poisson_lshape_faithful_small`：选择 corner-singularity / adaptive residual sampling 相关 PINN 文献，补充 L-shaped domain 的奇异性处理。
- `cylinder_wake_reconstruction`：选择 sparse-sensor reconstruction 或 autoencoder/POD 文献，补充传感器布局、latent decoder 和正则化提示。

后续新增 KB 时，应同步更新 `docs/index.md`、`docs/benchmark_plan.md`，并给
`tests/test_retrieval.py` 增加最小加载测试。

