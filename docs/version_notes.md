# 版本说明

[返回文档树](index.md) · 相关文档：[项目概览](../README.md)、[Ablation 说明](ablation.md)

## v0.1.0 MVP

当前版本目标是复刻 AgenticSciML 的多 Agent workflow，而不是复现论文分数。

已实现能力：

- Python 3.11 package scaffold。
- deterministic mock LLM adapter。
- optional OpenAI adapter。
- `AgentSpec` / `PromptTemplate` / artifact guard 基础合同层。
- runtime `AgentSpec` enforcement：agent 方法入口校验 `input_schema`，JSON 输出默认校验 `output_schema`。
- OpenAI Agents SDK 对齐的结构化输出校验、JSON retry、trace span 和 guardrail 事件。
- evaluator contract guardrail：检测 generated solution 是否篡改 `evaluate.py` 等受保护评估文件。
- static sandbox guardrail：运行前阻断网络模块、子进程、危险文件操作和明显绝对路径写入。
- checkpoint/resume：每轮关键阶段写 `checkpoint.json`，CLI 支持 `--resume` 继续已有 run。
- trace summary：每次 orchestrator 完成后写入 `trace_summary.json`，并提供 `agenticsciml trace-summary <run_dir>` 重新生成和检查 trace quality gate。
- run metadata：`run_metadata.json` 记录 wall time、champion、solution count，以及按 role 汇总的 LLM 调用次数和 prompt/response token 估算占位。
- benchmark catalog：6 类论文任务家族都有本地 deterministic engineering proxy，包括 `function_approx`、`poisson_lshape`、`burgers_pinn`、`antiderivative_operator`、`reaction_diffusion_operator`、`cylinder_wake_reconstruction`。
- evaluation contract：`solution.py` 定义 `MODEL`，支持 validate/train，`evaluate.py` 输出 `eval.json`。
- per-solution workspace 和 artifact persistence。
- solution tree、leaderboard、Mermaid tree 和 champion export。
- DataAnalyst、Evaluator、RootEngineer、Retriever、Proposer、Critic、Engineer、Debugger、ResultAnalyst、Selector 等角色 wrapper。
- 0-1 KB retrieval。
- proposer/critic 4-round proposal flow。
- 独立 `CriticAgent` artifact：`critic.md` 和 per-round transcript。
- root-only / no-KB / KB / random-KB ablation script。

已验证命令：

```bash
uv run --python 3.11 --extra dev pytest -q
uv run --python 3.11 --extra dev pytest tests/test_benchmark_catalog.py -q
uv run --python 3.11 --extra dev agenticsciml benchmarks
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 1
uv run --python 3.11 --extra dev agenticsciml trace-summary runs/<experiment_id>
uv run --python 3.11 --extra dev python scripts/run_ablation.py --output-dir runs/ablation-smoke
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --dry-run
```

边界：

- 不保证论文报告的 improvement factor。
- 不假设官方完整代码、完整 prompt 或模型配置可用。
- mock mode 只验证 workflow shape，不代表真实 SciML 表现。
- 新增 benchmark 是小规模工程代理，不是论文原始全预算实验。
- 当前 sandbox 是本地 workspace 隔离，不是强安全容器。

后续版本建议：

- `v0.2.0`：支持 run resume、阶段 checkpoint 恢复和更完整的 run metadata。
- `v0.3.0`：增强 sandbox，限制网络、文件系统和资源消耗。
- `v0.4.0`：把 Poisson PINN、Burgers PINN 和 operator benchmarks 升级到更接近论文设置的数据规模与训练预算。
- `v0.5.0`：真实 LLM ablation，比较 root-only、no-KB、KB、no-critic、no-debugger。
- `v0.6.0`：加入成本统计、token 估算、trace viewer、trace grading 报表和失败案例报告。
