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
- structured output failure handling：LLM JSON parse/API/schema 失败会在 agent 边界转换为 `StructuredOutputError`，写入 guardrail trace，并在预算内重试。
- evaluator contract guardrail：检测 generated solution 是否篡改 `evaluate.py` 等受保护评估文件。
- benchmark-aware contract：`ProblemBundle` 和 `BenchmarkContractFactory` 为每个 benchmark 生成 hash-stable `evaluation_contract.json`，不再对所有任务静默使用 function approximation 默认值。
- source-bound contract hash：`evaluation_contract.json` hash 覆盖 `evaluate.py`、`Data_config.json` 和 problem bundle digest；resume/load 会从磁盘重读 benchmark source，并拒绝 stale、tampered、缺失 contract、checkpoint/node contract mismatch 或 node contract metadata 缺失。
- benchmark source manifest：contract 记录 `BenchmarkSourceManifest`，覆盖 `Problem.md`、`Requirements.md`、`Evaluation.md`、`Data_config.json`、`evaluate.py`、`generate_data.py`、`guidelines.md`，以及 repo 中已有或 seed 0 生成的 `train_data.npz` / `val_data.npz` digest。
- manifest integrity guardrail：`EvaluationContract.from_dict()` 会校验 manifest 内容与 `benchmark_source_manifest_digest` 一致；生成数据 digest 时复用 sanitized subprocess env，不继承宿主 secrets。
- manifest schema governance：`BenchmarkSourceManifest` 记录 `schema_version`、`digest_algorithm`、`data_source_mode` 和 normalized `generator_command`；partial train/validation data artifact 会 fail closed。
- atomic storage writes：`ExperimentStorage` 对 JSON、transcript、report 和 solution text artifact 使用同目录临时文件 + `os.replace()` 原子写入。
- validation leak guardrail：`val_data.npz` 放在 run-private `private_eval/solution_*/` 目录，generated `solution.py` 的 validate/train cwd 下不存在 evaluator-private 目录或验证集。
- prediction-only evaluation：可信代码只把 `x_val` 写入 `predict_input.npz`，generated `solution.py --mode=predict` 写 `predictions.npz`，`evaluate.py` 使用私有 `u_val` 计算分数且不 import `solution.py`。
- clean subprocess env：generated solution validate/train/predict/evaluate 使用最小安全环境，不继承宿主 API key、代理、SSH agent、真实 `HOME` 等变量。
- agent context hardening：RootEngineer / Engineer prompt 显式包含 `ProblemBundle`、`EvaluationContract` JSON、`guidelines.md` 和可用分析上下文。
- patch-based mutation：Engineer 输出包含 `parent_digest` 和 patch/file map；Python 端校验 parent digest 后才写入 `solution.py`。
- failed mutation containment：Engineer patch/schema 失败会生成 failed child node、`engineering_error.md` 和 guardrail trace，不再中断整次 run。
- contract-aware debugger：Debugger prompt 显式包含当前代码、`parent_digest`、`ProblemBundle`、`EvaluationContract` JSON、`guidelines.md`、失败阶段和错误日志；修复只能通过 digest-checked unified diff patch 修改 `solution.py`。
- search policy metadata：`SolutionNode` 持久化 `method_tags`、`failure_kind`、`score_delta_from_parent`、`num_debug_attempts`、`benchmark_name` 和 `contract_hash`，供 selector、retriever 和 ablation 使用。
- deterministic parent selection：先由 Python `SearchPolicy` 保证 best available node、recent improvement、diverse underexplored node 和 `max_children_per_node` 约束，再允许 LLM selector 做补充。
- benchmark-aware retrieval query：`RetrievalQueryBuilder` 使用 benchmark family/metric/description、parent analysis、failure kind、method tags 和 leaderboard top-k 生成检索 query。
- KB ablation switches：`use_kb=False` 不注入 KB，`random_kb=True` 使用 seed-controlled random KB retrieval。
- ablation runner：`agenticsciml ablate` 和 `scripts/run_ablation.py` 生成 `ablation_runs.csv`、`ablation_summary.csv`、`ablation_report.md`，支持 `root_only`、`no_kb`、`kb`、`random_kb`、`no_critic`、`no_debugger`。
- ablation metrics：每个 variant 汇总 champion score、root score、champion/root improvement、valid solution rate、timeout count、debug success count、LLM call count 和 wall time。
- static sandbox guardrail：运行前阻断网络模块、子进程、危险文件操作和明显绝对路径写入。
- checkpoint/resume：每轮关键阶段写 `checkpoint.json`，CLI 支持 `--resume` 继续已有 run。
- trace summary：每次 orchestrator 完成后写入 `trace_summary.json`，并提供 `agenticsciml trace-summary <run_dir>` 重新生成和检查 trace quality gate。
- run metadata：`run_metadata.json` 记录 wall time、champion、solution count，以及按 role 汇总的 LLM 调用次数和 prompt/response token 估算占位。
- benchmark catalog：6 类论文任务家族都有本地 deterministic engineering proxy，包括 `function_approx`、`poisson_lshape`、`burgers_pinn`、`antiderivative_operator`、`reaction_diffusion_operator`、`cylinder_wake_reconstruction`。
- evaluation contract：`solution.py` 定义 `MODEL`，支持 validate/train/predict，predict 写 `predictions.npz`，`evaluate.py` 输出 `eval.json`。
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
uv run --python 3.11 --extra dev pytest tests/test_patch_mutation.py tests/test_orchestrator_cli.py -q
uv run --python 3.11 --extra dev agenticsciml benchmarks
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --mock --max-iterations 1
uv run --python 3.11 --extra dev agenticsciml trace-summary runs/<experiment_id>
uv run --python 3.11 --extra dev agenticsciml ablate examples/function_approx --seeds 0 --variants root_only,kb --output-dir runs/ablation-smoke
uv run --python 3.11 --extra dev python scripts/run_ablation.py --output-dir runs/ablation-smoke
uv run --python 3.11 --extra dev agenticsciml run examples/function_approx --dry-run
```

边界：

- 不保证论文报告的 improvement factor。
- 不假设官方完整代码、完整 prompt 或模型配置可用。
- mock mode 只验证 workflow shape，不代表真实 SciML 表现。
- 新增 benchmark 是小规模工程代理，不是论文原始全预算实验。
- 当前 sandbox 是本地 workspace 隔离，不是强安全容器。
- prediction-only protocol 降低验证标签泄漏风险，但还不是 OS/container 级强隔离；同用户进程仍不能视为恶意代码安全沙箱。

后续版本建议：

- `v0.2.0`：支持 run resume、阶段 checkpoint 恢复和更完整的 run metadata。
- `v0.3.0`：增强 sandbox，限制网络、文件系统和资源消耗。
- `v0.4.0`：把 Poisson PINN、Burgers PINN 和 operator benchmarks 升级到更接近论文设置的数据规模与训练预算。
- `v0.5.0`：真实 LLM ablation，比较 root-only、no-KB、KB、no-critic、no-debugger。
- `v0.6.0`：加入成本统计、token 估算、trace viewer、trace grading 报表和失败案例报告。
