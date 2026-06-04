# Ablation Notes

[返回文档树](index.md) · 相关文档：[多 Agent 设计方法](multi_agent_design.md)、[版本说明](version_notes.md)

当前 ablation 是 workflow 与报告链路检查，不是论文分数声明。

## Variants

默认 variants：

1. `root_only`：只生成 root baseline，`max_iterations=0`。
2. `no_kb`：运行多 Agent mutation，但不注入 KB。
3. `kb`：运行 lexical KB retrieval。
4. `random_kb`：运行 seed-controlled random KB retrieval。
5. `no_critic`：跳过 `CriticAgent` 调用，保留 proposer final proposal。
6. `no_debugger`：跳过 debugger loop。
7. `branch_context`：`parallel_mutations=2`，开启 same-parent fanout branch context。
8. `no_branch_context`：`parallel_mutations=2`，关闭 branch context，用于后续真实 LLM smoke 对比。

`no_critic`、`no_debugger` 和 `no_branch_context` 必须映射到真实 workflow
开关，不能只在报表里改标签。

## Commands

CLI：

```bash
uv run --python 3.11 --extra dev agenticsciml ablate examples/function_approx \
  --seeds 0 1 2 \
  --variants root_only,no_kb,kb,random_kb,no_critic,no_debugger,branch_context,no_branch_context \
  --output-dir runs/ablation
```

脚本：

```bash
uv run --python 3.11 --extra dev python scripts/run_ablation.py \
  --benchmark-dir examples/function_approx \
  --seeds 0 1 2 \
  --variants root_only,no_kb,kb,random_kb,no_critic,no_debugger,branch_context,no_branch_context \
  --output-dir runs/ablation
```

真实 LLM ablation 必须显式使用 `--real`。在没有凭证或预算确认前，先用
`--dry-run` 生成计划和预算 manifest；该路径不调用 provider，也不会写
`ablation_runs.csv` / `ablation_summary.csv`，因此不能被
`verify-ablation-evidence` 当作真实 ablation evidence：

```bash
uv run --python 3.11 --extra dev agenticsciml ablate examples/function_approx \
  --real \
  --dry-run \
  --seeds 0 1 2 \
  --variants root_only,kb,branch_context,no_branch_context \
  --output-dir runs/real-llm-ablation
```

真正调用 provider 时必须去掉 `--dry-run`，并使用 real LLM extra、凭证和预算
环境变量。该命令会在 provider call 前写出 `real_llm_ablation_plan.json` 与
`real_llm_ablation_manifest.json`，每个 run 还会写 `llm_call_ledger.jsonl`：

```bash
export OPENAI_API_KEY=<redacted>
export OPENAI_MODEL=gpt-5-mini
export AGENTICSCIML_MAX_LLM_CALLS=400
uv run --python 3.11 --extra real-llm agenticsciml ablate examples/function_approx \
  --real \
  --seeds 0 1 2 \
  --variants root_only,kb,branch_context,no_branch_context \
  --output-dir runs/real-llm-ablation
```

验证已有 ablation 输出是否能作为 readiness evidence：

```bash
uv run --python 3.11 --extra dev agenticsciml verify-ablation-evidence runs/ablation \
  --verified-by ablation-reviewer \
  --expected-seeds 0 1 2 \
  --expected-variants root_only,kb,random_kb
```

该命令读取 `ablation_runs.csv`、`ablation_summary.csv` 和可选
`ablation_report.md`，写出 `multi_seed_ablation_verified_manifest.json`。manifest
只验证本地输出形状、seed 覆盖和 non-baseline ablation variant 覆盖；它不是科学发现声明。

## Outputs

- `ablation_runs.csv`：每个 variant/seed 的 run-level 指标。
- `ablation_summary.csv`：按 variant 聚合的 median、mean、std、best、worst、IQR、valid runs。
- `ablation_report.md`：人类可读报告。
- `real_llm_ablation_plan.json`：real/dry-run 模式的计划文件，列出 benchmark、seed、
  variant、evolution config 和预期 artifact。
- `real_llm_ablation_manifest.json`：real/dry-run 模式的 provider/model、package、
  budget 和 plan hash manifest。
- `multi_seed_ablation_verified_manifest.json`：可选的验证 manifest，供 run config 的
  `multi_seed_ablation.ablation_output_dir` 或 `--multi-seed-ablation-json` 指向原始
  ablation 输出后由 orchestrator 重新生成并附加到 run artifact。
- `runs/<variant>-seed-<seed>/`：每个实验的完整 run artifacts。

run-level 指标至少包含：

- evidence mode
- scientific claim boundary
- champion score
- root score
- champion/root improvement
- valid solution rate
- timeout count
- debug success count
- branch context count
- branch intents
- LLM call count
- wall time

## Boundary

Mock-mode ablation 只验证：

- pipeline 能否稳定运行；
- variants 是否真的改变 workflow；
- reporting 是否能聚合多个 seed；
- artifact 是否足够支持后续真实 LLM 复盘。

当前 CSV 和报告会显式标注 `evidence_mode=mock_workflow_shape` 与
`scientific_claim=not_supported`。不要把 mock-mode improvement 当作 SciML
结论，更不能把它解释成 emergent discovery。真实 LLM ablation 至少需要多
seed、固定预算、成本统计和失败样本审查。

真实 LLM ablation CSV 会标注 `evidence_mode=real_llm_ablation` 和
`scientific_claim=not_supported`；每个底层 run 自己的 evidence boundary 保存在
`run_evidence_mode` / `run_scientific_claim`。即使 provider call 成功，ablation 输出也只
是 workflow contrast evidence，不能绕过 paper workflow readiness、domain review 或
paper-like benchmark gate。

## Readiness Evidence

`multi_seed_ablation` 可以继续记录外部 reviewer 声明，但更强的路径是提供
`ablation_output_dir`、`verified_by`、`expected_seeds` 和 `expected_variants`。orchestrator
会读取本地 ablation 输出，生成 `reports/multi_seed_ablation_verified_manifest.json`，
并把它纳入 `reports/multi_seed_ablation_evidence.json`。只有至少两个 seed、至少一个
非 `root_only` ablation variant、每个 ablation variant 都覆盖至少两个 seed，且 verifier
存在时，该证据才会通过 readiness 的 multi-seed/ablation gate。
