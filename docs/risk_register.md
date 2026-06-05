# Stage A Risk Register

[返回文档树](index.md) · 相关文档：[Ablation 说明](ablation.md)、[Paper Workflow Readiness](paper_workflow_readiness.md)、[Real Problem Evidence Closure](real_problem_evidence_closure.md)、[Paper Gap Report](paper_gap_report.md)

本文档记录 faithful-small Stage A 证据链的风险状态。它只绑定已存在的 PR、CI、测试和 run artifact；
不把 `faithful-small`、`real_llm` workflow coverage 或 planning artifact 升级成论文分数复现、
科学发现或真实问题已经解决的声明。

## 当前结论

- Stage A 计划覆盖已经完成：`root_only`、`no_kb`、`kb`、`random_kb` 四个 variant，
  seed `0..4`，共 `20/20` 个计划 run 已被 collector 聚合。
- 完整 bundle 的 verifier 通过：`verified=true`、`blockers=[]`、`warnings=[]`。
- 论文差距、真实问题 closure 和 paper workflow readiness 仍保持 `blocked`。
- 当前可支持的声明仅限：faithful-small Stage A workflow evidence 已被收集、检查并保留 claim boundary。

核心 evidence bundle：

```text
runs/faithful-small-stage-a-collected-complete-main-postreport-20260605
```

## 已关闭风险

| 风险 | 状态 | 证据 |
| --- | --- | --- |
| Stage A batch 身份不稳定，可能把不同计划或 benchmark 的 batch 拼在一起 | closed | PR [#9](https://github.com/err0r4eve3/agenticsciml-repro/pull/9) 合入；`batch_collection_manifest.json` 记录 `full_stage_plan_hash`、`benchmark_content_hash` 和 collected/missing run counts |
| secret hygiene report 曾保存 secret-derived value hash，可能形成敏感值验证器 | closed | PR [#10](https://github.com/err0r4eve3/agenticsciml-repro/pull/10) 合入；完整 bundle 的 `secret_hygiene_report.json` 为 `passed=true`、`finding_count=0`，且不含 `value_sha256`、`value_hash`、`secret_value_fingerprint` 或 `secret_derived_hash` |
| 完整 collector 输出缺少 human-readable report，reviewer 只能读 CSV/JSON | closed | PR [#11](https://github.com/err0r4eve3/agenticsciml-repro/pull/11) 合入；post-merge bundle 含 `ablation_report.md`，`verify-ablation-evidence` 的 `warnings=[]` |
| paper gap report 可能把完整 ablation bundle 误当成 completed orchestrator run | closed | PR [#12](https://github.com/err0r4eve3/agenticsciml-repro/pull/12) 合入；`tests/test_paper_gap_report.py::test_paper_gap_report_keeps_complete_ablation_bundle_fail_closed` 保持 `status=blocked` |
| Stage A 完整覆盖缺少单一 claim-boundary artifact | closed | `stage_a_claim_card.json` 绑定 collector、verifier、paper gap、secret hygiene 和 plan hash；其中 `scientific_claim=not_supported`、`paper_score_reproduction=false`、`scientific_discovery_claim=false` |
| 下一阶段 60 轮迭代目标不可审计 | closed | `iteration-campaign/iteration_campaign.json` 生成 60 个 planned/blocked rounds；`verify-iteration-campaign --fail-on-issues` 通过且 `issue_count=0` |
| real-problem closure 无法接收已完成 run audit，因此即使有真实 run 也会永久阻塞 `completed_run_audit` | closed | `plan-real-problem-closure --completed-run-dir <run>` 会验证 `run_metadata.json`、`trace_summary.json` 和 `reports/scientific_discovery_readiness.json`；只有 trace gate 通过且 claim gate 不冲突时才解除该模块 |

## 仍开放风险

| 风险 | 当前状态 | 下一步 |
| --- | --- | --- |
| Stage A 仍不是论文分数复现 | open | 补 `paper-like` benchmark、paper-equivalent data/evaluator provenance、private-label protocol 和 hash-bound manifest |
| 当前 workflow 证据不支持科学发现声明 | open | 完成真实 provider run、`scientific_discovery_readiness`、领域审核和失败样本复核后再升级 claim |
| paper workflow readiness 仍 blocked | open | 补真实多模态 provider、异构 selector evidence、资源蓝图、领域审批、paper-equivalent KB 和 paper-like benchmark |
| real-problem closure 仍 blocked | open | 让 `paper_like_benchmark`、`paper_equivalent_kb` 和 `domain_approval` 全部具备 proof artifact；`completed_run_audit` 只能证明 completed run artifact 一致，不能替代这些外部证据 |
| legacy batch1 缺少新 manifest 字段 | accepted | collector 必须继续显式使用 `--allow-legacy-missing-full-stage-hash`，并在 manifest 中保留 `legacy_batch_manifest_count=1`；不要把该 batch 当作新 schema 原生证据 |
| 运行 artifact 未提交到 Git | accepted | run artifacts 仍作为本地 evidence bundle 保存，不提交到仓库；长期 claim 必须引用 artifact path、hash 和验证命令 |

## 当前可复查命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli verify-ablation-evidence \
  runs/faithful-small-stage-a-collected-complete-main-postreport-20260605 \
  --expected-seeds 0 1 2 3 4 \
  --expected-variants root_only,no_kb,kb,random_kb
```

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-paper-workflow \
  examples/function_approx_faithful_small \
  --ablation-output-dir runs/faithful-small-stage-a-collected-complete-main-postreport-20260605 \
  --expected-seeds 0 1 2 3 4 \
  --expected-variants root_only,no_kb,kb,random_kb \
  --fail-on-blockers
```

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-real-problem-closure \
  examples/function_approx_faithful_small \
  --ablation-output-dir runs/faithful-small-stage-a-collected-complete-main-postreport-20260605 \
  --expected-seeds 0 1 2 3 4 \
  --expected-variants root_only,no_kb,kb,random_kb \
  --fail-on-blockers
```

## 对外表述边界

可以说：

- faithful-small Stage A 的计划 run 已完成收集和 verifier 检查。
- collector、secret hygiene、report 输出和 paper-gap fail-closed regression 已经合入 main。
- paper workflow readiness 和 real-problem closure artifact 明确列出了剩余 blocker。

不能说：

- 已复现 AgenticSciML 论文分数。
- 已证明多 Agent 能解决真实科学问题。
- faithful-small Stage A 的 score trend 等价于科学发现或 paper-like benchmark 结论。
