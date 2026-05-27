# Scientific Discovery Evidence Digest

[返回文档树](index.md) · 相关文档：[Benchmark Fidelity Levels](fidelity_levels.md)、[Method Substrate 合约](method_substrate.md)、[ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md)、[ChatUI 实验操作台](chatui_console.md)

## NotebookLM 来源

NotebookLM notebook：`Agentic AI for Scientific Computing and Finite Element Methods`
（`0a34715f-2901-4d1a-8037-47acc3a6e86f`），当前可见更新时间：
`2026-05-25T06:32:52Z`。本轮只把这些来源映射成工程证据门槛；没有把
NotebookLM summary 当作 evaluator 事实。

本轮参考的来源标题包括：

- `AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning - arXiv`
- `[2511.07262] AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning - arXiv`
- `ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms - arXiv`
- `[2512.03476v2] ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms`
- `[2605.11117v1] GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms`
- `ALL-FEM: Agentic Large Language models Fine-tuned for Finite Element Methods - arXiv`
- `Towards an AI Fluid Scientist: LLM-Powered Scientific Discovery in Experimental Fluid Mechanics - arXiv`
- `Towards a science of scaling agent systems: When and why agent systems work`
- `Towards Agentic Intelligence for Materials Science - arXiv`

## 工程映射

本轮选择流体/PDE 优先，不启动 real LLM，也不把 `mock` 或 `faithful-small`
结果包装成新发现。工程目标是补齐进入真实科学证据讨论前的机器可审计条件：

- `reports/scientific_discovery_readiness.json/md`：fail-closed 检查
  paper-like benchmark、real LLM、异构 selector、paper-equivalent KB、真实图像输入、
  领域审核、多 seed/ablation、失败归因、专家蓝图和资源约束。
- `solutions/<id>/visual_audit_report.json` 与 `reports/visual_audit_manifest.json`：
  记录 field / residual proxy / boundary proxy SVG artifact，并固定
  `privacy_boundary=prediction_only_no_validation_labels`。`visual_audit_mode=real`
  只有在真实 provider capability 支持图像输入且图像请求成功时，才允许记录
  `actual_image_inputs_used=true`。
- `solutions/<id>/method_experience_record.json` 与
  `reports/method_experience_cache.json`：把 operator assignment、policy fidelity、
  mutation status、score/failure kind 和 method fingerprint 写入本地经验层。
- `selector_votes.json`：记录每票的 provider、model、adapter 和 capabilities；
  `paper_workflow` 仍要求非 mock 的异构 selector 证据。
- `reports/domain_approval.json`：记录 reviewer、review notes、审批状态和 claim boundary；
  只证明人工审批记录存在，不覆盖其他 readiness gate。
- `reports/paper_like_benchmark_dossier.json`：把 benchmark fidelity、fidelity matrix、
  contract hash 和 paper benchmark approval 放入独立 dossier；不能把 `faithful-small`
  提升成 `paper-like`。
- `reports/selector_heterogeneity.json`：汇总 selector runtime votes、真实非 mock 成员、
  provider/model diversity 和 blocker；配置了 panel 但没有 runtime votes 时仍 blocked。
- `reports/multi_seed_ablation_evidence.json`：记录外部或 planner 附带的多 seed/ablation
  manifest。必须 `verified=true`、记录 `verified_by`/reviewer、至少两个 seed、至少一个
  ablation variant；orchestrator 会把声明 manifest 固化为 run artifact 后，才通过
  scientific readiness 的对应检查。
- Problem Intake / Web / CLI：新增 `visual_audit_mode`、`resource_constraints` 和
  `expert_blueprint_id` / `multi_seed_ablation`，但 custom benchmark 仍只能生成
  `workflow_proxy` scaffold。

## Claim Gate

`scientific_claim_supported=true` 只有在 completed run 同时满足以下条件时才允许：

- real LLM mode。
- `paper-like` benchmark 且人工明确批准 paper benchmark。
- 领域 evaluator 审批、reviewer 和 review notes 完整。
- 至少两个非 mock、真实异构 provider/model 的 selector 证据。
- paper-equivalent KB provenance。
- real vision provider 实际接收 image input，并在 artifact 中记录
  `actual_image_inputs_used=true`。
- 多 seed 或 ablation evidence 已写入 planner/run 证据，并带有已验证 manifest。
- method experience 记录覆盖成功、失败、plateau 或 policy-fidelity mismatch 等归因。

否则 readiness report 必须 `status=blocked`，并且 `trace_summary.json` 不允许
`run_metadata.json` 夸大 scientific readiness。

## 不做项

- 不 fine-tune。
- 不声明自主科学发现。
- 不声明真实实验闭环。
- 不把 `faithful-small` 当作 `paper-like`。
- 不把 mock 分数、LLM 文本、ChatUI 文案、视觉 SVG 或 method cache 当作科学结论。
- 不引入自由 group chat；中心化 Python orchestrator 继续负责 evaluator、selector、
  champion 和 artifact schema 的验证瓶颈。
