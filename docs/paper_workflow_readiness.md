# Paper Workflow Readiness

[返回文档树](index.md) · 相关文档：[Scientific Discovery Evidence Digest](scientific_discovery_evidence.md)、[LLM Problem Context Pack](llm_problem_context_pack.md)、[Real LLM 运行](real_run.md)、[Ablation 说明](ablation.md)

`plan-paper-workflow` 用于把真实科学证据链的剩余缺口变成一个可审计执行包。它不调用模型、
不读取密钥值、不升级任何 claim；只检查当前环境和本地证据是否已经满足进入
`paper_workflow` 运行的最低条件。

## 命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-paper-workflow \
  examples/cylinder_wake_reconstruction_faithful_small \
  --output-dir runs/paper-workflow-readiness \
  --selector-panel-json '[{"model":"gpt-5-mini"},{"model":"deepseek-v4-pro","base_url":"https://api.deepseek.com"}]' \
  --resource-constraints-json '{"cpu":"local","gpu":false,"timeout_s":120,"dependency_limits":["numpy"],"data_limits":"faithful-small"}' \
  --expert-blueprint-id fluid_pde
```

规划 60 轮可审计工程迭代：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-iteration-campaign \
  examples/cylinder_wake_reconstruction_faithful_small \
  --rounds 60 \
  --batch-size 10 \
  --output-dir runs/iteration-campaign
```

记录某一轮已经由具体 artifact 验证：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli record-iteration-round \
  runs/iteration-campaign/iteration_campaign.json \
  --round 6 \
  --evidence-path runs/iteration-campaign/round_006_evidence.json \
  --validation-command "PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_iteration_campaign.py -q" \
  --validation-exit-code 0 \
  --validation-output-path runs/iteration-campaign/round_006_validation.log
```

重新校验 campaign 已完成轮次的 record 和 evidence digest：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli verify-iteration-campaign \
  runs/iteration-campaign/iteration_campaign.json \
  --fail-on-issues
```

从 completed run 的 runtime selector votes 生成 paper workflow 可引用的 selector evidence packet：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli generate-selector-evidence \
  runs/<run_id>
```

生成 paper workflow readiness 时引用已有 selector evidence packet：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli plan-paper-workflow \
  examples/cylinder_wake_reconstruction_faithful_small \
  --selector-evidence-json runs/<run_id>/reports/selector_evidence_packet.json \
  --output-dir runs/paper-workflow-readiness
```

生成或随 plan 输出 reference capability matrix：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli build-reference-capability-matrix \
  --problem-intake-json planning/problem_intake.json \
  --expert-blueprint-id fluid_pde \
  --output-dir runs/reference-capability-matrix
```

生成或随 plan 输出未来 real LLM agent 可消费的 problem context pack：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli build-llm-problem-context \
  --problem-intake-json planning/problem_intake.json \
  --resource-constraints-json '{"cpu":"local","gpu":false,"timeout_s":120,"dependency_limits":["numpy"],"data_limits":"faithful-small"}' \
  --expert-blueprint-id fluid_pde \
  --output-dir runs/llm-problem-context
```

最终验收 60 轮是否全部完成时，增加完整性要求：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli verify-iteration-campaign \
  runs/iteration-campaign/iteration_campaign.json \
  --require-complete \
  --fail-on-issues
```

可选输入：

- `--domain-approval-json <path>`：领域审核 packet。
- `--selector-evidence-json <path>`：已生成的 runtime selector evidence packet。
- `--problem-intake-json <path>`：包含 hypothesis、observable、metric、failure modes、
  physical constraints 和 domain review checklist 的真实问题 intake。
- `--ablation-output-dir <path>`：已有 ablation 输出目录。
- `--expected-seeds 0 1 2`：期望 seed 覆盖。
- `--expected-variants root_only,kb,random_kb`：期望 variant 覆盖。
- `--fail-on-blockers`：生成 artifact 后，如果仍 blocked 则返回非零。

## 输出

- `paper_workflow_readiness.json`
- `paper_workflow_readiness.md`
- `domain_approval_template.json`
- `paper_benchmark_manifest_template.json`
- `paper_workflow_commands.md`
- `iteration_campaign.json`
- `iteration_campaign.md`
- `iteration_round_XXX_record.json`
- `iteration_campaign_verification.json`
- `selector_evidence_packet.json`
- `selector_evidence_packet.md`
- `reference_capability_matrix.json`
- `reference_capability_matrix.md`
- `llm_problem_context_pack.json`
- `llm_problem_context_pack.md`

## Gate

当前 gate 同时检查：

- `OPENAI_API_KEY` 是否存在，但不记录值。
- LLM 预算环境变量是否至少配置一项。
- provider capability 是否支持真实 image input。
- selector panel 是否至少有两个真实异构 provider/model candidate。
- benchmark 是否是 `paper-like` 且存在完整 `paper_benchmark_manifest.json`。
- KB 是否标记 `paper_kb_equivalent=true`。
- domain approval packet 是否通过固定 checklist。
- ablation 输出是否经 `ablation_runs.csv` / `ablation_summary.csv` 验证。
- `resource_constraints` 和 `expert_blueprint_id` 是否完整。

## 边界

该命令是预执行门禁，不是运行证据。即使所有预执行 gate 通过，也必须完成真实 run，
并由 `reports/scientific_discovery_readiness.json`、`trace_summary.json`、selector votes、
visual audit、domain approval、ablation evidence 和 paper-like benchmark dossier 共同
通过，才允许讨论 paper workflow 级别 claim。默认仍保持
`scientific_claim_supported=false`。

`plan-iteration-campaign` 只规划轮次和 blocker，不把计划本身计入证据。每一轮仍必须
落到代码、文档、测试、run artifact 或外部审批材料，才能改变 readiness。
`generate-selector-evidence` 只读取 completed run 中的 `reports/selector_votes.json` 和
可选的 `reports/selector_heterogeneity.json`，再生成
`reports/selector_evidence_packet.json/md`。它要求至少两个非 mock、不同 runtime member
的 selector votes，并且 provider 或 actual model 异构；重复单一路径投票、mock vote 或
缺失 selector artifact 都会保持 blocked。该 packet 只证明 selector 证据，不替代 evaluator
分数，也不会把 `scientific_claim_supported` 改成 true。
`plan-paper-workflow`、`plan-real-problem-closure` 和 `plan-iteration-campaign` 可通过
`--selector-evidence-json` 引用该 packet；ready packet 只解除 `heterogeneous_real_selector`
这一项，不会绕过 real LLM、多模态、paper-like benchmark、domain review 或 ablation gate。
这些命令也可通过 `--problem-intake-json` 生成并嵌入 `reference_capability_matrix`。
matrix 只报告参考文献机制覆盖度和 intake 结构完整性；即使 rubric ready，
`scientific_claim_supported` 也保持 false。
这些命令还会生成并嵌入 `llm_problem_context_pack`。该 pack 把完整 intake、专家蓝图、
资源边界和 reference matrix 转成 `data_analyst`、`root_engineer`、`proposer`、`critic`、
`engineer`、`debugger`、`selector`、`result_analyst` 和 `visual_audit` 的角色任务包。
它用于提升未来真实 LLM 执行质量，不代表离线测试已经替代 LLM 或解决真实问题。
`record-iteration-round` 只允许记录未被 readiness blocker 卡住的 planned 轮次；它会
写入证据文件 digest、验证命令、验证退出码、验证输出 digest 和轮次 record，并更新 campaign 的
`completed_rounds` / `remaining_rounds`。该记录仍是工程进度证据，不是科学发现证据。
`verify-iteration-campaign` 会重新读取每个 completed round 的 record 和 evidence 文件，
校验 SHA-256 digest、validation output digest、validation exit code 与 campaign 计数，
检测记录缺失、record 元数据篡改、证据篡改、验证输出篡改、record/evidence/validation
output 路径逃逸、record 文件名不规范、round 编号重复/缺失、未知状态、target/batch 篡改、
batch 摘要篡改和进度计数不一致。它还会校验 campaign 顶层 metadata：`campaign_version`、
`claim_boundary`、`paper_workflow_readiness_status` 和 `status` 必须与当前 blocker 状态一致，
防止把仍 blocked 的计划手工改成 ready 或科学发现证据。每个 completed round 的 record
也必须保留固定 claim boundary，不能把单轮工程证据改写成科学发现声明。若某轮
`requires_external_asset=true` 且仍携带 `blocked_by`，即使手工补齐 record 和 digest，
也不能被验收为 completed。Campaign artifact 会保存 `paper_workflow_readiness_checks`；
verifier 以 failed checks 重建顶层 `readiness_blockers`、每轮预期 `blocked_by` 和
外部资产轮次的 blocked 状态，防止先清空 blocker 列表或抹掉 round blocker 再记录完成。
它只验证工程证据完整性，不会把 blocked 的外部资产或科学 claim 判为已完成。默认模式允许校验部分完成的 campaign；
`--require-complete` 会额外要求所有轮次都已完成，用于最终 60 轮验收，不能用来绕过任何
readiness blocker。
