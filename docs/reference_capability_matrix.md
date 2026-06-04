# Reference Capability Matrix

[返回文档树](index.md) · 相关文档：[Scientific Discovery Evidence Digest](scientific_discovery_evidence.md)、[LLM Problem Context Pack](llm_problem_context_pack.md)、[Paper Workflow Readiness](paper_workflow_readiness.md)、[Real Problem Evidence Closure](real_problem_evidence_closure.md)

`build-reference-capability-matrix` 是离线证据规划工具。它把 NotebookLM digest 和本地文档中
可工程化的参考机制转成固定 matrix，并检查 problem intake 是否包含真实问题讨论所需的
最小结构字段。它不调用模型、不读取密钥、不运行实验，也不会改变任何 claim gate。

## 命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli build-reference-capability-matrix \
  --problem-intake-json planning/problem_intake.json \
  --expert-blueprint-id fluid_pde \
  --output-dir runs/reference-capability-matrix
```

输出：

- `reference_capability_matrix.json`
- `reference_capability_matrix.md`

需要把 matrix 进一步转成真实 LLM agent 可消费的角色任务包时，使用
[`build-llm-problem-context`](llm_problem_context_pack.md)。该命令仍不调用模型，只整理
role task、输入 schema、输出 schema、禁区和 stop condition。

## Problem Intake Rubric

rubric 要求 problem intake 至少记录：

- `hypothesis`：待验证科学假设或工程假设。
- `observable` / `observables`：可观测量、输入/输出或测量对象。
- `metric`：评价指标。
- `failure_modes`：预期失败类型和负样本审查入口。
- `physical_constraints`：PDE/fluid/FEM 的边界、残差、守恒、平滑或结构约束。
- `domain_review_checklist`：领域审查 checklist。
- `expert_blueprint_id`：例如 `fluid_pde`、`numerical_methods` 或 `operator_learning`。

这些字段完整只表示“适合离线规划”，不表示科学结论成立。

## Reference Mechanisms

当前 matrix 固定覆盖五类可离线落地机制：

- 科学问题拆解：把真实问题拆成 hypothesis / observable / metric / failure mode / review checklist。
- 流体/PDE 视觉与物理审计：生成 prediction-only field、residual proxy、boundary 和 smoothness artifact。
- ATHENA/GRAFT action-reward trace：记录 action path、solution artifact、reward、fingerprint 和 local experience。
- code-orchestrated agent boundary：让 Python orchestrator 保持状态机、contract、sandbox、selector 和 claim gate 所有权。
- domain / negative-result closure：保留领域审批、失败样本、多 seed/ablation 和真实实验闭环缺口。

每条 mechanism 都包含：

- `reference_mechanism`
- `project_module_mapping`
- `required_artifact`
- `fail_closed_blocker`
- `no_key_local_implementation_path`
- `future_real_run_requirement`

## 边界

该 matrix 是 reference-to-engineering map，不是 run 证据。它能帮助发现 readiness 缺口，
但不能替代 real LLM、真实多模态输入、paper-like benchmark、领域审批、多 seed/ablation、
completed trace summary 或真实实验闭环。`scientific_claim_supported` 固定为 `false`。
