# Method Substrate 合约

[返回文档树](index.md) · 相关文档：[ATHENA / GRAFT-ATHENA 方法映射](athena_graft_methods.md)、[论文机制笔记](paper_notes.md)、[论文算法 Reference Primitives](paper_algorithm_primitives.md)、[Benchmark 与实验设计](benchmark_plan.md)

## 范围

`src/agenticsciml/method_substrate.py` 是底层 deterministic local contract，
`src/agenticsciml/method_templates.py` 是其上的 inert template layer。两者用于把
ATHENA / GRAFT-ATHENA 中适合本项目的 workflow 机制落成可测试对象：

- `MethodAction` 表示结构化动作 `A_n`，例如 architecture、PDE constraint、
  optimizer 或 diagnostics。
- `ExpertBlueprint` 约束某类 action 的允许 family、必需参数和禁用参数。
- `MethodPath` 把有序 actions 视为一个方法路径，并用 canonical JSON 生成稳定
  fingerprint；`source_scope` 等 provenance 字段不参与 fingerprint identity。
- `ScientificReward` 表示由 trusted evaluator artifact 得到的标量奖励 `R_n`。
- `ExperienceSubstrate` 用本地 JSON cache 按 fingerprint 保存和检索经验记录；同一
  fingerprint 可以保留多条 reward 记录，避免后续实验覆盖早期证据。
- `MethodTemplateLibrary` 将 NotebookLM/Pro 复审后的 ATHENA / GRAFT 方法机制保存为
  source-grounded template，并能在不调用 LLM 的情况下实例化为 `MethodPath`。
- `asr_trace_record()` 生成 `A_n -> S_n -> R_n` 映射 payload，但不写 run artifact、
  不修改 orchestrator state。

## Claim Boundary

该合约只证明本地 repo 能记录和验证方法路径、蓝图约束、奖励值和经验 cache。
它不表示已经实现 GRAFT 的概率树学习、跨领域自改进、自动扩展 action space、
paper-score reproduction、或 autonomous scientific discovery。

所有分数、champion、benchmark fidelity 和 scientific claim 仍只能来自 evaluator、
run artifacts、trace summary 和 claim gate。

## 验证

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_method_substrate.py tests/test_method_templates.py -q
```

测试覆盖：

- canonical parameter order 不影响 fingerprint；
- action 顺序会影响 fingerprint；
- blueprint 会拒绝缺失必需参数、出现禁用参数或 family 不匹配；
- reward 拒绝 `NaN` / `Infinity`；
- method parameters 拒绝非 JSON 值和非有限数值；
- experience cache 可通过 method fingerprint 稳定 roundtrip，并保留同一
  fingerprint 的多条记录。
- template library 可稳定实例化 method path，并保持 no-overclaim 边界。
