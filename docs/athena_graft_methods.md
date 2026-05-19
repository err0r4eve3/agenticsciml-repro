# ATHENA / GRAFT-ATHENA 方法映射

[返回文档树](index.md) · 相关文档：[Method Substrate 合约](method_substrate.md)、[论文机制笔记](paper_notes.md)、[Benchmark 与实验设计](benchmark_plan.md)

## 来源范围

本文只总结两篇新增参考文献中适合当前 reproduction 项目的方法机制：

- ATHENA `arXiv:2512.03476v2`：HENA loop、expert blueprints、
  `A_n -> S_n -> R_n`。
- GRAFT-ATHENA `arXiv:2605.11117v1`：factored method path、method
  fingerprint、knowledge substrate。

## 方法机制表

| method id | 来源 | 本地解释 | 当前落点 | 边界 |
| --- | --- | --- | --- | --- |
| `hena_asr_mapping` | ATHENA | 显式记录 structural action、solution artifact 和 reward 的链路 | `asr_trace_record()` 生成结构化 payload | 只做 traceability，不证明闭环自主发现 |
| `expert_blueprint_constraint` | ATHENA | 用专家蓝图约束 action family、必需参数和禁用参数 | `ExpertBlueprint` 与 template library validation | 只是规则校验，不代表系统理解物理 |
| `factored_method_path` | GRAFT-ATHENA | 把方法视为 ordered action path | `MethodPath` 和 template-instantiated path | 不声称解决组合空间维度灾难 |
| `method_fingerprint_cache` | GRAFT-ATHENA | 用 canonical path identity 生成 cache key | `MethodPath.fingerprint()` 与 `ExperienceSubstrate` | 只做 exact hash lookup，不做 metric-space embedding |
| `local_experience_record_template` | GRAFT-ATHENA | 保存同一 method fingerprint 的历史 reward | `ExperienceSubstrate` 多记录 cache 与 best-reward helper | 不声称跨领域自改进或自主扩展 action space |

## 当前实现

`src/agenticsciml/method_templates.py` 提供保守模板库：

- `athena_graft_method_library()` 返回内置 templates 和 safe blueprint。
- `builtin_method_template_library()` 是当前内置模板库的通用入口。
- `MethodTemplate.instantiate()` 将 source-grounded template 转为 `MethodAction`。
- `MethodTemplateLibrary.method_path()` 从 template id 生成 `MethodPath`，并执行
  blueprint validation。
- `asr_trace_record()` 返回 `A/S/R` 映射 payload，但不写 run artifact、不修改
  orchestrator state。

这些模板是 planning / traceability aids，不是 evaluated algorithms。
内置模板显式保持 `runtime_enabled=False`、`evaluated_algorithm=False`、
`paper_score_claim=False` 和 `autonomous_discovery_claim=False`。

## 暂不实现

- Contextual bandit 在线策略更新。
- GRAFT 概率树学习或 I-map policy factorization。
- Metric-space embedding / nearest-neighbor similarity retrieval。
- Agent 自主写入新 action template 或修改 registry。
- 任何 paper-score reproduction、super-human performance 或 autonomous discovery
  claim。

## 验证

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_method_templates.py -q
```

测试覆盖 template id 稳定性、claim boundary、deterministic instantiation、
override 对 fingerprint 的影响、blueprint validation、ASR payload 和 best-reward
查询边界。
