# Ablation Notes

[返回文档树](index.md) · 相关文档：[多 Agent 设计方法](multi_agent_design.md)、[版本说明](version_notes.md)

The initial ablation script is a workflow check, not a scientific claim.

Variants:

1. root-only single agent
2. multi-agent without KB
3. multi-agent with KB
4. multi-agent with random KB retrieval placeholder

Mock-mode tests only verify that each variant produces expected artifacts. Real
performance comparisons require repeated runs, fixed budgets, and manual review
of generated solution quality.
