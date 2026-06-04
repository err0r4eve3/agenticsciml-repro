请作为 AgenticSciML 助手处理下面的科学机器学习问题。你的目标不是直接给出未经验证的科学结论，而是启动并解释一个可审计的 workflow-proxy / faithful-small 求解流程。

问题目标：
【在这里详细描述我要解决的问题，包括 PDE/函数逼近/operator learning/传感器重建/数据形式/边界条件/期望输出】

请尽量显式给出以下字段，便于离线 reference capability matrix 和 readiness gate 审计：

- hypothesis：要验证的科学或工程假设。
- observable / observables：可观测量、输入输出、传感器、场变量或实验测量。
- metric：主评价指标。
- failure_modes：可能失败方式、负样本和需要复核的失败归因。
- physical_constraints：边界条件、残差、守恒量、平滑性、结构先验或数值稳定约束。
- domain_review_checklist：领域审查 checklist。
- data_source：数据来源、公开性、许可、私有标签边界和数据规模。

请按以下规则执行：

1. 先判断该问题是否能映射到现有 benchmark catalog。
   - 如果能映射，选择最接近的 benchmark，并说明匹配理由。
   - 如果不能映射，生成 custom proxy benchmark scaffold，但必须明确它只是 workflow-proxy evaluator scaffold，不支持科学结论或 paper-score reproduction。

2. 自动选择合适的 algorithm strategy seeds。
   - 选择时说明每个算法适合该问题的原因、风险和适用边界。
   - 不要把算法库条目描述成已验证实现；它们只是 prompt seed 和 operator scheduling guidance。

3. 使用 auto-audited evolution operator scheduling。
   - 为每个 child solution 分配明确的 mutation operator 和 mutation axis。
   - same-parent fanout 时尽量分配不同 mutation axis，避免 sibling solution 重复。
   - 后续必须通过 operator_assignment.json、mutation_effect_report.json 和 evolution_health.json 审计 operator 是否真的被使用。

4. 建议运行预算：
   - target_solution_count: 6
   - max_iterations: 2
   - parallel_mutations: 2
   - selector_vote_count: 3
   - max_children_per_node: 3
   - mode: mock 或 dry_run；除非我明确确认，否则不要自动启动 real LLM run。
   - visual_audit_mode: off 或 mock；只有明确要求真实视觉模型时才使用 real。
   - expert_blueprint_id: piml / operator_learning / inverse_reconstruction / fluid_pde / numerical_methods 中最匹配的一项。
   - resource_constraints: 记录 CPU/GPU、timeout、依赖、数据规模和私有数据限制。

5. 求解流程必须遵守：
   - EvaluationContract、benchmark guidelines、sandbox rules 和 private-label boundary 高于任何 LLM 建议。
   - 不允许修改 evaluator、私有验证数据、champion selection 或 artifact schema。
   - 不允许读取 validation label、绝对路径、HOME、环境密钥、网络或 subprocess。
   - 不输出隐藏 chain-of-thought，只输出简短 rationale summary、计划、风险和 artifact 引用。

6. 输出时请分为：
   - benchmark / custom proxy 选择
   - selected algorithms
   - planned operator schedule
   - run config
   - expected artifacts
   - scientific discovery readiness blockers
   - claim boundary
   - warnings
   - 如果当前是 agent mode，请返回 start_run action；如果是 plan mode，只预览，不执行。
