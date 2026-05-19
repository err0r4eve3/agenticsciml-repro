# 学长审计 Issue 留档

[返回文档树](index.md)

本文记录 `issue.pdf` 中 4 个审计问题的闭环处理。结论统一按“问题是否成立、证据、根因、修复方案、验证命令、修复后答复”组织。所有新增 artifact 都是 workflow evidence，不改变 evaluator score，也不支持论文级科学结论。

## Issue 1：KB 检索到了但没有真正用上

### 原始问题

问题成立。审计指出系统检索到 `Budgeted PINN Mutation`，但生成代码没有体现 sample count、collocation count、depth/width、residual weight、training schedule 等建议。

### 复现/证据

原先每个 child solution 主要保存 `retrieved_kb.md`、`proposal.md`、`engineering_summary.md` 和 `solution.py`。这些文件能证明“检索到了 KB”，但没有机器可检查字段证明 proposal 或 engineer 真的采纳了 KB。

### 根因

KB 只作为 prompt 上下文和 markdown artifact 保存，缺少从“检索 -> proposal 采纳 -> engineer 声明 -> 代码静态信号”的审计链路。

### 修复方案

- 每个 child solution 新增 `kb_application_report.json`。
- `ProposerAgent` 输出增加可选 `kb_application` summary。
- `EngineerAgent` prompt 增加 KB checklist，并保存 `engineering_response.json`，其中可包含 `implemented_kb_points`。
- 对 PINN / collocation 类 KB 条目做轻量静态检查，覆盖 `sample_count`、`collocation_count`、`depth_width`、`residual_weight`、`training_schedule`。
- Web `/api/runs/{id}/solutions` 返回 `kb_application` 摘要；前端 solution table 显示 `not_retrieved`、`retrieved_only`、`proposed`、`implemented`、`unverified` 等状态。
- 如果 engineer 声称实现 KB 点但 `solution.py` 没有对应静态信号，状态降为 `unverified`，并写入 `unverified_implemented_points` / `missing_static_evidence`。

### 验证命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_retrieval.py::test_kb_application_report_warns_when_budgeted_pinn_entry_is_not_adopted -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_retrieval.py::test_kb_application_report_marks_claimed_but_unverified_adoption_as_warning tests/test_retrieval.py::test_kb_application_report_passes_when_budgeted_pinn_signals_are_actually_present -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_llm_and_agents.py::test_agents_save_transcripts_and_structured_outputs -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_orchestrator_cli.py::test_full_mock_pipeline_generates_tree_and_champion -q
```

### 修复后答复

这个问题成立。原先系统只证明“检索到了 KB”，不能证明“用上了 KB”。现在每个 child solution 会生成 `kb_application_report.json`，记录 KB 中哪些建议被 proposal 采纳、engineer 声明实现，以及代码中是否有可检查信号。如果只检索但没有采纳，系统会明确标为 `retrieved_only`；如果只声明实现但代码没有静态证据，会标为 `unverified` 并在 UI 中提示。

## Issue 2：evaluation 和数据不应该每个 solution 重复拷贝

### 原始问题

问题成立。审计指出每个 solution 目录下重复出现 evaluation/data 文件，目录混乱，也不利于区分固定输入和 solution 自身输出。

### 复现/证据

原先 `prepare_solution_workspace()` 会为每个 solution 复制 benchmark 文档、`train_data.npz`，并把 `evaluate.py` / `val_data.npz` 放入 per-solution private evaluator 目录。

### 根因

旧布局优先满足 private-label 隔离，但没有把固定 run input 抽到 run-level，导致重复文件和 artifact browser 噪音。

### 修复方案

- orchestrator 新 run 使用 `run_inputs/public/` 和 `run_inputs/private_eval/`。
- `run_inputs/public/` 保存公共 benchmark 文档和 `train_data.npz`。
- `run_inputs/private_eval/` 保存 `evaluate.py` 和 `val_data.npz`。
- solution workspace 通过 symlink 暴露公共输入；如果平台不支持 symlink，fallback 为 copy，并在 manifest 中标记。
- `train_and_evaluate()` 显式接收 `private_eval_dir`，private evaluator 不进入 solution workspace。
- Web artifact browser 阻止浏览 `run_inputs/private_eval/` 原始内容。

### 验证命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_execution.py::test_run_level_inputs_deduplicate_public_data_and_keep_private_eval_out_of_solution -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_execution.py::test_private_eval_dir_is_only_passed_explicitly_to_train_and_evaluate tests/test_execution.py::test_public_input_copy_fallback_preserves_deduplication_contract_when_symlink_fails -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_web_api.py::test_selector_votes_and_solutions_are_read_only_evidence -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_web_api.py::test_artifact_browser_rejects_private_eval_inputs -q
```

### 修复后答复

这个问题成立。原先为了 private-label 隔离，每个 solution 都准备了一套 evaluator/data，导致重复。修复后固定输入提升到 run-level `run_inputs/`，solution 目录只保留自身代码和输出。private validation 仍不会暴露给 generated solution，Web artifact browser 也拒绝直接浏览 `run_inputs/private_eval/`。

## Issue 3：后续 solution 分数完全一样，进化效果不明显

### 原始问题

问题成立。审计指出后续 solution 分数完全相同，leaderboard 只能显示停滞，不能解释是代码重复、mutation 无效，还是 evaluator 区分度不足。

### 复现/证据

原先 run artifact 中没有 per-mutation code digest、proposal digest、diff line count、duplicate-of 或 plateau 统计。只能从 score 观察停滞，无法归因。

### 根因

进化健康没有独立报告，leaderboard 混合展示 score 和 status，但没有审计 mutation 是否真的改变代码。

### 修复方案

- 每个 child solution 新增 `mutation_effect_report.json`。
- run 级新增 `reports/evolution_health.json`。
- 记录 code digest、proposal digest、diff line count、score delta、duplicate-of、mutation status。
- duplicate parent code 标记为 `duplicate_parent`。
- 代码改变但分数不动标记为 `changed_but_score_plateau`。
- Web solution table 展示 mutation 状态，前端 evidence 面板展示 unique code、duplicate、plateau、best improvement。

### 验证命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_orchestrator_cli.py::test_duplicate_child_code_is_marked_in_mutation_and_evolution_health -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_orchestrator_cli.py::test_plateau_without_duplicate_code_is_explained_in_mutation_and_evolution_health -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_orchestrator_cli.py::test_full_mock_pipeline_generates_tree_and_champion -q
```

### 修复后答复

这个问题成立。原先 leaderboard 只能看到分数相同，不能判断是代码没变、mutation 无效，还是 evaluator 区分度不足。修复后每个 child solution 都有 `mutation_effect_report.json`，run 级有 `evolution_health.json`，可以明确标出 duplicate、plateau 和 best improvement。

## Issue 4：不同任务 data analysis 几乎一样，像固定模板

### 原始问题

问题成立。审计指出不同任务的 `data_analysis.md` 文本相似，容易让人怀疑 data analyst 是否真的读取任务数据和 benchmark 文档。

### 复现/证据

原先 Data Analyst 虽然读取 training observation 和 EDA summary，但最终主要保存 LLM 自由文本 `data_analysis.md`，缺少结构化、benchmark-specific 字段。

### 根因

自然语言报告没有机器可检查的任务差异字段。即使 prompt 中有 observation manifest，artifact 层也不强制保存 benchmark family、metric、array keys、task-specific observations。

### 修复方案

- 新增 `reports/data_analysis_structured.json`。
- 字段包含 `benchmark_name`、`benchmark_family`、`fidelity_level`、`problem_summary`、`evaluation_metric`、`training_arrays`、`training_array_keys`、`task_specific_observations`、`modeling_implications`、`risks`、`private_label_boundary`。
- `data_analysis.md` 从 structured JSON 渲染，再附加 LLM summary。
- mock mode 也生成 deterministic benchmark-specific analysis，能区分 function approximation、Poisson、Burgers、operator learning、reaction-diffusion、cylinder wake 等任务。
- trace summary 增加 `data_analysis_specificity` warning，缺少 benchmark、array keys 或 task-specific observations 时提示。

### 验证命令

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_llm_and_agents.py::test_data_analyst_structured_output_is_benchmark_specific -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_llm_and_agents.py::test_data_analysis_structured_json_schema_rejects_generic_template_output -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_reports_data_analysis_specificity_warnings -q
```

### 修复后答复

这个问题成立。原先 data analysis 主要是一段自然语言，容易看起来像模板。修复后会额外写出 `data_analysis_structured.json`，明确包含 benchmark、metric、array keys 和 task-specific observations。不同 benchmark 的结构化分析必须不同，否则 trace summary 会提示 specificity 不足。

## 验证汇总

本轮 targeted 验证覆盖：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_llm_and_agents.py::test_data_analyst_writes_training_observation_artifacts tests/test_llm_and_agents.py::test_data_analyst_structured_output_is_benchmark_specific tests/test_llm_and_agents.py::test_data_analysis_structured_json_schema_rejects_generic_template_output tests/test_retrieval.py::test_kb_application_report_warns_when_budgeted_pinn_entry_is_not_adopted tests/test_retrieval.py::test_kb_application_report_marks_claimed_but_unverified_adoption_as_warning tests/test_retrieval.py::test_kb_application_report_passes_when_budgeted_pinn_signals_are_actually_present -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_execution.py::test_run_level_inputs_deduplicate_public_data_and_keep_private_eval_out_of_solution tests/test_execution.py::test_private_eval_dir_is_only_passed_explicitly_to_train_and_evaluate tests/test_execution.py::test_public_input_copy_fallback_preserves_deduplication_contract_when_symlink_fails -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_trace_reporting.py::test_trace_summary_reports_data_analysis_specificity_warnings -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_orchestrator_cli.py::test_full_mock_pipeline_generates_tree_and_champion tests/test_orchestrator_cli.py::test_duplicate_child_code_is_marked_in_mutation_and_evolution_health tests/test_orchestrator_cli.py::test_plateau_without_duplicate_code_is_explained_in_mutation_and_evolution_health -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_web_api.py::test_selector_votes_and_solutions_are_read_only_evidence tests/test_web_api.py::test_artifact_browser_rejects_private_eval_inputs -q
```

整轮回归验证还应包含：

```bash
PYTHONPATH=src uv run --python 3.11 --extra dev pytest tests/test_execution.py tests/test_llm_and_agents.py tests/test_orchestrator_cli.py tests/test_trace_reporting.py tests/test_web_api.py -q
PYTHONPATH=src uv run --python 3.11 --extra dev pytest -q
cd frontend && npm run build
git diff --check
```

## 给学长/老师的逐条答复

1. KB 问题成立。已增加 `kb_application_report.json`，能区分 retrieved-only、proposed、implemented、unverified，避免把“检索到”或“口头声明实现”误说成“代码已采用”。
2. evaluation/data 重复问题成立。已改为 run-level `run_inputs/`，solution workspace 只保留自身代码和输出；private evaluator 继续隔离且 Web 不展示原始 private data。
3. 分数停滞问题成立。已增加 per-solution `mutation_effect_report.json` 和 run-level `evolution_health.json`，能区分 duplicate、changed-but-plateau 和 score movement。
4. data analysis 模板化问题成立。已增加 `data_analysis_structured.json`，并从结构化字段渲染 markdown；trace summary 会提示 specificity 不足。

剩余边界：这些修复增强的是 workflow 可审计性和展示可信度，不等于实现论文级科学发现，也不替代 domain evaluator 的专业审查。
