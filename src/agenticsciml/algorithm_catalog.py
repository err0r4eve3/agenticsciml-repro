from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    algorithm_id: str
    name: str
    family: str
    compatible_benchmark_families: tuple[str, ...]
    benchmark_examples: tuple[str, ...]
    status: str
    description: str
    claim_boundary: str
    safety_notes: str
    source_scope: str = "Local strategy catalog entry."
    implementation_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        detail = _detail_for_algorithm(self.algorithm_id)
        return {
            "id": self.algorithm_id,
            "name": self.name,
            "family": self.family,
            "compatible_benchmark_families": list(self.compatible_benchmark_families),
            "benchmark_examples": list(self.benchmark_examples),
            "status": self.status,
            "description": self.description,
            "description_zh": detail.description_zh,
            "features": list(detail.features),
            "features_zh": list(detail.features_zh),
            "problem_fit": list(detail.problem_fit),
            "problem_fit_zh": list(detail.problem_fit_zh),
            "claim_boundary": self.claim_boundary,
            "safety_notes": self.safety_notes,
            "safety_notes_zh": detail.safety_notes_zh,
            "source_scope": self.source_scope,
            "implementation_path": self.implementation_path,
        }


@dataclass(frozen=True, slots=True)
class AlgorithmBilingualDetail:
    description_zh: str
    features: tuple[str, ...]
    features_zh: tuple[str, ...]
    problem_fit: tuple[str, ...]
    problem_fit_zh: tuple[str, ...]
    safety_notes_zh: str


ALGORITHM_CLAIM_BOUNDARY = (
    "Algorithm catalog entries are planning and prompt-seeding aids. Scores, "
    "champions, and scientific claims still come only from benchmark evaluators "
    "and run artifacts."
)

ALGORITHM_DETAILS: dict[str, AlgorithmBilingualDetail] = {
    "paper_sigmoid_moe_gate": AlgorithmBilingualDetail(
        description_zh="论文 S1.1 的轻量参考 primitive：用可学习 sigmoid gate 混合两个专家函数。",
        features=("Two-expert gated composition", "Smooth regime interpolation", "Dependency-light NumPy primitive"),
        features_zh=("双专家门控组合", "适合平滑分区/状态插值", "轻量本地 NumPy primitive"),
        problem_fit=("Function approximation with mixed regimes", "Smooth scalar targets where local experts can specialize"),
        problem_fit_zh=("混合机制的函数逼近", "可由局部专家分别拟合的平滑标量目标"),
        safety_notes_zh="仅是 reference primitive；不代表论文分数或完整复现证据。",
    ),
    "paper_poisson_decomposition_sampler": AlgorithmBilingualDetail(
        description_zh="论文 S1.2 的参考 helper：已知 particular 解加学习 residual，并对角点附近采样加权。",
        features=("Particular-plus-residual decomposition", "Corner-biased collocation sampling", "Boundary-aware PDE residual seeding"),
        features_zh=("particular + residual 分解", "角点偏置 collocation sampling", "面向边界约束的 PDE residual seed"),
        problem_fit=("L-shaped Poisson or elliptic PDEs", "Geometry singularities and boundary-condition-heavy tasks"),
        problem_fit_zh=("L 形 Poisson 或椭圆 PDE", "几何奇异点明显、边界条件占主导的问题"),
        safety_notes_zh="采样权重只是本地确定性 helper，不代表 paper-scale L-shaped Poisson 复现。",
    ),
    "paper_burgers_staged_pinn_schedule": AlgorithmBilingualDetail(
        description_zh="论文 S1.3 的 Burgers PINN 分阶段训练和 residual 权重调度参考。",
        features=("Staged training schedule", "Residual-weight ramping", "PINN phase-control primitive"),
        features_zh=("分阶段训练日程", "residual loss 权重渐进", "PINN 训练阶段控制 primitive"),
        problem_fit=("Burgers-style PINNs with shocks", "Tasks that need residual, boundary, and initial-condition balancing"),
        problem_fit_zh=("带 shock 或强非线性的 Burgers PINN", "需要平衡 residual、边界和初值损失的问题"),
        safety_notes_zh="该 schedule 只记录训练意图；科学结论仍需要 orchestrator run artifacts。",
    ),
    "paper_linear_bias_free_deeponet": AlgorithmBilingualDetail(
        description_zh="论文 S1.4 的 branch/trunk operator primitive，branch map 线性且无 bias。",
        features=("Branch/trunk factorization", "Linear bias-free branch map", "Small-budget operator-learning seed"),
        features_zh=("branch/trunk 分解", "线性无 bias 的 branch map", "小预算 operator-learning seed"),
        problem_fit=("Antiderivative and function-to-function operator maps", "Small operator datasets where linear branch structure is plausible"),
        problem_fit_zh=("反导数和函数到函数的 operator map", "线性 branch 结构合理的小样本 operator 数据"),
        safety_notes_zh="线性性质只在本地测试；泛化和论文分数必须来自 evaluator artifact。",
    ),
    "paper_reaction_diffusion_fno_helpers": AlgorithmBilingualDetail(
        description_zh="论文 S1.5 的 reaction-diffusion helper：derivative-enhanced loss、spectral smoothing 和 hard BC/IC。",
        features=("Derivative-enhanced loss hooks", "Spectral smoothing helper", "Hard boundary/initial-condition enforcement"),
        features_zh=("derivative-enhanced loss hook", "spectral smoothing helper", "hard BC/IC 约束辅助"),
        problem_fit=("Reaction-diffusion operator learning", "Gridded spatiotemporal PDEs with boundary and initial-condition constraints"),
        problem_fit_zh=("reaction-diffusion operator learning", "带边界/初始条件约束的网格化时空 PDE"),
        safety_notes_zh="这些 helper 不是完整 neural operator 训练栈，也不代表论文分数。",
    ),
    "paper_cylinder_bandlimited_filter": AlgorithmBilingualDetail(
        description_zh="论文 S1.6 的 decoder filtering 参考：Gaussian low-pass 与 bandlimit-preserving activation。",
        features=("Gaussian low-pass filtering", "Bandlimit-preserving activation", "Decoder artifact suppression"),
        features_zh=("Gaussian low-pass filtering", "保持 bandlimit 的 activation", "抑制 decoder 高频伪影"),
        problem_fit=("Cylinder wake sparse-sensor reconstruction", "Sparse probes to dense field recovery with high-frequency artifact risk"),
        problem_fit_zh=("cylinder wake 稀疏传感器重建", "由 sparse probes 重建 dense field 且需控制高频伪影的问题"),
        safety_notes_zh="只是 filtering primitive，不是完整 paper-scale cylinder wake reconstruction model。",
    ),
    "baseline_mlp_regressor": AlgorithmBilingualDetail(
        description_zh="紧凑前馈网络基线，用于平滑或中等振荡目标的 sanity check。",
        features=("Compact feed-forward baseline", "Low implementation overhead", "Broad first-pass sanity check"),
        features_zh=("紧凑前馈基线", "实现和调参成本低", "适合作为第一轮 sanity check"),
        problem_fit=("Smooth or moderately oscillatory scalar regression", "Simple operator or sensor reconstruction baselines"),
        problem_fit_zh=("平滑或中等振荡的标量回归", "简单 operator / sensor reconstruction baseline"),
        safety_notes_zh="只能作为 generated-solution strategy seed；评分仍以 evaluator 为准。",
    ),
    "fourier_feature_mlp": AlgorithmBilingualDetail(
        description_zh="在回归头前加入正弦特征 lifting，增强高频或周期目标拟合能力。",
        features=("Sinusoidal feature lifting", "High-frequency inductive bias", "Compact regression head"),
        features_zh=("正弦特征 lifting", "高频归纳偏置", "紧凑回归头"),
        problem_fit=("Periodic or high-frequency function approximation", "PDE fields with oscillatory structure"),
        problem_fit_zh=("周期或高频函数逼近", "带振荡结构的 PDE field"),
        safety_notes_zh="频率和超参必须从训练可见数据推导，不能窥探私有验证标签。",
    ),
    "piecewise_local_basis": AlgorithmBilingualDetail(
        description_zh="用分段 polynomial、RBF 或 spline-like 局部基函数拟合非连续或局部变化目标。",
        features=("Local polynomial/RBF/spline-style bases", "Segmented approximation", "Interpretable local fit"),
        features_zh=("局部 polynomial/RBF/spline-like basis", "分段近似", "局部拟合较可解释"),
        problem_fit=("Discontinuous or piecewise-smooth functions", "Targets with local irregularities"),
        problem_fit_zh=("非连续或分段平滑函数", "局部不规则性明显的目标"),
        safety_notes_zh="分段边界只能从训练数据推断。",
    ),
    "pinn_residual_minimizer": AlgorithmBilingualDetail(
        description_zh="面向 PDE 任务的 physics-informed residual、边界和初值 loss 模式。",
        features=("Residual, boundary, and initial-condition losses", "Physics-informed objective", "Contract-visible collocation pattern"),
        features_zh=("residual / 边界 / 初值 loss", "physics-informed objective", "contract-visible collocation pattern"),
        problem_fit=("Poisson and Burgers PDE tasks", "Problems with known governing residuals and boundary constraints"),
        problem_fit_zh=("Poisson、Burgers 等 PDE 任务", "控制方程 residual 和边界约束已知的问题"),
        safety_notes_zh="residual points 和私有标签必须继续由 evaluator/contract 控制。",
    ),
    "finite_difference_residual_probe": AlgorithmBilingualDetail(
        description_zh="用 finite-difference stencil 检查候选解的 residual 和边界违反情况。",
        features=("Stencil residual diagnostics", "Boundary-violation probing", "Patch guidance without replacing evaluator scores"),
        features_zh=("stencil residual 诊断", "边界违反探测", "用于指导 patch 但不替代 evaluator"),
        problem_fit=("Debugging PDE candidate solutions", "Poisson, Burgers, or reaction-diffusion residual checks"),
        problem_fit_zh=("调试 PDE 候选解", "Poisson、Burgers、reaction-diffusion residual 检查"),
        safety_notes_zh="诊断结果可指导修复，但不能替代 trusted evaluator。",
    ),
    "weak_form_pinn": AlgorithmBilingualDetail(
        description_zh="用积分或 test-function residual 处理 pointwise derivative 噪声较大的 PDE。",
        features=("Integral/test-function residual", "Lower sensitivity to derivative noise", "Quadrature-based objective"),
        features_zh=("积分 / test-function residual", "对导数噪声更不敏感", "quadrature-based objective"),
        problem_fit=("PDE weak-form tasks", "Noisy derivatives or irregular domains"),
        problem_fit_zh=("PDE weak form 问题", "导数噪声较大或几何不规则的任务"),
        safety_notes_zh="quadrature rules 和 sampled residual points 必须可复现并对 contract 可见。",
    ),
    "xpinn_domain_decomposition": AlgorithmBilingualDetail(
        description_zh="将不规则区域或不同物理 regime 拆成子模型，并加入 interface consistency penalty。",
        features=("Subdomain experts", "Interface consistency penalties", "Regime or geometry split"),
        features_zh=("子域专家模型", "interface consistency penalty", "按 regime 或几何拆分"),
        problem_fit=("Irregular or L-shaped PDE domains", "Shock, discontinuity, or multi-regime PDE behavior"),
        problem_fit_zh=("不规则或 L 形 PDE 区域", "shock、非连续或多 regime PDE 行为"),
        safety_notes_zh="domain split 必须来自 benchmark-visible geometry，而不是私有 evaluator 输出。",
    ),
    "deeponet_operator": AlgorithmBilingualDetail(
        description_zh="面向 function-to-function regression 的 branch/trunk factorization 模式。",
        features=("Branch/trunk factorization", "Function-to-function regression", "Lightweight operator-learning seed"),
        features_zh=("branch/trunk factorization", "函数到函数回归", "轻量 operator-learning seed"),
        problem_fit=("Antiderivative and operator maps", "Small operator datasets and reaction-diffusion surrogates"),
        problem_fit_zh=("反导数和 operator map", "小规模 operator 数据和 reaction-diffusion surrogate"),
        safety_notes_zh="生成实现必须保持 train/predict 分离，不能读取 evaluator 私有文件。",
    ),
    "fno_lite_operator": AlgorithmBilingualDetail(
        description_zh="小型 Fourier neural operator 风格 spectral mixing，用于网格 operator 任务。",
        features=("Spectral mixing", "Gridded-operator inductive bias", "Compact FNO-inspired block"),
        features_zh=("spectral mixing", "网格 operator 归纳偏置", "紧凑 FNO-inspired block"),
        problem_fit=("Gridded PDE fields", "Reaction-diffusion or operator tasks with spatial smoothness"),
        problem_fit_zh=("网格化 PDE field", "具有空间平滑性的 reaction-diffusion / operator task"),
        safety_notes_zh="tensor 尺寸必须有界；除非 benchmark metadata 允许，否则不能假设 GPU。",
    ),
    "kernel_surrogate_regression": AlgorithmBilingualDetail(
        description_zh="Ridge、RBF、GP-style 或 Nyström surrogate，用作小型确定性数据集强基线。",
        features=("Ridge/RBF/GP/Nystrom-style surrogate", "Strong small-data baseline", "Deterministic hyperparameter surface"),
        features_zh=("Ridge/RBF/GP/Nystrom-style surrogate", "小数据强基线", "确定性超参搜索空间"),
        problem_fit=("Small deterministic datasets", "Smooth regression and sensor/operator baselines"),
        problem_fit_zh=("小型确定性数据集", "平滑回归和 sensor/operator baseline"),
        safety_notes_zh="kernel scale 和 regularization 不能通过私有验证标签选择。",
    ),
    "low_rank_operator_regression": AlgorithmBilingualDetail(
        description_zh="用 SVD/PCA 低秩基和 ridge/least-squares 系数重建 operator 或 field。",
        features=("SVD/PCA basis", "Ridge or least-squares coefficients", "Latent field reconstruction"),
        features_zh=("SVD/PCA basis", "ridge 或 least-squares 系数", "latent field reconstruction"),
        problem_fit=("Low-rank operator or sensor reconstruction", "Cylinder wake-like fields with dominant modes"),
        problem_fit_zh=("低秩 operator 或 sensor reconstruction", "存在 dominant modes 的 cylinder-wake-like field"),
        safety_notes_zh="basis construction 只能使用训练可见数组。",
    ),
    "sparse_sensor_reconstructor": AlgorithmBilingualDetail(
        description_zh="将 sparse probe values 映射到 dense field，可用 local basis、kernel 或 latent coefficients。",
        features=("Sparse probes to dense fields", "Local basis/kernel/latent coefficient options", "Inverse-problem seed"),
        features_zh=("sparse probes 到 dense fields", "local basis/kernel/latent coefficient 可选", "inverse-problem seed"),
        problem_fit=("Sparse sensor reconstruction", "Cylinder wake field recovery and probe-to-field maps"),
        problem_fit_zh=("稀疏传感器重建", "cylinder wake field recovery 和 probe-to-field map"),
        safety_notes_zh="没有更强 artifact 前，不能把 proxy reconstruction 称为 paper-level wake recovery 证据。",
    ),
    "sindy_sparse_discovery": AlgorithmBilingualDetail(
        description_zh="用 sparse library regression 从可见数据发现紧凑 residual 或 dynamics 项。",
        features=("Sparse library regression", "Compact dynamics hypotheses", "Interpretable residual terms"),
        features_zh=("sparse library regression", "紧凑 dynamics hypothesis", "可解释 residual terms"),
        problem_fit=("Dynamics discovery", "Burgers or reaction-diffusion residual terms and sensor-derived dynamics"),
        problem_fit_zh=("动力学发现", "Burgers / reaction-diffusion residual terms 和 sensor-derived dynamics"),
        safety_notes_zh="发现的方程只是 hypothesis，必须经 benchmark contract 和 trace artifacts 验证。",
    ),
    "score_aware_ensemble": AlgorithmBilingualDetail(
        description_zh="组合兼容候选预测，或用 leaderboard 诊断结果生成后续分支 seed。",
        features=("Compatible prediction blending", "Leaderboard-aware diagnostics", "Follow-up branch seeding"),
        features_zh=("兼容预测融合", "leaderboard-aware diagnostics", "后续分支 seed 生成"),
        problem_fit=("Post-run refinement", "Runs with multiple compatible candidates and selector-vote evidence"),
        problem_fit_zh=("run 后期 refinement", "已有多个兼容候选和 selector-vote evidence 的 run"),
        safety_notes_zh="champion selection 仍归 deterministic repository policy，而不是自由聊天意图。",
    ),
}


ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        algorithm_id="paper_sigmoid_moe_gate",
        name="Paper Sigmoid-Gated MoE Reference",
        family="paper_champion_strategy",
        compatible_benchmark_families=("function_approx",),
        benchmark_examples=("function_approx_faithful_small",),
        status="reference_implementation",
        description="Dependency-light reference primitive for the paper's S1.1 learnable sigmoid-gated two-expert composition.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Reference primitive only; it does not provide paper-score reproduction evidence.",
        source_scope="AgenticSciML arXiv v2 Results/S1.1 champion strategy summary.",
        implementation_path="agenticsciml.paper_algorithms:sigmoid_moe_prediction",
    ),
    AlgorithmSpec(
        algorithm_id="paper_poisson_decomposition_sampler",
        name="Paper Poisson Decomposition Reference",
        family="paper_champion_strategy",
        compatible_benchmark_families=("poisson",),
        benchmark_examples=("poisson_lshape_faithful_small",),
        status="reference_implementation",
        description="Reference helpers for known-particular plus learned-residual composition and corner-biased collocation sampling.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Sampling weights are local deterministic helpers, not evidence of paper-scale L-shaped Poisson reproduction.",
        source_scope="AgenticSciML arXiv v2 Results/S1.2 champion strategy summary.",
        implementation_path="agenticsciml.paper_algorithms:particular_plus_residual",
    ),
    AlgorithmSpec(
        algorithm_id="paper_burgers_staged_pinn_schedule",
        name="Paper Burgers Staged PINN Reference",
        family="paper_champion_strategy",
        compatible_benchmark_families=("burgers_pinn",),
        benchmark_examples=("burgers_pinn_faithful_small",),
        status="reference_implementation",
        description="Reference schedule and residual-weight helpers for the paper's staged Burgers PINN mutation pattern.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="The schedule records phase intent; it is not paper-score evidence without orchestrator run artifacts.",
        source_scope="AgenticSciML arXiv v2 Results/S1.3 champion strategy summary.",
        implementation_path="agenticsciml.paper_algorithms:burgers_three_phase_schedule",
    ),
    AlgorithmSpec(
        algorithm_id="paper_linear_bias_free_deeponet",
        name="Paper Linear Bias-Free DeepONet Reference",
        family="paper_champion_strategy",
        compatible_benchmark_families=("operator_learning",),
        benchmark_examples=("antiderivative_operator_faithful_small",),
        status="reference_implementation",
        description="Reference branch/trunk operator primitive whose branch map is linear and bias-free.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Linearity is tested locally; paper-score or generalization claims require benchmark evaluation artifacts.",
        source_scope="AgenticSciML arXiv v2 Results/S1.4 champion strategy summary.",
        implementation_path="agenticsciml.paper_algorithms:linear_bias_free_deeponet_prediction",
    ),
    AlgorithmSpec(
        algorithm_id="paper_reaction_diffusion_fno_helpers",
        name="Paper Reaction-Diffusion FNO Helper Reference",
        family="paper_champion_strategy",
        compatible_benchmark_families=("reaction_diffusion", "operator_learning"),
        benchmark_examples=("reaction_diffusion_operator_faithful_small",),
        status="reference_implementation",
        description="Reference helpers for derivative-enhanced loss, spectral smoothing, and hard BC/IC enforcement on spatiotemporal grids.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="FNO-style helpers are not a complete neural operator training stack or paper-score result.",
        source_scope="AgenticSciML arXiv v2 Results/S1.5 champion strategy summary.",
        implementation_path="agenticsciml.paper_algorithms:derivative_enhanced_loss",
    ),
    AlgorithmSpec(
        algorithm_id="paper_cylinder_bandlimited_filter",
        name="Paper Cylinder Bandlimited Decoder Filter Reference",
        family="paper_champion_strategy",
        compatible_benchmark_families=("sensor_reconstruction",),
        benchmark_examples=("cylinder_wake_reconstruction", "cylinder_wake_reconstruction_faithful_small"),
        status="reference_implementation",
        description="Reference Gaussian low-pass and bandlimit-preserving activation helpers for U-FNO/CNO-inspired decoder filtering.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Filtering primitive only; it is not a complete paper-scale cylinder wake reconstruction model.",
        source_scope="AgenticSciML arXiv v2 Results/S1.6 champion strategy summary.",
        implementation_path="agenticsciml.paper_algorithms:bandlimit_preserving_activation",
    ),
    AlgorithmSpec(
        algorithm_id="baseline_mlp_regressor",
        name="Baseline MLP Regressor",
        family="neural_baseline",
        compatible_benchmark_families=("function_approx", "operator_learning", "sensor_reconstruction"),
        benchmark_examples=("function_approx", "antiderivative_operator"),
        status="strategy_blueprint",
        description="Compact NumPy/PyTorch-style feed-forward baseline for smooth or moderately oscillatory targets.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Use only as a generated-solution strategy seed; evaluator remains the scoring authority.",
    ),
    AlgorithmSpec(
        algorithm_id="fourier_feature_mlp",
        name="Fourier Feature MLP",
        family="spectral_features",
        compatible_benchmark_families=("function_approx", "burgers_pinn", "reaction_diffusion"),
        benchmark_examples=("function_approx_faithful_small", "burgers_pinn_faithful_small"),
        status="strategy_blueprint",
        description="Sinusoidal feature lifting before regression to improve high-frequency or periodic target fits.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Do not tune against private validation labels; feature frequencies must be derived from train-visible data.",
    ),
    AlgorithmSpec(
        algorithm_id="piecewise_local_basis",
        name="Piecewise Local Basis",
        family="classical_regression",
        compatible_benchmark_families=("function_approx",),
        benchmark_examples=("function_approx", "function_approx_faithful_small"),
        status="strategy_blueprint",
        description="Segmented polynomial, radial-basis, or spline-like local approximation for discontinuous functions.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Boundary locations must be inferred from training data only.",
    ),
    AlgorithmSpec(
        algorithm_id="pinn_residual_minimizer",
        name="PINN Residual Minimizer",
        family="physics_informed",
        compatible_benchmark_families=("poisson", "burgers_pinn"),
        benchmark_examples=("poisson_lshape_faithful_small", "burgers_pinn_faithful_small"),
        status="strategy_blueprint",
        description="Physics-informed residual, boundary, and initial-condition loss pattern for PDE-style tasks.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Residual points and private labels must stay evaluator-controlled when the contract requires it.",
    ),
    AlgorithmSpec(
        algorithm_id="finite_difference_residual_probe",
        name="Finite-Difference Residual Probe",
        family="physics_diagnostics",
        compatible_benchmark_families=("poisson", "burgers_pinn", "reaction_diffusion"),
        benchmark_examples=("poisson_lshape_faithful_small", "reaction_diffusion_operator_faithful_small"),
        status="strategy_blueprint",
        description="Use finite-difference stencil checks to diagnose residual and boundary violations in candidate solutions.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Diagnostics may guide patches but must not replace the trusted evaluator.",
    ),
    AlgorithmSpec(
        algorithm_id="weak_form_pinn",
        name="Weak-Form PINN",
        family="physics_informed",
        compatible_benchmark_families=("poisson", "burgers_pinn", "reaction_diffusion"),
        benchmark_examples=("poisson_lshape_faithful_small", "reaction_diffusion_operator_faithful_small"),
        status="strategy_blueprint",
        description="Integral or test-function residual formulation for PDE tasks where pointwise derivatives are noisy.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Quadrature rules and sampled residual points must be contract-visible and reproducible.",
    ),
    AlgorithmSpec(
        algorithm_id="xpinn_domain_decomposition",
        name="XPINN Domain Decomposition",
        family="physics_informed",
        compatible_benchmark_families=("poisson", "burgers_pinn", "reaction_diffusion"),
        benchmark_examples=("poisson_lshape_faithful_small", "burgers_pinn_faithful_small"),
        status="strategy_blueprint",
        description="Split irregular domains or regimes into submodels with interface consistency penalties.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Domain splits must be generated from benchmark-visible geometry, not private evaluator outputs.",
    ),
    AlgorithmSpec(
        algorithm_id="deeponet_operator",
        name="DeepONet-Style Operator",
        family="operator_learning",
        compatible_benchmark_families=("operator_learning", "reaction_diffusion"),
        benchmark_examples=("antiderivative_operator_faithful_small", "reaction_diffusion_operator_faithful_small"),
        status="strategy_blueprint",
        description="Branch/trunk factorization pattern for function-to-function regression under small local budgets.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Generated implementations must respect train/predict mode separation and private evaluator files.",
    ),
    AlgorithmSpec(
        algorithm_id="fno_lite_operator",
        name="FNO-Lite Operator",
        family="operator_learning",
        compatible_benchmark_families=("operator_learning", "reaction_diffusion"),
        benchmark_examples=("reaction_diffusion_operator_faithful_small",),
        status="strategy_blueprint",
        description="Small Fourier-neural-operator-inspired spectral mixing for gridded operator tasks.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Keep tensor sizes bounded; do not assume GPU availability unless benchmark metadata allows it.",
    ),
    AlgorithmSpec(
        algorithm_id="kernel_surrogate_regression",
        name="Kernel Surrogate Regression",
        family="classical_regression",
        compatible_benchmark_families=("function_approx", "operator_learning", "sensor_reconstruction"),
        benchmark_examples=("function_approx", "antiderivative_operator"),
        status="strategy_blueprint",
        description="Ridge, RBF, Gaussian-process-style, or Nyström surrogate baseline for small deterministic datasets.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Kernel scales and regularization must be selected without peeking at private validation labels.",
    ),
    AlgorithmSpec(
        algorithm_id="low_rank_operator_regression",
        name="Low-Rank Operator Regression",
        family="classical_operator",
        compatible_benchmark_families=("operator_learning", "reaction_diffusion", "sensor_reconstruction"),
        benchmark_examples=("antiderivative_operator", "cylinder_wake_reconstruction_faithful_small"),
        status="strategy_blueprint",
        description="SVD/PCA-style low-rank basis reconstruction paired with ridge or least-squares coefficients.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Basis construction must use training-visible arrays only.",
    ),
    AlgorithmSpec(
        algorithm_id="sparse_sensor_reconstructor",
        name="Sparse Sensor Reconstructor",
        family="inverse_problem",
        compatible_benchmark_families=("sensor_reconstruction",),
        benchmark_examples=("cylinder_wake_reconstruction", "cylinder_wake_reconstruction_faithful_small"),
        status="strategy_blueprint",
        description="Map sparse probe values to dense fields using local bases, kernels, or learned latent coefficients.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Do not describe proxy reconstruction as paper-level wake-recovery evidence without stronger artifacts.",
    ),
    AlgorithmSpec(
        algorithm_id="sindy_sparse_discovery",
        name="SINDy Sparse Discovery",
        family="symbolic_dynamics",
        compatible_benchmark_families=("burgers_pinn", "reaction_diffusion", "sensor_reconstruction"),
        benchmark_examples=("burgers_pinn_faithful_small", "cylinder_wake_reconstruction_faithful_small"),
        status="strategy_blueprint",
        description="Sparse library regression pattern for discovering compact residual or dynamics terms from visible data.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Discovered equations are hypotheses until evaluated by the benchmark contract and trace artifacts.",
    ),
    AlgorithmSpec(
        algorithm_id="score_aware_ensemble",
        name="Score-Aware Ensemble",
        family="selection_policy",
        compatible_benchmark_families=(
            "function_approx",
            "poisson",
            "burgers_pinn",
            "operator_learning",
            "reaction_diffusion",
            "sensor_reconstruction",
        ),
        benchmark_examples=("function_approx", "reaction_diffusion_operator"),
        status="strategy_blueprint",
        description="Combine compatible candidate predictions or use leaderboard diagnostics to seed follow-up branches.",
        claim_boundary=ALGORITHM_CLAIM_BOUNDARY,
        safety_notes="Champion selection still belongs to deterministic repository policy, not free-form chat intent.",
    ),
)


def list_algorithms(family: str | None = None) -> list[AlgorithmSpec]:
    if not family:
        return list(ALGORITHMS)
    family_key = family.strip().lower()
    return [
        spec
        for spec in ALGORITHMS
        if spec.family.lower() == family_key
        or family_key in {item.lower() for item in spec.compatible_benchmark_families}
    ]


def _detail_for_algorithm(algorithm_id: str) -> AlgorithmBilingualDetail:
    try:
        return ALGORITHM_DETAILS[algorithm_id]
    except KeyError as exc:
        raise KeyError(f"missing bilingual algorithm detail for {algorithm_id!r}") from exc
