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
        return {
            "id": self.algorithm_id,
            "name": self.name,
            "family": self.family,
            "compatible_benchmark_families": list(self.compatible_benchmark_families),
            "benchmark_examples": list(self.benchmark_examples),
            "status": self.status,
            "description": self.description,
            "claim_boundary": self.claim_boundary,
            "safety_notes": self.safety_notes,
            "source_scope": self.source_scope,
            "implementation_path": self.implementation_path,
        }


ALGORITHM_CLAIM_BOUNDARY = (
    "Algorithm catalog entries are planning and prompt-seeding aids. Scores, "
    "champions, and scientific claims still come only from benchmark evaluators "
    "and run artifacts."
)


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
