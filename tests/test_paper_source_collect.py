from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticsciml.paper_source_collect import build_paper_source_collection, parse_arxiv_atom


ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2606.02427</id>
    <updated>2026-06-01T16:04:21Z</updated>
    <published>2026-06-01T16:04:21Z</published>
    <title>Spectral Audit of In-Context Operator Networks</title>
    <summary>Existing evaluations of neural operators rely primarily on prediction error.</summary>
    <author><name>Zhiwei Gao</name></author>
    <author><name>George Em Karniadakis</name></author>
    <category term="math.NA" />
  </entry>
</feed>
"""


def test_parse_arxiv_atom_builds_bilingual_real_problem_candidate() -> None:
    entries = parse_arxiv_atom(ATOM_FIXTURE)
    collection = build_paper_source_collection(
        entries=entries,
        query="au:Karniadakis",
        max_results=1,
        status="collected",
        error=None,
    )

    assert collection["status"] == "collected"
    assert collection["candidate_count"] == 1
    assert collection["issue_count"] == 0
    assert entries[0]["id"] == "2606.02427"
    assert entries[0]["published"] == "2026-06-01"
    assert entries[0]["real_problem_zh"]


def test_parse_arxiv_atom_prefers_solver_problem_for_krylov_operator_paper() -> None:
    feed = ATOM_FIXTURE.replace(
        "Spectral Audit of In-Context Operator Networks",
        "NSPOD: Accelerating Krylov solvers via DeepONet-learned POD subspaces",
    ).replace(
        "Existing evaluations of neural operators rely primarily on prediction error.",
        "A neural operator preconditioner accelerates Krylov linear solvers for PDE systems.",
    )

    entry = parse_arxiv_atom(feed)[0]

    assert entry["real_problem"] == (
        "Evaluate whether learned solver aids accelerate PDE linear systems while preserving convergence and generalization."
    )
    assert entry["real_problem_zh"] == "评估学习型求解器辅助是否能加速 PDE 线性系统，同时保持收敛性和泛化能力。"


def test_parse_arxiv_atom_uses_specific_problem_for_recent_pde_paper_themes() -> None:
    cases = [
        (
            "Curvature-Aware Optimization for High-Accuracy Physics-Informed Neural Networks",
            "Natural Gradient, BFGS, Broyden, and quasi-Newton optimizers accelerate PINN convergence for PDEs.",
            "Audit optimizer and conditioning choices for high-accuracy PINN convergence on challenging PDE or ODE systems.",
            "审计 optimizer 与 conditioning 选择对高精度 PINN 在困难 PDE 或 ODE 系统上收敛的影响。",
        ),
        (
            "PINNs in PDE Constrained Optimal Control Problems: Direct vs Indirect Methods",
            "The study compares direct PINNs with indirect adjoint optimality-system formulations.",
            "Compare PINN control formulations against adjoint or optimality-system baselines for PDE-constrained control.",
            "对比 PINN 控制 formulation 与伴随法或最优性系统基线在 PDE 约束控制中的表现。",
        ),
        (
            "Adjoint Method versus Physics-Informed Neural Networks in PDE-Constrained Inverse Problems",
            "A fair comparison matches optimizer, unknown parameterization, and regularization for adjoint optimization and PINNs in PDE-constrained inverse problems.",
            "Compare adjoint optimization and PINNs for PDE-constrained inverse problems under matched formulations, parameterizations, regularization, cost, and warm-start tradeoffs.",
            "对比 PDE-constrained inverse problem 中的 adjoint optimization 与 PINN，并检查 formulation、parameterization、regularization、成本和 warm-start 取舍。",
        ),
        (
            "State-space models are accurate and efficient neural operators for dynamical systems",
            "Mamba state-space models improve neural operator learning for long-range dependencies, extrapolation, and chaotic dynamics, with a pharmacology application under limited data.",
            "Validate state-space or Mamba neural operators for dynamical systems under long-range dependency, extrapolation, chaotic rollout, and computational-efficiency constraints.",
            "验证 dynamical system 的 state-space 或 Mamba neural operator，并检查 long-range dependency、外推、chaotic rollout 和计算效率约束。",
        ),
        (
            "Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines",
            "Frequency-resolved diagnostics expose high-frequency failure modes in neural operators.",
            "Diagnose high-frequency failure modes in physics-informed or operator learning and test mitigation controls.",
            "诊断 physics-informed 或 operator learning 中的高频失效模式，并测试缓解控制。",
        ),
        (
            "Spectral Audit of In-Context Operator Networks",
            "A Jacobian-based spectral audit views the query-function derivative as a learned tangent operator and detects high-frequency degradation, sensitivity failures, and prompt-operator inconsistencies.",
            "Audit in-context operator networks with Jacobian-based spectral tangent-operator diagnostics for local PDE mechanism fidelity, stability, sensitivity, and prompt consistency beyond prediction error.",
            "用基于 Jacobian 的 spectral tangent-operator 诊断审计 in-context operator network，并检查 local PDE mechanism fidelity、稳定性、敏感性和 prompt consistency，而不只看预测误差。",
        ),
        (
            "UFO: A Domain-Unification-Free Operator Framework for Generalized Operator Learning",
            "A cross-domain neural operator enables discretization decoupling across irregular sampling, spectral mismatch, nonlinear dynamics, and stochastic high-frequency fields.",
            "Validate cross-domain neural operator frameworks for discretization-decoupled generalized operator learning under irregular sampling, spectral mismatch, distribution shift, and arbitrary-resolution query constraints.",
            "验证 discretization-decoupled generalized operator learning 的 cross-domain neural operator framework，并检查 irregular sampling、spectral mismatch、distribution shift 和任意分辨率查询约束。",
        ),
        (
            "Mitigating Spectral Bias in Neural Operators via High-Frequency Scaling for Physical Systems",
            "High-frequency scaling mitigates spectral bias in convolutional neural operators for turbulence and two-phase flow systems, with diffusion models used as conditioned baselines.",
            "Diagnose spectral-bias high-frequency failure modes in convolutional neural operators and validate HFS controls for multiscale single- and two-phase fluid systems against diffusion-conditioned baselines.",
            "诊断 convolutional neural operator 的 spectral-bias 高频失效模式，并验证 HFS 控制在多尺度单相/两相流体系统中相对 diffusion-conditioned baseline 的效果。",
        ),
        (
            "Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks",
            "AIVT reconstructs turbulent temperature fields from sparse Lagrangian velocity measurements at DNS-level fidelity.",
            "Validate physics-informed KAN field-inference workflows for turbulent velocity and temperature reconstruction under sparse Lagrangian measurements and DNS-fidelity constraints.",
            "验证 turbulent velocity 与 temperature reconstruction 的 physics-informed KAN field-inference 工作流，并检查稀疏 Lagrangian measurements 和 DNS-fidelity 约束。",
        ),
        (
            "Integrating Neural Operators with Diffusion Models Improves Spectral Representation in Turbulence Modeling",
            "Diffusion models correct neural operator forecasts to improve turbulent-flow spectra and long-horizon autoregressive rollout stability.",
            "Validate diffusion-corrected neural operator surrogates for turbulent-flow spectral fidelity, high-frequency structure, and long-horizon rollout stability.",
            "验证 turbulent-flow spectral fidelity 的 diffusion-corrected neural operator surrogate，并检查高频结构和长周期 rollout 稳定性。",
        ),
        (
            "Physics-Informed Laplace Neural Operator for Solving Partial Differential Equations",
            "PILNO uses virtual inputs for small-data and out-of-distribution PDE operator generalization.",
            "Validate physics-informed neural operator surrogates for data-efficient PDE solving under small-data and out-of-distribution generalization constraints.",
            "验证数据稀缺和 OOD 泛化约束下用于 PDE 求解的 physics-informed neural operator surrogate。",
        ),
        (
            "Uncertainty Quantification in PINNs for Turbulent Flows: Bayesian Inference and Repulsive Ensembles",
            "Bayesian PINNs, Monte Carlo dropout, and repulsive deep ensembles calibrate turbulent-flow inverse modeling uncertainty.",
            "Audit uncertainty calibration for physics-informed turbulent-flow inverse modeling under Bayesian, dropout, and ensemble tradeoffs.",
            "审计 physics-informed 湍流反问题建模中的不确定性校准，并权衡 Bayesian、dropout 和 ensemble 方法。",
        ),
        (
            "From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning",
            "This review covers network architectures, optimization techniques, uncertainty quantification, and PIKAN applications.",
            "Audit PINN-to-PIKAN Kolmogorov-Arnold SciML review coverage under architecture, optimization, uncertainty, and application-diversity constraints.",
            "审计 PINN-to-PIKAN Kolmogorov-Arnold SciML review 覆盖，并检查架构、优化、不确定性和应用多样性边界。",
        ),
        (
            "Scalable Bayesian Physics-Informed Kolmogorov-Arnold Networks",
            "Uncertainty quantification combines dropout Tikhonov ensemble Kalman inversion with Chebyshev KANs.",
            "Audit Bayesian KAN or PIKAN uncertainty workflows under gradient-free inference, overfitting, stability, and parameter-efficiency constraints.",
            "审计 Bayesian KAN 或 PIKAN 不确定性工作流，并检查 gradient-free inference、过拟合、稳定性和参数效率约束。",
        ),
        (
            "NeuroSEM: A hybrid framework for simulating multiphysics problems by coupling PINNs and spectral elements",
            "NeuroSEM integrates PINNs with the Spectral Element Method solver for Rayleigh-Bénard convection, noisy data, and turbulence-aware multiphysics simulation.",
            "Validate hybrid PINN and spectral-element workflows for coupled multiphysics simulation under data assimilation, solver integration, and turbulence robustness constraints.",
            "验证混合 PINN 与 spectral-element 工作流在耦合 multiphysics 模拟中的数据同化、求解器集成和湍流鲁棒性。",
        ),
        (
            "Retrofitting Earth System Models with Cadence-Limited Neural Operator Updates",
            "Online bias correction tendencies retrofit ESM predictions under runtime cadence limits.",
            "Audit online ML bias-correction for Earth-system models under stability, portability, and runtime cadence constraints.",
            "审计 Earth-system model 的在线 ML 偏差校正，同时检查稳定性、可迁移性和运行 cadence 约束。",
        ),
        (
            "Adversarial Physics-Informed Machine Learning for Robust Optimal Safe Predefined-Time Stabilization",
            "A differential game and HJI equation learn safe stabilization under adversarial disturbances.",
            "Assess robust safe-control learning for nonlinear dynamical systems under adversarial disturbances and stability constraints.",
            "评估非线性动力系统在对抗扰动和稳定性约束下的鲁棒安全控制学习。",
        ),
        (
            "A Neural-Operator Preconditioned Newton Method for Accelerated Nonlinear Solvers",
            "A fixed-point neural operator preconditions Newton iterations for nonlinear systems.",
            "Evaluate neural preconditioning for nonlinear solvers under convergence, robustness, and instability constraints.",
            "评估非线性求解器中的神经预条件方法，并检查收敛性、鲁棒性和不稳定边界。",
        ),
        (
            "Spectrally Safe Neural Operator Warm-Starts for Large-Scale Newton Solvers",
            "A label-free energy fine-tuning step keeps warm-started Newton Jacobians positive definite for Krylov-compatible large-scale PDE solves.",
            "Audit spectrally safe neural-operator warm starts for large-scale Newton PDE solvers under Jacobian definiteness, Krylov compatibility, label-free energy fine-tuning, and speedup constraints.",
            "审计 large-scale Newton PDE solver 的 spectrally safe neural-operator warm start，并检查 Jacobian definiteness、Krylov compatibility、label-free energy fine-tuning 和加速约束。",
        ),
        (
            "From LIF to QIF: Toward Differentiable Spiking Neurons for Scientific Machine Learning",
            "QIF spiking neural networks support stable gradients for operator learning and PDE solving.",
            "Evaluate differentiable spiking-neuron models for stable SciML regression, operator learning, and PDE solving.",
            "评估可微分 spiking-neuron 模型在 SciML 回归、operator learning 和 PDE 求解中的稳定性。",
        ),
        (
            "AMORE: Adaptive Multi-Output Operator Network for Stiff Chemical Kinetics",
            "Adaptive losses predict thermochemical states for combustion, hypersonics, and reactive transport systems.",
            "Validate multi-output neural operator surrogates for stiff chemical kinetics under conservation, stiffness, and CFD-coupling constraints.",
            "验证 stiff chemical kinetics 的多输出神经算子 surrogate，并检查守恒、刚性和 CFD 耦合约束。",
        ),
        (
            "Automatic discovery of optimal meta-solvers for time-dependent nonlinear PDEs",
            "Multi-objective optimization discovers Pareto optimal meta-solvers for Newton and IMEX time integration.",
            "Audit automated meta-solver discovery for time-dependent PDEs under accuracy, speed, memory, and preference-selection tradeoffs.",
            "审计 time-dependent PDE 的自动 meta-solver 发现，并权衡精度、速度、内存和偏好选择。",
        ),
        (
            "FMEnets: Flow, Material, and Energy networks for non-ideal plug flow reactor design",
            "A physics-informed framework predicts flow, material and energy states for reactor design with independent optimizers.",
            "Validate multi-output neural operator surrogates for stiff chemical kinetics under conservation, stiffness, and CFD-coupling constraints.",
            "验证 stiff chemical kinetics 的多输出神经算子 surrogate，并检查守恒、刚性和 CFD 耦合约束。",
        ),
        (
            "Learning and discovering multiple solutions using physics-informed neural networks with random initialization and deep ensemble",
            "PINNs discover multiple solutions and solution multiplicity in nonlinear DEs where Newton iteration is sensitive.",
            "Audit PINN discovery of multiple nonlinear ODE/PDE solutions under initialization, ensemble diversity, and solver-refinement constraints.",
            "审计 PINN 对非线性 ODE/PDE 多解的发现能力，并检查初始化、ensemble 多样性和求解器细化约束。",
        ),
        (
            "DeepSeek vs. ChatGPT vs. Claude: A Comparative Study for Scientific Computing and Scientific Machine Learning Tasks",
            "Large Language Models are compared on scientific computing and scientific machine learning decision tasks.",
            "Benchmark LLM capability for scientific computing and SciML tasks while preserving model-specific failure modes and decision points.",
            "评测 LLM 在 scientific computing 和 SciML 任务中的能力，同时保留模型特定失败模式和决策点。",
        ),
        (
            "Agentic Risk-Aware Set-Based Engineering Design",
            "A multi-agent LLM framework performs airfoil design with CVaR risk filtering and high-fidelity CFD evidence.",
            "Audit risk-aware multi-agent engineering-design workflows under CVaR filtering, human-in-the-loop review, tool validation, and CFD evidence constraints.",
            "审计 risk-aware multi-agent engineering-design 工作流，并检查 CVaR 过滤、human-in-the-loop review、工具验证和 CFD 证据约束。",
        ),
        (
            "GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms",
            "GRAFT projects combinatorial decisions into adaptive factored trees and expands its own action space across production solvers.",
            "Audit self-improving agentic scientific-discovery workflows under reusable method substrates, action-space expansion, probabilistic-tree policy, and production-solver evidence constraints.",
            "审计 self-improving agentic scientific-discovery 工作流，并检查可复用 method substrate、action-space expansion、probabilistic-tree policy 和 production-solver 证据约束。",
        ),
        (
            "DeepVIVONet: Using deep neural operators to optimize sensor locations with application to vortex-induced vibrations",
            "Sparse spatio-temporal measurements reconstruct and forecast vortex-induced vibrations of a marine riser.",
            "Optimize sparse sensor placement for vortex-induced vibration reconstruction and forecasting under transfer-learning constraints.",
            "优化 vortex-induced vibration 重建与 forecasting 的稀疏传感器布置，并检查 transfer-learning 约束。",
        ),
        (
            "Fusion-DeepONet: A Data-Efficient Neural Operator for Geometry-Dependent Hypersonic and Supersonic Flows",
            "Shape optimization uses a geometry-dependent surrogate to predict hypersonic and supersonic flow fields on arbitrary grids with scarce data.",
            "Validate data-efficient neural operator surrogates for geometry-dependent hypersonic or supersonic flow prediction on scarce data.",
            "验证稀缺数据下 geometry-dependent hypersonic 或 supersonic flow 预测的高效神经算子 surrogate。",
        ),
        (
            "KKANs: Kurkova-Kolmogorov-Arnold Networks and Their Learning Dynamics",
            "Kolmogorov-Arnold networks are analyzed with information bottleneck theory and geometric complexity.",
            "Audit KAN-style SciML architectures through approximation behavior, learning dynamics, and signal-to-noise generalization controls.",
            "通过逼近行为、learning dynamics 和 signal-to-noise 泛化控制审计 KAN-style SciML 架构。",
        ),
        (
            "A Digital Twin for Diesel Engines: Operator-infused Physics-Informed Neural Networks with Transfer Learning for Engine Health Monitoring",
            "Operator-infused PINNs identify diesel engine parameters for health monitoring and maintenance forecasting.",
            "Evaluate operator-infused PINN digital twins for diesel-engine health monitoring under transfer learning and uncertainty constraints.",
            "评估 diesel-engine health monitoring 的 operator-infused PINN digital twin，并检查 transfer learning 和不确定性约束。",
        ),
        (
            "Predicting Crack Nucleation and Propagation in Brittle Materials Using Deep Operator Networks with Diverse Trunk Architectures",
            "DeepONet solves brittle fracture phase-field problems and compares a Kolmogorov-Arnold trunk architecture.",
            "Validate DeepONet surrogates for brittle-fracture crack nucleation and propagation under phase-field physics constraints.",
            "验证 brittle-fracture crack nucleation 与 propagation 的 DeepONet surrogate，并检查 phase-field 物理约束。",
        ),
        (
            "Physics-Informed Machine Learning in Biomedical Science and Engineering",
            "Biomedical PIML reviews PINNs, NODEs and neural operators for biofluid mechanics, cell signaling, and uncertainty quantification.",
            "Audit physics-informed biomedical modeling under data scarcity, interpretability, uncertainty, and multiscale physiology constraints.",
            "审计 physics-informed biomedical modeling 在数据稀缺、可解释性、不确定性和多尺度生理约束下的可靠性。",
        ),
        (
            "Representation Meets Optimization: Training PINNs and PIKANs for Gray-Box Discovery in Systems Pharmacology",
            "PIKANs use Kolmogorov-Arnold networks and optimizers for systems pharmacology modeling.",
            "Audit physics-informed biomedical modeling under data scarcity, interpretability, uncertainty, and multiscale physiology constraints.",
            "审计 physics-informed biomedical modeling 在数据稀缺、可解释性、不确定性和多尺度生理约束下的可靠性。",
        ),
        (
            "Learning Turbulent Flows with Generative Models: Super-resolution, Forecasting, and Sparse Flow Reconstruction",
            "Generative neural operators improve turbulent-flow super-resolution and sparse reconstruction; see https://vivekoommen.github.io/gen4turb/.",
            "Evaluate generative operator models for turbulent-flow super-resolution, forecasting, and sparse reconstruction without losing fine-scale structure.",
            "评估湍流 super-resolution、forecasting 和 sparse reconstruction 的生成式算子模型，避免丢失细尺度结构。",
        ),
        (
            "PI-SONet: A Physics-Informed Symplectic Operator Network for Real-Time Optimal Control of Multi-Agent Systems",
            "A symplectic operator network learns real-time optimal control of multi-agent systems.",
            "Assess physics-informed operator networks for real-time optimal control of multi-agent dynamical systems under stability constraints.",
            "评估 real-time optimal control 中的 physics-informed operator network，并检查多智能体动力系统稳定性约束。",
        ),
        (
            "Generalizable turbulence closures across bluff-body shapes by PINN-based solver-agnostic training",
            "The RANS residual trains closures without deriving adjoints; the final closure must remain solver-stable across bluff-body wakes.",
            "Assess PINN-trained turbulence closures across bluff-body shapes under solver stability, geometry generalization, and closure-model constraints.",
            "评估跨 bluff-body 形状的 PINN-trained turbulence closure，并检查求解器稳定性、几何泛化和 closure-model 约束。",
        ),
        (
            "Connecting the geometry and dynamics of many-body complex systems with message passing neural operators",
            "ROMA learns multiscale evolution operators for many-body complex systems with message passing and noisy input-output data.",
            "Validate message-passing neural operators for many-body complex-system geometry and dynamics under scalability, stability, and multiscale-fidelity constraints.",
            "验证 many-body complex-system 几何与动力学的 message-passing neural operator，并检查可扩展性、稳定性和多尺度保真度约束。",
        ),
        (
            "Synergistic Learning with Multi-Task DeepONet for Efficient PDE Problem Solving",
            "MT-DeepONet applies multi-task learning to PDEs with multiple source terms and parameterized geometries.",
            "Validate multi-task DeepONet workflows for efficient PDE problem solving under task coupling, geometry generalization, accuracy, and training-cost constraints.",
            "验证 efficient PDE problem solving 的 multi-task DeepONet 工作流，并检查任务耦合、几何泛化、精度和训练成本约束。",
        ),
    ]

    for title, summary, expected, expected_zh in cases:
        feed = ATOM_FIXTURE.replace("Spectral Audit of In-Context Operator Networks", title).replace(
            "Existing evaluations of neural operators rely primarily on prediction error.",
            summary,
        )
        entry = parse_arxiv_atom(feed)[0]
        assert entry["real_problem"] == expected
        assert entry["real_problem_zh"] == expected_zh


def test_collect_paper_sources_cli_writes_failure_cache_without_network(tmp_path: Path, cli_env: dict[str, str]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "collect-paper-sources",
            "--output-dir",
            str(tmp_path),
            "--max-results",
            "0",
            "--fail-on-issues",
        ],
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert result.returncode == 1
    path = Path(result.stdout.strip())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "collection_failed"
    assert payload["issue_count"] == 1
