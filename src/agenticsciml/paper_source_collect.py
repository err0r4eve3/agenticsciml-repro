from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from agenticsciml.storage import _atomic_write_text


ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_ARXIV_QUERY = 'au:"George Em Karniadakis" AND (all:"neural operator" OR all:"physics-informed" OR all:"agent")'
DEFAULT_SOURCE_LIMIT = 50
COLLECTION_JSON = "paper_source_collection.json"


def write_paper_source_collection(
    *,
    output_dir: Path,
    query: str = DEFAULT_ARXIV_QUERY,
    max_results: int = DEFAULT_SOURCE_LIMIT,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        feed = fetch_arxiv_feed(query=query, max_results=max_results, timeout_s=timeout_s)
        payload = build_paper_source_collection(
            entries=parse_arxiv_atom(feed),
            query=query,
            max_results=max_results,
            status="collected",
            error=None,
        )
    except Exception as exc:
        payload = build_paper_source_collection(
            entries=[],
            query=query,
            max_results=max_results,
            status="collection_failed",
            error=str(exc),
        )
    path = output_dir / COLLECTION_JSON
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return {"collection": payload, "paths": {"collection_json": str(path.resolve())}}


def fetch_arxiv_feed(*, query: str, max_results: int, timeout_s: float) -> str:
    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(max_results),
        }
    )
    request = urllib.request.Request(
        f"{ARXIV_API_URL}?{params}",
        headers={"User-Agent": "agenticsciml-repro/0.1 paper-source-collector"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def parse_arxiv_atom(feed: str) -> list[dict[str, Any]]:
    root = ET.fromstring(feed)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        authors = [
            _text(author.find("atom:name", ns))
            for author in entry.findall("atom:author", ns)
            if _text(author.find("atom:name", ns))
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ns)
            if category.attrib.get("term")
        ]
        paper_url = _text(entry.find("atom:id", ns))
        title = _normalize(_text(entry.find("atom:title", ns)))
        summary = _normalize(_text(entry.find("atom:summary", ns)))
        entries.append(
            {
                "id": paper_url.rsplit("/", 1)[-1],
                "title": title,
                "url": paper_url,
                "authors": authors,
                "published": _date(_text(entry.find("atom:published", ns))),
                "updated": _date(_text(entry.find("atom:updated", ns))),
                "summary": summary,
                "categories": categories,
                "real_problem": _real_problem(title, summary),
                "real_problem_zh": _real_problem_zh(title, summary),
            }
        )
    return entries


def build_paper_source_collection(
    *,
    entries: list[dict[str, Any]],
    query: str,
    max_results: int,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    issues = []
    if status != "collected":
        issues.append("arxiv source collection failed")
    for entry in entries:
        missing = [key for key in ("title", "url", "authors", "published", "real_problem", "real_problem_zh") if not entry.get(key)]
        if missing:
            issues.append(f"{entry.get('id') or 'unknown'} missing {','.join(missing)}")
    return {
        "artifact_type": "agenticsciml_paper_source_collection",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "arxiv_api",
        "uses_network": True,
        "query": query,
        "max_results": max_results,
        "status": status,
        "error": error,
        "candidate_count": len(entries),
        "issue_count": len(issues),
        "issues": issues,
        "candidates": entries,
    }


def _text(element: ET.Element[str] | None) -> str:
    return "" if element is None or element.text is None else element.text.strip()


def _date(value: str) -> str | None:
    return value[:10] if value else None


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _real_problem(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(term in text for term in ("deepseek", "chatgpt", "claude", "comparative study")) and (
        "scientific computing" in text or "scientific machine learning" in text
    ):
        return "Benchmark LLM capability for scientific computing and SciML tasks while preserving model-specific failure modes and decision points."
    if "multi-agent systems" in text and "control" in text:
        return "Assess physics-informed operator networks for real-time optimal control of multi-agent dynamical systems under stability constraints."
    if any(term in text for term in ("knowledge-guided multi-agent", "graph ontologist", "design knowledge graph")):
        return "Audit knowledge-guided multi-agent engineering-design workflows under domain-graph construction, requirement formulation, candidate generation, systems-engineer review, manager validation, and tool-backed optimization constraints."
    if any(term in text for term in ("risk-aware set-based", "set-based engineering design", "cvar")):
        return "Audit risk-aware multi-agent engineering-design workflows under CVaR filtering, human-in-the-loop review, tool validation, and CFD evidence constraints."
    if any(term in text for term in ("graft-athena", "adaptive factored trees", "self-improving agentic", "autonomous discovery", "action space")):
        return "Audit self-improving agentic scientific-discovery workflows under reusable method substrates, action-space expansion, probabilistic-tree policy, and production-solver evidence constraints."
    if "athena" in text and any(term in text for term in ("hierarchical evolutionary", "contextual bandit", "numerical algorithms", "scientific pipelines")):
        return "Audit agentic numerical-algorithm discovery workflows under conceptual scaffolding, contextual-bandit policy, solver construction, symbolic-numeric orchestration, verification, and repair constraints."
    if "agents' last exam" in text or "sustained performance measurement" in text:
        return "Audit long-horizon agent benchmarks for economically valuable real-world workflows under sustained performance, verifiable outcome, domain coverage, and deployment-gap constraints."
    if "agent" in text or "llm" in text:
        return "Audit long-horizon agent or LLM workflows with verifiable outcomes and explicit failure boundaries."
    if "multiple solutions" in text or "solution multiplicity" in text:
        return "Audit PINN discovery of multiple nonlinear ODE/PDE solutions under initialization, ensemble diversity, and solver-refinement constraints."
    if any(term in text for term in ("fmenets", "plug flow reactor", "reactor design")):
        return "Validate physics-informed Flow-Material-Energy networks for non-ideal plug-flow reactor design under coupled Navier-Stokes, material-balance, energy-balance, sparse inverse-measurement, and finite-element comparison constraints."
    if any(term in text for term in ("stiff chemical", "chemical kinetics", "combustion", "reactive transport", "thermochemical")):
        return "Validate multi-output neural operator surrogates for stiff chemical kinetics under conservation, stiffness, and CFD-coupling constraints."
    if ("state-space model" in text or "mamba" in text) and ("operator learning" in text or "neural operator" in text):
        return "Validate state-space or Mamba neural operators for dynamical systems under long-range dependency, extrapolation, chaotic rollout, and computational-efficiency constraints."
    if "systems pharmacology" in text:
        return "Audit PINN and PIKAN gray-box systems-pharmacology discovery under representation choice, optimizer schedule, training configuration, numerical precision, scalability, and data-sparse non-unique inverse-problem constraints."
    if any(term in text for term in ("biomedical", "biofluid", "biosolid", "medical imaging", "cell signaling", "physiological", "pharmacology")):
        return "Audit physics-informed biomedical modeling under data scarcity, interpretability, uncertainty, and multiscale physiology constraints."
    if any(term in text for term in ("bayesian", "dteki", "hmc", "ensemble kalman", "tikhonov")) and any(
        term in text for term in ("kolmogorov-arnold", "pikan", "chebyshev kan")
    ):
        return "Audit Bayesian KAN or PIKAN uncertainty workflows under gradient-free inference, overfitting, stability, and parameter-efficiency constraints."
    if "physics-informed neural networks and extensions" in text or (
        "review" in text and "physics-informed neural networks" in text and "governing differential equations" in text
    ):
        return "Audit PINN review and extension coverage for governing-equation discovery under residual formulation, practical-extension, benchmark-example, and claim-boundary constraints."
    if "from pinns to pikans" in text or ("recent advances" in text and "pikan" in text):
        return "Audit PINN-to-PIKAN Kolmogorov-Arnold SciML review coverage under architecture, optimization, uncertainty, and application-diversity constraints."
    if any(term in text for term in ("uncertainty quantification", "bayesian", "dropout", "repulsive ensemble")) and (
        "pinn" in text or "physics-informed" in text or "turbulence" in text or "turbulent" in text
    ):
        return "Audit uncertainty calibration for physics-informed turbulent-flow inverse modeling under Bayesian, dropout, and ensemble tradeoffs."
    if "meta-solver" in text or "meta solver" in text or "multi-objective" in text or "pareto" in text:
        return "Audit neural-operator-assisted meta-solver discovery for PDE-discretization linear systems under Jacobi/Gauss-Seidel/Krylov composition, DeepONet coarse preconditioning, Pareto metrics, preference selection, and spectrum-split error constraints."
    if any(term in text for term in ("sensor location", "sensor placement", "vortex-induced", "marine riser", "deepvivonet")):
        return "Optimize sparse sensor placement for vortex-induced vibration reconstruction and forecasting under transfer-learning constraints."
    if any(term in text for term in ("hypersonic", "supersonic", "reentry", "arbitrary grids", "geometry-dependent")):
        return "Validate data-efficient neural operator surrogates for geometry-dependent hypersonic or supersonic flow prediction on scarce data."
    if any(term in text for term in ("turbulence closure", "turbulence closures", "bluff-body", "rans residual", "reynolds stress", "reynolds force")):
        return "Assess PINN-trained turbulence closures across bluff-body shapes under solver stability, geometry generalization, and closure-model constraints."
    if "adjoint" in text and "pinn" in text and "inverse problem" in text:
        return "Compare adjoint optimization and PINNs for PDE-constrained inverse problems under matched formulations, parameterizations, regularization, cost, and warm-start tradeoffs."
    if "optimizing the optimizer" in text or (
        any(term in text for term in ("kolmogorov-arnold", "pikan"))
        and any(term in text for term in ("self-scaled", "bfgs", "broyden", "quasi-newton", "line search"))
        and ("pde" in text or "deeponet" in text or "operator learning" in text)
    ):
        return "Audit self-scaled quasi-Newton optimizer selection across PINN, PIKAN, and DeepONet training under nonlinear loss landscapes, saddle points, PDE benchmark diversity, line-search strategy, and accuracy-vs-efficiency constraints."
    if any(term in text for term in ("optimizer", "natural gradient", "bfgs", "broyden", "curvature-aware")):
        return "Audit optimizer and conditioning choices for high-accuracy PINN convergence on challenging PDE or ODE systems."
    if "nspod" in text or "neural subspace proper orthogonal decomposition" in text:
        return "Validate NSPOD DeepONet-learned POD preconditioners for Krylov linear solvers under multigrid-like subspace quality, unstructured CAD geometry, solid-mechanics PDE, AMG comparison, and convergence-speed constraints."
    if "geometry-aware neural preconditioner" in title.lower() or "hybrid iterative solvers with geometry-aware" in title.lower():
        return "Validate geometry-aware neural preconditioners for hybrid iterative parametric-PDE solvers under unstructured-mesh geometry transfer, relaxation/Krylov coupling, robustness, and efficiency constraints."
    if "warm-start" in text and "newton" in text and ("spectral" in text or "jacobian" in text):
        return "Audit spectrally safe neural-operator warm starts for large-scale Newton PDE solvers under Jacobian definiteness, Krylov compatibility, label-free energy fine-tuning, and speedup constraints."
    if "newton" in text or "nonlinear solver" in text or "nonlinear system" in text:
        return "Evaluate neural preconditioning for nonlinear solvers under convergence, robustness, and instability constraints."
    if any(term in text for term in ("krylov", "preconditioner", "linear solver", "linear system")):
        return "Evaluate whether learned solver aids accelerate PDE linear systems while preserving convergence and generalization."
    if "earth system" in text or "esm" in text or "bias correction" in text or "cadence-limited" in text:
        return "Audit online ML bias-correction for Earth-system models under stability, portability, and runtime cadence constraints."
    if "stabilization" in text or "hji" in text or "lyapunov" in text or "differential game" in text:
        return "Assess robust safe-control learning for nonlinear dynamical systems under adversarial disturbances and stability constraints."
    if any(term in text for term in ("lagrangian velocity", "velocimetry-thermometry", "temperature fields")):
        return "Validate physics-informed KAN field-inference workflows for turbulent velocity and temperature reconstruction under sparse Lagrangian measurements and DNS-fidelity constraints."
    if "solitary wave" in text or "initial-value iterative" in text:
        return "Validate two-stage initial-value iterative PINNs for solitary-wave simulation in nonlinear wave equations under initial-data-only, theoretical-guarantee, and traditional-solver comparison constraints."
    if "optimal control" in text or "adjoint" in text or "direct vs indirect" in text:
        return "Compare PINN control formulations against adjoint or optimality-system baselines for PDE-constrained control."
    if "high-frequency scaling" in text or ("hfs" in text and "spectral bias" in text):
        return "Diagnose spectral-bias high-frequency failure modes in convolutional neural operators and validate HFS controls for multiscale single- and two-phase fluid systems against diffusion-conditioned baselines."
    if "diffusion model" in text and ("turbulence" in text or "turbulent" in text):
        return "Validate diffusion-corrected neural operator surrogates for turbulent-flow spectral fidelity, high-frequency structure, and long-horizon rollout stability."
    if "in-context operator" in text and ("spectral audit" in text or "tangent operator" in text or "jacobian" in text):
        return "Audit in-context operator networks with Jacobian-based spectral tangent-operator diagnostics for local PDE mechanism fidelity, stability, sensitivity, and prompt consistency beyond prediction error."
    if "domain-unification-free" in text or "discretization decoupling" in text or "discretization-decoupled" in text:
        return "Validate cross-domain neural operator frameworks for discretization-decoupled generalized operator learning under irregular sampling, spectral mismatch, distribution shift, and arbitrary-resolution query constraints."
    if "spectral bias" in text or "high-frequency" in text or "frequency-resolved" in text:
        return "Diagnose high-frequency failure modes in physics-informed or operator learning and test mitigation controls."
    if "laplace neural operator" in text or "pilno" in text or "virtual inputs" in text or "out-of-distribution" in text:
        return "Validate physics-informed neural operator surrogates for data-efficient PDE solving under small-data and out-of-distribution generalization constraints."
    if ("neurosem" in text or "spectral element" in text or "sem solver" in text) and (
        "pinn" in text or "physics-informed" in text or "multiphysics" in text
    ):
        return "Validate hybrid PINN and spectral-element workflows for coupled multiphysics simulation under data assimilation, solver integration, and turbulence robustness constraints."
    if any(term in text for term in ("crack nucleation", "crack propagation", "brittle", "fracture", "phase-field")):
        return "Validate DeepONet surrogates for brittle-fracture crack nucleation and propagation under phase-field physics constraints."
    if any(term in text for term in ("kolmogorov-arnold", "kkan", "kan", "information bottleneck", "geometric complexity")):
        return "Audit KAN-style SciML architectures through approximation behavior, learning dynamics, and signal-to-noise generalization controls."
    if any(term in text for term in ("spiking", "lif", "qif", "integrate-and-fire", "snn")):
        return "Evaluate differentiable spiking-neuron models for stable SciML regression, operator learning, and PDE solving."
    if any(term in text for term in ("diesel engine", "engine health", "maintenance forecasting", "parameter identification", "mean value diesel")):
        return "Evaluate operator-infused PINN digital twins for diesel-engine health monitoring under transfer learning and uncertainty constraints."
    if ("turbulent" in text or "turbulence" in text) and any(
        term in text for term in ("generative", "super-resolution", "forecasting", "sparse", "reconstruction")
    ):
        return "Evaluate generative operator models for turbulent-flow super-resolution, forecasting, and sparse reconstruction without losing fine-scale structure."
    if any(term in text for term in ("message passing", "message-passing", "many-body", "renormalized operators", "multiscale attention")):
        return "Validate message-passing neural operators for many-body complex-system geometry and dynamics under scalability, stability, and multiscale-fidelity constraints."
    if (
        any(term in text for term in ("multi-task", "multitask", "mt-deeponet", "synergistic learning"))
        and ("deeponet" in text or "operator network" in text)
        and ("pde" in text or "partial differential" in text or "geometry" in text)
    ):
        return "Validate multi-task DeepONet workflows for efficient PDE problem solving under task coupling, geometry generalization, accuracy, and training-cost constraints."
    if "operator" in text:
        return "Evaluate neural operator reliability beyond average prediction error using stability and fidelity diagnostics."
    if "turbulence" in text or "fluid" in text:
        return "Assess whether learned fluid models remain stable and generalize under solver-facing constraints."
    if "inverse" in text or "physics-informed" in text or "pinn" in text:
        return "Plan PDE-constrained SciML workflows while separating representation, optimization, and physics-residual failures."
    return "Track a new scientific machine learning result as planning context without treating it as evaluator evidence."


def _real_problem_zh(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(term in text for term in ("deepseek", "chatgpt", "claude", "comparative study")) and (
        "scientific computing" in text or "scientific machine learning" in text
    ):
        return "评测 LLM 在 scientific computing 和 SciML 任务中的能力，同时保留模型特定失败模式和决策点。"
    if "multi-agent systems" in text and "control" in text:
        return "评估 real-time optimal control 中的 physics-informed operator network，并检查多智能体动力系统稳定性约束。"
    if any(term in text for term in ("knowledge-guided multi-agent", "graph ontologist", "design knowledge graph")):
        return "审计 knowledge-guided multi-agent engineering-design 工作流，并检查 domain-graph construction、requirement formulation、candidate generation、systems-engineer review、manager validation 和 tool-backed optimization 约束。"
    if any(term in text for term in ("risk-aware set-based", "set-based engineering design", "cvar")):
        return "审计 risk-aware multi-agent engineering-design 工作流，并检查 CVaR 过滤、human-in-the-loop review、工具验证和 CFD 证据约束。"
    if any(term in text for term in ("graft-athena", "adaptive factored trees", "self-improving agentic", "autonomous discovery", "action space")):
        return "审计 self-improving agentic scientific-discovery 工作流，并检查可复用 method substrate、action-space expansion、probabilistic-tree policy 和 production-solver 证据约束。"
    if "athena" in text and any(term in text for term in ("hierarchical evolutionary", "contextual bandit", "numerical algorithms", "scientific pipelines")):
        return "审计 agentic numerical-algorithm discovery 工作流，并检查 conceptual scaffolding、contextual-bandit policy、solver construction、symbolic-numeric orchestration、verification 和 repair 约束。"
    if "agents' last exam" in text or "sustained performance measurement" in text:
        return "审计 economically valuable real-world workflow 的长周期 agent benchmark，并检查 sustained performance、可验证结果、domain coverage 和 deployment-gap 约束。"
    if "agent" in text or "llm" in text:
        return "审计长周期 agent 或 LLM 工作流，要求结果可验证并显式记录失败边界。"
    if "multiple solutions" in text or "solution multiplicity" in text:
        return "审计 PINN 对非线性 ODE/PDE 多解的发现能力，并检查初始化、ensemble 多样性和求解器细化约束。"
    if any(term in text for term in ("fmenets", "plug flow reactor", "reactor design")):
        return "验证 non-ideal plug-flow reactor design 的 physics-informed Flow-Material-Energy network，并检查 coupled Navier-Stokes、material balance、energy balance、sparse inverse measurement 和 finite-element 对比约束。"
    if any(term in text for term in ("stiff chemical", "chemical kinetics", "combustion", "reactive transport", "thermochemical")):
        return "验证 stiff chemical kinetics 的多输出神经算子 surrogate，并检查守恒、刚性和 CFD 耦合约束。"
    if ("state-space model" in text or "mamba" in text) and ("operator learning" in text or "neural operator" in text):
        return "验证 dynamical system 的 state-space 或 Mamba neural operator，并检查 long-range dependency、外推、chaotic rollout 和计算效率约束。"
    if "systems pharmacology" in text:
        return "审计 PINN 与 PIKAN 的 gray-box systems-pharmacology discovery，并检查 representation choice、optimizer schedule、training configuration、数值精度、可扩展性和数据稀缺非唯一反问题约束。"
    if any(term in text for term in ("biomedical", "biofluid", "biosolid", "medical imaging", "cell signaling", "physiological", "pharmacology")):
        return "审计 physics-informed biomedical modeling 在数据稀缺、可解释性、不确定性和多尺度生理约束下的可靠性。"
    if any(term in text for term in ("bayesian", "dteki", "hmc", "ensemble kalman", "tikhonov")) and any(
        term in text for term in ("kolmogorov-arnold", "pikan", "chebyshev kan")
    ):
        return "审计 Bayesian KAN 或 PIKAN 不确定性工作流，并检查 gradient-free inference、过拟合、稳定性和参数效率约束。"
    if "physics-informed neural networks and extensions" in text or (
        "review" in text and "physics-informed neural networks" in text and "governing differential equations" in text
    ):
        return "审计 PINN review 与 extensions 覆盖，并检查 governing-equation discovery、residual formulation、practical extension、benchmark example 和 claim-boundary 约束。"
    if "from pinns to pikans" in text or ("recent advances" in text and "pikan" in text):
        return "审计 PINN-to-PIKAN Kolmogorov-Arnold SciML review 覆盖，并检查架构、优化、不确定性和应用多样性边界。"
    if any(term in text for term in ("uncertainty quantification", "bayesian", "dropout", "repulsive ensemble")) and (
        "pinn" in text or "physics-informed" in text or "turbulence" in text or "turbulent" in text
    ):
        return "审计 physics-informed 湍流反问题建模中的不确定性校准，并权衡 Bayesian、dropout 和 ensemble 方法。"
    if "meta-solver" in text or "meta solver" in text or "multi-objective" in text or "pareto" in text:
        return "审计 PDE-discretization linear system 的 neural-operator-assisted meta-solver discovery，并检查 Jacobi/Gauss-Seidel/Krylov composition、DeepONet coarse preconditioning、Pareto metrics、preference selection 和 spectrum-split error 约束。"
    if any(term in text for term in ("sensor location", "sensor placement", "vortex-induced", "marine riser", "deepvivonet")):
        return "优化 vortex-induced vibration 重建与 forecasting 的稀疏传感器布置，并检查 transfer-learning 约束。"
    if any(term in text for term in ("hypersonic", "supersonic", "reentry", "arbitrary grids", "geometry-dependent")):
        return "验证稀缺数据下 geometry-dependent hypersonic 或 supersonic flow 预测的高效神经算子 surrogate。"
    if any(term in text for term in ("turbulence closure", "turbulence closures", "bluff-body", "rans residual", "reynolds stress", "reynolds force")):
        return "评估跨 bluff-body 形状的 PINN-trained turbulence closure，并检查求解器稳定性、几何泛化和 closure-model 约束。"
    if "adjoint" in text and "pinn" in text and "inverse problem" in text:
        return "对比 PDE-constrained inverse problem 中的 adjoint optimization 与 PINN，并检查 formulation、parameterization、regularization、成本和 warm-start 取舍。"
    if "optimizing the optimizer" in text or (
        any(term in text for term in ("kolmogorov-arnold", "pikan"))
        and any(term in text for term in ("self-scaled", "bfgs", "broyden", "quasi-newton", "line search"))
        and ("pde" in text or "deeponet" in text or "operator learning" in text)
    ):
        return "审计 PINN、PIKAN 与 DeepONet training 中的 self-scaled quasi-Newton optimizer selection，并检查 nonlinear loss landscape、saddle point、PDE benchmark diversity、line-search strategy 和 accuracy-vs-efficiency 约束。"
    if any(term in text for term in ("optimizer", "natural gradient", "bfgs", "broyden", "curvature-aware")):
        return "审计 optimizer 与 conditioning 选择对高精度 PINN 在困难 PDE 或 ODE 系统上收敛的影响。"
    if "nspod" in text or "neural subspace proper orthogonal decomposition" in text:
        return "验证 Krylov linear solver 的 NSPOD DeepONet-learned POD preconditioner，并检查 multigrid-like subspace quality、unstructured CAD geometry、solid-mechanics PDE、AMG 对比和收敛加速约束。"
    if "geometry-aware neural preconditioner" in title.lower() or "hybrid iterative solvers with geometry-aware" in title.lower():
        return "验证 hybrid iterative parametric-PDE solver 的 geometry-aware neural preconditioner，并检查 unstructured-mesh geometry transfer、relaxation/Krylov coupling、鲁棒性和效率约束。"
    if "warm-start" in text and "newton" in text and ("spectral" in text or "jacobian" in text):
        return "审计 large-scale Newton PDE solver 的 spectrally safe neural-operator warm start，并检查 Jacobian definiteness、Krylov compatibility、label-free energy fine-tuning 和加速约束。"
    if "newton" in text or "nonlinear solver" in text or "nonlinear system" in text:
        return "评估非线性求解器中的神经预条件方法，并检查收敛性、鲁棒性和不稳定边界。"
    if any(term in text for term in ("krylov", "preconditioner", "linear solver", "linear system")):
        return "评估学习型求解器辅助是否能加速 PDE 线性系统，同时保持收敛性和泛化能力。"
    if "earth system" in text or "esm" in text or "bias correction" in text or "cadence-limited" in text:
        return "审计 Earth-system model 的在线 ML 偏差校正，同时检查稳定性、可迁移性和运行 cadence 约束。"
    if "stabilization" in text or "hji" in text or "lyapunov" in text or "differential game" in text:
        return "评估非线性动力系统在对抗扰动和稳定性约束下的鲁棒安全控制学习。"
    if any(term in text for term in ("lagrangian velocity", "velocimetry-thermometry", "temperature fields")):
        return "验证 turbulent velocity 与 temperature reconstruction 的 physics-informed KAN field-inference 工作流，并检查稀疏 Lagrangian measurements 和 DNS-fidelity 约束。"
    if "solitary wave" in text or "initial-value iterative" in text:
        return "验证 nonlinear wave equation 中 solitary-wave simulation 的 two-stage initial-value iterative PINN，并检查仅初值数据、理论保证和传统求解器对比约束。"
    if "optimal control" in text or "adjoint" in text or "direct vs indirect" in text:
        return "对比 PINN 控制 formulation 与伴随法或最优性系统基线在 PDE 约束控制中的表现。"
    if "high-frequency scaling" in text or ("hfs" in text and "spectral bias" in text):
        return "诊断 convolutional neural operator 的 spectral-bias 高频失效模式，并验证 HFS 控制在多尺度单相/两相流体系统中相对 diffusion-conditioned baseline 的效果。"
    if "diffusion model" in text and ("turbulence" in text or "turbulent" in text):
        return "验证 turbulent-flow spectral fidelity 的 diffusion-corrected neural operator surrogate，并检查高频结构和长周期 rollout 稳定性。"
    if "in-context operator" in text and ("spectral audit" in text or "tangent operator" in text or "jacobian" in text):
        return "用基于 Jacobian 的 spectral tangent-operator 诊断审计 in-context operator network，并检查 local PDE mechanism fidelity、稳定性、敏感性和 prompt consistency，而不只看预测误差。"
    if "domain-unification-free" in text or "discretization decoupling" in text or "discretization-decoupled" in text:
        return "验证 discretization-decoupled generalized operator learning 的 cross-domain neural operator framework，并检查 irregular sampling、spectral mismatch、distribution shift 和任意分辨率查询约束。"
    if "spectral bias" in text or "high-frequency" in text or "frequency-resolved" in text:
        return "诊断 physics-informed 或 operator learning 中的高频失效模式，并测试缓解控制。"
    if "laplace neural operator" in text or "pilno" in text or "virtual inputs" in text or "out-of-distribution" in text:
        return "验证数据稀缺和 OOD 泛化约束下用于 PDE 求解的 physics-informed neural operator surrogate。"
    if ("neurosem" in text or "spectral element" in text or "sem solver" in text) and (
        "pinn" in text or "physics-informed" in text or "multiphysics" in text
    ):
        return "验证混合 PINN 与 spectral-element 工作流在耦合 multiphysics 模拟中的数据同化、求解器集成和湍流鲁棒性。"
    if any(term in text for term in ("crack nucleation", "crack propagation", "brittle", "fracture", "phase-field")):
        return "验证 brittle-fracture crack nucleation 与 propagation 的 DeepONet surrogate，并检查 phase-field 物理约束。"
    if any(term in text for term in ("kolmogorov-arnold", "kkan", "kan", "information bottleneck", "geometric complexity")):
        return "通过逼近行为、learning dynamics 和 signal-to-noise 泛化控制审计 KAN-style SciML 架构。"
    if any(term in text for term in ("spiking", "lif", "qif", "integrate-and-fire", "snn")):
        return "评估可微分 spiking-neuron 模型在 SciML 回归、operator learning 和 PDE 求解中的稳定性。"
    if any(term in text for term in ("diesel engine", "engine health", "maintenance forecasting", "parameter identification", "mean value diesel")):
        return "评估 diesel-engine health monitoring 的 operator-infused PINN digital twin，并检查 transfer learning 和不确定性约束。"
    if ("turbulent" in text or "turbulence" in text) and any(
        term in text for term in ("generative", "super-resolution", "forecasting", "sparse", "reconstruction")
    ):
        return "评估湍流 super-resolution、forecasting 和 sparse reconstruction 的生成式算子模型，避免丢失细尺度结构。"
    if any(term in text for term in ("message passing", "message-passing", "many-body", "renormalized operators", "multiscale attention")):
        return "验证 many-body complex-system 几何与动力学的 message-passing neural operator，并检查可扩展性、稳定性和多尺度保真度约束。"
    if (
        any(term in text for term in ("multi-task", "multitask", "mt-deeponet", "synergistic learning"))
        and ("deeponet" in text or "operator network" in text)
        and ("pde" in text or "partial differential" in text or "geometry" in text)
    ):
        return "验证 efficient PDE problem solving 的 multi-task DeepONet 工作流，并检查任务耦合、几何泛化、精度和训练成本约束。"
    if "operator" in text:
        return "用稳定性和保真度诊断评估神经算子可靠性，而不只看平均预测误差。"
    if "turbulence" in text or "fluid" in text:
        return "评估学习到的流体模型在面向求解器约束下是否稳定并能泛化。"
    if "inverse" in text or "physics-informed" in text or "pinn" in text:
        return "规划 PDE 约束 SciML 工作流，并区分表示、优化和物理残差失败。"
    return "把新的科学机器学习结果作为规划上下文跟踪，不把它当作 evaluator 证据。"
