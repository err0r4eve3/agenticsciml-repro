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
    if "agent" in text or "llm" in text:
        return "Audit long-horizon agent or LLM workflows with verifiable outcomes and explicit failure boundaries."
    if "multiple solutions" in text or "solution multiplicity" in text or "deep ensemble" in text:
        return "Audit PINN discovery of multiple nonlinear ODE/PDE solutions under initialization, ensemble diversity, and solver-refinement constraints."
    if "meta-solver" in text or "meta solver" in text or "multi-objective" in text or "pareto" in text:
        return "Audit automated meta-solver discovery for time-dependent PDEs under accuracy, speed, memory, and preference-selection tradeoffs."
    if "newton" in text or "nonlinear solver" in text or "nonlinear system" in text:
        return "Evaluate neural preconditioning for nonlinear solvers under convergence, robustness, and instability constraints."
    if any(term in text for term in ("krylov", "preconditioner", "linear solver", "linear system")):
        return "Evaluate whether learned solver aids accelerate PDE linear systems while preserving convergence and generalization."
    if "earth system" in text or "esm" in text or "bias correction" in text or "cadence-limited" in text:
        return "Audit online ML bias-correction for Earth-system models under stability, portability, and runtime cadence constraints."
    if "stabilization" in text or "hji" in text or "lyapunov" in text or "differential game" in text:
        return "Assess robust safe-control learning for nonlinear dynamical systems under adversarial disturbances and stability constraints."
    if "optimal control" in text or "adjoint" in text or "direct vs indirect" in text:
        return "Compare PINN control formulations against adjoint or optimality-system baselines for PDE-constrained control."
    if "spectral bias" in text or "high-frequency" in text or "frequency-resolved" in text:
        return "Diagnose high-frequency failure modes in physics-informed or operator learning and test mitigation controls."
    if any(term in text for term in ("sensor location", "sensor placement", "vortex-induced", "marine riser", "deepvivonet")):
        return "Optimize sparse sensor placement for vortex-induced vibration reconstruction and forecasting under transfer-learning constraints."
    if any(term in text for term in ("stiff chemical", "chemical kinetics", "combustion", "reactive transport", "thermochemical", "plug flow reactor", "reactor design")):
        return "Validate multi-output neural operator surrogates for stiff chemical kinetics under conservation, stiffness, and CFD-coupling constraints."
    if any(term in text for term in ("biomedical", "biofluid", "biosolid", "medical imaging", "cell signaling", "physiological", "pharmacology")):
        return "Audit physics-informed biomedical modeling under data scarcity, interpretability, uncertainty, and multiscale physiology constraints."
    if any(term in text for term in ("hypersonic", "supersonic", "reentry", "arbitrary grids", "geometry-dependent")):
        return "Validate data-efficient neural operator surrogates for geometry-dependent hypersonic or supersonic flow prediction on scarce data."
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
    if any(term in text for term in ("optimizer", "optimization", "natural gradient", "bfgs", "broyden", "curvature-aware")):
        return "Audit optimizer and conditioning choices for high-accuracy PINN convergence on challenging PDE or ODE systems."
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
    if "agent" in text or "llm" in text:
        return "审计长周期 agent 或 LLM 工作流，要求结果可验证并显式记录失败边界。"
    if "multiple solutions" in text or "solution multiplicity" in text or "deep ensemble" in text:
        return "审计 PINN 对非线性 ODE/PDE 多解的发现能力，并检查初始化、ensemble 多样性和求解器细化约束。"
    if "meta-solver" in text or "meta solver" in text or "multi-objective" in text or "pareto" in text:
        return "审计 time-dependent PDE 的自动 meta-solver 发现，并权衡精度、速度、内存和偏好选择。"
    if "newton" in text or "nonlinear solver" in text or "nonlinear system" in text:
        return "评估非线性求解器中的神经预条件方法，并检查收敛性、鲁棒性和不稳定边界。"
    if any(term in text for term in ("krylov", "preconditioner", "linear solver", "linear system")):
        return "评估学习型求解器辅助是否能加速 PDE 线性系统，同时保持收敛性和泛化能力。"
    if "earth system" in text or "esm" in text or "bias correction" in text or "cadence-limited" in text:
        return "审计 Earth-system model 的在线 ML 偏差校正，同时检查稳定性、可迁移性和运行 cadence 约束。"
    if "stabilization" in text or "hji" in text or "lyapunov" in text or "differential game" in text:
        return "评估非线性动力系统在对抗扰动和稳定性约束下的鲁棒安全控制学习。"
    if "optimal control" in text or "adjoint" in text or "direct vs indirect" in text:
        return "对比 PINN 控制 formulation 与伴随法或最优性系统基线在 PDE 约束控制中的表现。"
    if "spectral bias" in text or "high-frequency" in text or "frequency-resolved" in text:
        return "诊断 physics-informed 或 operator learning 中的高频失效模式，并测试缓解控制。"
    if any(term in text for term in ("sensor location", "sensor placement", "vortex-induced", "marine riser", "deepvivonet")):
        return "优化 vortex-induced vibration 重建与 forecasting 的稀疏传感器布置，并检查 transfer-learning 约束。"
    if any(term in text for term in ("stiff chemical", "chemical kinetics", "combustion", "reactive transport", "thermochemical", "plug flow reactor", "reactor design")):
        return "验证 stiff chemical kinetics 的多输出神经算子 surrogate，并检查守恒、刚性和 CFD 耦合约束。"
    if any(term in text for term in ("biomedical", "biofluid", "biosolid", "medical imaging", "cell signaling", "physiological", "pharmacology")):
        return "审计 physics-informed biomedical modeling 在数据稀缺、可解释性、不确定性和多尺度生理约束下的可靠性。"
    if any(term in text for term in ("hypersonic", "supersonic", "reentry", "arbitrary grids", "geometry-dependent")):
        return "验证稀缺数据下 geometry-dependent hypersonic 或 supersonic flow 预测的高效神经算子 surrogate。"
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
    if any(term in text for term in ("optimizer", "optimization", "natural gradient", "bfgs", "broyden", "curvature-aware")):
        return "审计 optimizer 与 conditioning 选择对高精度 PINN 在困难 PDE 或 ODE 系统上收敛的影响。"
    if "operator" in text:
        return "用稳定性和保真度诊断评估神经算子可靠性，而不只看平均预测误差。"
    if "turbulence" in text or "fluid" in text:
        return "评估学习到的流体模型在面向求解器约束下是否稳定并能泛化。"
    if "inverse" in text or "physics-informed" in text or "pinn" in text:
        return "规划 PDE 约束 SciML 工作流，并区分表示、优化和物理残差失败。"
    return "把新的科学机器学习结果作为规划上下文跟踪，不把它当作 evaluator 证据。"
