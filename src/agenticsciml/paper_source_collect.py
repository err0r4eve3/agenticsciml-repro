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
    if "agent" in text or "llm" in text:
        return "Audit long-horizon agent or LLM workflows with verifiable outcomes and explicit failure boundaries."
    if "operator" in text:
        return "Evaluate neural operator reliability beyond average prediction error using stability and fidelity diagnostics."
    if "turbulence" in text or "fluid" in text:
        return "Assess whether learned fluid models remain stable and generalize under solver-facing constraints."
    if "inverse" in text or "physics-informed" in text or "pinn" in text:
        return "Plan PDE-constrained SciML workflows while separating representation, optimization, and physics-residual failures."
    return "Track a new scientific machine learning result as planning context without treating it as evaluator evidence."


def _real_problem_zh(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if "agent" in text or "llm" in text:
        return "审计长周期 agent 或 LLM 工作流，要求结果可验证并显式记录失败边界。"
    if "operator" in text:
        return "用稳定性和保真度诊断评估神经算子可靠性，而不只看平均预测误差。"
    if "turbulence" in text or "fluid" in text:
        return "评估学习到的流体模型在面向求解器约束下是否稳定并能泛化。"
    if "inverse" in text or "physics-informed" in text or "pinn" in text:
        return "规划 PDE 约束 SciML 工作流，并区分表示、优化和物理残差失败。"
    return "把新的科学机器学习结果作为规划上下文跟踪，不把它当作 evaluator 证据。"
