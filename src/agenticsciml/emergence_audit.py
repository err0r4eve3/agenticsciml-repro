from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agenticsciml.algorithm_catalog import AlgorithmSpec, list_algorithms
from agenticsciml.retrieval.kb_store import KnowledgeBase, KnowledgeBaseEntry
from agenticsciml.state import SolutionNode


AUDITOR_VERSION = "emergence_audit.v1"
DIRECT_OVERLAP_THRESHOLD = 0.34
WEAK_OVERLAP_THRESHOLD = 0.18

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

PRIOR_RESULT_TERMS = {
    "analysis",
    "boundary",
    "delta",
    "error",
    "failure",
    "improve",
    "improved",
    "parent",
    "plateau",
    "residual",
    "score",
    "sibling",
    "weakness",
}


def audit_solution_emergence(
    *,
    node: SolutionNode,
    parent_node: SolutionNode | None,
    root_node: SolutionNode | None,
    benchmark_dir: Path,
    strategy_seed_ids: list[str],
) -> dict[str, Any]:
    workspace = Path(node.workspace)
    artifact_refs = _artifact_refs(workspace)
    proposal_text = _read_text(workspace / "proposal.md")
    analysis_text = _read_text(workspace / "analysis.md")
    engineering_text = _read_text(workspace / "engineering_summary.md")
    strategy_text = "\n".join(
        [
            proposal_text,
            analysis_text,
            engineering_text,
            " ".join(node.method_tags),
        ]
    )
    kb_entries = _load_kb_entries(benchmark_dir)
    algorithms = list_algorithms()
    kb_overlap = _overlap_report(strategy_text, [_kb_source_payload(entry) for entry in kb_entries])
    catalog_overlap = _overlap_report(
        strategy_text,
        [
            _algorithm_source_payload(algorithm, selected=algorithm.algorithm_id in strategy_seed_ids)
            for algorithm in algorithms
        ],
    )
    prior_result_evidence = _prior_result_evidence(strategy_text, parent_node)
    score_evidence = _score_evidence(node, parent_node, root_node)
    policy_fidelity_evidence = _policy_fidelity_evidence(workspace / "policy_fidelity_report.json")
    blocking_gaps = _blocking_gaps(
        node=node,
        proposal_text=proposal_text,
        kb_overlap=kb_overlap,
        catalog_overlap=catalog_overlap,
        prior_result_evidence=prior_result_evidence,
        score_evidence=score_evidence,
        policy_fidelity_evidence=policy_fidelity_evidence,
    )
    claim_level = _claim_level(
        kb_overlap=kb_overlap,
        catalog_overlap=catalog_overlap,
        prior_result_evidence=prior_result_evidence,
        score_evidence=score_evidence,
        policy_fidelity_evidence=policy_fidelity_evidence,
        blocking_gaps=blocking_gaps,
    )
    return {
        "schema_version": 1,
        "auditor_version": AUDITOR_VERSION,
        "node_id": node.node_id,
        "claim_level": claim_level,
        "claim_boundary": (
            "This deterministic audit can only label candidate emergence. "
            "It does not prove paper-level emergent discovery without real LLM "
            "multi-seed ablation, physical/visual consistency review, and "
            "source-grounded external review."
        ),
        "kb_overlap": kb_overlap,
        "catalog_overlap": catalog_overlap,
        "prior_result_evidence": prior_result_evidence,
        "score_evidence": score_evidence,
        "policy_fidelity_evidence": policy_fidelity_evidence,
        "blocking_gaps": blocking_gaps,
        "artifact_refs": artifact_refs,
    }


def _load_kb_entries(benchmark_dir: Path) -> list[KnowledgeBaseEntry]:
    kb_dir = benchmark_dir / "kb"
    if not (kb_dir / "index.json").exists():
        return []
    return KnowledgeBase.load(kb_dir).all()


def _artifact_refs(workspace: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for name in (
        "proposal.md",
        "retrieved_kb.md",
        "critic.md",
        "engineering_summary.md",
        "analysis.md",
        "eval.json",
        "policy_fidelity_report.json",
    ):
        if (workspace / name).exists():
            refs[name] = name
    return refs


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _kb_source_payload(entry: KnowledgeBaseEntry) -> dict[str, str]:
    return {
        "id": entry.entry_id,
        "title": entry.title,
        "text": "\n".join([entry.title, entry.description, entry.content]),
    }


def _algorithm_source_payload(algorithm: AlgorithmSpec, *, selected: bool) -> dict[str, str]:
    payload = algorithm.to_dict()
    text_parts = [
        algorithm.algorithm_id,
        algorithm.name,
        algorithm.family,
        algorithm.description,
        algorithm.claim_boundary,
        algorithm.safety_notes,
        str(payload.get("description_zh", "")),
        " ".join(str(item) for item in payload.get("features", [])),
        " ".join(str(item) for item in payload.get("problem_fit", [])),
    ]
    return {
        "id": algorithm.algorithm_id,
        "title": algorithm.name,
        "text": "\n".join(text_parts),
        "selected": "true" if selected else "false",
    }


def _overlap_report(strategy_text: str, sources: list[dict[str, str]]) -> dict[str, Any]:
    strategy_tokens = _tokens(strategy_text)
    matches: list[dict[str, Any]] = []
    for source in sources:
        source_tokens = _tokens(source["text"])
        score = _overlap_score(strategy_tokens, source_tokens)
        if score > 0:
            matches.append(
                {
                    "id": source["id"],
                    "title": source["title"],
                    "score": round(score, 4),
                    "selected": source.get("selected") == "true",
                }
            )
    matches.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    direct = [item["id"] for item in matches if float(item["score"]) >= DIRECT_OVERLAP_THRESHOLD]
    weak = [item["id"] for item in matches if float(item["score"]) >= WEAK_OVERLAP_THRESHOLD]
    selected_direct = [
        item["id"]
        for item in matches
        if item.get("selected") and float(item["score"]) >= WEAK_OVERLAP_THRESHOLD
    ]
    return {
        "checked_source_count": len(sources),
        "max_score": float(matches[0]["score"]) if matches else 0.0,
        "top_matches": matches[:5],
        "direct_match_entry_ids": direct,
        "weak_match_entry_ids": weak,
        "selected_seed_match_ids": selected_direct,
        "direct_overlap_threshold": DIRECT_OVERLAP_THRESHOLD,
    }


def _tokens(text: str) -> set[str]:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.lower())
    normalized = {token.replace("-", "_").replace("+", "") for token in raw_tokens}
    return {token for token in normalized if token not in STOPWORDS and len(token) >= 3}


def _overlap_score(strategy_tokens: set[str], source_tokens: set[str]) -> float:
    if not strategy_tokens or not source_tokens:
        return 0.0
    return len(strategy_tokens & source_tokens) / min(len(strategy_tokens), len(source_tokens))


def _prior_result_evidence(strategy_text: str, parent_node: SolutionNode | None) -> dict[str, Any]:
    tokens = _tokens(strategy_text)
    signals: list[str] = []
    if parent_node is not None:
        signals.append("has_parent")
    for term in sorted(PRIOR_RESULT_TERMS):
        if term in tokens:
            signals.append(f"text:{term}")
    return {
        "present": parent_node is not None and len(signals) >= 3,
        "signals": signals,
    }


def _score_evidence(
    node: SolutionNode,
    parent_node: SolutionNode | None,
    root_node: SolutionNode | None,
) -> dict[str, Any]:
    score = node.score
    parent_score = parent_node.score if parent_node else None
    root_score = root_node.score if root_node else None
    improved_over_parent = bool(score and parent_score and score.better_than(parent_score))
    improved_over_root = bool(score and root_score and score.better_than(root_score))
    return {
        "status": node.status,
        "metric": score.metric if score else None,
        "score": score.value if score else None,
        "higher_is_better": score.higher_is_better if score else None,
        "parent_score": parent_score.value if parent_score else None,
        "root_score": root_score.value if root_score else None,
        "score_delta_from_parent": node.score_delta_from_parent,
        "improved_over_parent": improved_over_parent,
        "improved_over_root": improved_over_root,
    }


def _policy_fidelity_evidence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "passing": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "passing": bool(payload.get("execution_allowed")),
        "status": payload.get("status"),
    }


def _blocking_gaps(
    *,
    node: SolutionNode,
    proposal_text: str,
    kb_overlap: dict[str, Any],
    catalog_overlap: dict[str, Any],
    prior_result_evidence: dict[str, Any],
    score_evidence: dict[str, Any],
    policy_fidelity_evidence: dict[str, Any],
) -> list[str]:
    gaps: list[str] = []
    if not proposal_text.strip():
        gaps.append("proposal_missing")
    if node.parent_id is None:
        gaps.append("root_solution_not_emergent")
    if node.status != "evaluated":
        gaps.append("solution_not_evaluated")
    if kb_overlap["direct_match_entry_ids"]:
        gaps.append("direct_kb_overlap")
    if catalog_overlap["direct_match_entry_ids"] or catalog_overlap["selected_seed_match_ids"]:
        gaps.append("direct_catalog_or_seed_overlap")
    if not prior_result_evidence["present"]:
        gaps.append("prior_result_evidence_missing")
    if not score_evidence["improved_over_parent"]:
        gaps.append("parent_improvement_missing")
    if not policy_fidelity_evidence["available"]:
        gaps.append("policy_fidelity_missing")
    elif not policy_fidelity_evidence["passing"]:
        gaps.append("policy_fidelity_not_passing")
    return gaps


def _claim_level(
    *,
    kb_overlap: dict[str, Any],
    catalog_overlap: dict[str, Any],
    prior_result_evidence: dict[str, Any],
    score_evidence: dict[str, Any],
    policy_fidelity_evidence: dict[str, Any],
    blocking_gaps: list[str],
) -> str:
    if not blocking_gaps:
        return "candidate_emergent"
    if kb_overlap["direct_match_entry_ids"]:
        return "kb_direct"
    if catalog_overlap["direct_match_entry_ids"] or catalog_overlap["selected_seed_match_ids"]:
        return "catalog_seeded"
    if prior_result_evidence["present"] and score_evidence["improved_over_parent"]:
        return "prior_result_adapted"
    if policy_fidelity_evidence["available"] and not policy_fidelity_evidence["passing"]:
        return "implementation_unfaithful"
    return "not_assessed"
