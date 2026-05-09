from __future__ import annotations

import re

from agenticsciml.retrieval.kb_store import KnowledgeBase, KnowledgeBaseEntry


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def retrieve_top_entry(
    kb: KnowledgeBase,
    query: str,
    threshold: float = 1.0,
) -> KnowledgeBaseEntry | None:
    query_tokens = _tokens(query)
    best_entry: KnowledgeBaseEntry | None = None
    best_score = float("-inf")
    for entry in kb.all():
        haystack = _tokens(f"{entry.title} {entry.description} {entry.content}")
        score = len(query_tokens & haystack)
        if score > best_score or (score == best_score and best_entry and entry.entry_id < best_entry.entry_id):
            best_entry = entry
            best_score = float(score)
    if best_entry is None or best_score < threshold:
        return None
    return best_entry
