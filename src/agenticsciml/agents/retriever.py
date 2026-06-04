from __future__ import annotations

from agenticsciml.agents.base import AgentBase
from agenticsciml.retrieval.kb_store import KnowledgeBase, KnowledgeBaseEntry
from agenticsciml.retrieval.lexical import retrieve_top_entry
from agenticsciml.state import AgentMessage


class RetrieverAgent(AgentBase):
    role = "retriever"

    def retrieve(
        self,
        solution_id: str,
        kb: KnowledgeBase,
        query: str,
        enabled: bool = True,
        random_mode: bool = False,
        random_seed: int = 0,
    ) -> KnowledgeBaseEntry | None:
        self.require_inputs({"solution_id": solution_id, "query": query, "enabled": enabled})
        prompt = (
            "Select at most one KB entry relevant to the parent weakness. "
            f"Mode: {'random' if random_mode else 'lexical'}. Query: {query}"
        )
        if not enabled:
            entry = None
        elif random_mode:
            entry = kb.random_entry(seed=random_seed, salt=query)
        else:
            entry = retrieve_top_entry(kb, query, threshold=1.0)
        response = entry.entry_id if entry else "none"
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, response)])
        self.storage.save_solution_text(solution_id, "retrieval_query.txt", query)
        manifest = kb.manifest()
        self.storage.save_json(
            f"solutions/{solution_id}/retrieved_kb.json",
            {
                "schema_version": 1,
                "enabled": enabled,
                "retrieval_mode": "disabled" if not enabled else "random" if random_mode else "lexical",
                "query": query,
                "selected_entry_id": entry.entry_id if entry else None,
                "selected_entry": entry.provenance() if entry else None,
                "kb_manifest": manifest,
                "paper_kb_equivalent": bool(manifest.get("paper_kb_equivalent")),
                "coverage_status": manifest.get("coverage_status"),
            },
        )
        if entry:
            self.storage.save_solution_text(
                solution_id,
                "retrieved_kb.md",
                f"# Retrieved KB: {entry.title}\n\n{entry.content}\n",
            )
        return entry
