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
    ) -> KnowledgeBaseEntry | None:
        self.require_inputs({"solution_id": solution_id, "query": query, "enabled": enabled})
        prompt = (
            "Select at most one KB entry relevant to the parent weakness. "
            f"Query: {query}"
        )
        entry = retrieve_top_entry(kb, query, threshold=1.0) if enabled else None
        response = entry.entry_id if entry else "none"
        self._save_messages(solution_id, [AgentMessage(self.role, prompt, response)])
        if entry:
            self.storage.save_solution_text(
                solution_id,
                "retrieved_kb.md",
                f"# Retrieved KB: {entry.title}\n\n{entry.content}\n",
            )
        return entry
