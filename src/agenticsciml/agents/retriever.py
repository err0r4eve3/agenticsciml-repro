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
        if entry:
            self.storage.save_solution_text(
                solution_id,
                "retrieved_kb.md",
                f"# Retrieved KB: {entry.title}\n\n{entry.content}\n",
            )
        return entry
