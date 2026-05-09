from pathlib import Path

from agenticsciml.retrieval.kb_store import KnowledgeBase
from agenticsciml.retrieval.lexical import retrieve_top_entry


def test_kb_loads_entries() -> None:
    kb = KnowledgeBase.load(Path("examples/function_approx/kb"))

    entry = kb.get("fourier_features")

    assert entry.title == "Fourier Features"
    assert "oscillation" in entry.description.lower()


def test_lexical_retrieval_is_deterministic() -> None:
    kb = KnowledgeBase.load(Path("examples/function_approx/kb"))

    first = retrieve_top_entry(kb, "discontinuity oscillation instability", threshold=0.0)
    second = retrieve_top_entry(kb, "discontinuity oscillation instability", threshold=0.0)

    assert first is not None
    assert first.entry_id == second.entry_id
    assert first.entry_id in {"mixture_of_experts", "fourier_features", "weighted_loss", "gradient_clipping"}
