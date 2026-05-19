import json
from pathlib import Path

from agenticsciml.audit_reports import build_kb_application_report
from agenticsciml.agents.retriever import RetrieverAgent
from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.retrieval.kb_store import KnowledgeBase
from agenticsciml.retrieval.lexical import retrieve_top_entry
from agenticsciml.retrieval.query_builder import RetrievalQueryBuilder
from agenticsciml.state import AnalysisReport, Proposal, SolutionNode, SolutionScore
from agenticsciml.storage import ExperimentStorage


def test_kb_loads_entries() -> None:
    kb = KnowledgeBase.load(Path("examples/function_approx/kb"))

    entry = kb.get("fourier_features")

    assert entry.title == "Fourier Features"
    assert "oscillation" in entry.description.lower()
    manifest = kb.manifest()
    assert manifest["coverage_status"] == "local_kb_seed"
    assert manifest["paper_kb_equivalent"] is False
    assert manifest["paper_reference_entry_count"] == 70
    assert manifest["provenance_complete"] is False
    assert manifest["provenance_complete_count"] < manifest["entry_count"]
    assert "fourier_features" in manifest["missing_provenance_entry_ids"]


def test_burgers_kb_loads_source_grounded_entries() -> None:
    kb = KnowledgeBase.load(Path("examples/burgers_pinn/kb"))

    residual = kb.get("pinn_residual_objective")
    collocation = kb.get("collocation_and_scaling")
    budget = kb.get("budgeted_pinn_mutation")

    assert "10.1016/j.jcp.2018.10.045" in residual.content
    assert "continuous_time_inference (Burgers)" in collocation.content
    assert "OpenAI Agents SDK official docs" in budget.content
    assert "guardrail" in budget.description.lower()


def test_kb_application_report_warns_when_budgeted_pinn_entry_is_not_adopted(tmp_path: Path) -> None:
    kb = KnowledgeBase.load(Path("examples/burgers_pinn/kb"))
    entry = kb.get("budgeted_pinn_mutation")
    workspace = tmp_path / "solution_001"
    workspace.mkdir()
    (workspace / "solution.py").write_text("def predict(x):\n    return x\n", encoding="utf-8")
    proposal = Proposal(
        title="Unrelated linear tweak",
        diagnosis="Parent is simple.",
        mutation_plan=["Keep the same linear map."],
        expected_effect="No budgeted PINN change.",
        risks=[],
    )

    report = build_kb_application_report(
        solution_id="solution_001",
        kb_entry=entry,
        proposal=proposal,
        workspace=workspace,
    )

    assert report["retrieved_entry_id"] == "budgeted_pinn_mutation"
    assert report["status"] == "retrieved_only"
    assert report["warnings"]
    assert set(report["static_evidence"]) >= {
        "sample_count",
        "collocation_count",
        "depth_width",
        "residual_weight",
        "training_schedule",
    }


def test_lexical_retrieval_is_deterministic() -> None:
    kb = KnowledgeBase.load(Path("examples/function_approx/kb"))

    first = retrieve_top_entry(kb, "discontinuity oscillation instability", threshold=0.0)
    second = retrieve_top_entry(kb, "discontinuity oscillation instability", threshold=0.0)

    assert first is not None
    assert first.entry_id == second.entry_id
    assert first.entry_id in {"mixture_of_experts", "fourier_features", "weighted_loss", "gradient_clipping"}


def test_retrieval_query_differs_across_benchmarks() -> None:
    parent = SolutionNode(
        node_id="solution_000",
        parent_id=None,
        workspace="/tmp/solution_000",
        score=SolutionScore(metric="relative_l2", value=0.4, higher_is_better=False),
        method_tags=["mlp", "smooth_activation"],
        failure_kind="underfit",
    )
    report = AnalysisReport(
        node_id="solution_000",
        summary="The model smooths sharp residual regions and misses local structure.",
    )
    function_query = RetrievalQueryBuilder.build(
        ProblemBundle.load(Path("examples/function_approx")),
        parent=parent,
        parent_analysis=report,
        leaderboard=[parent],
    )
    poisson_query = RetrievalQueryBuilder.build(
        ProblemBundle.load(Path("examples/poisson_lshape")),
        parent=parent,
        parent_analysis=report,
        leaderboard=[parent],
    )

    assert function_query != poisson_query
    assert "function_approx" in function_query
    assert "poisson_lshape" in poisson_query
    assert "underfit" in function_query
    assert "smooth_activation" in function_query


def test_retriever_respects_no_kb_mode(tmp_path: Path) -> None:
    kb = KnowledgeBase.load(Path("examples/function_approx/kb"))
    storage = ExperimentStorage.create(tmp_path, "no-kb")
    storage.create_solution_workspace("solution_001")
    agent = RetrieverAgent(llm=None, storage=storage)  # type: ignore[arg-type]

    entry = agent.retrieve("solution_001", kb, "discontinuity oscillation", enabled=False)

    assert entry is None
    assert not (storage.run_dir / "solutions" / "solution_001" / "retrieved_kb.md").exists()
    retrieved = storage.run_dir / "solutions" / "solution_001" / "retrieved_kb.json"
    assert retrieved.exists()
    payload = json.loads(retrieved.read_text(encoding="utf-8"))
    assert payload["retrieval_mode"] == "disabled"
    assert payload["paper_kb_equivalent"] is False


def test_retriever_random_kb_is_deterministic(tmp_path: Path) -> None:
    kb = KnowledgeBase.load(Path("examples/function_approx/kb"))
    storage = ExperimentStorage.create(tmp_path, "random-kb")
    storage.create_solution_workspace("solution_001")
    storage.create_solution_workspace("solution_002")
    agent = RetrieverAgent(llm=None, storage=storage)  # type: ignore[arg-type]

    first = agent.retrieve(
        "solution_001",
        kb,
        "same query",
        enabled=True,
        random_mode=True,
        random_seed=17,
    )
    second = agent.retrieve(
        "solution_002",
        kb,
        "same query",
        enabled=True,
        random_mode=True,
        random_seed=17,
    )

    assert first is not None
    assert second is not None
    assert first.entry_id == second.entry_id
    retrieved = storage.run_dir / "solutions" / "solution_001" / "retrieved_kb.json"
    payload = json.loads(retrieved.read_text(encoding="utf-8"))
    assert payload["retrieval_mode"] == "random"
    assert payload["selected_entry"]["entry_id"] == first.entry_id
    assert payload["kb_manifest"]["coverage_status"] == "local_kb_seed"
