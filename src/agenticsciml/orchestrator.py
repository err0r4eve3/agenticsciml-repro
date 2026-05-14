from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agenticsciml.agents import (
    DataAnalystAgent,
    DebuggerAgent,
    EngineerAgent,
    EvaluatorAgent,
    ProposerAgent,
    ResultAnalystAgent,
    RetrieverAgent,
    RootEngineerAgent,
    SelectorAgent,
)
from agenticsciml.agents.base import StructuredOutputError
from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.config import EvaluationContract, ExperimentConfig
from agenticsciml.evidence import evidence_metadata_for_run
from agenticsciml.execution.sandbox import prepare_solution_workspace, train_and_evaluate
from agenticsciml.llm.base import LLMClient
from agenticsciml.patching import PatchApplicationError
from agenticsciml.retrieval.kb_store import KnowledgeBase
from agenticsciml.retrieval.query_builder import RetrievalQueryBuilder
from agenticsciml.reporting import (
    write_leaderboard,
    write_trace_summary,
    write_tree_json,
    write_tree_mermaid,
)
from agenticsciml.search_policy import SearchPolicy
from agenticsciml.state import (
    AnalysisReport,
    Proposal,
    SOLUTION_TREE_SCHEMA_VERSION,
    SolutionNode,
    SolutionScore,
    format_solution_id,
    solution_id_index,
    validate_solution_node_artifact_paths,
    validate_solution_tree_artifact_payload,
)
from agenticsciml.storage import ExperimentStorage
from agenticsciml.trace_contracts import FanoutTraceMetadata


BRANCH_INTENTS = (
    "features_or_architecture",
    "training_stability",
    "loss_weighting_or_sampling",
    "regularization_or_simplicity",
)


class AgenticSciMLOrchestrator:
    def __init__(self, config: ExperimentConfig, llm: LLMClient):
        if config.benchmark_dir is None:
            raise ValueError("ExperimentConfig.benchmark_dir is required.")
        run_dir = config.output_dir / config.experiment_id
        if not config.resume and run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Use --resume or choose a new experiment-id."
            )
        self.config = config
        self.llm = llm
        self.storage = ExperimentStorage.create(config.output_dir, config.experiment_id)
        self.nodes: list[SolutionNode] = []
        self.analysis_by_node: dict[str, AnalysisReport] = {}
        self._analysis_lock = threading.RLock()

        self.data_analyst = DataAnalystAgent(llm, self.storage)
        self.evaluator = EvaluatorAgent(llm, self.storage)
        self.root_engineer = RootEngineerAgent(llm, self.storage)
        self.retriever = RetrieverAgent(llm, self.storage)
        self.proposer = ProposerAgent(llm, self.storage)
        self.engineer = EngineerAgent(llm, self.storage)
        self.debugger = DebuggerAgent(llm, self.storage)
        self.result_analyst = ResultAnalystAgent(llm, self.storage)
        self.selector = SelectorAgent(llm, self.storage)
        self.problem_bundle = ProblemBundle.load(config.benchmark_dir)
        self.contract: EvaluationContract | None = None
        self.loaded_checkpoint: dict[str, object] | None = None
        self._next_solution_index: int | None = None

    def run(self) -> Path:
        started = time.monotonic()
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.run.start",
            {
                "experiment_id": self.config.experiment_id,
                "run_state": "partial",
                "benchmark_dir": str(self.config.benchmark_dir),
                "max_iterations": self.config.evolution.max_iterations,
                "branch_context_enabled": self.config.evolution.use_branch_context,
                **self._evidence_metadata(),
            },
        )
        self.storage.save_json("config.json", self.config.to_dict())
        resumed = self._load_checkpoint_if_requested()
        if resumed:
            contract = self._load_or_create_contract()
            self.contract = contract
            self._validate_loaded_checkpoint(contract)
        else:
            data_report = self.data_analyst.analyze(self.config.benchmark_dir)
            contract = self.evaluator.create_contract(self.problem_bundle, data_report)
            self.contract = contract

            root = self._create_root(contract, data_report)
            self.nodes.append(root)
            self._save_checkpoint("root_created")

        for _ in range(self.config.evolution.max_iterations):
            parents = self._select_parents()
            for parent, child in self._create_children_for_parents(parents, contract):
                parent.children.append(child.node_id)
                self.nodes.append(child)
                self._save_checkpoint("child_created")

        self._write_reports(started)
        self._save_checkpoint("completed")
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.run.end",
            {
                "experiment_id": self.config.experiment_id,
                "run_state": "exported",
                "solution_count": len(self.nodes),
                "wall_time_s": time.monotonic() - started,
            },
        )
        write_trace_summary(self.storage.run_dir)
        return self.storage.run_dir

    def _load_or_create_contract(self) -> EvaluationContract:
        contract_path = self.storage.run_dir / "evaluation_contract.json"
        if contract_path.exists():
            contract = EvaluationContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
            BenchmarkContractFactory.verify_contract(self.problem_bundle, contract)
            return contract
        if self.config.resume:
            raise ValueError(f"Cannot resume without evaluation contract: {contract_path}")
        data_report = ""
        data_report_path = self.storage.run_dir / "reports" / "data_analysis.md"
        if data_report_path.exists():
            data_report = data_report_path.read_text(encoding="utf-8")
        return self.evaluator.create_contract(self.problem_bundle, data_report)

    def _load_checkpoint_if_requested(self) -> bool:
        if not self.config.resume:
            return False
        checkpoint_path = self.storage.run_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume without checkpoint: {checkpoint_path}")
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.loaded_checkpoint = payload
        node_issues = validate_solution_tree_artifact_payload(payload, context="checkpoint.json")
        node_payloads = payload.get("nodes", [])
        if isinstance(node_payloads, list):
            for node in node_payloads:
                if isinstance(node, dict):
                    node_id = node.get("node_id", "<unknown>")
                    node_issues.extend(
                        validate_solution_node_artifact_paths(
                            node,
                            run_dir=self.storage.run_dir,
                            context=f"checkpoint.json node {node_id}",
                        )
                    )
        if node_issues:
            raise ValueError("Invalid checkpoint solution tree: " + "; ".join(node_issues))
        self.nodes = [SolutionNode.from_dict(node) for node in node_payloads]
        self.analysis_by_node = self._load_analysis_reports(self.nodes)
        self._next_solution_index = self._compute_next_solution_index()
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.resume.loaded",
            {
                "node_count": len(self.nodes),
                "checkpoint_phase": payload.get("phase"),
            },
        )
        return True

    def _load_analysis_reports(self, nodes: list[SolutionNode]) -> dict[str, AnalysisReport]:
        reports: dict[str, AnalysisReport] = {}
        for node in nodes:
            analysis_path = Path(node.analysis_path) if node.analysis_path else Path(node.workspace) / "analysis.md"
            if analysis_path.exists():
                text = analysis_path.read_text(encoding="utf-8")
                summary = text.split("\n\n", 1)[1].strip() if "\n\n" in text else text.strip()
                reports[node.node_id] = AnalysisReport(node_id=node.node_id, summary=summary)
        return reports

    def _save_checkpoint(self, phase: str) -> None:
        self.storage.save_json(
            "checkpoint.json",
            {
                "phase": phase,
                "schema_version": SOLUTION_TREE_SCHEMA_VERSION,
                "experiment_id": self.config.experiment_id,
                "benchmark_name": self.problem_bundle.benchmark_name,
                "contract_hash": self.contract.contract_hash if self.contract else "",
                "nodes": [node.to_dict() for node in self.nodes],
                "analysis_node_ids": sorted(self.analysis_by_node),
            },
        )
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.checkpoint.saved",
            {
                "phase": phase,
                "node_count": len(self.nodes),
            },
        )

    def _validate_loaded_checkpoint(self, contract: EvaluationContract) -> None:
        payload = self.loaded_checkpoint or {}
        if payload.get("benchmark_name") != self.problem_bundle.benchmark_name:
            raise ValueError(
                "Checkpoint benchmark mismatch: "
                f"stored {payload.get('benchmark_name')}, expected {self.problem_bundle.benchmark_name}"
            )
        if payload.get("contract_hash") != contract.contract_hash:
            raise ValueError(
                "Checkpoint contract hash mismatch: "
                f"stored {payload.get('contract_hash')}, expected {contract.contract_hash}"
            )
        for node in self.nodes:
            if node.benchmark_name != contract.benchmark_name:
                raise ValueError(
                    f"Node {node.node_id} benchmark mismatch: "
                    f"{node.benchmark_name} != {contract.benchmark_name}"
                )
            if node.contract_hash != contract.contract_hash:
                raise ValueError(
                    f"Node {node.node_id} contract hash mismatch: "
                    f"{node.contract_hash} != {contract.contract_hash}"
                )

    def _next_solution_id(self) -> str:
        return self._reserve_solution_ids(1)[0]

    def _reserve_solution_ids(self, count: int) -> list[str]:
        if count < 0:
            raise ValueError(f"Solution id reservation count must be non-negative: {count}")
        if count == 0:
            return []
        next_index = (
            self._next_solution_index
            if self._next_solution_index is not None
            else self._compute_next_solution_index()
        )
        reserved: list[str] = []
        occupied = self._occupied_solution_ids()
        while len(reserved) < count:
            solution_id = format_solution_id(next_index)
            next_index += 1
            if solution_id in occupied:
                continue
            reserved.append(solution_id)
            occupied.add(solution_id)
        self._next_solution_index = next_index
        return reserved

    def _compute_next_solution_index(self) -> int:
        indexes: list[int] = []
        for node in self.nodes:
            index = solution_id_index(node.node_id)
            if index is None:
                raise ValueError(f"Cannot allocate solution ids with malformed node_id: {node.node_id}")
            indexes.append(index)
        if self.storage.solutions_dir.exists():
            for workspace in self.storage.solutions_dir.iterdir():
                if not workspace.is_dir():
                    continue
                index = solution_id_index(workspace.name)
                if index is None:
                    if workspace.name.startswith("solution_"):
                        raise ValueError(
                            f"Cannot allocate solution ids with malformed workspace name: {workspace.name}"
                        )
                    continue
                indexes.append(index)
        return max(indexes, default=-1) + 1

    def _occupied_solution_ids(self) -> set[str]:
        occupied = {node.node_id for node in self.nodes}
        if self.storage.solutions_dir.exists():
            occupied.update(
                workspace.name
                for workspace in self.storage.solutions_dir.iterdir()
                if workspace.is_dir() and solution_id_index(workspace.name) is not None
            )
        return occupied

    def _guidelines_text(self) -> str:
        path = self.config.benchmark_dir / "guidelines.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _create_root(self, contract: EvaluationContract, data_report: str | None) -> SolutionNode:
        solution_id = self._next_solution_id()
        workspace = self.storage.create_solution_workspace(solution_id)
        prepare_solution_workspace(self.config.benchmark_dir, workspace)
        self.root_engineer.generate(
            solution_id,
            problem_bundle=self.problem_bundle,
            contract=contract,
            guidelines=self._guidelines_text(),
            data_report=data_report,
        )
        return self._execute_analyze_node(
            solution_id,
            None,
            workspace,
            contract,
            parent_node=None,
            method_tags=["root_baseline"],
        )

    def _create_children_for_parents(
        self,
        parents: list[SolutionNode],
        contract: EvaluationContract,
    ) -> list[tuple[SolutionNode, SolutionNode]]:
        available = [
            parent
            for parent in parents
            if len(parent.children) < self.config.evolution.max_children_per_node
        ]
        if not available:
            return []

        mutation_budget = max(1, self.config.evolution.parallel_mutations)
        selected = self._mutation_parent_slots(available, mutation_budget)
        max_workers = min(len(selected), mutation_budget)
        solution_ids = self._reserve_solution_ids(len(selected))
        jobs = list(zip(selected, solution_ids))
        execution_mode = "parallel" if max_workers > 1 else "sequential"
        batch_started = time.monotonic()
        fanout_trace = FanoutTraceMetadata.from_pairs(
            [(parent.node_id, solution_id) for parent, solution_id in jobs]
        )
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.parallel_children.start",
            {
                "execution_mode": execution_mode,
                "child_count": len(jobs),
                "max_workers": max_workers,
                **fanout_trace.to_dict(),
            },
        )

        branch_contexts = (
            self._branch_contexts(fanout_trace)
            if self.config.evolution.use_branch_context
            else {solution_id: {} for _, solution_id in jobs}
        )

        if max_workers == 1:
            children = [
                (parent, self._run_child_job(parent, contract, solution_id, branch_contexts[solution_id]))
                for parent, solution_id in jobs
            ]
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agenticsciml-child") as executor:
                futures = [
                    executor.submit(
                        self._run_child_job,
                        parent,
                        contract,
                        solution_id,
                        branch_contexts[solution_id],
                    )
                    for parent, solution_id in jobs
                ]
                children = []
                for (parent, solution_id), future in zip(jobs, futures):
                    try:
                        child = future.result()
                    except Exception as exc:  # pragma: no cover - defensive guard for real LLM/tool failures.
                        child = self._failed_child_from_exception(parent, solution_id, contract, exc)
                    children.append((parent, child))

        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.parallel_children.end",
            {
                "execution_mode": execution_mode,
                "child_count": len(children),
                "max_workers": max_workers,
                "duration_s": time.monotonic() - batch_started,
                **fanout_trace.to_dict(),
            },
        )
        return children

    def _mutation_parent_slots(
        self,
        parents: list[SolutionNode],
        mutation_budget: int,
    ) -> list[SolutionNode]:
        slots: list[SolutionNode] = []
        child_counts = {parent.node_id: len(parent.children) for parent in parents}
        while len(slots) < mutation_budget:
            added = False
            for parent in parents:
                if child_counts[parent.node_id] >= self.config.evolution.max_children_per_node:
                    continue
                slots.append(parent)
                child_counts[parent.node_id] += 1
                added = True
                if len(slots) >= mutation_budget:
                    break
            if not added:
                break
        return slots

    def _branch_contexts(self, fanout_trace: FanoutTraceMetadata) -> dict[str, dict[str, object]]:
        contexts: dict[str, dict[str, object]] = {}
        parent_child_ids = fanout_trace.parent_to_children
        parent_branch_offsets: dict[str, int] = {}
        for edge in fanout_trace.parent_child_edges:
            parent_offset = parent_branch_offsets.get(edge.parent_id, 0)
            parent_branch_offsets[edge.parent_id] = parent_offset + 1
            siblings = [
                child_id
                for child_id in parent_child_ids.get(edge.parent_id, [])
                if child_id != edge.child_id
            ]
            branch_intent = BRANCH_INTENTS[parent_offset % len(BRANCH_INTENTS)]
            contexts[edge.child_id] = {
                "fanout_slot_index": edge.slot_index,
                "parent_branch_index": parent_offset,
                "parent_fanout_child_count": len(parent_child_ids.get(edge.parent_id, [])),
                "sibling_branch_ids": siblings,
                "branch_intent": branch_intent,
                "diversity_instruction": (
                    "Prefer a mutation strategy that is meaningfully distinct from sibling "
                    "branches from the same parent while keeping the fixed evaluator contract."
                ),
            }
        return contexts

    def _run_child_job(
        self,
        parent: SolutionNode,
        contract: EvaluationContract,
        solution_id: str,
        branch_context: dict[str, object] | None = None,
    ) -> SolutionNode:
        started = time.monotonic()
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.child_mutation.start",
            {
                "solution_id": solution_id,
                "parent_id": parent.node_id,
                "branch_context_enabled": self.config.evolution.use_branch_context,
                "branch_context": branch_context or {},
            },
        )
        try:
            child = self._create_child(parent, contract, solution_id=solution_id, branch_context=branch_context)
        except Exception as exc:  # pragma: no cover - defensive guard for real LLM/tool failures.
            child = self._failed_child_from_exception(parent, solution_id, contract, exc)
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.child_mutation.end",
            {
                "solution_id": child.node_id,
                "parent_id": parent.node_id,
                "status": child.status,
                "failure_kind": child.failure_kind,
                "branch_context_enabled": self.config.evolution.use_branch_context,
                "branch_context": branch_context or {},
                "duration_s": time.monotonic() - started,
            },
        )
        return child

    def _failed_child_from_exception(
        self,
        parent: SolutionNode,
        solution_id: str,
        contract: EvaluationContract,
        exc: Exception,
    ) -> SolutionNode:
        workspace = self.storage.create_solution_workspace(solution_id)
        self.storage.save_solution_text(
            solution_id,
            "orchestration_error.md",
            f"# Orchestration Error\n\n{type(exc).__name__}: {exc}\n",
        )
        self.storage.save_solution_text(
            solution_id,
            "proposal.md",
            "# Proposal Unavailable\n\nChild creation failed before a proposal could be completed.\n",
        )
        self.storage.record_trace(
            "guardrail_span",
            "child_creation:exception",
            {
                "solution_id": solution_id,
                "parent_id": parent.node_id,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        report = self.result_analyst.analyze(solution_id, workspace)
        with self._analysis_lock:
            self.analysis_by_node[solution_id] = report
        return SolutionNode(
            node_id=solution_id,
            parent_id=parent.node_id,
            workspace=str(workspace),
            score=None,
            status="failed",
            proposal_path=str(workspace / "proposal.md"),
            analysis_path=str(workspace / "analysis.md"),
            error=str(exc),
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
            method_tags=["mutation_error"],
            failure_kind="orchestration_error",
            score_delta_from_parent=None,
            num_debug_attempts=0,
        )

    def _create_child(
        self,
        parent: SolutionNode,
        contract: EvaluationContract,
        solution_id: str | None = None,
        branch_context: dict[str, object] | None = None,
    ) -> SolutionNode:
        solution_id = solution_id or self._next_solution_id()
        workspace = self.storage.create_solution_workspace(solution_id)
        prepare_solution_workspace(self.config.benchmark_dir, workspace)
        branch_context = branch_context or {}
        self.storage.save_json(Path("solutions") / solution_id / "branch_context.json", branch_context)

        parent_workspace = Path(parent.workspace)
        parent_analysis = self.analysis_by_node.get(parent.node_id)
        parent_summary = parent_analysis.summary if parent_analysis else parent.status
        related_reports = self._related_reports(parent)
        kb_text = None
        kb_entry = None
        query = RetrievalQueryBuilder.build(
            self.problem_bundle,
            parent=parent,
            parent_analysis=parent_analysis,
            leaderboard=self.nodes,
        )
        if self.config.evolution.use_kb:
            kb_dir = self.config.benchmark_dir / "kb"
            kb = KnowledgeBase.load(kb_dir)
            kb_entry = self.retriever.retrieve(
                solution_id,
                kb,
                query,
                enabled=True,
                random_mode=self.config.evolution.random_kb,
                random_seed=self.config.evolution.random_seed,
            )
            kb_text = kb_entry.content if kb_entry else None
        else:
            self.retriever.retrieve(solution_id, KnowledgeBase({}), query, enabled=False)

        proposal = self.proposer.debate(
            solution_id=solution_id,
            parent_summary=parent_summary,
            kb_entry=kb_text,
            related_reports=related_reports,
            use_critic=self.config.evolution.use_critic,
            branch_context=branch_context,
        )
        parent_code = self.engineer.read_parent_code(parent_workspace)
        method_tags = self._method_tags(proposal, kb_entry.entry_id if kb_entry else None)
        branch_intent = branch_context.get("branch_intent")
        if isinstance(branch_intent, str) and branch_intent:
            method_tags.append(f"branch:{branch_intent}")
        try:
            self.engineer.mutate(
                solution_id,
                parent_code,
                proposal,
                problem_bundle=self.problem_bundle,
                contract=contract,
                guidelines=self._guidelines_text(),
                parent_analysis=parent_analysis,
                branch_context=branch_context,
            )
        except (PatchApplicationError, StructuredOutputError) as exc:
            self.storage.save_solution_text(
                solution_id,
                "engineering_error.md",
                f"# Engineering Error\n\n{type(exc).__name__}: {exc}\n",
            )
            self.storage.record_trace(
                "guardrail_span",
                "engineer:patch_application",
                {
                    "solution_id": solution_id,
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            report = self.result_analyst.analyze(solution_id, workspace)
            with self._analysis_lock:
                self.analysis_by_node[solution_id] = report
            return SolutionNode(
                node_id=solution_id,
                parent_id=parent.node_id,
                workspace=str(workspace),
                score=None,
                status="failed",
                proposal_path=str(workspace / "proposal.md"),
                analysis_path=str(workspace / "analysis.md"),
                error=str(exc),
                benchmark_name=contract.benchmark_name,
                contract_hash=contract.contract_hash,
                method_tags=method_tags,
                failure_kind="engineering_error",
                score_delta_from_parent=None,
                num_debug_attempts=0,
            )
        return self._execute_analyze_node(
            solution_id,
            parent.node_id,
            workspace,
            contract,
            parent_node=parent,
            method_tags=method_tags,
        )

    def _execute_analyze_node(
        self,
        solution_id: str,
        parent_id: str | None,
        workspace: Path,
        contract: EvaluationContract,
        parent_node: SolutionNode | None = None,
        method_tags: list[str] | None = None,
    ) -> SolutionNode:
        result = train_and_evaluate(workspace, contract, timeout_s=self.config.evolution.timeout_s)
        self.storage.record_trace(
            "tool_span",
            "train_and_evaluate",
            {
                "solution_id": solution_id,
                "exit_code": result.exit_code,
                "duration_s": result.duration_s,
                "timed_out": result.timed_out,
            },
        )
        debug_attempts = 0
        while (
            self.config.evolution.use_debugger
            and result.exit_code != 0
            and debug_attempts < self.config.evolution.max_debug_retries
        ):
            attempt_index = debug_attempts + 1
            try:
                changed = self.debugger.debug(
                    solution_id,
                    workspace,
                    result.stdout + "\n" + result.stderr,
                    problem_bundle=self.problem_bundle,
                    contract=contract,
                    guidelines=self._guidelines_text(),
                    failure_phase=_failure_phase(result.command),
                )
            except (PatchApplicationError, StructuredOutputError) as exc:
                self.storage.save_solution_text(
                    solution_id,
                    "debugger_error.md",
                    f"# Debugger Error\n\n{type(exc).__name__}: {exc}\n",
                )
                self.storage.record_trace(
                    "guardrail_span",
                    "debugger:patch_application",
                    {
                        "solution_id": solution_id,
                        "attempt": attempt_index,
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                debug_attempts = attempt_index
                break
            debug_attempts = attempt_index
            if not changed:
                break
            result = train_and_evaluate(workspace, contract, timeout_s=self.config.evolution.timeout_s)
            self.storage.record_trace(
                "tool_span",
                "train_and_evaluate.retry",
                {
                    "solution_id": solution_id,
                    "retry": attempt_index,
                    "exit_code": result.exit_code,
                    "duration_s": result.duration_s,
                    "timed_out": result.timed_out,
                },
            )

        score = None
        status = "failed"
        error = None
        eval_path = workspace / "eval.json"
        if result.exit_code == 0 and eval_path.exists():
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            score = SolutionScore(
                metric=str(payload["metric"]),
                value=float(payload["score"]),
                higher_is_better=bool(payload.get("higher_is_better", False)),
            )
            status = "evaluated"
        else:
            error = result.stderr or result.stdout[-1000:]

        report = self.result_analyst.analyze(solution_id, workspace)
        with self._analysis_lock:
            self.analysis_by_node[solution_id] = report
        score_delta = None
        if parent_node and parent_node.score and score:
            score_delta = score.value - parent_node.score.value
        return SolutionNode(
            node_id=solution_id,
            parent_id=parent_id,
            workspace=str(workspace),
            score=score,
            status=status,
            proposal_path=str(workspace / "proposal.md"),
            analysis_path=str(workspace / "analysis.md"),
            error=error,
            benchmark_name=contract.benchmark_name,
            contract_hash=contract.contract_hash,
            method_tags=method_tags or [],
            failure_kind=self._failure_kind(result.timed_out, status, error),
            score_delta_from_parent=score_delta,
            num_debug_attempts=debug_attempts,
        )

    def _select_parents(self) -> list[SolutionNode]:
        available = [
            node
            for node in self.nodes
            if len(node.children) < self.config.evolution.max_children_per_node
        ]
        if not available:
            return []
        policy = SearchPolicy(
            max_children_per_node=self.config.evolution.max_children_per_node,
            random_seed=self.config.evolution.random_seed,
            include_random=True,
        )
        selected = policy.select(self.nodes, max_to_select=self.config.evolution.parallel_mutations)
        if len(available) <= self.config.evolution.parallel_mutations:
            return selected

        best = selected[0] if selected else self._best_node(available)
        selected = [best]
        vote_result = self.selector.select_with_votes(
            candidates=[node.to_dict() for node in available],
            best_node_id=best.node_id,
            max_to_select=self.config.evolution.parallel_mutations,
            vote_count=self.config.evolution.selector_vote_count,
        )
        self.storage.record_trace(
            "agent_span",
            "selector_votes",
            {
                "best_node_id": best.node_id,
                "selected_parent_ids": vote_result.selected_parent_ids,
                "vote_counts": vote_result.vote_counts,
                "vote_count": self.config.evolution.selector_vote_count,
            },
        )
        by_id = {node.node_id: node for node in available}
        selected_ids = {node.node_id for node in selected}
        for node_id in vote_result.selected_parent_ids:
            if len(selected) >= self.config.evolution.parallel_mutations:
                break
            node = by_id.get(node_id)
            if node and node.node_id not in selected_ids:
                selected.append(node)
                selected_ids.add(node.node_id)
        for node in policy.select(self.nodes, max_to_select=self.config.evolution.parallel_mutations):
            if len(selected) >= self.config.evolution.parallel_mutations:
                break
            if node.node_id not in selected_ids:
                selected.append(node)
                selected_ids.add(node.node_id)
        return selected[: self.config.evolution.parallel_mutations]

    def _best_node(self, nodes: list[SolutionNode] | None = None) -> SolutionNode:
        candidates = nodes or self.nodes
        best: SolutionNode | None = None
        for node in candidates:
            if node.score is None:
                continue
            if best is None or node.score.better_than(best.score):
                best = node
        return best or candidates[0]

    def _failure_kind(self, timed_out: bool, status: str, error: str | None) -> str | None:
        if status == "evaluated":
            return None
        if timed_out:
            return "timeout"
        text = (error or "").lower()
        if "contract violation" in text or "validate" in text:
            return "contract_error"
        if "syntaxerror" in text:
            return "invalid_code"
        return "runtime_error"

    def _method_tags(self, proposal: Proposal, kb_entry_id: str | None) -> list[str]:
        tags: list[str] = []
        if kb_entry_id:
            tags.append(kb_entry_id)
        text = " ".join(
            [
                proposal.title,
                proposal.diagnosis,
                proposal.expected_effect,
                " ".join(proposal.mutation_plan),
                " ".join(proposal.risks),
            ]
        ).lower()
        keyword_tags = {
            "fourier": "fourier_features",
            "mixture": "mixture_of_experts",
            "expert": "mixture_of_experts",
            "weighted": "weighted_loss",
            "weighting": "weighted_loss",
            "clip": "gradient_clipping",
            "adaptive": "adaptive_activation",
        }
        for keyword, tag in keyword_tags.items():
            if keyword in text and tag not in tags:
                tags.append(tag)
        return tags or ["mutation"]

    def _related_reports(self, parent: SolutionNode) -> list[str]:
        reports: list[str] = []
        if parent.parent_id:
            parent_node = self._node_by_id(parent.parent_id)
            if parent_node:
                for sibling_id in parent_node.children:
                    if sibling_id != parent.node_id:
                        report = self.analysis_by_node.get(sibling_id)
                        if report:
                            reports.append(report.summary)
        report = self.analysis_by_node.get(parent.node_id)
        if report:
            reports.append(report.summary)
        return reports[:4]

    def _node_by_id(self, node_id: str) -> SolutionNode | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def _write_reports(self, started: float) -> None:
        write_tree_json(self.storage.run_dir, self.nodes)
        write_tree_mermaid(self.storage.run_dir, self.nodes)
        write_leaderboard(self.storage.run_dir, self.nodes)
        best = self._best_node()
        champion_dir = self.storage.run_dir / "champion"
        champion_dir.mkdir(exist_ok=True)
        best_workspace = Path(best.workspace)
        for filename in ["solution.py", "analysis.md", "eval.json"]:
            source = best_workspace / filename
            if source.exists():
                shutil.copy2(source, champion_dir / filename)
        self.storage.save_json(
            "run_metadata.json",
            {
                "run_state": "exported",
                "wall_time_s": time.monotonic() - started,
                "benchmark_name": self.problem_bundle.benchmark_name,
                "solution_count": len(self.nodes),
                "champion": best.node_id,
                "branch_context_enabled": self.config.evolution.use_branch_context,
                **self._evidence_metadata(),
                "llm_calls": self._llm_call_summary(),
            },
        )
        self.storage.record_trace(
            "tool_span",
            "export_reports",
            {
                "champion": best.node_id,
                "solution_count": len(self.nodes),
            },
        )

    def _llm_call_summary(self) -> dict[str, object]:
        trace_path = self.storage.run_dir / "trace.jsonl"
        by_role: dict[str, int] = {}
        total = 0
        prompt_token_estimate = 0
        response_token_estimate = 0
        duration_s = 0.0
        if not trace_path.exists():
            return {
                "total": total,
                "by_role": by_role,
                "prompt_token_estimate": prompt_token_estimate,
                "response_token_estimate": response_token_estimate,
                "duration_s": duration_s,
            }
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event_type") != "generation_span":
                continue
            metadata = event.get("metadata", {})
            role = str(metadata.get("spec_role") or event.get("name") or "unknown")
            by_role[role] = by_role.get(role, 0) + 1
            total += 1
            prompt_token_estimate += int(metadata.get("prompt_token_estimate", 0))
            response_token_estimate += int(metadata.get("response_token_estimate", 0))
            duration_s += float(metadata.get("duration_s", 0.0))
        return {
            "total": total,
            "by_role": dict(sorted(by_role.items())),
            "prompt_token_estimate": prompt_token_estimate,
            "response_token_estimate": response_token_estimate,
            "duration_s": duration_s,
        }

    def _evidence_metadata(self) -> dict[str, object]:
        return evidence_metadata_for_run(
            use_mock=self.config.use_mock,
            fidelity_level=self.problem_bundle.benchmark_spec.fidelity_level,
        )


def _failure_phase(command: list[str]) -> str:
    joined = " ".join(command)
    if "--mode=validate" in joined:
        return "validate"
    if "--mode=train" in joined:
        return "train"
    if "--mode=predict" in joined:
        return "predict"
    if "evaluate.py" in joined:
        return "evaluate"
    return "unknown"
