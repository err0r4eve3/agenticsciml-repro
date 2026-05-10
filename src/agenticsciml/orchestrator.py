from __future__ import annotations

import json
import shutil
import time
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
from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.config import EvaluationContract, ExperimentConfig
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
from agenticsciml.state import AnalysisReport, Proposal, SolutionNode, SolutionScore
from agenticsciml.storage import ExperimentStorage


class AgenticSciMLOrchestrator:
    def __init__(self, config: ExperimentConfig, llm: LLMClient):
        if config.benchmark_dir is None:
            raise ValueError("ExperimentConfig.benchmark_dir is required.")
        self.config = config
        self.llm = llm
        self.storage = ExperimentStorage.create(config.output_dir, config.experiment_id)
        self.nodes: list[SolutionNode] = []
        self.analysis_by_node: dict[str, AnalysisReport] = {}

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

    def run(self) -> Path:
        started = time.monotonic()
        self.storage.record_trace(
            "workflow_span",
            "agenticsciml.run.start",
            {
                "experiment_id": self.config.experiment_id,
                "benchmark_dir": str(self.config.benchmark_dir),
                "max_iterations": self.config.evolution.max_iterations,
            },
        )
        self.storage.save_json("config.json", self.config.to_dict())
        resumed = self._load_checkpoint_if_requested()
        if resumed:
            contract = self._load_or_create_contract()
        else:
            data_report = self.data_analyst.analyze(self.config.benchmark_dir)
            contract = self.evaluator.create_contract(self.problem_bundle, data_report)

            root = self._create_root(contract, data_report)
            self.nodes.append(root)
            self._save_checkpoint("root_created")

        for _ in range(self.config.evolution.max_iterations):
            parents = self._select_parents()
            for parent in parents:
                if len(parent.children) >= self.config.evolution.max_children_per_node:
                    continue
                child = self._create_child(parent, contract)
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
                "solution_count": len(self.nodes),
                "wall_time_s": time.monotonic() - started,
            },
        )
        write_trace_summary(self.storage.run_dir)
        return self.storage.run_dir

    def _load_or_create_contract(self) -> EvaluationContract:
        contract_path = self.storage.run_dir / "evaluation_contract.json"
        if contract_path.exists():
            return EvaluationContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
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
        self.nodes = [SolutionNode.from_dict(node) for node in payload.get("nodes", [])]
        self.analysis_by_node = self._load_analysis_reports(self.nodes)
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
                "experiment_id": self.config.experiment_id,
                "benchmark_name": self.problem_bundle.benchmark_name,
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

    def _next_solution_id(self) -> str:
        return f"solution_{len(self.nodes):03d}"

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

    def _create_child(self, parent: SolutionNode, contract: EvaluationContract) -> SolutionNode:
        solution_id = self._next_solution_id()
        workspace = self.storage.create_solution_workspace(solution_id)
        prepare_solution_workspace(self.config.benchmark_dir, workspace)

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
        )
        parent_code = self.engineer.read_parent_code(parent_workspace)
        method_tags = self._method_tags(proposal, kb_entry.entry_id if kb_entry else None)
        try:
            self.engineer.mutate(
                solution_id,
                parent_code,
                proposal,
                problem_bundle=self.problem_bundle,
                contract=contract,
                guidelines=self._guidelines_text(),
                parent_analysis=parent_analysis,
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
        if len(self.nodes) < self.config.evolution.parallel_mutations:
            return selected

        best = selected[0] if selected else self._best_node(available)
        llm_selected_ids = self.selector.select(
            candidates=[node.to_dict() for node in available],
            best_node_id=best.node_id,
            max_to_select=self.config.evolution.parallel_mutations,
        )
        by_id = {node.node_id: node for node in available}
        selected_ids = {node.node_id for node in selected}
        for node_id in llm_selected_ids:
            if len(selected) >= self.config.evolution.parallel_mutations:
                break
            node = by_id.get(node_id)
            if node and node.node_id not in selected_ids:
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
                "wall_time_s": time.monotonic() - started,
                "benchmark_name": self.problem_bundle.benchmark_name,
                "solution_count": len(self.nodes),
                "champion": best.node_id,
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
