from __future__ import annotations

import csv
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agenticsciml.algorithm_catalog import ALGORITHM_CLAIM_BOUNDARY, list_algorithms
from agenticsciml.benchmarks import REPO_ROOT, benchmark_for_path, list_benchmarks
from agenticsciml.config import (
    AgentConfig,
    EvolutionConfig,
    ExperimentConfig,
    agent_role_default_model_settings,
)
from agenticsciml.custom_benchmarks import (
    CUSTOM_APPROVAL_SCOPE,
    CUSTOM_BENCHMARK_CLAIM_BOUNDARY,
    CUSTOM_EVIDENCE_LEVEL,
    CUSTOM_EVALUATOR_TRUST_LEVEL,
    CUSTOM_SYNTHESIS_LEVEL,
    create_custom_benchmark_bundle,
)
from agenticsciml.evidence import CLAIM_GATE_BLOCKED
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.orchestrator import AgenticSciMLOrchestrator
from agenticsciml.paper_tasks import list_paper_tasks
from agenticsciml.readiness import build_readiness_report


RunMode = Literal["mock", "real", "dry_run"]
ClaimLevel = Literal["workflow_proxy", "paper_workflow"]
WorkspaceScope = Literal["repo", "account", "run", "solution"]
AssistantMode = Literal["ask", "plan", "agent"]
ReasoningEffort = Literal["low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = ("low", "medium", "high", "xhigh")
DEFAULT_ACCOUNT_ID = "local"
ACCOUNT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
PLANNER_VERSION = "problem_intake_keyword_planner.v1"

PLANNED_AGENT_CALLS = (
    "data_analyst",
    "evaluator",
    "root_engineer",
    "selector",
    "retriever",
    "proposer",
    "critic",
    "engineer",
    "debugger",
    "result_analyst",
)

def _agent_role(role: str, label: str, kind: str) -> dict[str, object]:
    return {
        "role": role,
        "label": label,
        "kind": kind,
        "default_model_settings": agent_role_default_model_settings(role),
    }


AGENT_ROLES: tuple[dict[str, object], ...] = (
    _agent_role("data_analyst", "Data Analyst", "analysis"),
    _agent_role("evaluator", "Evaluator", "contract"),
    _agent_role("root_engineer", "Root Engineer", "generation"),
    _agent_role("retriever", "Retriever", "retrieval"),
    _agent_role("proposer", "Proposer", "planning"),
    _agent_role("critic", "Critic", "review"),
    _agent_role("engineer", "Engineer", "patch"),
    _agent_role("debugger", "Debugger", "repair"),
    _agent_role("result_analyst", "Result Analyst", "analysis"),
    _agent_role("selector", "Selector", "selection"),
)

ASSISTANT_MODE_MODEL_SETTINGS: dict[AssistantMode, dict[str, object]] = {
    "ask": {"reasoning_effort": "medium", "temperature": 0.2},
    "plan": {"reasoning_effort": "high", "temperature": 0.35},
    "agent": {"reasoning_effort": "high", "temperature": 0.1},
}


class AgentModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None


class RunStartRequest(BaseModel):
    benchmark: str = "function_approx"
    benchmark_dir: str | None = None
    mode: RunMode = "mock"
    account_id: str | None = None
    max_iterations: int = Field(default=1, ge=0)
    parallel_mutations: int = Field(default=2, ge=1)
    timeout_s: int = Field(default=60, ge=1)
    output_dir: str = "runs"
    experiment_id: str | None = None
    no_kb: bool = False
    random_kb: bool = False
    random_seed: int = 0
    no_branch_context: bool = False
    target_solution_count: int | None = Field(default=None, ge=1)
    selector_vote_count: int = Field(default=3, ge=1)
    max_children_per_node: int = Field(default=10, ge=1)
    agent_models: dict[str, AgentModelRequest] = Field(default_factory=dict)
    selector_panel: list[AgentModelRequest] = Field(default_factory=list, max_length=16)
    selected_algorithm_ids: list[str] = Field(default_factory=list)
    manual_strategy_locks: list[dict[str, Any]] = Field(default_factory=list)
    branch_context: dict[str, Any] = Field(default_factory=dict)
    problem_intake: dict[str, Any] = Field(default_factory=dict)
    planner_snapshot: dict[str, Any] = Field(default_factory=dict)
    claim_level: ClaimLevel = "workflow_proxy"
    domain_evaluator_approved: bool = False
    domain_reviewer: str | None = Field(default=None, max_length=120)
    domain_review_notes: str | None = Field(default=None, max_length=2000)
    paper_benchmark_approved: bool = False
    auto_approve_evaluation: bool = True
    resume: bool = False
    background: bool = False
    real_confirmed: bool = False


class ProblemIntakeRequest(BaseModel):
    problem_statement: str = Field(min_length=20, max_length=12000)
    requirements: str = Field(default="", max_length=12000)
    evaluation_criteria: str = Field(default="", max_length=8000)
    data_description: str = Field(default="", max_length=8000)
    mode: RunMode = "mock"
    account_id: str | None = None
    target_solution_count: int = Field(default=5, ge=1, le=200)
    parallel_mutations: int = Field(default=2, ge=1, le=32)
    selector_vote_count: int = Field(default=3, ge=1, le=32)
    max_children_per_node: int = Field(default=10, ge=1, le=200)
    selected_algorithm_ids: list[str] = Field(default_factory=list)
    agent_models: dict[str, AgentModelRequest] = Field(default_factory=dict)
    selector_panel: list[AgentModelRequest] = Field(default_factory=list, max_length=16)
    allow_custom_benchmark: bool = False
    claim_level: ClaimLevel = "workflow_proxy"
    domain_evaluator_approved: bool = False
    domain_reviewer: str | None = Field(default=None, max_length=120)
    domain_review_notes: str | None = Field(default=None, max_length=2000)
    paper_benchmark_approved: bool = False


class RunReadinessRequest(BaseModel):
    benchmark: str = "function_approx"
    benchmark_dir: str | None = None
    mode: RunMode = "mock"
    account_id: str | None = None
    target_solution_count: int | None = Field(default=None, ge=1)
    max_iterations: int = Field(default=1, ge=0)
    parallel_mutations: int = Field(default=2, ge=1)
    selector_vote_count: int = Field(default=3, ge=1)
    max_children_per_node: int = Field(default=10, ge=1)
    selector_panel: list[AgentModelRequest] = Field(default_factory=list, max_length=16)
    selected_algorithm_ids: list[str] = Field(default_factory=list)
    manual_strategy_locks: list[dict[str, Any]] = Field(default_factory=list)
    branch_context: dict[str, Any] = Field(default_factory=dict)
    problem_intake: dict[str, Any] = Field(default_factory=dict)
    planner_snapshot: dict[str, Any] = Field(default_factory=dict)
    claim_level: ClaimLevel = "workflow_proxy"
    domain_evaluator_approved: bool = False
    domain_reviewer: str | None = Field(default=None, max_length=120)
    domain_review_notes: str | None = Field(default=None, max_length=2000)
    paper_benchmark_approved: bool = False
    real_confirmed: bool = False


class SolverChatRequest(BaseModel):
    message: str
    active_run_id: str | None = None
    selected_benchmark: str = "function_approx"
    mode: RunMode = "mock"
    assistant_mode: AssistantMode = "ask"
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    workspace_scope: WorkspaceScope = "account"
    account_id: str | None = None
    output_dir: str = "runs"
    target_solution_count: int = Field(default=3, ge=1, le=200)
    parallel_mutations: int = Field(default=1, ge=1, le=32)
    selector_vote_count: int = Field(default=3, ge=1, le=32)
    max_children_per_node: int = Field(default=10, ge=1, le=200)
    selected_algorithm_ids: list[str] = Field(default_factory=list)
    agent_models: dict[str, AgentModelRequest] = Field(default_factory=dict)
    selector_panel: list[AgentModelRequest] = Field(default_factory=list, max_length=16)
    claim_level: ClaimLevel = "workflow_proxy"


class AccountCreateRequest(BaseModel):
    account_id: str = Field(min_length=1, max_length=48)
    display_name: str | None = Field(default=None, max_length=80)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    run_dir: str
    benchmark: str
    mode: str
    status: str
    started_at: float
    ended_at: float | None = None
    error: str | None = None


_RUNS: dict[str, RunRecord] = {}
_RUNS_LOCK = threading.RLock()


def create_app() -> FastAPI:
    app = FastAPI(title="AgenticSciML ChatUI Console", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "repo_root": str(REPO_ROOT)}

    @app.get("/api/benchmarks")
    def benchmarks() -> dict[str, object]:
        specs = list_benchmarks()
        return {"benchmarks": [spec.to_dict() for spec in specs]}

    @app.get("/api/algorithms")
    def algorithms(family: str | None = Query(default=None)) -> dict[str, object]:
        return {
            "algorithms": [spec.to_dict() for spec in list_algorithms(family)],
            "claim_boundary": ALGORITHM_CLAIM_BOUNDARY,
        }

    @app.get("/api/paper-tasks")
    def paper_tasks() -> dict[str, object]:
        return {"tasks": list_paper_tasks()}

    @app.get("/api/agent-roles")
    def agent_roles() -> dict[str, object]:
        return {
            "roles": list(AGENT_ROLES),
            "reasoning_effort_note": (
                "reasoning_effort is persisted for audit and passed to providers "
                "that expose a compatible reasoning_effort parameter."
            ),
        }

    @app.post("/api/problem-intake/plan")
    def problem_intake_plan(request: ProblemIntakeRequest) -> dict[str, object]:
        _validate_agent_model_roles(request.agent_models)
        _validate_algorithm_ids(request.selected_algorithm_ids)
        return _problem_intake_plan_payload(request)

    @app.post("/api/run-readiness/preview")
    def run_readiness_preview(request: RunReadinessRequest) -> dict[str, object]:
        benchmark_dir = _resolve_benchmark_dir(
            request.benchmark,
            request.benchmark_dir,
            account_id=request.account_id,
        )
        return _readiness_report_for_request(request, benchmark_dir)

    @app.get("/api/accounts")
    def accounts() -> dict[str, object]:
        return {"accounts": _list_accounts()}

    @app.post("/api/accounts")
    def create_account(request: AccountCreateRequest) -> dict[str, object]:
        return {"account": _ensure_account(request.account_id, request.display_name)}

    @app.get("/api/runs")
    def runs(
        account_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        base = _resolve_output_dir(output_dir, account_id=account_id)
        records = []
        if base.exists():
            for child in sorted(base.iterdir()):
                if child.is_dir():
                    records.append(_describe_run(child.name, base))
        return {"runs": records}

    @app.post("/api/runs")
    def start_run(request: RunStartRequest) -> dict[str, object]:
        _validate_run_start_request(request)
        run_id = request.experiment_id or _default_experiment_id(request.mode)
        output_dir = _resolve_output_dir(request.output_dir, account_id=request.account_id)
        benchmark_dir = _resolve_benchmark_dir(
            request.benchmark,
            request.benchmark_dir,
            account_id=request.account_id,
        )
        benchmark_spec = benchmark_for_path(benchmark_dir)
        benchmark_name = benchmark_spec.name if benchmark_spec else benchmark_dir.name
        run_dir = output_dir / run_id
        readiness_report = _readiness_report_for_request(request, benchmark_dir)
        claim_gate = readiness_report.get("claim_gate") if isinstance(readiness_report, dict) else {}
        if isinstance(claim_gate, dict) and claim_gate.get("status") == CLAIM_GATE_BLOCKED:
            raise HTTPException(
                status_code=400,
                detail="claim gate blocked launch: " + "; ".join(str(item) for item in claim_gate.get("reasons", [])),
            )

        if request.mode == "dry_run":
            run_budget = _effective_run_budget(request)
            record = RunRecord(
                run_id=run_id,
                run_dir=str(run_dir),
                benchmark=benchmark_name,
                mode=request.mode,
                status="dry_run",
                started_at=time.time(),
                ended_at=time.time(),
            )
            _store_record(record)
            return {
                "run_id": run_id,
                "status": "dry_run",
                "run_dir": str(run_dir),
                "planned_agent_calls": list(PLANNED_AGENT_CALLS),
                "run_budget": run_budget,
                "agent_models": {
                    role: config.to_dict()
                    for role, config in _agent_configs_from_request(request).items()
                },
                "selector_panel": [
                    config.to_dict()
                    for config in _selector_panel_from_request(request)
                ],
                "selected_algorithm_ids": _normalized_algorithm_ids(request.selected_algorithm_ids),
                "readiness_report": readiness_report,
                "claim_gate": readiness_report.get("claim_gate"),
            }

        record = RunRecord(
            run_id=run_id,
            run_dir=str(run_dir),
            benchmark=benchmark_name,
            mode=request.mode,
            status="running",
            started_at=time.time(),
        )
        _store_record(record)

        if request.background:
            thread = threading.Thread(
                target=_run_orchestrator,
                args=(request, run_id, benchmark_dir, output_dir, record, readiness_report),
                daemon=True,
            )
            thread.start()
            return {"run_id": run_id, "status": "running", "run_dir": str(run_dir)}

        _run_orchestrator(request, run_id, benchmark_dir, output_dir, record, readiness_report)
        return _describe_run(run_id, output_dir)

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: str, request: RunStartRequest) -> dict[str, object]:
        resume_request = _merge_resume_request(run_id, request)
        return start_run(resume_request)

    @app.get("/api/runs/{run_id}")
    def get_run(
        run_id: str,
        account_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        return _describe_run(run_id, _resolve_output_dir(output_dir, account_id=account_id))

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str,
        account_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
        follow: bool = Query(default=True),
        timeout_s: float = Query(default=30.0, ge=0.1, le=300.0),
    ) -> StreamingResponse:
        run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir, account_id=account_id))
        return StreamingResponse(
            _stream_trace_events(run_id, run_dir, follow=follow, timeout_s=timeout_s),
            media_type="text/event-stream",
        )

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def get_artifact(
        run_id: str,
        artifact_path: str,
        account_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir, account_id=account_id))
        target = _safe_run_child(run_dir, artifact_path)
        if target.is_dir():
            return {
                "path": artifact_path,
                "kind": "directory",
                "entries": _artifact_entries(target, run_dir),
            }
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        content = _read_text_artifact(target)
        return {
            "path": artifact_path,
            "kind": "file",
            "size_bytes": target.stat().st_size,
            "content": content,
        }

    @app.get("/api/runs/{run_id}/selector-votes")
    def selector_votes(
        run_id: str,
        account_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir, account_id=account_id))
        return _selector_votes_payload(run_id, run_dir)

    @app.get("/api/runs/{run_id}/solutions")
    def run_solutions(
        run_id: str,
        account_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir, account_id=account_id))
        return _solutions_payload(run_id, run_dir)

    @app.get("/api/code-server/url")
    def code_server_url(
        scope: WorkspaceScope = Query(default="repo"),
        account_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
        solution_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        return _code_server_payload(
            scope,
            account_id=account_id,
            run_id=run_id,
            solution_id=solution_id,
            output_dir=output_dir,
        )

    @app.get("/api/code-server/workspaces")
    def code_server_workspaces(
        account_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        return {"workspaces": _code_server_workspaces(account_id=account_id, run_id=run_id, output_dir=output_dir)}

    @app.get("/api/solver/settings")
    def solver_settings() -> dict[str, object]:
        return _solver_settings_payload()

    @app.post("/api/solver/chat")
    def solver_chat(request: SolverChatRequest) -> dict[str, object]:
        return _solver_chat_response(request)

    frontend_dist = REPO_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app

def _run_orchestrator(
    request: RunStartRequest,
    run_id: str,
    benchmark_dir: Path,
    output_dir: Path,
    record: RunRecord,
    readiness_report: dict[str, object],
) -> None:
    try:
        run_budget = _effective_run_budget(request)
        evolution = EvolutionConfig(
            max_iterations=int(run_budget["max_iterations"]),
            parallel_mutations=int(run_budget["parallel_mutations"]),
            max_children_per_node=request.max_children_per_node,
            timeout_s=request.timeout_s,
            use_kb=not request.no_kb,
            random_kb=request.random_kb,
            random_seed=request.random_seed,
            use_branch_context=not request.no_branch_context,
            selector_vote_count=request.selector_vote_count,
        )
        config = ExperimentConfig(
            experiment_id=run_id,
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            evolution=evolution,
            use_mock=request.mode == "mock",
            agents=_agent_configs_from_request(request),
            selector_panel=_selector_panel_from_request(request),
            strategy_seed_ids=_normalized_algorithm_ids(request.selected_algorithm_ids),
            problem_intake=_normalized_mapping(request.problem_intake),
            planner_snapshot=_normalized_mapping(request.planner_snapshot),
            readiness_report=dict(readiness_report),
            claim_level=request.claim_level,
            domain_evaluator_approved=request.domain_evaluator_approved,
            domain_reviewer=request.domain_reviewer,
            domain_review_notes=request.domain_review_notes,
            paper_benchmark_approved=request.paper_benchmark_approved,
            auto_approve_evaluation=request.auto_approve_evaluation,
            resume=request.resume,
        )
        llm = MockLLMClient() if request.mode == "mock" else OpenAIAdapter()
        run_dir = AgenticSciMLOrchestrator(config, llm).run()
        record.run_dir = str(run_dir)
        record.status = "completed"
        record.ended_at = time.time()
    except Exception as exc:
        record.status = "failed"
        record.ended_at = time.time()
        record.error = str(exc)
    finally:
        _store_record(record)


def _effective_run_budget(request: RunStartRequest | RunReadinessRequest) -> dict[str, object]:
    parallel_mutations = request.parallel_mutations
    max_iterations = request.max_iterations
    if request.target_solution_count is not None:
        target_children = max(0, request.target_solution_count - 1)
        max_iterations = math.ceil(target_children / parallel_mutations) if target_children else 0
    planned_solution_budget = 1 + (max_iterations * parallel_mutations)
    return {
        "target_solution_count": request.target_solution_count,
        "planned_solution_budget": planned_solution_budget,
        "max_iterations": max_iterations,
        "parallel_mutations": parallel_mutations,
        "selector_vote_count": request.selector_vote_count,
        "max_children_per_node": request.max_children_per_node,
        "selector_panel_member_count": len(request.selector_panel),
    }


def _readiness_report_for_request(
    request: RunStartRequest | RunReadinessRequest,
    benchmark_dir: Path,
) -> dict[str, object]:
    benchmark_spec = benchmark_for_path(benchmark_dir)
    if benchmark_spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown benchmark: {benchmark_dir}")
    return build_readiness_report(
        benchmark=benchmark_spec,
        algorithms=list(list_algorithms()),
        selected_algorithm_ids=_normalized_algorithm_ids(request.selected_algorithm_ids),
        mode=request.mode,
        run_budget=_effective_run_budget(request),
        problem_intake=_normalized_mapping(request.problem_intake),
        planner_snapshot=_normalized_mapping(request.planner_snapshot),
        manual_strategy_locks=list(request.manual_strategy_locks),
        branch_context=_normalized_mapping(request.branch_context),
        selector_panel=_selector_panel_payload_from_models(request.selector_panel),
        claim_level=request.claim_level,
        domain_evaluator_approved=request.domain_evaluator_approved,
        domain_reviewer=request.domain_reviewer,
        domain_review_notes=request.domain_review_notes,
        paper_benchmark_approved=request.paper_benchmark_approved,
        real_confirmed=request.real_confirmed,
        real_mode_enabled=_real_web_runs_enabled(),
    )


def _problem_intake_plan_payload(request: ProblemIntakeRequest) -> dict[str, object]:
    text = _intake_text(request)
    benchmark_candidates = _rank_benchmarks_for_problem(text)
    recommended = benchmark_candidates[0]["benchmark"]
    recommended_name = str(recommended["name"])
    algorithm_rankings = _rank_algorithms_for_problem(
        text,
        recommended_name,
        request.selected_algorithm_ids,
    )
    selected_algorithm_ids = [
        str(item["algorithm"]["id"])
        for item in algorithm_rankings
        if item["selected"]
    ][:6]
    max_iterations = math.ceil(
        max(0, request.target_solution_count - 1) / request.parallel_mutations
    )
    run_budget = {
        "target_solution_count": request.target_solution_count,
        "planned_solution_budget": 1 + max_iterations * request.parallel_mutations,
        "max_iterations": max_iterations,
        "parallel_mutations": request.parallel_mutations,
        "selector_vote_count": request.selector_vote_count,
        "max_children_per_node": request.max_children_per_node,
    }
    warnings = [
        (
            "This planner maps the problem to the current local benchmark catalog; "
            "custom mode creates only an auto-generated proxy evaluator."
        )
        if request.allow_custom_benchmark
        else "This planner maps the problem to the current local benchmark catalog; it does not create a new evaluator.",
        "Selected algorithms are strategy seeds for prompts and audit, not proven implementations.",
    ]
    if request.mode == "real":
        warnings.append("Real mode still requires Web API real_confirmed=true and server-side real-mode enablement.")
    problem_intake = _problem_intake_snapshot(request)
    planner_snapshot = _planner_snapshot(
        request,
        recommended=recommended,
        benchmark_candidates=benchmark_candidates,
        algorithm_rankings=algorithm_rankings,
        selected_algorithm_ids=selected_algorithm_ids,
        run_budget={**run_budget, "mode": request.mode},
    )
    custom_problem_package = None
    action_payload: dict[str, object] = {
        "benchmark": recommended_name,
        "mode": request.mode,
        "target_solution_count": request.target_solution_count,
        "max_iterations": max_iterations,
        "parallel_mutations": request.parallel_mutations,
        "selector_vote_count": request.selector_vote_count,
        "max_children_per_node": request.max_children_per_node,
        "selected_algorithm_ids": selected_algorithm_ids,
        "problem_intake": problem_intake,
        "planner_snapshot": planner_snapshot,
        "claim_level": request.claim_level,
        "domain_evaluator_approved": request.domain_evaluator_approved,
        "domain_reviewer": request.domain_reviewer,
        "domain_review_notes": request.domain_review_notes,
        "paper_benchmark_approved": request.paper_benchmark_approved,
        "agent_models": {
            role: config.to_dict()
            for role, config in _agent_configs_from_problem_request(request).items()
        },
        "selector_panel": _selector_panel_payload_from_models(request.selector_panel),
        "background": True,
    }
    if request.account_id:
        action_payload["account_id"] = _resolve_account_id(request.account_id)
    run_allowed = True
    if request.allow_custom_benchmark:
        custom_problem_package = _custom_problem_package(
            request,
            benchmark_candidates=benchmark_candidates,
            algorithm_rankings=algorithm_rankings,
        )
        action_payload["benchmark"] = str(custom_problem_package["benchmark"])
        action_payload["auto_approve_evaluation"] = True
        planner_snapshot["generated_custom_benchmark"] = {
            "benchmark": custom_problem_package["benchmark"],
            "benchmark_dir": custom_problem_package["benchmark_dir"],
            "status": custom_problem_package["status"],
            "fidelity_level": "proxy",
            "synthesis_level": custom_problem_package["synthesis_level"],
            "evidence_level": custom_problem_package["evidence_level"],
            "approval_scope": custom_problem_package["approval_scope"],
            "evaluator_trust_level": custom_problem_package["evaluator_trust_level"],
            "domain_evaluator_present": custom_problem_package["domain_evaluator_present"],
            "metric_validated_by_domain_expert": custom_problem_package[
                "metric_validated_by_domain_expert"
            ],
            "paper_benchmark_equivalent": custom_problem_package["paper_benchmark_equivalent"],
            "requires_replacement_for_scientific_claim": custom_problem_package[
                "requires_replacement_for_scientific_claim"
            ],
            "domain_evidence_review_required": custom_problem_package["domain_evidence_review_required"],
            "paper_level_claim_supported": custom_problem_package["paper_level_claim_supported"],
        }
        planner_snapshot["claim_boundary"] = (
            "Custom problem intake generated a deterministic workflow-proxy evaluator scaffold. "
            "The resulting custom proxy benchmark bundle can run as workflow proxy evidence only; domain review is "
            "required before any scientific or paper-level claim."
        )
        warnings.append(
            "Created a workflow-proxy evaluator scaffold for workflow testing; domain review is required before scientific claims."
        )
    actions: list[dict[str, object]] = [
        {
            "type": "start_run",
            "payload": action_payload,
        }
    ]
    return {
        "problem_summary": _compact_summary(request.problem_statement),
        "problem_intake": problem_intake,
        "planner_snapshot": planner_snapshot,
        "recommended_benchmark": recommended,
        "benchmark_candidates": benchmark_candidates[:6],
        "algorithm_rankings": algorithm_rankings[:8],
        "selected_algorithm_ids": selected_algorithm_ids,
        "run_config": {
            **run_budget,
            "mode": request.mode,
        },
        "agent_models": {
            role: config.to_dict()
            for role, config in _agent_configs_from_problem_request(request).items()
        },
        "selector_panel": _selector_panel_payload_from_models(request.selector_panel),
        "custom_problem_package": custom_problem_package,
        "run_allowed": run_allowed,
        "actions": actions,
        "warnings": warnings,
        "claim_boundary": (
            "Planner recommendations and strategy seeds are not scientific evidence. "
            "Only completed run artifacts, evaluator scores, trace summaries, and tests support local claims."
        ),
    }


def _problem_intake_snapshot(request: ProblemIntakeRequest) -> dict[str, object]:
    return {
        "schema_version": 1,
        "problem_statement": request.problem_statement,
        "requirements": request.requirements,
        "evaluation_criteria": request.evaluation_criteria,
        "data_description": request.data_description,
        "problem_summary": _compact_summary(request.problem_statement),
    }


def _planner_snapshot(
    request: ProblemIntakeRequest,
    *,
    recommended: dict[str, object],
    benchmark_candidates: list[dict[str, object]],
    algorithm_rankings: list[dict[str, object]],
    selected_algorithm_ids: list[str],
    run_budget: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "planner_version": PLANNER_VERSION,
        "recommended_benchmark": recommended,
        "benchmark_candidates": benchmark_candidates[:6],
        "algorithm_rankings": algorithm_rankings[:12],
        "manual_selected_algorithm_ids": _normalized_algorithm_ids(request.selected_algorithm_ids),
        "selected_algorithm_ids": list(selected_algorithm_ids),
        "selected_seed_snapshot": _algorithm_seed_snapshot(selected_algorithm_ids),
        "run_config": dict(run_budget),
        "claim_boundary": (
            "Problem-intake planning is a controlled mapping to local benchmark and strategy seed catalogs. "
            "It is not evaluator synthesis and is not scientific evidence."
        ),
    }


def _custom_problem_package(
    request: ProblemIntakeRequest,
    *,
    benchmark_candidates: list[dict[str, object]],
    algorithm_rankings: list[dict[str, object]],
) -> dict[str, object]:
    selected_algorithms = [
        item["algorithm"]
        for item in algorithm_rankings
        if item.get("selected")
    ][:6]
    bundle = create_custom_benchmark_bundle(
        _custom_benchmarks_dir(request.account_id),
        problem_statement=request.problem_statement,
        requirements=request.requirements,
        evaluation_criteria=request.evaluation_criteria,
        data_description=request.data_description,
    )
    return {
        "schema_version": 1,
        "status": "custom_proxy_benchmark_scaffolded",
        "synthesis_level": CUSTOM_SYNTHESIS_LEVEL,
        "evidence_level": CUSTOM_EVIDENCE_LEVEL,
        "approval_scope": CUSTOM_APPROVAL_SCOPE,
        "evaluator_trust_level": CUSTOM_EVALUATOR_TRUST_LEVEL,
        "domain_evaluator_present": False,
        "metric_validated_by_domain_expert": False,
        "paper_benchmark_equivalent": False,
        "requires_replacement_for_scientific_claim": True,
        "run_allowed": True,
        "approval_required": False,
        "workflow_run_approval_required": False,
        "domain_evidence_review_required": True,
        "paper_level_claim_supported": False,
        "scientific_claim_supported": False,
        "human_review_recommended": True,
        "benchmark": bundle.benchmark,
        "benchmark_dir": str(bundle.benchmark_dir),
        "problem_summary": _compact_summary(request.problem_statement),
        "created_files": list(bundle.files),
        "required_files": [
            "Problem.md",
            "Requirements.md",
            "Evaluation.md",
            "Data_config.json",
            "Benchmark_spec.json",
            "evaluator_synthesis.json",
            "evaluator_synthesis.md",
            "generate_data.py",
            "evaluate.py",
            "guidelines.md",
            "eda/data_eda.py",
            "eda/data_eda_seed0.json",
            "eda/data_overview_seed0.svg",
        ],
        "evaluator_contract_requirements": [
            "Define deterministic train and validation data generation or checked-in data artifacts.",
            "Keep validation labels and validation file paths evaluator-only.",
            "Expose a prediction-only evaluate.py contract with a private metric.",
            "Add benchmark fidelity metadata and paper-gap notes before any run.",
            "Treat the generated evaluator scaffold as proxy workflow evidence until a human replaces or approves the domain metric.",
        ],
        "nearest_catalog_candidates": benchmark_candidates[:3],
        "strategy_seed_suggestions": selected_algorithms,
        "claim_boundary": CUSTOM_BENCHMARK_CLAIM_BOUNDARY,
    }


def _algorithm_seed_snapshot(algorithm_ids: list[str]) -> list[dict[str, object]]:
    algorithms_by_id = {algorithm.algorithm_id: algorithm for algorithm in list_algorithms()}
    snapshots: list[dict[str, object]] = []
    for algorithm_id in _normalized_algorithm_ids(algorithm_ids):
        algorithm = algorithms_by_id.get(algorithm_id)
        if algorithm is None:
            continue
        payload = algorithm.to_dict()
        snapshots.append(
            {
                "id": payload["id"],
                "name": payload["name"],
                "family": payload["family"],
                "status": payload["status"],
                "compatible_benchmark_families": payload["compatible_benchmark_families"],
                "benchmark_examples": payload["benchmark_examples"],
                "description": payload["description"],
                "claim_boundary": payload["claim_boundary"],
                "safety_notes": payload["safety_notes"],
                "source_scope": payload.get("source_scope"),
                "implementation_path": payload.get("implementation_path"),
            }
        )
    return snapshots


def _agent_configs_from_problem_request(request: ProblemIntakeRequest) -> dict[str, AgentConfig]:
    return {
        role: AgentConfig(
            role=role,
            model=agent_model.model.strip(),
            temperature=agent_model.temperature,
            reasoning_effort=agent_model.reasoning_effort,
        )
        for role, agent_model in request.agent_models.items()
    }


def _selector_panel_payload_from_models(models: list[AgentModelRequest]) -> list[dict[str, object]]:
    return [
        {
            "model": model.model.strip(),
            "temperature": model.temperature,
            "reasoning_effort": model.reasoning_effort,
        }
        for model in models
    ]


def _intake_text(request: ProblemIntakeRequest) -> str:
    return " ".join(
        [
            request.problem_statement,
            request.requirements,
            request.evaluation_criteria,
            request.data_description,
        ]
    ).lower()


def _compact_summary(text: str, limit: int = 360) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _rank_benchmarks_for_problem(text: str) -> list[dict[str, object]]:
    ranked = []
    for benchmark in list_benchmarks():
        score = 0
        reasons: list[str] = []
        haystack = " ".join(
            [
                benchmark.name,
                benchmark.paper_section,
                benchmark.paper_task_name,
                benchmark.family,
                benchmark.metric,
                benchmark.description,
            ]
        ).lower()
        for keyword, weight in _benchmark_keywords(benchmark.name):
            if keyword in text:
                score += weight
                reasons.append(keyword)
        score += _token_overlap_score(text, haystack, weight=2)
        if benchmark.fidelity_level == "faithful-small":
            score += 3
        ranked.append(
            {
                "benchmark": benchmark.to_dict(),
                "score": score,
                "rationale": (
                    "Matched " + ", ".join(sorted(set(reasons))[:6])
                    if reasons
                    else "Fallback catalog candidate"
                ),
            }
        )
    return sorted(ranked, key=lambda item: (-int(item["score"]), str(item["benchmark"]["name"])))


def _rank_algorithms_for_problem(
    text: str,
    benchmark_name: str,
    manual_algorithm_ids: list[str],
) -> list[dict[str, object]]:
    manual = set(manual_algorithm_ids)
    ranked = []
    for algorithm in list_algorithms():
        payload = algorithm.to_dict()
        score = 0
        reasons: list[str] = []
        if algorithm.algorithm_id in manual:
            score += 100
            reasons.append("manual")
        if benchmark_name in algorithm.benchmark_examples:
            score += 35
            reasons.append("benchmark_example")
        haystack = " ".join(
            [
                algorithm.algorithm_id,
                algorithm.name,
                algorithm.family,
                algorithm.description,
                " ".join(algorithm.compatible_benchmark_families),
                " ".join(algorithm.benchmark_examples),
            ]
        ).lower()
        overlap = _token_overlap_score(text, haystack, weight=3)
        if overlap:
            score += overlap
            reasons.append("problem_text")
        selected = algorithm.algorithm_id in manual
        ranked.append(
            {
                "algorithm": payload,
                "score": score,
                "selected": selected,
                "source": "manual" if algorithm.algorithm_id in manual else "auto",
                "rationale": ", ".join(reasons) if reasons else "Catalog fallback",
            }
        )
    auto_ranked = sorted(ranked, key=lambda item: (-int(item["score"]), str(item["algorithm"]["id"])))
    selected_count = sum(1 for item in auto_ranked if item["selected"])
    for item in auto_ranked:
        if selected_count >= 4:
            break
        if item["score"] > 0 and not item["selected"]:
            item["selected"] = True
            selected_count += 1
    return auto_ranked


def _benchmark_keywords(benchmark_name: str) -> tuple[tuple[str, int], ...]:
    if "cylinder" in benchmark_name:
        return (
            ("cylinder", 18),
            ("wake", 18),
            ("sensor", 14),
            ("sparse", 10),
            ("vorticity", 14),
            ("reconstruction", 12),
            ("shred", 10),
        )
    if "reaction_diffusion" in benchmark_name:
        return (
            ("reaction", 16),
            ("diffusion", 16),
            ("spatiotemporal", 12),
            ("source", 8),
            ("operator", 8),
            ("fno", 8),
        )
    if "antiderivative" in benchmark_name:
        return (
            ("antiderivative", 18),
            ("integral", 14),
            ("operator", 10),
            ("deeponet", 8),
            ("function-to-function", 8),
        )
    if "burgers" in benchmark_name:
        return (
            ("burgers", 20),
            ("viscous", 10),
            ("pinn", 10),
            ("initial condition", 8),
            ("boundary condition", 8),
        )
    if "poisson" in benchmark_name:
        return (
            ("poisson", 20),
            ("l-shaped", 16),
            ("laplace", 8),
            ("boundary", 8),
            ("pinn", 8),
            ("residual", 8),
        )
    return (
        ("function", 8),
        ("approximation", 10),
        ("discontinuous", 16),
        ("piecewise", 12),
        ("oscillatory", 8),
        ("regression", 8),
    )


def _token_overlap_score(text: str, haystack: str, *, weight: int) -> int:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9_+-]{4,}", text)
        if token not in {"with", "from", "that", "this", "into", "using", "need", "must"}
    }
    haystack_tokens = set(re.findall(r"[a-z0-9_+-]{4,}", haystack))
    return len(tokens & haystack_tokens) * weight


def _store_record(record: RunRecord) -> None:
    with _RUNS_LOCK:
        _RUNS[_record_key(record.run_dir)] = record


def _validate_run_start_request(request: RunStartRequest) -> None:
    _validate_agent_model_roles(request.agent_models)
    _validate_algorithm_ids(request.selected_algorithm_ids)
    if request.mode == "real":
        if not request.real_confirmed:
            raise HTTPException(
                status_code=400,
                detail="real mode requires real_confirmed=true on the run request",
            )
        if not _real_web_runs_enabled():
            raise HTTPException(
                status_code=403,
                detail="real mode is disabled for the Web API; set AGENTICSCIML_ENABLE_REAL_WEB_RUNS=1 on the server",
            )


def _validate_agent_model_roles(agent_models: dict[str, AgentModelRequest]) -> None:
    known_roles = {str(role["role"]) for role in AGENT_ROLES}
    unknown_roles = sorted(set(agent_models) - known_roles)
    if unknown_roles:
        raise HTTPException(
            status_code=400,
            detail="Unknown agent role override(s): " + ", ".join(unknown_roles),
        )


def _validate_algorithm_ids(algorithm_ids: list[str]) -> None:
    known_ids = {algorithm.algorithm_id for algorithm in list_algorithms()}
    unknown_ids = sorted(set(algorithm_ids) - known_ids)
    if unknown_ids:
        raise HTTPException(
            status_code=400,
            detail="Unknown algorithm id(s): " + ", ".join(unknown_ids),
        )


def _normalized_algorithm_ids(algorithm_ids: list[str]) -> list[str]:
    normalized: list[str] = []
    for algorithm_id in algorithm_ids:
        if algorithm_id not in normalized:
            normalized.append(algorithm_id)
    return normalized


def _agent_configs_from_request(request: RunStartRequest) -> dict[str, AgentConfig]:
    return {
        role: AgentConfig(
            role=role,
            model=agent_model.model.strip(),
            temperature=agent_model.temperature,
            reasoning_effort=agent_model.reasoning_effort,
        )
        for role, agent_model in request.agent_models.items()
    }


def _selector_panel_from_request(
    request: RunStartRequest | RunReadinessRequest,
) -> list[AgentConfig]:
    return [
        AgentConfig(
            role=f"selector_{index:03d}",
            model=agent_model.model.strip(),
            temperature=agent_model.temperature,
            reasoning_effort=agent_model.reasoning_effort,
        )
        for index, agent_model in enumerate(request.selector_panel, start=1)
    ]


def _merge_resume_request(run_id: str, request: RunStartRequest) -> RunStartRequest:
    output_dir = _resolve_output_dir(request.output_dir, account_id=request.account_id)
    run_dir = _resolve_run_dir(run_id, output_dir)
    existing_config = _read_existing_experiment_config(run_dir)
    updates: dict[str, Any] = {"experiment_id": run_id, "resume": True}
    explicitly_set = set(request.model_fields_set)

    if existing_config is not None:
        if "benchmark" not in explicitly_set and "benchmark_dir" not in explicitly_set:
            if existing_config.benchmark_dir is not None:
                benchmark_spec = benchmark_for_path(existing_config.benchmark_dir)
                if benchmark_spec is not None:
                    updates["benchmark"] = benchmark_spec.name
                    updates["benchmark_dir"] = None
                elif request.account_id is None:
                    updates["benchmark_dir"] = str(existing_config.benchmark_dir)
        if "selected_algorithm_ids" not in explicitly_set:
            updates["selected_algorithm_ids"] = list(existing_config.strategy_seed_ids)
        if "agent_models" not in explicitly_set:
            updates["agent_models"] = _agent_requests_from_configs(existing_config.agents)
        if "selector_panel" not in explicitly_set:
            updates["selector_panel"] = _agent_requests_from_config_list(existing_config.selector_panel)
        if "problem_intake" not in explicitly_set:
            updates["problem_intake"] = dict(existing_config.problem_intake)
        if "planner_snapshot" not in explicitly_set:
            updates["planner_snapshot"] = dict(existing_config.planner_snapshot)
        if "claim_level" not in explicitly_set:
            updates["claim_level"] = existing_config.claim_level
        if "domain_evaluator_approved" not in explicitly_set:
            updates["domain_evaluator_approved"] = existing_config.domain_evaluator_approved
        if "domain_reviewer" not in explicitly_set:
            updates["domain_reviewer"] = existing_config.domain_reviewer
        if "domain_review_notes" not in explicitly_set:
            updates["domain_review_notes"] = existing_config.domain_review_notes
        if "paper_benchmark_approved" not in explicitly_set:
            updates["paper_benchmark_approved"] = existing_config.paper_benchmark_approved
        if "auto_approve_evaluation" not in explicitly_set:
            updates["auto_approve_evaluation"] = existing_config.auto_approve_evaluation

    return request.model_copy(update=updates)


def _read_existing_experiment_config(run_dir: Path) -> ExperimentConfig | None:
    payload = _read_optional_json(run_dir / "config.json")
    if not payload:
        return None
    try:
        return ExperimentConfig.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def _agent_requests_from_configs(agents: dict[str, AgentConfig]) -> dict[str, AgentModelRequest]:
    requests: dict[str, AgentModelRequest] = {}
    for role, config in agents.items():
        reasoning_effort = (
            cast(ReasoningEffort, config.reasoning_effort)
            if config.reasoning_effort in REASONING_EFFORTS
            else None
        )
        requests[role] = AgentModelRequest(
            model=config.model,
            temperature=config.temperature,
            reasoning_effort=reasoning_effort,
        )
    return requests


def _agent_requests_from_config_list(agents: list[AgentConfig]) -> list[AgentModelRequest]:
    return [
        AgentModelRequest(
            model=config.model,
            temperature=config.temperature,
            reasoning_effort=(
                cast(ReasoningEffort, config.reasoning_effort)
                if config.reasoning_effort in REASONING_EFFORTS
                else None
            ),
        )
        for config in agents
    ]


def _normalized_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _record_for(run_dir: Path) -> RunRecord | None:
    with _RUNS_LOCK:
        return _RUNS.get(_record_key(run_dir))


def _record_key(run_dir: str | Path) -> str:
    return str(Path(run_dir).resolve(strict=False))


def _normalize_run_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    normalized = dict(metadata)
    champion = normalized.get("champion_node_id") or normalized.get("champion")
    if isinstance(champion, str) and champion:
        normalized["champion_node_id"] = champion
    return normalized


def _describe_run(run_id: str, output_dir: Path) -> dict[str, object]:
    run_dir = output_dir / run_id
    record = _record_for(run_dir)
    if not run_dir.exists() and record is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    metadata = _normalize_run_metadata(_read_optional_json(run_dir / "run_metadata.json"))
    trace_summary = _read_optional_json(run_dir / "trace_summary.json")
    status = record.status if record else _infer_run_status(run_dir, metadata)
    payload: dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "run_dir": str(run_dir),
        "record": asdict(record) if record else None,
        "metadata": metadata,
        "trace_summary": trace_summary,
        "leaderboard": _read_leaderboard(run_dir / "leaderboard.csv"),
        "artifacts": _artifact_entries(run_dir, run_dir) if run_dir.exists() else [],
    }
    if record and record.error:
        payload["error"] = record.error
    return payload


def _infer_run_status(run_dir: Path, metadata: dict[str, Any] | None) -> str:
    if metadata and metadata.get("run_state") in {"completed", "exported", "finalized"}:
        return "completed"
    if (run_dir / "trace.jsonl").exists() or (run_dir / "checkpoint.json").exists():
        return "partial"
    return "unknown"


def _resolve_output_dir(value: str, *, account_id: str | None = None) -> Path:
    if account_id and value == "runs":
        return _account_runs_dir(account_id)
    if account_id:
        raise HTTPException(
            status_code=400,
            detail="account-scoped Web requests must use the account runs directory",
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_run_dir(run_id: str, output_dir: Path) -> Path:
    run_dir = output_dir / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run_dir.resolve()


def _resolve_benchmark_dir(
    benchmark: str,
    benchmark_dir: str | None,
    *,
    account_id: str | None = None,
) -> Path:
    if account_id and benchmark_dir:
        raise HTTPException(
            status_code=400,
            detail="account-scoped Web requests must use benchmark catalog names, not benchmark_dir paths",
        )
    if account_id and _is_path_like_benchmark(benchmark):
        raise HTTPException(
            status_code=400,
            detail="account-scoped Web requests must use benchmark catalog names, not local paths",
        )
    if benchmark_dir:
        path = Path(benchmark_dir).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    candidate = Path(benchmark).expanduser()
    if candidate.exists() or "/" in benchmark:
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        return candidate.resolve()
    for spec in list_benchmarks():
        if spec.name == benchmark:
            return spec.path.resolve()
    custom_candidate = (_custom_benchmarks_dir(account_id) / benchmark).resolve(strict=False)
    if custom_candidate.exists():
        try:
            custom_spec = benchmark_for_path(custom_candidate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if custom_spec is not None:
            return custom_candidate
    raise HTTPException(status_code=404, detail=f"Unknown benchmark: {benchmark}")


def _is_path_like_benchmark(value: str) -> bool:
    candidate = Path(value).expanduser()
    return candidate.is_absolute() or "/" in value or "\\" in value or value in {".", ".."}


def _real_web_runs_enabled() -> bool:
    return os.environ.get("AGENTICSCIML_ENABLE_REAL_WEB_RUNS") == "1"


def _repo_workspace_enabled() -> bool:
    return os.environ.get("AGENTICSCIML_ALLOW_REPO_WORKSPACE") == "1"


def _resolve_account_id(value: str | None) -> str:
    account_id = (value or DEFAULT_ACCOUNT_ID).strip()
    if not ACCOUNT_ID_RE.fullmatch(account_id):
        raise HTTPException(
            status_code=400,
            detail="account_id must start with a letter or number and contain only letters, numbers, _ or -",
        )
    return account_id


def _accounts_root() -> Path:
    configured = os.environ.get("AGENTICSCIML_ACCOUNTS_ROOT")
    root = Path(configured).expanduser() if configured else REPO_ROOT / ".agenticsciml" / "accounts"
    return root.resolve()


def _account_root(account_id: str | None, *, create: bool = True) -> Path:
    resolved_id = _resolve_account_id(account_id)
    root = _accounts_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    account_root = (root / resolved_id).resolve(strict=False)
    if account_root != root and root not in account_root.parents:
        raise HTTPException(status_code=400, detail="account workspace escapes accounts root")
    if create:
        (account_root / "workspace").mkdir(parents=True, exist_ok=True)
        (account_root / "runs").mkdir(parents=True, exist_ok=True)
        (account_root / "benchmarks").mkdir(parents=True, exist_ok=True)
    return account_root


def _account_workspace_dir(account_id: str | None) -> Path:
    return (_account_root(account_id) / "workspace").resolve(strict=True)


def _account_runs_dir(account_id: str | None) -> Path:
    return (_account_root(account_id) / "runs").resolve(strict=True)


def _custom_benchmarks_dir(account_id: str | None) -> Path:
    if account_id:
        path = _account_root(account_id) / "benchmarks"
    else:
        path = REPO_ROOT / ".agenticsciml" / "custom_benchmarks"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _ensure_account(account_id: str | None, display_name: str | None = None) -> dict[str, object]:
    resolved_id = _resolve_account_id(account_id)
    root = _account_root(resolved_id)
    metadata_path = root / "account.json"
    if metadata_path.exists():
        existing = _read_optional_json(metadata_path) or {}
    else:
        existing = {}
    payload = {
        "schema_version": 1,
        "account_id": resolved_id,
        "display_name": display_name or existing.get("display_name") or resolved_id,
        "workspace_root": str((root / "workspace").resolve()),
        "runs_dir": str((root / "runs").resolve()),
        "isolation": "local_namespace",
        "auth": "not_implemented",
    }
    if existing == payload:
        return payload
    tmp_path = metadata_path.with_name(f".{metadata_path.name}.{threading.get_ident()}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, metadata_path)
    return payload


def _list_accounts() -> list[dict[str, object]]:
    accounts_by_id = {DEFAULT_ACCOUNT_ID: _ensure_account(DEFAULT_ACCOUNT_ID, "Local")}
    root = _accounts_root()
    if root.exists():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            try:
                account_id = _resolve_account_id(child.name)
            except HTTPException:
                continue
            accounts_by_id[account_id] = _ensure_account(account_id)
    return list(accounts_by_id.values())


def _default_experiment_id(mode: str) -> str:
    prefix = "mock" if mode == "mock" else mode
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_leaderboard(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _selector_votes_payload(run_id: str, run_dir: Path) -> dict[str, object]:
    selector_path = run_dir / "reports" / "selector_votes.json"
    selector_events = _selector_vote_event_count(run_dir)
    payload = _read_optional_json(selector_path) or {}
    votes = payload.get("votes")
    vote_counts = payload.get("vote_counts")
    selected_parent_ids = payload.get("selected_parent_ids")
    panel_members = payload.get("selector_panel_members")
    selector_diversity = payload.get("selector_diversity")
    return {
        "run_id": run_id,
        "available": selector_path.exists() and bool(payload),
        "path": "reports/selector_votes.json",
        "selector_vote_events": selector_events,
        "selector_voting_exercised": selector_events > 0,
        "schema_version": payload.get("schema_version"),
        "ensemble_mode": payload.get("ensemble_mode"),
        "selector_panel_members": panel_members if isinstance(panel_members, list) else [],
        "selector_diversity": selector_diversity if isinstance(selector_diversity, dict) else {},
        "selected_parent_ids": selected_parent_ids if isinstance(selected_parent_ids, list) else [],
        "vote_counts": vote_counts if isinstance(vote_counts, dict) else {},
        "votes": votes if isinstance(votes, list) else [],
        "claim_boundary": payload.get("claim_boundary"),
    }


def _selector_vote_event_count(run_dir: Path) -> int:
    selector_dir = run_dir / "reports" / "selector_votes"
    if not selector_dir.exists():
        return 0
    return sum(1 for path in selector_dir.glob("selection_*.json") if path.is_file())


def _solutions_payload(run_id: str, run_dir: Path) -> dict[str, object]:
    tree_payload = _read_optional_json(run_dir / "tree.json") or {}
    raw_nodes = tree_payload.get("nodes")
    nodes = raw_nodes if isinstance(raw_nodes, list) else []
    leaderboard = _read_leaderboard(run_dir / "leaderboard.csv")
    leaderboard_by_node = {
        str(row["node_id"]): row
        for row in leaderboard
        if isinstance(row, dict) and row.get("node_id")
    }
    solutions = [
        _solution_summary(run_dir, node, leaderboard_by_node)
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("node_id"), str)
    ]
    return {
        "run_id": run_id,
        "available": bool(nodes),
        "tree": {
            "root_id": tree_payload.get("root_id"),
            "node_count": len(solutions),
            "schema_version": tree_payload.get("schema_version"),
        },
        "solutions": solutions,
        "leaderboard": leaderboard,
        "figures": _local_figure_artifacts(run_dir),
    }


def _solution_summary(
    run_dir: Path,
    node: dict[str, Any],
    leaderboard_by_node: dict[str, dict[str, str]],
) -> dict[str, object]:
    node_id = str(node["node_id"])
    workspace = run_dir / "solutions" / node_id
    eval_payload = _read_optional_json(workspace / "eval.json") or {}
    score_payload = node.get("score") if isinstance(node.get("score"), dict) else {}
    leaderboard_row = leaderboard_by_node.get(node_id, {})
    score_value = _first_float(
        eval_payload.get("score"),
        score_payload.get("value"),
        leaderboard_row.get("score"),
    )
    higher_is_better = _first_bool(
        eval_payload.get("higher_is_better"),
        score_payload.get("higher_is_better"),
    )
    metric = _first_text(
        eval_payload.get("metric"),
        score_payload.get("metric"),
        leaderboard_row.get("metric"),
    )
    method_tags = node.get("method_tags")
    children = node.get("children")
    policy_fidelity = _policy_fidelity_summary(workspace / "policy_fidelity_report.json")
    emergence_audit = _emergence_audit_summary(workspace / "emergence_report.json")
    return {
        "node_id": node_id,
        "parent_id": node.get("parent_id"),
        "children": children if isinstance(children, list) else [],
        "status": node.get("status"),
        "metric": metric,
        "score": score_value,
        "loss": score_value if higher_is_better is False else None,
        "higher_is_better": higher_is_better,
        "score_delta_from_parent": _first_float(node.get("score_delta_from_parent")),
        "method_tags": method_tags if isinstance(method_tags, list) else [],
        "failure_kind": node.get("failure_kind"),
        "num_debug_attempts": node.get("num_debug_attempts"),
        "policy_fidelity": policy_fidelity,
        "emergence_audit": emergence_audit,
        "workspace": f"solutions/{node_id}",
        "artifacts": _solution_artifacts(run_dir, workspace),
    }


def _policy_fidelity_summary(path: Path) -> dict[str, object]:
    payload = _read_optional_json(path)
    if not isinstance(payload, dict):
        return {"available": False}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "available": True,
        "status": payload.get("status"),
        "execution_allowed": payload.get("execution_allowed"),
        "inspector_version": payload.get("inspector_version"),
        "failed_blocker_count": summary.get("failed_blocker_count", 0),
        "failed_warning_count": summary.get("failed_warning_count", 0),
        "auditable_lock_count": summary.get("auditable_lock_count", 0),
    }


def _emergence_audit_summary(path: Path) -> dict[str, object]:
    payload = _read_optional_json(path)
    if not isinstance(payload, dict):
        return {"available": False}
    gaps = payload.get("blocking_gaps")
    return {
        "available": True,
        "auditor_version": payload.get("auditor_version"),
        "claim_level": payload.get("claim_level"),
        "blocking_gap_count": len(gaps) if isinstance(gaps, list) else 0,
        "claim_boundary": payload.get("claim_boundary"),
    }


def _solution_artifacts(run_dir: Path, workspace: Path) -> list[dict[str, object]]:
    artifact_names = (
        "eval.json",
        "analysis.md",
        "proposal.md",
        "branch_context.json",
        "policy_fidelity_report.json",
        "emergence_report.json",
        "prediction_overview.svg",
    )
    return [
        _artifact_ref(run_dir, workspace / name)
        for name in artifact_names
        if (workspace / name).exists()
    ]


def _local_figure_artifacts(run_dir: Path) -> list[dict[str, object]]:
    figures: list[dict[str, object]] = []
    data_overview = run_dir / "reports" / "data_overview.svg"
    if data_overview.exists():
        figures.append(_artifact_ref(run_dir, data_overview))
    solutions_dir = run_dir / "solutions"
    if solutions_dir.exists():
        for figure_path in sorted(solutions_dir.glob("solution_*/prediction_overview.svg")):
            figures.append(_artifact_ref(run_dir, figure_path))
    return figures


def _artifact_ref(run_dir: Path, path: Path) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
        root = run_dir.resolve(strict=True)
    except OSError:
        return {"path": str(path), "kind": "missing", "size_bytes": None}
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="Artifact path escapes the run directory")
    return {
        "path": str(resolved.relative_to(root)),
        "kind": "file" if resolved.is_file() else "directory",
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _first_bool(*values: object) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _first_float(*values: object) -> float | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _artifact_entries(path: Path, run_dir: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        try:
            resolved = child.resolve(strict=True)
        except OSError:
            continue
        if run_dir.resolve() != resolved and run_dir.resolve() not in resolved.parents:
            continue
        entries.append(
            {
                "path": str(child.relative_to(run_dir)),
                "kind": "directory" if child.is_dir() else "file",
                "size_bytes": child.stat().st_size if child.is_file() else None,
            }
        )
    return entries


def _safe_run_child(run_dir: Path, artifact_path: str) -> Path:
    requested = Path(artifact_path)
    if requested.is_absolute() or any(part == ".." for part in requested.parts):
        raise HTTPException(status_code=400, detail="Artifact path must stay inside the run directory")
    target = (run_dir / requested).resolve(strict=False)
    run_root = run_dir.resolve(strict=True)
    if target != run_root and run_root not in target.parents:
        raise HTTPException(status_code=400, detail="Artifact path escapes the run directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    real_target = target.resolve(strict=True)
    if real_target != run_root and run_root not in real_target.parents:
        raise HTTPException(status_code=400, detail="Artifact path escapes the run directory")
    return real_target


def _read_text_artifact(path: Path) -> str:
    if path.stat().st_size > 1_000_000:
        raise HTTPException(status_code=413, detail="Artifact is too large for inline display")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="Artifact is not UTF-8 text") from exc


def _stream_trace_events(run_id: str, run_dir: Path, *, follow: bool, timeout_s: float):
    trace_path = run_dir / "trace.jsonl"
    start = time.monotonic()
    offset = 0
    while True:
        if trace_path.exists():
            with trace_path.open("r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if line:
                        yield f"event: trace\ndata: {line}\n\n"
                offset = f.tell()
        record = _record_for(run_dir)
        if not follow or (record and record.status in {"completed", "failed", "dry_run"}):
            yield "event: end\ndata: {}\n\n"
            return
        if time.monotonic() - start >= timeout_s:
            yield "event: heartbeat\ndata: {}\n\n"
            return
        time.sleep(0.5)


def _code_server_payload(
    scope: WorkspaceScope,
    *,
    account_id: str | None,
    run_id: str | None,
    solution_id: str | None,
    output_dir: str,
) -> dict[str, object]:
    workspace = _workspace_for_scope(
        scope,
        account_id=account_id,
        run_id=run_id,
        solution_id=solution_id,
        output_dir=output_dir,
    )
    resolved_account_id = _resolve_account_id(account_id) if account_id else None
    base_url = os.environ.get("AGENTICSCIML_CODE_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
    return {
        "scope": scope,
        "account_id": resolved_account_id,
        "run_id": run_id,
        "solution_id": solution_id,
        "workspace": str(workspace),
        "url": f"{base_url}/?folder={quote(str(workspace))}",
        "configured": bool(base_url),
        "auth_mode": "upstream_account",
        "warnings": [
            "Run code-server with --auth none only behind the ChatUI account/auth gateway or loopback-only access.",
            "Do not expose this sidecar directly or pass host secrets into its environment.",
        ],
        "command_hint": f"code-server --auth none --bind-addr 127.0.0.1:8080 {workspace}",
    }


def _workspace_for_scope(
    scope: WorkspaceScope,
    *,
    account_id: str | None,
    run_id: str | None,
    solution_id: str | None,
    output_dir: str,
) -> Path:
    if scope == "account":
        return _account_workspace_dir(account_id)
    if scope == "repo":
        if not _repo_workspace_enabled():
            raise HTTPException(
                status_code=403,
                detail="shared repo workspace is disabled for the Web API; set AGENTICSCIML_ALLOW_REPO_WORKSPACE=1 for local development",
            )
        return REPO_ROOT.resolve()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required for run or solution workspace scope")
    run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir, account_id=account_id))
    if scope == "run":
        return run_dir
    if solution_id in {None, "", "champion"}:
        workspace = run_dir / "champion"
    else:
        workspace = run_dir / "solutions" / solution_id
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace}")
    return workspace.resolve(strict=True)


def _code_server_workspaces(
    *,
    account_id: str | None,
    run_id: str | None,
    output_dir: str,
) -> list[dict[str, object]]:
    workspaces: list[dict[str, object]]
    if account_id:
        resolved_account_id = _resolve_account_id(account_id)
        account = _ensure_account(resolved_account_id)
        workspaces = [
            _workspace_option(
                f"account:{resolved_account_id}",
                label=f"{account['display_name']} / workspace",
                scope="account",
                account_id=resolved_account_id,
                run_id=None,
                solution_id=None,
                workspace=Path(str(account["workspace_root"])).resolve(),
                status="account-isolated",
            )
        ]
    else:
        resolved_account_id = None
        workspaces = []
        if _repo_workspace_enabled():
            workspaces.append(
                _workspace_option(
                    "repo",
                    label="Repository",
                    scope="repo",
                    account_id=None,
                    run_id=None,
                    solution_id=None,
                    workspace=REPO_ROOT.resolve(),
                    status="ready",
                )
            )
    base = _resolve_output_dir(output_dir, account_id=resolved_account_id)
    run_dirs: list[Path] = []
    if run_id:
        run_dirs = [_resolve_run_dir(run_id, base)]
    elif base.exists():
        run_dirs = [child.resolve() for child in sorted(base.iterdir(), reverse=True) if child.is_dir()]

    for run_dir in run_dirs:
        current_run_id = run_dir.name
        metadata = _normalize_run_metadata(_read_optional_json(run_dir / "run_metadata.json")) or {}
        workspaces.append(
            _workspace_option(
                f"run:{current_run_id}",
                label=current_run_id,
                scope="run",
                account_id=resolved_account_id,
                run_id=current_run_id,
                solution_id=None,
                workspace=run_dir,
                status=str(metadata.get("run_state") or _infer_run_status(run_dir, metadata)),
            )
        )
        champion_dir = run_dir / "champion"
        if champion_dir.exists():
            workspaces.append(
                _workspace_option(
                    f"champion:{current_run_id}",
                    label=f"{current_run_id} / champion",
                    scope="solution",
                    account_id=resolved_account_id,
                    run_id=current_run_id,
                    solution_id="champion",
                    workspace=champion_dir.resolve(),
                    status=str(metadata.get("champion_node_id") or "champion"),
                )
            )
        solutions_dir = run_dir / "solutions"
        if solutions_dir.exists():
            for solution_dir in sorted(solutions_dir.iterdir()):
                if not solution_dir.is_dir() or not solution_dir.name.startswith("solution_"):
                    continue
                workspaces.append(
                    _workspace_option(
                        f"solution:{current_run_id}:{solution_dir.name}",
                        label=f"{current_run_id} / {solution_dir.name}",
                        scope="solution",
                        account_id=resolved_account_id,
                        run_id=current_run_id,
                        solution_id=solution_dir.name,
                        workspace=solution_dir.resolve(),
                        status="solution",
                    )
                )
    return workspaces


def _workspace_option(
    workspace_id: str,
    *,
    label: str,
    scope: WorkspaceScope,
    account_id: str | None,
    run_id: str | None,
    solution_id: str | None,
    workspace: Path,
    status: str,
) -> dict[str, object]:
    base_url = os.environ.get("AGENTICSCIML_CODE_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
    return {
        "id": workspace_id,
        "label": label,
        "scope": scope,
        "account_id": account_id,
        "run_id": run_id,
        "solution_id": solution_id,
        "workspace": str(workspace),
        "status": status,
        "isolation": "account" if account_id else "shared",
        "url": f"{base_url}/?folder={quote(str(workspace))}",
        "configured": bool(base_url),
        "auth_mode": "upstream_account",
        "warnings": [
            "Run code-server with --auth none only behind the ChatUI account/auth gateway or loopback-only access.",
            "Do not expose this sidecar directly or pass host secrets into its environment.",
        ],
        "command_hint": f"code-server --auth none --bind-addr 127.0.0.1:8080 {workspace}",
    }


def _solver_chat_response(request: SolverChatRequest) -> dict[str, object]:
    text = request.message.lower()
    model_settings = _assistant_model_settings(request)
    proposed_actions: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    warnings: list[str] = []
    trace_refs: list[dict[str, object]] = []
    resolved_account_id = _resolve_account_id(request.account_id) if request.account_id else None
    agent_scope_allowed = True

    if request.mode == "real":
        warnings.append("Real LLM mode requires explicit credentials and keeps existing budget/claim boundaries.")
    if request.assistant_mode == "agent":
        if not resolved_account_id:
            agent_scope_allowed = False
            warnings.append("Agent mode requires account_id and can only operate inside the current account workspace.")
        if request.workspace_scope == "repo":
            agent_scope_allowed = False
            warnings.append("Agent mode cannot operate the shared repo workspace; use account, run, or solution scope.")

    if any(token in text for token in ("跑", "run", "start", "mock", "实验")):
        if request.assistant_mode in {"plan", "agent"} and _should_plan_problem_from_chat(request.message):
            try:
                _validate_agent_model_roles(request.agent_models)
                _validate_algorithm_ids(request.selected_algorithm_ids)
                plan = _problem_intake_plan_payload(_problem_intake_request_from_chat(request))
                action = dict(plan["actions"][0])
                payload = dict(action["payload"])
                payload["account_id"] = resolved_account_id
                payload["background"] = True
                action["payload"] = payload
                proposed_actions.append(action)
                artifacts.append(
                    {
                        "kind": "problem_intake_plan",
                        "recommended_benchmark": plan["recommended_benchmark"],
                        "selected_algorithm_ids": plan["selected_algorithm_ids"],
                        "run_config": plan["run_config"],
                    }
                )
                warnings.extend(str(warning) for warning in plan["warnings"])
            except HTTPException as exc:
                warnings.append(str(exc.detail))
        else:
            proposed_actions.append(
                {
                    "type": "start_run",
                    "payload": {
                        "benchmark": request.selected_benchmark,
                        "mode": request.mode,
                        "account_id": resolved_account_id,
                        "claim_level": request.claim_level,
                        "background": True,
                    },
                }
            )
    if any(token in text for token in ("resume", "恢复", "继续")) and request.active_run_id:
        proposed_actions.append({"type": "resume_run", "run_id": request.active_run_id})
    if agent_scope_allowed and _should_open_code_server_from_chat(request.message):
        try:
            proposed_actions.append(
                {
                    "type": "open_code_server",
                    "payload": _code_server_payload(
                        request.workspace_scope,
                        account_id=request.account_id,
                        run_id=request.active_run_id,
                        solution_id="champion" if "champion" in text else None,
                        output_dir=request.output_dir,
                    ),
                }
            )
        except HTTPException as exc:
            warnings.append(str(exc.detail))
    if any(token in text for token in ("trace", "解释", "summary", "总结", "leaderboard", "artifact")):
        if not request.active_run_id:
            warnings.append("Select an active run before asking for artifact or trace summaries.")
        else:
            run_dir = _resolve_run_dir(
                request.active_run_id,
                _resolve_output_dir(request.output_dir, account_id=request.account_id),
            )
            summary = _read_optional_json(run_dir / "trace_summary.json")
            leaderboard = _read_leaderboard(run_dir / "leaderboard.csv")
            artifacts.extend(_important_artifacts(run_dir))
            if summary:
                trace_refs.append(
                    {
                        "run_id": request.active_run_id,
                        "event_count": summary.get("event_count"),
                        "quality_gate": summary.get("quality_gate"),
                    }
                )
            if leaderboard:
                artifacts.append({"path": "leaderboard.csv", "top_row": leaderboard[0]})
    if "compare" in text or "比较" in text:
        warnings.append("Run comparison needs two explicit run ids; this MVP returns a prompt to choose the second run.")

    if not proposed_actions and not artifacts and not trace_refs and request.assistant_mode != "ask":
        proposed_actions.append({"type": "summarize_artifact", "payload": {"benchmark": request.selected_benchmark}})

    if request.assistant_mode == "ask":
        actions: list[dict[str, object]] = []
        if proposed_actions:
            warnings.append(
                "Ask mode does not return executable actions. "
                "Switch to plan to preview actions or agent to run them."
            )
        reply = _solver_ask_reply(request, proposed_actions, artifacts, warnings, trace_refs)
    else:
        actions = proposed_actions if agent_scope_allowed else []
        if request.assistant_mode == "plan" and actions:
            warnings.append("Plan mode returns proposed actions only. The frontend must not dispatch them automatically.")
        reply = _solver_reply(request.assistant_mode, actions, artifacts, warnings, trace_refs)

    return {
        "assistant_mode": request.assistant_mode,
        "model_settings": model_settings,
        "reply": reply,
        "actions": actions,
        "artifacts": artifacts,
        "warnings": warnings,
        "trace_refs": trace_refs,
    }


def _should_plan_problem_from_chat(message: str) -> bool:
    text = message.lower()
    if len(message.strip()) < 40:
        return False
    intent_tokens = (
        "自动",
        "自主",
        "选择",
        "评选",
        "求解",
        "解法",
        "算法",
        "benchmark",
        "problem",
        "scientific",
        "输入",
        "输出",
        "数据",
        "评价",
        "reconstruct",
        "predict",
        "approximation",
    )
    return any(token in text for token in intent_tokens)


def _should_open_code_server_from_chat(message: str) -> bool:
    text = message.lower()
    explicit_code_tokens = (
        "code-server",
        "code server",
        "vscode",
        "vs code",
        "workspace",
        "工作区",
        "代码工作区",
        "代码编辑",
        "编辑代码",
    )
    if any(token in text for token in explicit_code_tokens):
        return True
    open_tokens = ("打开", "进入", "跳转", "open")
    target_tokens = ("code", "代码", "workspace", "工作区", "champion", "solution", "解法")
    return any(token in text for token in open_tokens) and any(token in text for token in target_tokens)


def _problem_intake_request_from_chat(request: SolverChatRequest) -> ProblemIntakeRequest:
    return ProblemIntakeRequest(
        problem_statement=request.message,
        requirements="",
        evaluation_criteria="",
        data_description="",
        mode=request.mode,
        target_solution_count=request.target_solution_count,
        parallel_mutations=request.parallel_mutations,
        selector_vote_count=request.selector_vote_count,
        max_children_per_node=request.max_children_per_node,
        selected_algorithm_ids=request.selected_algorithm_ids,
        agent_models=request.agent_models,
        selector_panel=request.selector_panel,
        allow_custom_benchmark=False,
        claim_level=request.claim_level,
    )


def _solver_settings_payload() -> dict[str, object]:
    return {
        "default_assistant_mode": "ask",
        "reasoning_efforts": list(REASONING_EFFORTS),
        "temperature_range": [0.0, 2.0],
        "assistant_modes": {
            mode: dict(settings)
            for mode, settings in ASSISTANT_MODE_MODEL_SETTINGS.items()
        },
        "agent_role_defaults": {
            str(role["role"]): dict(role["default_model_settings"])
            for role in AGENT_ROLES
        },
    }


def _assistant_model_settings(request: SolverChatRequest) -> dict[str, object]:
    defaults = ASSISTANT_MODE_MODEL_SETTINGS[request.assistant_mode]
    has_override = request.reasoning_effort is not None or request.temperature is not None
    return {
        "reasoning_effort": request.reasoning_effort or defaults["reasoning_effort"],
        "temperature": request.temperature if request.temperature is not None else defaults["temperature"],
        "source": "request_override" if has_override else "mode_default",
    }


def _important_artifacts(run_dir: Path) -> list[dict[str, object]]:
    names = [
        "run_metadata.json",
        "trace_summary.json",
        "leaderboard.csv",
        "tree.json",
        "reports/data_analysis.md",
        "champion/analysis.md",
    ]
    return [
        {"path": name, "size_bytes": (run_dir / name).stat().st_size}
        for name in names
        if (run_dir / name).exists()
    ]


def _solver_ask_reply(
    request: SolverChatRequest,
    proposed_actions: list[dict[str, object]],
    artifacts: list[dict[str, object]],
    warnings: list[str],
    trace_refs: list[dict[str, object]],
) -> str:
    text = request.message.lower()
    if trace_refs:
        gate = trace_refs[0].get("quality_gate")
        return f"已读取 trace summary。quality_gate={gate}，相关 artifacts={len(artifacts)}。"
    if _contains_any(text, ("你是谁", "who are you", "身份", "自我介绍", "介绍一下你")):
        return (
            "我是 AgenticSciML 助手，负责解释本项目的 SciML 工作流、benchmark、算法策略、"
            "run、trace 和 artifact。Ask 模式下我只回答问题，不启动实验、不恢复 run，也不打开工作区。"
        )
    if _contains_any(text, ("你能做什么", "能做什么", "可以做什么", "功能", "帮助", "help", "capabilities")):
        return (
            "我可以解释 AgenticSciML 的项目结构、benchmark 与 claim boundary；解读 run metadata、"
            "leaderboard、trace_summary 和 artifacts；说明 mock、dry_run、real 的风险；也可以在 Plan 模式"
            "生成动作计划，在 Agent 模式按当前账号 workspace 执行受控动作。"
        )
    if _contains_any(text, ("benchmark", "基准", "算法", "algorithm", "策略")):
        return (
            "当前可讨论 benchmark、算法策略目录、运行模式和验证边界。算法目录只表示可选策略和 prompt seed，"
            "是否有效必须以实际 run artifact、leaderboard、trace_summary 和测试结果为准。"
        )
    if proposed_actions:
        return (
            "这条请求会触发受控动作。Ask 模式不会执行或返回可执行 action；"
            "需要预览步骤请切到 Plan，需要执行请切到 Agent。"
        )
    if warnings:
        return "我可以解释这个问题，但当前缺少必要上下文或存在安全边界；请查看 warnings 里的具体原因。"
    return (
        "我是 AgenticSciML 助手。你可以问项目结构、benchmark、算法策略、run 状态、trace、artifact、"
        "VS Code workspace，以及 mock/dry_run/real 模式边界。"
    )


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _solver_reply(
    assistant_mode: AssistantMode,
    actions: list[dict[str, object]],
    artifacts: list[dict[str, object]],
    warnings: list[str],
    trace_refs: list[dict[str, object]],
) -> str:
    if assistant_mode == "plan" and actions:
        return "Plan 模式：已生成建议动作，但不会自动执行。确认后可切到 Agent 执行。"
    if actions and actions[0].get("type") == "start_run":
        return "Agent 模式：已准备启动实验；ChatUI 应调用 start_run action，并继续监听 run events。"
    if trace_refs:
        gate = trace_refs[0].get("quality_gate")
        return f"已读取 trace summary。quality_gate={gate}，相关 artifacts={len(artifacts)}。"
    if warnings:
        return "请求需要补充上下文或受安全边界限制；请查看 warnings。"
    return "已解析请求，返回可执行 action 和相关 artifact 线索。"
