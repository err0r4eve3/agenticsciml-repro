from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agenticsciml.algorithm_catalog import ALGORITHM_CLAIM_BOUNDARY, list_algorithms
from agenticsciml.benchmarks import REPO_ROOT, benchmark_for_path, list_benchmarks
from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


RunMode = Literal["mock", "real", "dry_run"]
WorkspaceScope = Literal["repo", "account", "run", "solution"]
AssistantMode = Literal["ask", "plan", "agent"]
ReasoningEffort = Literal["low", "medium", "high"]
DEFAULT_ACCOUNT_ID = "local"
ACCOUNT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")

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

ASSISTANT_MODE_MODEL_SETTINGS: dict[AssistantMode, dict[str, object]] = {
    "ask": {"reasoning_effort": "medium", "temperature": 0.2},
    "plan": {"reasoning_effort": "high", "temperature": 0.35},
    "agent": {"reasoning_effort": "high", "temperature": 0.1},
}


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
    selector_vote_count: int = Field(default=3, ge=1)
    resume: bool = False
    background: bool = False
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

        if request.mode == "dry_run":
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
                args=(request, run_id, benchmark_dir, output_dir, record),
                daemon=True,
            )
            thread.start()
            return {"run_id": run_id, "status": "running", "run_dir": str(run_dir)}

        _run_orchestrator(request, run_id, benchmark_dir, output_dir, record)
        return _describe_run(run_id, output_dir)

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: str, request: RunStartRequest) -> dict[str, object]:
        resume_request = request.model_copy(update={"experiment_id": run_id, "resume": True})
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
) -> None:
    try:
        evolution = EvolutionConfig(
            max_iterations=request.max_iterations,
            parallel_mutations=request.parallel_mutations,
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


def _store_record(record: RunRecord) -> None:
    with _RUNS_LOCK:
        _RUNS[_record_key(record.run_dir)] = record


def _validate_run_start_request(request: RunStartRequest) -> None:
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


def _record_for(run_dir: Path) -> RunRecord | None:
    with _RUNS_LOCK:
        return _RUNS.get(_record_key(run_dir))


def _record_key(run_dir: str | Path) -> str:
    return str(Path(run_dir).resolve(strict=False))


def _describe_run(run_id: str, output_dir: Path) -> dict[str, object]:
    run_dir = output_dir / run_id
    record = _record_for(run_dir)
    if not run_dir.exists() and record is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    metadata = _read_optional_json(run_dir / "run_metadata.json")
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
    return account_root


def _account_workspace_dir(account_id: str | None) -> Path:
    return (_account_root(account_id) / "workspace").resolve(strict=True)


def _account_runs_dir(account_id: str | None) -> Path:
    return (_account_root(account_id) / "runs").resolve(strict=True)


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
        "warnings": [
            "Start code-server on 127.0.0.1 with password/token auth before opening this URL.",
            "Do not expose this sidecar publicly or pass host secrets into its environment.",
        ],
        "command_hint": f"PASSWORD=<local-token> code-server --bind-addr 127.0.0.1:8080 {workspace}",
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
        metadata = _read_optional_json(run_dir / "run_metadata.json") or {}
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
        "warnings": [
            "Start code-server on 127.0.0.1 with password/token auth before opening this URL.",
            "Do not expose this sidecar publicly or pass host secrets into its environment.",
        ],
        "command_hint": f"PASSWORD=<local-token> code-server --bind-addr 127.0.0.1:8080 {workspace}",
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
        proposed_actions.append(
            {
                "type": "start_run",
                "payload": {
                    "benchmark": request.selected_benchmark,
                    "mode": request.mode,
                    "account_id": resolved_account_id,
                    "background": True,
                },
            }
        )
    if any(token in text for token in ("resume", "恢复", "继续")) and request.active_run_id:
        proposed_actions.append({"type": "resume_run", "run_id": request.active_run_id})
    if agent_scope_allowed and any(token in text for token in ("code", "vscode", "代码", "打开", "champion", "solution")):
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


def _solver_settings_payload() -> dict[str, object]:
    return {
        "default_assistant_mode": "ask",
        "reasoning_efforts": ["low", "medium", "high"],
        "temperature_range": [0.0, 2.0],
        "assistant_modes": {
            mode: dict(settings)
            for mode, settings in ASSISTANT_MODE_MODEL_SETTINGS.items()
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
