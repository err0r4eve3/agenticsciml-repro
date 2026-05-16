from __future__ import annotations

import csv
import json
import os
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

from agenticsciml.benchmarks import REPO_ROOT, benchmark_for_path, list_benchmarks
from agenticsciml.config import EvolutionConfig, ExperimentConfig
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm.openai_adapter import OpenAIAdapter
from agenticsciml.orchestrator import AgenticSciMLOrchestrator


RunMode = Literal["mock", "real", "dry_run"]
WorkspaceScope = Literal["repo", "run", "solution"]

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


class RunStartRequest(BaseModel):
    benchmark: str = "function_approx"
    benchmark_dir: str | None = None
    mode: RunMode = "mock"
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


class SolverChatRequest(BaseModel):
    message: str
    active_run_id: str | None = None
    selected_benchmark: str = "function_approx"
    mode: RunMode = "mock"
    workspace_scope: WorkspaceScope = "repo"
    output_dir: str = "runs"


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

    @app.get("/api/runs")
    def runs(output_dir: str = Query(default="runs")) -> dict[str, object]:
        base = _resolve_output_dir(output_dir)
        records = []
        if base.exists():
            for child in sorted(base.iterdir()):
                if child.is_dir():
                    records.append(_describe_run(child.name, base))
        return {"runs": records}

    @app.post("/api/runs")
    def start_run(request: RunStartRequest) -> dict[str, object]:
        run_id = request.experiment_id or _default_experiment_id(request.mode)
        output_dir = _resolve_output_dir(request.output_dir)
        benchmark_dir = _resolve_benchmark_dir(request.benchmark, request.benchmark_dir)
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
    def get_run(run_id: str, output_dir: str = Query(default="runs")) -> dict[str, object]:
        return _describe_run(run_id, _resolve_output_dir(output_dir))

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str,
        output_dir: str = Query(default="runs"),
        follow: bool = Query(default=True),
        timeout_s: float = Query(default=30.0, ge=0.1, le=300.0),
    ) -> StreamingResponse:
        run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir))
        return StreamingResponse(
            _stream_trace_events(run_id, run_dir, follow=follow, timeout_s=timeout_s),
            media_type="text/event-stream",
        )

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def get_artifact(
        run_id: str,
        artifact_path: str,
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir))
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
        run_id: str | None = Query(default=None),
        solution_id: str | None = Query(default=None),
        output_dir: str = Query(default="runs"),
    ) -> dict[str, object]:
        return _code_server_payload(scope, run_id=run_id, solution_id=solution_id, output_dir=output_dir)

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
        _RUNS[record.run_id] = record


def _record_for(run_id: str) -> RunRecord | None:
    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def _describe_run(run_id: str, output_dir: Path) -> dict[str, object]:
    run_dir = output_dir / run_id
    record = _record_for(run_id)
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


def _resolve_output_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_run_dir(run_id: str, output_dir: Path) -> Path:
    run_dir = output_dir / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run_dir.resolve()


def _resolve_benchmark_dir(benchmark: str, benchmark_dir: str | None) -> Path:
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
        record = _record_for(run_id)
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
    run_id: str | None,
    solution_id: str | None,
    output_dir: str,
) -> dict[str, object]:
    workspace = _workspace_for_scope(scope, run_id=run_id, solution_id=solution_id, output_dir=output_dir)
    base_url = os.environ.get("AGENTICSCIML_CODE_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
    return {
        "scope": scope,
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
    run_id: str | None,
    solution_id: str | None,
    output_dir: str,
) -> Path:
    if scope == "repo":
        return REPO_ROOT.resolve()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required for run or solution workspace scope")
    run_dir = _resolve_run_dir(run_id, _resolve_output_dir(output_dir))
    if scope == "run":
        return run_dir
    if solution_id in {None, "", "champion"}:
        workspace = run_dir / "champion"
    else:
        workspace = run_dir / "solutions" / solution_id
    if not workspace.exists():
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace}")
    return workspace.resolve(strict=True)


def _solver_chat_response(request: SolverChatRequest) -> dict[str, object]:
    text = request.message.lower()
    actions: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    warnings: list[str] = []
    trace_refs: list[dict[str, object]] = []

    if request.mode == "real":
        warnings.append("Real LLM mode requires explicit credentials and keeps existing budget/claim boundaries.")

    if any(token in text for token in ("跑", "run", "start", "mock", "实验")):
        actions.append(
            {
                "type": "start_run",
                "payload": {
                    "benchmark": request.selected_benchmark,
                    "mode": request.mode,
                    "background": True,
                },
            }
        )
    if any(token in text for token in ("resume", "恢复", "继续")) and request.active_run_id:
        actions.append({"type": "resume_run", "run_id": request.active_run_id})
    if any(token in text for token in ("code", "vscode", "代码", "打开", "champion", "solution")):
        try:
            actions.append(
                {
                    "type": "open_code_server",
                    "payload": _code_server_payload(
                        request.workspace_scope,
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
            run_dir = _resolve_run_dir(request.active_run_id, _resolve_output_dir(request.output_dir))
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

    if not actions and not artifacts and not trace_refs:
        actions.append({"type": "summarize_artifact", "payload": {"benchmark": request.selected_benchmark}})

    return {
        "reply": _solver_reply(actions, artifacts, warnings, trace_refs),
        "actions": actions,
        "artifacts": artifacts,
        "warnings": warnings,
        "trace_refs": trace_refs,
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


def _solver_reply(
    actions: list[dict[str, object]],
    artifacts: list[dict[str, object]],
    warnings: list[str],
    trace_refs: list[dict[str, object]],
) -> str:
    if actions and actions[0].get("type") == "start_run":
        return "已准备启动实验；ChatUI 应调用 start_run action，并继续监听 run events。"
    if trace_refs:
        gate = trace_refs[0].get("quality_gate")
        return f"已读取 trace summary。quality_gate={gate}，相关 artifacts={len(artifacts)}。"
    if warnings:
        return "请求需要补充上下文或受安全边界限制；请查看 warnings。"
    return "已解析请求，返回可执行 action 和相关 artifact 线索。"
