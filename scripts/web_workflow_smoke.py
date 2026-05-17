#!/usr/bin/env python3
"""Live smoke test for the AgenticSciML ChatUI and code-server workflow.

This script intentionally drives the public Web API over HTTP instead of
importing application internals. It is meant for local or deployed smoke checks
after the FastAPI service and optional code-server sidecar are already running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SmokeFailure(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="FastAPI base URL")
    parser.add_argument("--account-id", default="codex-smoke", help="account namespace to use")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="run id to create; defaults to a timestamped smoke id",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=180.0,
        help="HTTP timeout for run creation and artifact reads",
    )
    parser.add_argument(
        "--skip-code-server-live",
        action="store_true",
        help="only validate code-server API payloads, not the live sidecar HTTP response",
    )
    parser.add_argument(
        "--allow-non-loopback-code-server-url",
        action="store_true",
        help="allow deployed HTTPS code-server URLs when the payload still declares upstream account auth",
    )
    parser.add_argument(
        "--expect-repo-root-contains",
        default=None,
        help="fail if /api/health repo_root does not contain this string; useful for remote release drift checks",
    )
    args = parser.parse_args()

    try:
        summary = run_smoke(args)
    except SmokeFailure as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    account_id = args.account_id
    other_account_id = make_other_account_id(account_id)
    experiment_id = args.experiment_id or f"web-smoke-{time.strftime('%Y%m%d-%H%M%S')}"

    health = request_json(base_url, "GET", "/api/health", timeout_s=args.timeout_s)
    expect(health.get("ok") is True, "health endpoint did not return ok=true")
    if args.expect_repo_root_contains:
        expect(
            args.expect_repo_root_contains in str(health.get("repo_root", "")),
            f"health repo_root does not contain expected release marker {args.expect_repo_root_contains!r}",
        )

    account = request_json(
        base_url,
        "POST",
        "/api/accounts",
        {"account_id": account_id, "display_name": "Codex Smoke"},
        timeout_s=args.timeout_s,
    )["account"]
    expect(account["account_id"] == account_id, "account create returned the wrong account_id")
    other_account = request_json(
        base_url,
        "POST",
        "/api/accounts",
        {"account_id": other_account_id, "display_name": "Codex Smoke Other"},
        timeout_s=args.timeout_s,
    )["account"]
    expect(other_account["account_id"] == other_account_id, "other account create returned the wrong account_id")

    settings = request_json(base_url, "GET", "/api/solver/settings", timeout_s=args.timeout_s)
    expect(settings["default_assistant_mode"] == "ask", "default assistant mode must remain ask")
    expect(settings["assistant_modes"]["agent"]["reasoning_effort"] == "high", "agent mode should use high reasoning")
    negative_checks = run_preflight_negative_checks(base_url, account_id, timeout_s=args.timeout_s)

    ask = solver_chat(
        base_url,
        account_id,
        {
            "message": "你是谁",
            "assistant_mode": "ask",
        },
        timeout_s=args.timeout_s,
    )
    expect(ask["assistant_mode"] == "ask", "ask response reported the wrong assistant_mode")
    expect(ask["actions"] == [], "ask mode returned executable actions")
    expect("AgenticSciML 助手" in ask["reply"], "ask mode identity reply lost project-specific behavior")

    plan = solver_chat(
        base_url,
        account_id,
        {
            "message": "打开当前账号代码工作区",
            "assistant_mode": "plan",
            "workspace_scope": "account",
        },
        timeout_s=args.timeout_s,
    )
    plan_action_types = [action["type"] for action in plan["actions"]]
    expect("open_code_server" in plan_action_types, "plan mode did not return an open_code_server action")
    expect(any("Plan mode" in warning for warning in plan["warnings"]), "plan mode warning is missing")

    code_payload = request_json(
        base_url,
        "GET",
        "/api/code-server/url",
        params={"scope": "account", "account_id": account_id},
        timeout_s=args.timeout_s,
    )
    validate_code_server_payload(
        code_payload,
        allow_non_loopback_url=args.allow_non_loopback_code_server_url,
    )
    if not args.skip_code_server_live:
        validate_code_server_live(code_payload["url"], timeout_s=min(args.timeout_s, 30.0))

    workspaces_before = request_json(
        base_url,
        "GET",
        "/api/code-server/workspaces",
        params={"account_id": account_id},
        timeout_s=args.timeout_s,
    )["workspaces"]
    expect(
        any(workspace["id"] == f"account:{account_id}" for workspace in workspaces_before),
        "account workspace is missing from code-server workspace list",
    )

    agent = solver_chat(
        base_url,
        account_id,
        {
            "message": (
                "请用 Agent 模式自主选择 benchmark 和解法求解："
                "Reconstruct two-dimensional cylinder wake vorticity fields from sparse noisy temporal sensors. "
                "Use lagged sensor history and preserve smooth band-limited spatial structure. "
                "运行 mock 实验，解法量 3，并保留 trace evidence。"
            ),
            "assistant_mode": "agent",
            "workspace_scope": "account",
            "target_solution_count": 3,
            "parallel_mutations": 1,
            "selector_vote_count": 3,
        },
        timeout_s=args.timeout_s,
    )
    expect(agent["assistant_mode"] == "agent", "agent response reported the wrong assistant_mode")
    expect([action["type"] for action in agent["actions"]] == ["start_run"], "agent did not return exactly one start_run action")
    action_payload = dict(agent["actions"][0]["payload"])
    expect(
        action_payload["benchmark"] == "cylinder_wake_reconstruction_faithful_small",
        "agent planner did not choose the cylinder wake faithful-small benchmark",
    )
    expect(
        "paper_cylinder_bandlimited_filter" in action_payload["selected_algorithm_ids"],
        "agent planner did not include the expected cylinder wake algorithm seed",
    )

    run_payload = {
        **action_payload,
        "experiment_id": experiment_id,
        "background": False,
    }
    run = request_json(base_url, "POST", "/api/runs", run_payload, timeout_s=args.timeout_s)
    expect(run["run_id"] == experiment_id, "run creation returned the wrong run_id")
    expect(run["status"] == "completed", f"run did not complete: {run.get('status')}")
    expect(run["trace_summary"]["quality_gate"]["passed"] is True, "trace quality gate did not pass")

    run_detail = request_json(
        base_url,
        "GET",
        f"/api/runs/{experiment_id}",
        params={"account_id": account_id},
        timeout_s=args.timeout_s,
    )
    metadata = run_detail["metadata"]
    expect(metadata["benchmark_name"] == "cylinder_wake_reconstruction_faithful_small", "metadata benchmark mismatch")
    expect(metadata["solution_count"] == 3, "target solution count was not realized by the mock run")
    expect(metadata["champion_node_id"], "run metadata is missing champion_node_id")

    votes = request_json(
        base_url,
        "GET",
        f"/api/runs/{experiment_id}/selector-votes",
        params={"account_id": account_id},
        timeout_s=args.timeout_s,
    )
    expect(votes["available"] is True, "selector votes artifact is missing")

    solutions = request_json(
        base_url,
        "GET",
        f"/api/runs/{experiment_id}/solutions",
        params={"account_id": account_id},
        timeout_s=args.timeout_s,
    )
    expect(solutions["available"] is True, "solutions payload is unavailable")
    expect(solutions["tree"]["node_count"] == 3, "solutions endpoint returned the wrong node count")
    expect(
        all(item["loss"] is not None for item in solutions["solutions"]),
        "one or more solution summaries are missing loss values",
    )

    trace = solver_chat(
        base_url,
        account_id,
        {
            "message": "解释这个 trace summary 和 leaderboard",
            "assistant_mode": "ask",
            "active_run_id": experiment_id,
            "workspace_scope": "account",
        },
        timeout_s=args.timeout_s,
    )
    expect(trace["actions"] == [], "trace summary in ask mode returned executable actions")
    expect(trace["trace_refs"], "trace summary did not return trace refs")
    expect(trace["trace_refs"][0]["quality_gate"]["passed"] is True, "trace ref quality gate did not pass")

    workspaces_after = request_json(
        base_url,
        "GET",
        "/api/code-server/workspaces",
        params={"account_id": account_id, "run_id": experiment_id},
        timeout_s=args.timeout_s,
    )["workspaces"]
    workspace_ids = {workspace["id"] for workspace in workspaces_after}
    expect(f"run:{experiment_id}" in workspace_ids, "run workspace is missing after smoke run")
    expect(f"champion:{experiment_id}" in workspace_ids, "champion workspace is missing after smoke run")
    negative_checks.update(
        run_cross_account_negative_checks(
            base_url,
            run_id=experiment_id,
            other_account_id=other_account_id,
            timeout_s=args.timeout_s,
        )
    )

    return {
        "ok": True,
        "base_url": base_url,
        "repo_root": health.get("repo_root"),
        "account_id": account_id,
        "other_account_id": other_account_id,
        "run_id": experiment_id,
        "benchmark": metadata["benchmark_name"],
        "champion_node_id": metadata["champion_node_id"],
        "solution_count": metadata["solution_count"],
        "selector_vote_available": votes["available"],
        "negative_checks": negative_checks,
        "code_server": {
            "auth_mode": code_payload["auth_mode"],
            "workspace": code_payload["workspace"],
            "live_checked": not args.skip_code_server_live,
        },
    }


def make_other_account_id(account_id: str) -> str:
    candidate = f"{account_id}-other"
    if len(candidate) <= 48:
        return candidate
    return "codex-smoke-other"


def run_preflight_negative_checks(base_url: str, account_id: str, *, timeout_s: float) -> dict[str, Any]:
    real_error = expect_http_status(
        base_url,
        "POST",
        "/api/runs",
        expected_status=400,
        payload={
            "benchmark": "function_approx",
            "mode": "real",
            "account_id": account_id,
            "background": False,
        },
        timeout_s=timeout_s,
    )
    expect("real_confirmed" in str(real_error.get("detail", "")), "real-mode preflight did not fail on real_confirmed")

    repo_scope = solver_chat(
        base_url,
        account_id,
        {
            "message": "打开共享 repo 代码",
            "assistant_mode": "agent",
            "workspace_scope": "repo",
        },
        timeout_s=timeout_s,
    )
    expect(repo_scope["actions"] == [], "agent mode returned actions for shared repo scope")
    expect(
        any("shared repo workspace" in warning for warning in repo_scope["warnings"]),
        "agent mode did not warn about shared repo workspace",
    )
    return {
        "real_mode_requires_confirmation": True,
        "agent_repo_scope_blocked": True,
    }


def run_cross_account_negative_checks(
    base_url: str,
    *,
    run_id: str,
    other_account_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    checks = {
        "run_detail": (f"/api/runs/{run_id}", None),
        "selector_votes": (f"/api/runs/{run_id}/selector-votes", None),
        "solutions": (f"/api/runs/{run_id}/solutions", None),
        "run_metadata_artifact": (f"/api/runs/{run_id}/artifacts/run_metadata.json", None),
        "code_server_workspaces": ("/api/code-server/workspaces", {"run_id": run_id}),
    }
    statuses: dict[str, int] = {}
    for name, (path, extra_params) in checks.items():
        params = {"account_id": other_account_id}
        if extra_params:
            params.update(extra_params)
        expect_http_status(
            base_url,
            "GET",
            path,
            expected_status=404,
            params=params,
            timeout_s=timeout_s,
        )
        statuses[name] = 404
    return {
        "cross_account_run_denied": True,
        "cross_account_expected_statuses": statuses,
    }


def solver_chat(base_url: str, account_id: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    request_payload = {
        "selected_benchmark": "function_approx",
        "mode": "mock",
        "workspace_scope": "account",
        "account_id": account_id,
        **payload,
    }
    return request_json(base_url, "POST", "/api/solver/chat", request_payload, timeout_s=timeout_s)


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    status, parsed = request_payload(base_url, method, path, payload, params=params, timeout_s=timeout_s)
    if status != 200:
        raise SmokeFailure(f"{method} {path} returned HTTP {status}: {parsed}")
    if not isinstance(parsed, dict):
        raise SmokeFailure(f"{method} {path} returned a non-object JSON payload")
    return parsed


def expect_http_status(
    base_url: str,
    method: str,
    path: str,
    *,
    expected_status: int,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    status, parsed = request_payload(base_url, method, path, payload, params=params, timeout_s=timeout_s)
    expect(status == expected_status, f"{method} {path} returned HTTP {status}; expected {expected_status}: {parsed}")
    return parsed if isinstance(parsed, dict) else {"body": parsed}


def request_payload(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: float,
) -> tuple[int, Any]:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{base_url}{path}{query}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except URLError as exc:
        raise SmokeFailure(f"{method} {path} failed: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        if status == 200:
            raise SmokeFailure(f"{method} {path} returned non-JSON body") from exc
        parsed = body
    return status, parsed


def validate_code_server_payload(payload: dict[str, Any], *, allow_non_loopback_url: bool) -> None:
    url = str(payload.get("url", ""))
    command_hint = str(payload.get("command_hint", ""))
    expect(payload["auth_mode"] == "upstream_account", "code-server payload must use upstream account auth")
    if allow_non_loopback_url:
        expect(url.startswith("https://"), "deployed code-server URL must use HTTPS")
    else:
        expect(url.startswith("http://127.0.0.1:8080/"), "code-server URL must default to loopback")
    expect("PASSWORD=" not in url, "code-server URL leaked a PASSWORD value")
    expect("PASSWORD=" not in command_hint, "code-server command leaked a PASSWORD value")
    expect("--auth password" not in command_hint, "code-server command reintroduced sidecar password auth")
    expect("--auth none" in command_hint, "code-server command does not disable sidecar password")
    expect("0.0.0.0" not in command_hint, "code-server command binds to a public interface")
    expect("--bind-addr 127.0.0.1:8080" in command_hint, "code-server command is not loopback-bound")


def validate_code_server_live(url: str, *, timeout_s: float) -> None:
    request = Request(url, headers={"Accept": "text/html"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read(131072).decode("utf-8", errors="replace")
            status = response.status
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise SmokeFailure(f"code-server returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SmokeFailure(f"code-server sidecar request failed: {exc.reason}") from exc
    expect(status == 200, f"code-server returned unexpected HTTP status {status}")
    lower = body.lower()
    login_markers = ('name="password"', 'id="password"', "enter password", "login")
    expect(not any(marker in lower for marker in login_markers), "code-server still appears to show a password login")


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


if __name__ == "__main__":
    raise SystemExit(main())
