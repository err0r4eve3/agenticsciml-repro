from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from agenticsciml.web.app import create_app


def test_web_benchmarks_match_catalog() -> None:
    client = TestClient(create_app())

    response = client.get("/api/benchmarks")

    assert response.status_code == 200
    names = {item["name"] for item in response.json()["benchmarks"]}
    assert "function_approx" in names


def test_web_mock_run_writes_required_artifacts(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "mock",
            "max_iterations": 0,
            "parallel_mutations": 1,
            "experiment_id": "web-test",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    run_dir = tmp_path / "web-test"
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "trace_summary.json").exists()
    assert payload["metadata"]["run_state"] == "exported"
    assert payload["trace_summary"]["quality_gate"]["passed"] is True


def test_web_artifacts_reject_path_escape(tmp_path: Path) -> None:
    client = TestClient(create_app())
    run_dir = tmp_path / "web-test"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text("", encoding="utf-8")
    (run_dir / "safe.txt").write_text("ok", encoding="utf-8")

    safe = client.get(
        "/api/runs/web-test/artifacts/safe.txt",
        params={"output_dir": str(tmp_path)},
    )
    assert safe.status_code == 200
    assert safe.json()["content"] == "ok"

    encoded_dotdot = client.get(
        "/api/runs/web-test/artifacts/%2E%2E/pyproject.toml",
        params={"output_dir": str(tmp_path)},
    )
    assert encoded_dotdot.status_code in {400, 404}

    if hasattr(os, "symlink"):
        os.symlink(tmp_path, run_dir / "escape")
        symlink_escape = client.get(
            "/api/runs/web-test/artifacts/escape",
            params={"output_dir": str(tmp_path)},
        )
        assert symlink_escape.status_code == 400


def test_solver_chat_returns_structured_actions_and_warnings(tmp_path: Path) -> None:
    client = TestClient(create_app())

    start = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )
    assert start.status_code == 200
    start_payload = start.json()
    assert start_payload["actions"][0]["type"] == "start_run"
    assert start_payload["actions"][0]["payload"]["background"] is True

    real_dry = client.post(
        "/api/solver/chat",
        json={
            "message": "解释 trace",
            "selected_benchmark": "function_approx",
            "mode": "real",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )
    assert real_dry.status_code == 200
    warnings = real_dry.json()["warnings"]
    assert any("Real LLM mode" in warning for warning in warnings)
    assert any("Select an active run" in warning for warning in warnings)


def test_code_server_url_uses_loopback_without_token(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.get("/api/code-server/url", params={"scope": "repo"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["url"].startswith("http://127.0.0.1:8080/")
    assert "PASSWORD=" not in payload["url"]
    assert payload["workspace"].endswith("New project 11")


def test_code_server_workspaces_list_independent_directories(tmp_path: Path) -> None:
    client = TestClient(create_app())
    run_dir = tmp_path / "web-test"
    (run_dir / "champion").mkdir(parents=True)
    (run_dir / "solutions" / "solution_000").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        '{"run_state": "exported", "champion_node_id": "solution_000"}',
        encoding="utf-8",
    )

    response = client.get(
        "/api/code-server/workspaces",
        params={"run_id": "web-test", "output_dir": str(tmp_path)},
    )

    assert response.status_code == 200
    workspaces = {item["id"]: item for item in response.json()["workspaces"]}
    assert workspaces["repo"]["workspace"].endswith("New project 11")
    assert workspaces["run:web-test"]["workspace"] == str(run_dir.resolve())
    assert workspaces["champion:web-test"]["workspace"] == str((run_dir / "champion").resolve())
    assert workspaces["solution:web-test:solution_000"]["workspace"] == str(
        (run_dir / "solutions" / "solution_000").resolve()
    )
    for item in workspaces.values():
        assert "PASSWORD=" not in item["url"]
