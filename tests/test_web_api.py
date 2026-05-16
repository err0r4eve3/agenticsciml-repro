from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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


def test_web_algorithms_expose_professional_catalog() -> None:
    client = TestClient(create_app())

    response = client.get("/api/algorithms")

    assert response.status_code == 200
    payload = response.json()
    algorithms = {item["id"]: item for item in payload["algorithms"]}
    assert "Algorithm catalog entries are planning" in payload["claim_boundary"]
    assert "pinn_residual_minimizer" in algorithms
    assert "weak_form_pinn" in algorithms
    assert "xpinn_domain_decomposition" in algorithms
    assert "deeponet_operator" in algorithms
    assert "kernel_surrogate_regression" in algorithms
    assert "sindy_sparse_discovery" in algorithms
    assert "paper_sigmoid_moe_gate" in algorithms
    assert "paper_linear_bias_free_deeponet" in algorithms
    assert algorithms["paper_sigmoid_moe_gate"]["status"] == "reference_implementation"
    assert algorithms["paper_sigmoid_moe_gate"]["implementation_path"]
    assert algorithms["pinn_residual_minimizer"]["status"] == "strategy_blueprint"
    assert len(algorithms) >= 20
    assert "paper-level" in algorithms["sparse_sensor_reconstructor"]["safety_notes"]


def test_solver_settings_expose_mode_defaults() -> None:
    client = TestClient(create_app())

    response = client.get("/api/solver/settings")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_assistant_mode"] == "ask"
    assert payload["reasoning_efforts"] == ["low", "medium", "high"]
    assert payload["temperature_range"] == [0.0, 2.0]
    assert payload["assistant_modes"] == {
        "ask": {"reasoning_effort": "medium", "temperature": 0.2},
        "plan": {"reasoning_effort": "high", "temperature": 0.35},
        "agent": {"reasoning_effort": "high", "temperature": 0.1},
    }


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


def test_web_real_run_requires_server_flag_and_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    monkeypatch.delenv("AGENTICSCIML_ENABLE_REAL_WEB_RUNS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app())

    disabled = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "real",
            "account_id": "alice",
            "experiment_id": "real-web-test",
            "real_confirmed": True,
        },
    )
    assert disabled.status_code == 403
    assert "disabled" in disabled.json()["detail"]

    monkeypatch.setenv("AGENTICSCIML_ENABLE_REAL_WEB_RUNS", "1")
    unconfirmed = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "real",
            "account_id": "alice",
            "experiment_id": "real-web-test",
        },
    )
    assert unconfirmed.status_code == 400
    assert "real_confirmed" in unconfirmed.json()["detail"]

    missing_key = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "real",
            "account_id": "alice",
            "experiment_id": "real-web-missing-key",
            "real_confirmed": True,
        },
    )
    assert missing_key.status_code == 200
    missing_key_payload = missing_key.json()
    assert missing_key_payload["status"] == "failed"
    assert "OPENAI_API_KEY" in missing_key_payload["record"]["error"]


def test_account_scoped_runs_reject_path_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    output_override = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "mode": "dry_run",
            "account_id": "alice",
            "experiment_id": "path-override",
            "output_dir": str(tmp_path),
        },
    )
    assert output_override.status_code == 400
    assert "account runs directory" in output_override.json()["detail"]

    benchmark_dir_override = client.post(
        "/api/runs",
        json={
            "benchmark": "function_approx",
            "benchmark_dir": "examples/function_approx",
            "mode": "dry_run",
            "account_id": "alice",
            "experiment_id": "benchmark-dir-override",
        },
    )
    assert benchmark_dir_override.status_code == 400
    assert "benchmark_dir" in benchmark_dir_override.json()["detail"]

    benchmark_path = client.post(
        "/api/runs",
        json={
            "benchmark": "examples/function_approx",
            "mode": "dry_run",
            "account_id": "alice",
            "experiment_id": "benchmark-path",
        },
    )
    assert benchmark_path.status_code == 400
    assert "catalog names" in benchmark_path.json()["detail"]


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


def test_solver_chat_defaults_to_ask_without_actions(tmp_path: Path) -> None:
    client = TestClient(create_app())

    identity = client.post(
        "/api/solver/chat",
        json={
            "message": "你是谁",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert identity.status_code == 200
    identity_payload = identity.json()
    assert identity_payload["assistant_mode"] == "ask"
    assert identity_payload["model_settings"] == {
        "reasoning_effort": "medium",
        "temperature": 0.2,
        "source": "mode_default",
    }
    assert identity_payload["actions"] == []
    assert identity_payload["warnings"] == []
    assert "AgenticSciML 助手" in identity_payload["reply"]

    capabilities = client.post(
        "/api/solver/chat",
        json={
            "message": "你能做什么",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert capabilities.status_code == 200
    capabilities_payload = capabilities.json()
    assert capabilities_payload["assistant_mode"] == "ask"
    assert capabilities_payload["actions"] == []
    assert capabilities_payload["warnings"] == []
    assert "benchmark" in capabilities_payload["reply"]
    assert "Plan 模式" in capabilities_payload["reply"]
    assert "Agent 模式" in capabilities_payload["reply"]

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
    assert start_payload["assistant_mode"] == "ask"
    assert start_payload["actions"] == []
    assert any("Ask mode" in warning for warning in start_payload["warnings"])


def test_solver_chat_mode_model_settings_are_distinct(tmp_path: Path) -> None:
    client = TestClient(create_app())

    expected = {
        "ask": {"reasoning_effort": "medium", "temperature": 0.2, "source": "mode_default"},
        "plan": {"reasoning_effort": "high", "temperature": 0.35, "source": "mode_default"},
        "agent": {"reasoning_effort": "high", "temperature": 0.1, "source": "mode_default"},
    }
    for assistant_mode, settings in expected.items():
        response = client.post(
            "/api/solver/chat",
            json={
                "message": "跑一个 mock 实验",
                "selected_benchmark": "function_approx",
                "mode": "mock",
                "assistant_mode": assistant_mode,
                "workspace_scope": "account",
                "account_id": "alice",
                "output_dir": str(tmp_path),
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["assistant_mode"] == assistant_mode
        assert payload["model_settings"] == settings

    override = client.post(
        "/api/solver/chat",
        json={
            "message": "你能做什么",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "ask",
            "reasoning_effort": "high",
            "temperature": 0.45,
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert override.status_code == 200
    assert override.json()["model_settings"] == {
        "reasoning_effort": "high",
        "temperature": 0.45,
        "source": "request_override",
    }


def test_solver_chat_plan_and_agent_modes_return_structured_actions(tmp_path: Path) -> None:
    client = TestClient(create_app())

    plan = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "plan",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )
    assert plan.status_code == 200
    plan_payload = plan.json()
    assert plan_payload["assistant_mode"] == "plan"
    assert plan_payload["actions"][0]["type"] == "start_run"
    assert any("Plan mode" in warning for warning in plan_payload["warnings"])

    start = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "account_id": "alice",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )
    assert start.status_code == 200
    start_payload = start.json()
    assert start_payload["assistant_mode"] == "agent"
    assert start_payload["actions"][0]["type"] == "start_run"
    assert start_payload["actions"][0]["payload"]["account_id"] == "alice"
    assert start_payload["actions"][0]["payload"]["background"] is True


def test_solver_chat_agent_rejects_shared_repo_scope(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/solver/chat",
        json={
            "message": "打开代码",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "account_id": "alice",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["actions"] == []
    assert any("shared repo workspace" in warning for warning in payload["warnings"])


def test_solver_chat_agent_requires_account_id(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/solver/chat",
        json={
            "message": "跑一个 mock 实验",
            "selected_benchmark": "function_approx",
            "mode": "mock",
            "assistant_mode": "agent",
            "workspace_scope": "account",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["actions"] == []
    assert any("requires account_id" in warning for warning in payload["warnings"])


def test_solver_chat_returns_real_mode_warnings(tmp_path: Path) -> None:
    client = TestClient(create_app())

    real_dry = client.post(
        "/api/solver/chat",
        json={
            "message": "解释 trace",
            "selected_benchmark": "function_approx",
            "mode": "real",
            "assistant_mode": "ask",
            "workspace_scope": "repo",
            "output_dir": str(tmp_path),
        },
    )
    assert real_dry.status_code == 200
    warnings = real_dry.json()["warnings"]
    assert any("Real LLM mode" in warning for warning in warnings)
    assert any("Select an active run" in warning for warning in warnings)


def test_code_server_url_rejects_repo_workspace_by_default() -> None:
    client = TestClient(create_app())

    response = client.get("/api/code-server/url", params={"scope": "repo"})

    assert response.status_code == 403
    assert "shared repo workspace is disabled" in response.json()["detail"]


def test_code_server_url_uses_loopback_without_token_when_repo_workspace_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ALLOW_REPO_WORKSPACE", "1")
    client = TestClient(create_app())

    response = client.get("/api/code-server/url", params={"scope": "repo"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["url"].startswith("http://127.0.0.1:8080/")
    assert "PASSWORD=" not in payload["url"]
    assert payload["workspace"].endswith("New project 11")


def test_code_server_workspaces_list_independent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTICSCIML_ALLOW_REPO_WORKSPACE", "1")
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


def test_account_workspaces_are_isolated_local_namespaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    created = client.post("/api/accounts", json={"account_id": "alice", "display_name": "Alice"})
    assert created.status_code == 200
    account = created.json()["account"]
    assert account["account_id"] == "alice"
    assert account["auth"] == "not_implemented"
    assert Path(account["workspace_root"]).is_relative_to(tmp_path / "accounts" / "alice")

    run_dir = Path(account["runs_dir"]) / "alice-run"
    (run_dir / "solutions" / "solution_000").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text('{"run_state": "partial"}', encoding="utf-8")

    response = client.get("/api/code-server/workspaces", params={"account_id": "alice"})

    assert response.status_code == 200
    workspaces = {item["id"]: item for item in response.json()["workspaces"]}
    assert "repo" not in workspaces
    assert workspaces["account:alice"]["scope"] == "account"
    assert workspaces["account:alice"]["isolation"] == "account"
    assert workspaces["account:alice"]["workspace"] == account["workspace_root"]
    assert workspaces["run:alice-run"]["workspace"] == str(run_dir.resolve())
    assert workspaces["solution:alice-run:solution_000"]["account_id"] == "alice"

    invalid = client.get("/api/code-server/workspaces", params={"account_id": "../alice"})
    assert invalid.status_code == 400


def test_accounts_endpoint_tolerates_parallel_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    with ThreadPoolExecutor(max_workers=6) as executor:
        responses = list(executor.map(lambda _: client.get("/api/accounts"), range(12)))

    assert {response.status_code for response in responses} == {200}
    assert all(response.json()["accounts"][0]["account_id"] == "local" for response in responses)


def test_account_run_records_do_not_collide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTICSCIML_ACCOUNTS_ROOT", str(tmp_path / "accounts"))
    client = TestClient(create_app())

    for account_id in ("alice", "bob"):
        response = client.post(
            "/api/runs",
            json={
                "benchmark": "function_approx",
                "mode": "dry_run",
                "account_id": account_id,
                "experiment_id": "same-run-id",
            },
        )
        assert response.status_code == 200

    alice = client.get("/api/runs/same-run-id", params={"account_id": "alice"})
    bob = client.get("/api/runs/same-run-id", params={"account_id": "bob"})

    assert alice.status_code == 200
    assert bob.status_code == 200
    alice_record = alice.json()["record"]
    bob_record = bob.json()["record"]
    assert alice_record["status"] == "dry_run"
    assert bob_record["status"] == "dry_run"
    assert "/alice/" in alice_record["run_dir"]
    assert "/bob/" in bob_record["run_dir"]
    assert alice_record["run_dir"] != bob_record["run_dir"]
