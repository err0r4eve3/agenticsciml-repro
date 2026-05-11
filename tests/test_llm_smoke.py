import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agenticsciml.evidence import EVIDENCE_MODE_REAL_LLM_SMOKE
from agenticsciml.llm.mock import MockLLMClient
from agenticsciml.llm_smoke import _paired_contrast_gate, _smoke_gate, run_llm_smoke, verify_llm_smoke_output


def _rewrite_parallel_child_max_workers(run_dir: Path, value: object) -> None:
    trace_path = run_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = False
    for event in events:
        if event.get("name") == "agenticsciml.parallel_children.start":
            event.setdefault("metadata", {})["max_workers"] = value
            changed = True
    assert changed, "expected parallel_children.start trace event"
    trace_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )


def _rewrite_run_metadata_llm_calls_total(run_dir: Path, value: object) -> None:
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("llm_calls", {})["total"] = value
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_first_ledger_field(run_dir: Path, field: str, value: object) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries, "expected at least one ledger entry"
    entries[0][field] = value
    ledger_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _delete_first_ledger_field(run_dir: Path, field: str) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries, "expected at least one ledger entry"
    entries[0].pop(field, None)
    ledger_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _duplicate_second_ledger_call_id(run_dir: Path) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) >= 2, "expected at least two ledger entries"
    entries[1]["call_id"] = entries[0]["call_id"]
    ledger_path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _delete_first_ledger_entry(run_dir: Path) -> None:
    ledger_path = run_dir / "llm_call_ledger.jsonl"
    lines = [line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "expected at least one ledger entry"
    ledger_path.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""), encoding="utf-8")


def test_llm_smoke_dry_run_writes_plan_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=True,
    )
    plan = json.loads(result.plan_json.read_text(encoding="utf-8"))
    report = result.report_md.read_text(encoding="utf-8")

    assert result.runs_csv is None
    assert result.manifest_json.exists()
    assert plan["evidence_mode"] == EVIDENCE_MODE_REAL_LLM_SMOKE
    assert plan["scientific_claim"] == "not_supported"
    assert [entry["variant"] for entry in plan["runs"]] == ["branch_context", "no_branch_context"]
    assert plan["runs"][0]["use_branch_context"] is True
    assert plan["runs"][1]["use_branch_context"] is False
    assert plan["runs"][0]["experiment_id"] == "smoke-branch_context-seed-0"
    assert "runs/smoke-branch_context-seed-0/tree.json" in plan["expected_artifacts"]
    assert "API calls: none" in report
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "dry_run"
    assert manifest["expected_llm_call_range"]["min"] > 0


def test_llm_smoke_dry_run_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown LLM smoke variant"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["unknown_variant"],
            dry_run=True,
        )


def test_llm_smoke_real_mode_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context"],
            dry_run=False,
        )


def test_llm_smoke_real_mode_requires_paired_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact paired variants"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )


def test_llm_smoke_real_mode_rejects_duplicate_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicates are not allowed"):
        run_llm_smoke(
            benchmark_dir=Path("examples/function_approx").resolve(),
            output_dir=tmp_path,
            variants=["branch_context", "no_branch_context", "branch_context"],
            dry_run=False,
            llm_client=MockLLMClient(),
        )


def test_llm_smoke_real_gate_with_scripted_llm(tmp_path: Path) -> None:
    result = run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    rows = list(csv.DictReader(result.runs_csv.open(encoding="utf-8")))  # type: ignore[union-attr]

    assert {row["variant"] for row in rows} == {"branch_context", "no_branch_context"}
    assert all(row["smoke_gate_passed"] == "True" for row in rows)
    assert all(row["trace_quality_gate_passed"] == "True" for row in rows)
    assert all(int(row["llm_calls"]) > 0 for row in rows)
    assert result.manifest_json.exists()
    no_branch = next(row for row in rows if row["variant"] == "no_branch_context")
    assert no_branch["branch_context_enabled"] == "False"
    assert no_branch["branch_intents"] == ""

    verification = verify_llm_smoke_output(tmp_path)
    verification_payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))
    assert verification.passed is True
    assert verification_payload["passed"] is True
    assert len(verification_payload["recomputed_rows"]) == 2
    for row in verification_payload["recomputed_rows"]:
        assert int(row["llm_ledger_calls"]) == int(row["llm_calls"])
        assert int(row["generation_span_count"]) == int(row["llm_calls"])
        assert row["llm_ledger_providers"] == "MockLLMClient"


def test_verify_llm_smoke_output_rejects_dry_run_only(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=True,
    )

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("dry-run outputs are not real-smoke evidence" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_header_only_runs_csv(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    header = rows_path.read_text(encoding="utf-8").splitlines()[0]
    rows_path.write_text(header + "\n", encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("must contain paired run rows" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_stale_external_run_dir(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    rows = list(csv.DictReader(rows_path.open(encoding="utf-8")))
    rows[0]["run_dir"] = str(tmp_path / "external-run")
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("does not match expected" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_manifest_mode_mismatch(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_mode"] = "dry_run"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("manifest execution_mode must be real" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_output_dir_mismatch(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    plan_path = tmp_path / "real_llm_smoke_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["output_dir"] = str(tmp_path / "other-bundle")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("plan output_dir does not match" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_manifest_output_dir_mismatch(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_dir"] = str(tmp_path / "other-bundle")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("manifest output_dir does not match" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_non_string_manifest_provider_model(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provider"] = True
    manifest["model"] = ""
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("manifest provider must be a non-empty string" in issue for issue in payload["issues"])
    assert any("manifest model must be a non-empty string" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_reports_malformed_seed_without_crashing(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    rows = list(csv.DictReader(rows_path.open(encoding="utf-8")))
    rows[0]["seed"] = "not-an-int"
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("row seed must be an integer" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_non_integer_plan_seed(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    plan_path = tmp_path / "real_llm_smoke_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["seed"] = 1.7
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("plan seed must be an integer, not float" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_malformed_expected_call_range(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_llm_call_range"] = {"min": "abc", "max": 0}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("expected_llm_call_range.min must be an integer string" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.max must be >= 1" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_missing_expected_call_range_bounds(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_llm_call_range"] = {}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("expected_llm_call_range.min is required" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.max is required" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_reports_manifest_schema_when_rows_are_damaged(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    rows_path = tmp_path / "real_llm_smoke_runs.csv"
    header = rows_path.read_text(encoding="utf-8").splitlines()[0]
    rows_path.write_text(header + "\n", encoding="utf-8")
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_llm_call_range"] = {"min": "bad", "max": 0}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("must contain paired run rows" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.min must be an integer string" in issue for issue in payload["issues"])
    assert any("expected_llm_call_range.max must be >= 1" in issue for issue in payload["issues"])


@pytest.mark.parametrize("bad_value", [1.5, True])
def test_verify_llm_smoke_output_rejects_non_integer_parallel_mutations(
    tmp_path: Path,
    bad_value: object,
) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    plan_path = tmp_path / "real_llm_smoke_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for entry in plan["runs"]:
        if entry["variant"] == "branch_context":
            entry["parallel_mutations"] = bad_value
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("branch_context: plan parallel_mutations must be an integer" in issue for issue in payload["issues"])


@pytest.mark.parametrize("bad_value", ["abc", True])
def test_verify_llm_smoke_output_rejects_malformed_trace_max_workers(
    tmp_path: Path,
    bad_value: object,
) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _rewrite_parallel_child_max_workers(tmp_path / "runs" / "smoke-branch_context-seed-0", bad_value)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("branch_context: trace max_workers must be an integer" in issue for issue in payload["issues"])


@pytest.mark.parametrize(
    ("bad_value", "expected_issue"),
    [
        ("bad", "branch_context: llm_calls.total must be an integer string"),
        (True, "branch_context: llm_calls.total must be an integer, not bool"),
        (0, "branch_context: llm_calls.total must be positive in real smoke"),
    ],
)
def test_verify_llm_smoke_output_rejects_malformed_llm_calls_total(
    tmp_path: Path,
    bad_value: object,
    expected_issue: str,
) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _rewrite_run_metadata_llm_calls_total(tmp_path / "runs" / "smoke-branch_context-seed-0", bad_value)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any(expected_issue in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_deleted_ledger_call(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _delete_first_ledger_entry(tmp_path / "runs" / "smoke-branch_context-seed-0")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger call count" in issue and "does not match" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_ledger_provider_mismatch(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _rewrite_first_ledger_field(tmp_path / "runs" / "smoke-branch_context-seed-0", "provider", "OtherProvider")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger providers" in issue and "manifest provider" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_manifest_model_mismatch(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    manifest_path = tmp_path / "real_llm_smoke_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = "other-model"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("ledger models" in issue and "manifest model" in issue for issue in payload["issues"])


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("prompt_hash", "not-a-sha", "prompt_hash must be a lowercase sha256 hex string"),
        ("provider", "", "provider must be a non-empty string"),
        ("success", False, "success must be true for completed smoke evidence"),
        ("prompt", "raw prompt text", "contains forbidden raw field"),
        ("raw_messages", [], "contains forbidden raw field"),
        ("completion_text", "raw completion text", "contains forbidden raw field"),
        ("request_payload", {}, "contains forbidden raw field"),
        ("unknown_key", "extra", "contains unknown ledger field"),
        ("duration_s", float("nan"), "duration_s must be a finite number"),
        ("temperature", float("inf"), "temperature must be a finite number"),
        ("span_kind", "tool_span", "span_kind must be generation_span"),
    ],
)
def test_verify_llm_smoke_output_rejects_invalid_ledger_entry_schema(
    tmp_path: Path,
    field: str,
    value: object,
    expected_issue: str,
) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _rewrite_first_ledger_field(tmp_path / "runs" / "smoke-branch_context-seed-0", field, value)

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any(expected_issue in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_missing_ledger_hash(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _delete_first_ledger_field(tmp_path / "runs" / "smoke-branch_context-seed-0", "response_hash")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("response_hash must be a lowercase sha256 hex string" in issue for issue in payload["issues"])


def test_verify_llm_smoke_output_rejects_duplicate_ledger_call_id(tmp_path: Path) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )
    _duplicate_second_ledger_call_id(tmp_path / "runs" / "smoke-branch_context-seed-0")

    verification = verify_llm_smoke_output(tmp_path)
    payload = json.loads(verification.verification_json.read_text(encoding="utf-8"))

    assert verification.passed is False
    assert any("call_id must be llm_call_000002" in issue for issue in payload["issues"])
    assert any("call_id must be unique" in issue for issue in payload["issues"])


def test_cli_verify_smoke_llm_command(tmp_path: Path, cli_env: dict[str, str]) -> None:
    run_llm_smoke(
        benchmark_dir=Path("examples/function_approx").resolve(),
        output_dir=tmp_path,
        variants=["branch_context", "no_branch_context"],
        dry_run=False,
        llm_client=MockLLMClient(),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "verify-smoke-llm",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=cli_env,
    )

    assert Path(result.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_verification.json"


def test_smoke_gate_requires_branch_context_prompt_delivery(tmp_path: Path) -> None:
    run_dir = tmp_path
    child = run_dir / "solutions" / "solution_001" / "transcripts"
    child.mkdir(parents=True)
    (run_dir / "solutions" / "solution_001" / "branch_context.json").write_text(
        json.dumps(
            {
                "branch_intent": "features_or_architecture",
                "sibling_branch_ids": ["solution_002"],
                "diversity_instruction": "distinct branch",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (child / "proposal_debate.json").write_text(
        json.dumps(
            [
                {
                    "role": "proposer",
                    "prompt": "No branch context was delivered in this request.",
                    "response": "branch_intent sibling_branch_ids diversity_instruction",
                }
            ]
        ),
        encoding="utf-8",
    )
    (child / "engineer.json").write_text("[]", encoding="utf-8")
    (run_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "name": "agenticsciml.child_mutation.start",
                "metadata": {"branch_context": {"branch_intent": "features_or_architecture"}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    gate = _smoke_gate(
        run_dir,
        "branch_context",
        {"branch_context_enabled": True, "llm_calls": {"total": 1}},
        {"quality_gate": {"passed": True}},
        [{"node_id": "solution_001", "parent_id": "solution_000"}],
        {"branch:features_or_architecture"},
    )

    assert gate["passed"] is False
    assert any("request prompts missing" in issue for issue in gate["issues"])


def test_paired_contrast_gate_rejects_single_variant() -> None:
    gate = _paired_contrast_gate(
        [
            {
                "variant": "branch_context",
                "smoke_gate_passed": True,
                "branch_context_enabled": True,
                "llm_calls": 1,
                "branch_intents": "features_or_architecture",
            }
        ]
    )

    assert gate["passed"] is False
    assert "missing required variant no_branch_context" in gate["issues"]


def test_cli_smoke_llm_dry_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = {**cli_env}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context,no_branch_context",
            "--dry-run",
            "--max-iterations",
            "1",
            "--parallel-mutations",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    report_path = Path(result.stdout.strip().splitlines()[-1])
    assert report_path.name == "real_llm_smoke_report.md"
    assert (tmp_path / "real_llm_smoke_plan.json").exists()


def test_cli_smoke_llm_defaults_to_dry_run(tmp_path: Path, cli_env: dict[str, str]) -> None:
    env = {**cli_env}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticsciml.cli",
            "smoke-llm",
            "examples/function_approx",
            "--variants",
            "branch_context",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert Path(result.stdout.strip().splitlines()[-1]).name == "real_llm_smoke_report.md"
    plan = json.loads((tmp_path / "real_llm_smoke_plan.json").read_text(encoding="utf-8"))
    assert plan["execution_mode"] == "dry_run"
