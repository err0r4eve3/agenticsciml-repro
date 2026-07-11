from __future__ import annotations

import copy
import ast
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agenticsciml.benchmarks import BenchmarkContractFactory, ProblemBundle
from agenticsciml.config import EvaluationContract
from agenticsciml.llm.budget import (
    _hash_payload,
    _hash_text,
    estimate_orchestrator_llm_call_range,
)
from agenticsciml.observations import render_structured_data_analysis


PreRootResumeStage = Literal["initialized", "data_ready", "contract_ready"]
MIN_ROOT_REAL_LLM_CALLS = estimate_orchestrator_llm_call_range(
    max_iterations=0,
    parallel_mutations=1,
)["min"]

_BUDGET_LIMIT_FIELDS = (
    "max_prompt_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "max_calls",
    "max_cost_usd",
)
_BUDGET_RATE_FIELD = "cost_per_1k_tokens_usd"
_BUDGET_FIELDS = {*_BUDGET_LIMIT_FIELDS, _BUDGET_RATE_FIELD}


@dataclass(frozen=True, slots=True)
class PreRootResumeState:
    stage: PreRootResumeStage
    completed_llm_calls: int
    data_report: str | None = None
    contract: EvaluationContract | None = None
    approval_status: str | None = None


def inspect_pre_root_resume_state(
    run_dir: Path,
    problem_bundle: ProblemBundle,
) -> PreRootResumeState:
    """Validate and classify a checkpoint-free run without inventing progress."""

    reports_dir = run_dir / "reports"
    report_path = reports_dir / "data_analysis.md"
    structured_path = reports_dir / "data_analysis_structured.json"
    contract_path = run_dir / "evaluation_contract.json"
    approval_path = run_dir / "evaluation_approval.json"

    _reject_post_root_artifacts(run_dir)

    report_exists = report_path.is_file()
    structured_exists = structured_path.is_file()
    if report_exists != structured_exists:
        raise ValueError(
            "Pre-root data analysis artifacts are incomplete; both data_analysis.md and "
            "data_analysis_structured.json are required"
        )
    if (contract_path.exists() or approval_path.exists()) and not report_exists:
        raise ValueError("Pre-root evaluation artifacts exist without completed data analysis")
    if approval_path.exists() and not contract_path.is_file():
        raise ValueError("Pre-root evaluation approval exists without an evaluation contract")

    data_report: str | None = None
    if report_exists:
        try:
            data_report = report_path.read_text(encoding="utf-8")
            structured = _load_json_object(structured_path, "structured data analysis")
        except (OSError, UnicodeError) as exc:
            raise ValueError("Pre-root data analysis artifacts are unreadable") from exc
        if not data_report.strip():
            raise ValueError("Pre-root data analysis report is empty")
        if structured.get("schema_version") != 1:
            raise ValueError("Pre-root structured data analysis schema is unsupported")
        if structured.get("benchmark_name") != problem_bundle.benchmark_name:
            raise ValueError("Pre-root structured data analysis benchmark mismatch")
        if not isinstance(structured.get("training_array_keys"), list) or not structured.get(
            "training_array_keys"
        ):
            raise ValueError("Pre-root structured data analysis has no training array keys")
        if structured.get("private_label_boundary") != "training_data_only_no_validation_labels":
            raise ValueError("Pre-root structured data analysis private-label boundary is invalid")
        if not isinstance(structured.get("llm_report_summary"), str):
            raise ValueError("Pre-root structured data analysis LLM summary is invalid")
        if data_report != render_structured_data_analysis(structured):
            raise ValueError("Pre-root data analysis Markdown does not match structured JSON")
        data_message = _load_transcript(
            run_dir / "transcripts" / "data_analyst.json", "data_analyst"
        )
        if data_message["response"] != structured.get("llm_report_summary"):
            raise ValueError("Pre-root data transcript response does not match structured JSON")

    contract: EvaluationContract | None = None
    if contract_path.is_file():
        try:
            contract = EvaluationContract.from_dict(
                _load_json_object(contract_path, "evaluation contract")
            )
            BenchmarkContractFactory.verify_contract(problem_bundle, contract)
        except Exception as exc:
            raise ValueError("Pre-root evaluation contract is stale or invalid") from exc
        guidelines_path = reports_dir / "evaluation_contract.md"
        expected_guidelines = BenchmarkContractFactory.create_guidelines(problem_bundle, contract)
        try:
            stored_guidelines = guidelines_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("Pre-root evaluation contract report is missing") from exc
        if stored_guidelines != expected_guidelines:
            raise ValueError("Pre-root evaluation contract report does not match contract")
        _load_transcript(run_dir / "transcripts" / "evaluator.json", "evaluator")

    approval_status: str | None = None
    if approval_path.is_file():
        approval = _load_json_object(approval_path, "evaluation approval")
        if approval.get("schema_version") != 1:
            raise ValueError("Pre-root evaluation approval schema is unsupported")
        if contract is None:
            raise ValueError("Pre-root evaluation approval exists without a valid contract")
        if approval.get("benchmark_name") != contract.benchmark_name:
            raise ValueError("Pre-root evaluation approval benchmark mismatch")
        if approval.get("contract_hash") != contract.contract_hash:
            raise ValueError("Pre-root evaluation approval contract mismatch")
        if not isinstance(approval.get("approval_required"), bool):
            raise ValueError("Pre-root evaluation approval metadata is invalid")
        approval_status = approval.get("status")
        if approval_status not in {"approved", "auto_approved", "pending", "rejected"}:
            raise ValueError("Pre-root evaluation approval status is invalid")

    if contract is not None:
        return PreRootResumeState(
            stage="contract_ready",
            completed_llm_calls=2,
            data_report=data_report,
            contract=contract,
            approval_status=approval_status,
        )
    if data_report is not None:
        return PreRootResumeState(
            stage="data_ready",
            completed_llm_calls=1,
            data_report=data_report,
        )
    return PreRootResumeState(stage="initialized", completed_llm_calls=0)


def validate_pre_root_real_llm_evidence(
    run_dir: Path,
    state: PreRootResumeState,
) -> None:
    """Bind completed pre-root stages to successful ledger and generation evidence."""

    validate_real_llm_ledger_trace_consistency(
        run_dir,
        minimum_calls=state.completed_llm_calls,
    )

    required = []
    if state.completed_llm_calls >= 1:
        required.append(("data_analyst", None))
    if state.completed_llm_calls >= 2:
        required.append(("evaluator", "evaluator"))
    if not required:
        return

    ledger_path = run_dir / "llm_call_ledger.jsonl"
    trace_path = run_dir / "trace.jsonl"
    ledger = _load_jsonl_objects(ledger_path, "LLM call ledger")
    traces = _load_jsonl_objects(trace_path, "trace")
    ledger_by_id: dict[str, dict[str, object]] = {}
    for row in ledger:
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in ledger_by_id:
            raise ValueError("Pre-root LLM ledger has an invalid or duplicate call_id")
        ledger_by_id[call_id] = row

    for role, schema_name in required:
        transcript = _load_transcript(run_dir / "transcripts" / f"{role}.json", role)
        expected_prompt_hash = _hash_text(transcript["prompt"])
        if schema_name is None:
            expected_response_hash = _hash_text(transcript["response"])
        else:
            try:
                response_payload = ast.literal_eval(transcript["response"])
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Pre-root {role} transcript response is not structured") from exc
            if not isinstance(response_payload, dict):
                raise ValueError(f"Pre-root {role} transcript response is not a JSON object")
            expected_response_hash = _hash_payload(response_payload)
        matching_call_ids = {
            metadata.get("llm_call_id")
            for event in traces
            if event.get("event_type") == "generation_span"
            and event.get("name") == role
            and isinstance((metadata := event.get("metadata")), dict)
            and isinstance(metadata.get("llm_call_id"), str)
        }
        valid_rows = [
            ledger_by_id[call_id]
            for call_id in matching_call_ids
            if call_id in ledger_by_id
            and ledger_by_id[call_id].get("success") is True
            and ledger_by_id[call_id].get("schema_name") == schema_name
            and ledger_by_id[call_id].get("prompt_hash") == expected_prompt_hash
            and ledger_by_id[call_id].get("response_hash") == expected_response_hash
        ]
        if not valid_rows:
            raise ValueError(
                f"Pre-root {role} artifacts are not bound to a successful ledger/trace call"
            )


def validate_real_llm_ledger_trace_consistency(
    run_dir: Path,
    *,
    minimum_calls: int,
) -> dict[str, int]:
    """Require a one-to-one binding between billable ledger rows and generation traces."""

    ledger_path = run_dir / "llm_call_ledger.jsonl"
    trace_path = run_dir / "trace.jsonl"
    ledger = (
        _load_jsonl_objects(ledger_path, "LLM call ledger")
        if ledger_path.is_file()
        else []
    )
    traces = _load_jsonl_objects(trace_path, "trace") if trace_path.is_file() else []

    ledger_by_id: dict[str, dict[str, object]] = {}
    for row in ledger:
        call_id = row.get("call_id")
        if (
            not isinstance(call_id, str)
            or re.fullmatch(r"llm_call_(\d{6})", call_id) is None
            or call_id in ledger_by_id
        ):
            raise ValueError("Real LLM ledger has an invalid or duplicate call_id")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"Real LLM ledger row {call_id} has invalid success metadata")
        _validate_real_llm_call_metadata(row, call_id, source="ledger")
        ledger_by_id[call_id] = row

    expected_ids = {f"llm_call_{index:06d}" for index in range(1, len(ledger_by_id) + 1)}
    if set(ledger_by_id) != expected_ids:
        raise ValueError("Real LLM ledger call IDs are not contiguous from llm_call_000001")

    trace_by_id: dict[str, dict[str, object]] = {}
    for event in traces:
        if event.get("event_type") != "generation_span":
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        call_id = metadata.get("llm_call_id")
        if call_id is None:
            continue
        if (
            not isinstance(call_id, str)
            or re.fullmatch(r"llm_call_(\d{6})", call_id) is None
            or call_id in trace_by_id
        ):
            raise ValueError("Real LLM trace has an invalid or duplicate llm_call_id")
        _validate_real_llm_call_metadata(metadata, call_id, source="trace")
        trace_by_id[call_id] = metadata

    ledger_ids = set(ledger_by_id)
    trace_ids = set(trace_by_id)
    if ledger_ids != trace_ids:
        raise ValueError(
            "Real LLM ledger/trace call IDs mismatch: "
            f"ledger_only={sorted(ledger_ids - trace_ids)}, "
            f"trace_only={sorted(trace_ids - ledger_ids)}"
        )
    if len(ledger_ids) < minimum_calls:
        raise ValueError(
            "Real LLM ledger/trace evidence requires at least "
            f"{minimum_calls} bound calls; found {len(ledger_ids)}"
        )

    for call_id in sorted(ledger_ids):
        ledger_row = ledger_by_id[call_id]
        trace_metadata = trace_by_id[call_id]
        for field in ("method", "schema_name", "provider", "model", "adapter_type"):
            if ledger_row.get(field) != trace_metadata.get(field):
                raise ValueError(
                    f"Real LLM ledger/trace {field} mismatch for {call_id}"
                )
    return {
        "ledger_call_count": len(ledger_ids),
        "trace_call_count": len(trace_ids),
    }


def _validate_real_llm_call_metadata(
    metadata: dict[str, object],
    call_id: str,
    *,
    source: str,
) -> None:
    method = metadata.get("method")
    allowed_methods = {
        "complete_text",
        "complete_json",
        "complete_json_with_images",
    }
    if method not in allowed_methods:
        raise ValueError(
            f"Real LLM {source} row {call_id} has invalid method metadata"
        )
    for field in ("provider", "model", "adapter_type"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Real LLM {source} row {call_id} has invalid {field} metadata"
            )
    schema_name = metadata.get("schema_name")
    if method == "complete_text":
        if schema_name is not None:
            raise ValueError(
                f"Real LLM {source} row {call_id} has invalid schema_name metadata"
            )
    elif not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError(
            f"Real LLM {source} row {call_id} has invalid schema_name metadata"
        )


def validate_resume_conditions_compatible(
    stored_conditions: dict[str, object],
    current_conditions: dict[str, object],
) -> None:
    """Allow only an auditable relaxation of run-level LLM ceilings on resume."""

    stored = copy.deepcopy(stored_conditions)
    current = copy.deepcopy(current_conditions)
    _normalize_legacy_clean_source_revision(stored, current)
    stored_budget = _pop_budget_limits(stored)
    current_budget = _pop_budget_limits(current)
    if stored != current:
        raise ValueError("Resume experiment conditions changed outside LLM budget limits")
    if stored_budget is None and current_budget is None:
        return
    if stored_budget is None or current_budget is None:
        raise ValueError("Resume LLM budget configuration is missing")
    _validate_budget_relaxation(stored_budget, current_budget)


def read_source_revision(benchmark_dir: Path) -> dict[str, object]:
    """Bind a run to its commit and effective local Python source tree."""

    cwd = benchmark_dir.resolve()
    try:
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        ).resolve()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        listed = subprocess.run(
            [
                "git",
                "ls-files",
                "-co",
                "--exclude-standard",
                "-z",
                "--",
                "src",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=5,
        ).stdout
        if not commit:
            raise ValueError("empty git commit")
        digest, file_count = _runtime_source_digest(repo_root, listed)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return {
            "commit": "unknown",
            "dirty": "unknown",
            "runtime_source_digest": "unknown",
            "runtime_source_file_count": "unknown",
        }
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "runtime_source_digest": digest,
        "runtime_source_file_count": file_count,
    }


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid {label}: expected a JSON object")
    return payload


def _load_jsonl_objects(path: Path, label: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Invalid or missing {label}: {path}") from exc
    if not lines:
        raise ValueError(f"Invalid or missing {label}: {path}")
    payloads: list[dict[str, object]] = []
    for line in lines:
        try:
            payload = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid {label}: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid {label}: expected JSON objects")
        payloads.append(payload)
    return payloads


def _load_transcript(path: Path, role: str) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Pre-root {role} transcript is invalid or missing") from exc
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Pre-root {role} transcript is empty")
    if not all(
        isinstance(message, dict)
        and message.get("role") == role
        and isinstance(message.get("prompt"), str)
        and isinstance(message.get("response"), str)
        for message in payload
    ):
        raise ValueError(f"Pre-root {role} transcript payload is invalid")
    message = payload[0]
    assert isinstance(message, dict)
    return {
        "prompt": str(message["prompt"]),
        "response": str(message["response"]),
    }


def _reject_post_root_artifacts(run_dir: Path) -> None:
    forbidden = (
        "tree.json",
        "run_metadata.json",
        "leaderboard.csv",
        "checkpoint.json",
    )
    present = [name for name in forbidden if (run_dir / name).exists()]
    solutions_dir = run_dir / "solutions"
    if solutions_dir.is_dir() and any(solutions_dir.iterdir()):
        present.append("solutions/*")
    champion_dir = run_dir / "champion"
    if champion_dir.is_dir() and any(champion_dir.iterdir()):
        present.append("champion/*")
    if present:
        raise ValueError(
            "Checkpoint-free run contains post-root artifacts: " + ", ".join(present)
        )


def _pop_budget_limits(conditions: dict[str, object]) -> dict[str, object] | None:
    runtime = conditions.get("llm_runtime")
    if not isinstance(runtime, dict):
        return None
    budget = runtime.pop("budget_limits", None)
    if budget is None:
        return None
    if not isinstance(budget, dict):
        raise ValueError("Resume LLM budget limits are invalid")
    return budget


def _normalize_legacy_clean_source_revision(
    stored: dict[str, object],
    current: dict[str, object],
) -> None:
    stored_source = stored.get("source_revision")
    current_source = current.get("source_revision")
    if not isinstance(stored_source, dict) or not isinstance(current_source, dict):
        return
    if "runtime_source_digest" in stored_source:
        return
    if stored_source.get("dirty") is not False:
        return
    if stored_source.get("commit") != current_source.get("commit"):
        return
    current_source.pop("runtime_source_digest", None)
    current_source.pop("runtime_source_file_count", None)


def _runtime_source_digest(repo_root: Path, nul_paths: bytes) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for raw_path in sorted(path for path in nul_paths.split(b"\0") if path):
        relative = Path(os.fsdecode(raw_path))
        absolute = repo_root / relative
        if absolute.is_symlink():
            content = os.readlink(absolute).encode("utf-8", errors="surrogateescape")
            kind = b"symlink"
        elif absolute.is_file():
            content = absolute.read_bytes()
            kind = b"file"
        else:
            raise ValueError(f"runtime source path is not a file: {relative}")
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(kind)
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    if count == 0:
        raise ValueError("runtime source manifest is empty")
    return digest.hexdigest(), count


def _validate_budget_relaxation(
    stored: dict[str, object],
    current: dict[str, object],
) -> None:
    if set(stored) != _BUDGET_FIELDS or set(current) != _BUDGET_FIELDS:
        raise ValueError("Resume LLM budget limit schema is invalid")
    _budget_number_or_none(stored[_BUDGET_RATE_FIELD], _BUDGET_RATE_FIELD)
    _budget_number_or_none(current[_BUDGET_RATE_FIELD], _BUDGET_RATE_FIELD)
    if stored[_BUDGET_RATE_FIELD] != current[_BUDGET_RATE_FIELD]:
        raise ValueError("Resume LLM budget cost rate cannot change")
    for field in _BUDGET_LIMIT_FIELDS:
        old = _budget_number_or_none(stored[field], field)
        new = _budget_number_or_none(current[field], field)
        if old is None:
            if new is not None:
                raise ValueError(f"Resume LLM budget {field} cannot be tightened")
            continue
        if new is not None and new < old:
            raise ValueError(f"Resume LLM budget {field} cannot be tightened")


def _budget_number_or_none(value: object, field: str) -> int | float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Resume LLM budget {field} is invalid")
    return value
