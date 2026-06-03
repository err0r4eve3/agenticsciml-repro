from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agenticsciml.ablation_evidence import build_multi_seed_ablation_verified_manifest
from agenticsciml.benchmarks import ProblemBundle
from agenticsciml.evidence import SCIENTIFIC_CLAIM_NOT_SUPPORTED
from agenticsciml.llm.capabilities import capabilities_for_openai_compatible
from agenticsciml.retrieval.kb_store import kb_manifest_for_dir
from agenticsciml.storage import _atomic_write_text


PAPER_WORKFLOW_READINESS_SCHEMA_VERSION = 1
DOMAIN_CHECKLIST_IDS = (
    "evaluator_private_labels_hidden",
    "metric_matches_domain_target",
    "paper_claim_boundary_reviewed",
    "failure_samples_reviewed",
    "physical_constraints_reviewed",
)
PAPER_BENCHMARK_MANIFEST = "paper_benchmark_manifest.json"


def build_paper_workflow_readiness_bundle(
    *,
    benchmark_dir: Path,
    selector_panel: list[dict[str, object]] | None = None,
    selector_evidence_path: Path | None = None,
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env_map = dict(env or {})
    problem_bundle = ProblemBundle.load(benchmark_dir)
    benchmark = problem_bundle.benchmark_spec
    provider = _provider_readiness(env_map)
    selector = _selector_readiness(
        selector_panel or [],
        provider_api_key_present=provider["api_key_present"],
        selector_evidence_path=selector_evidence_path,
    )
    benchmark_gate = _paper_benchmark_readiness(benchmark_dir, benchmark.to_dict())
    domain = _domain_approval_readiness(domain_approval_path)
    ablation = _ablation_readiness(
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds or [],
        expected_variants=expected_variants or [],
    )
    kb_manifest = kb_manifest_for_dir(problem_bundle.benchmark_dir / "kb")
    resource_gate = _resource_readiness(resource_constraints or {}, expert_blueprint_id)

    checks = [
        _check(
            "real_llm_credentials",
            "provider",
            provider["api_key_present"],
            "OPENAI_API_KEY is present for real LLM mode.",
            "Set OPENAI_API_KEY without writing it into artifacts.",
            provider,
        ),
        _check(
            "real_llm_budget",
            "provider",
            provider["budget_configured"],
            "At least one explicit LLM budget gate is configured.",
            "Set AGENTICSCIML_MAX_LLM_CALLS, AGENTICSCIML_MAX_TOTAL_TOKENS, or AGENTICSCIML_MAX_COST_USD.",
            provider["budget"],
        ),
        _check(
            "real_multimodal_provider",
            "multimodal",
            provider["api_key_present"] and provider["capabilities"]["supports_image_inputs"] is True,
            "Native vision-capable provider is available for actual image input audit.",
            "Use an image-capable native Responses provider; OpenAI-compatible chat adapters remain text-only.",
            provider["capabilities"],
        ),
        _check(
            "heterogeneous_real_selector",
            "selector",
            selector["heterogeneous_selector_candidate"] is True
            or selector["runtime_selector_evidence_ready"] is True,
            "Selector panel has at least two real heterogeneous provider/model candidates.",
            "Configure at least two selector members with distinct provider/model paths and real credentials.",
            selector,
        ),
        _check(
            "paper_like_benchmark",
            "benchmark",
            benchmark_gate["paper_like_ready"] is True,
            "Benchmark is paper-like and has a complete paper benchmark manifest.",
            "Add paper-equivalent data/evaluator provenance and keep private-label protocol hash-bound.",
            benchmark_gate,
        ),
        _check(
            "paper_equivalent_kb",
            "knowledge_base",
            kb_manifest.get("paper_kb_equivalent") is True,
            "Knowledge base provenance is marked paper-equivalent.",
            "Attach paper/code/data provenance with versioned source digests before paper_workflow claims.",
            kb_manifest,
        ),
        _check(
            "domain_approval_packet",
            "domain_review",
            domain["approved"] is True,
            "Domain approval packet passes required checklist.",
            "Fill the domain approval template with reviewer, notes, and every required checklist item.",
            domain,
        ),
        _check(
            "multi_seed_ablation_output",
            "ablation",
            ablation["verified_multi_seed_ablation"] is True,
            "A verified multi-seed ablation output bundle is available.",
            "Run ablation across at least two seeds and one non-baseline variant, then verify the output.",
            ablation,
        ),
        _check(
            "resource_and_blueprint",
            "problem_intake",
            resource_gate["ready"] is True,
            "Resource constraints and expert blueprint are recorded.",
            "Provide expert_blueprint_id plus CPU/GPU/timeout/dependency/data limits.",
            resource_gate,
        ),
    ]
    blockers = [check for check in checks if not check["passed"]]
    return {
        "schema_version": PAPER_WORKFLOW_READINESS_SCHEMA_VERSION,
        "readiness_version": "paper_workflow_readiness.v1",
        "status": "ready" if not blockers else "blocked",
        "ready_to_execute_real_run": not blockers,
        "scientific_claim_supported": False,
        "benchmark": benchmark.to_dict(),
        "provider_readiness": provider,
        "selector_readiness": selector,
        "paper_benchmark_readiness": benchmark_gate,
        "domain_approval_readiness": domain,
        "multi_seed_ablation_readiness": ablation,
        "kb_manifest": kb_manifest,
        "resource_readiness": resource_gate,
        "checks": checks,
        "blockers": [
            {
                "check_id": check["check_id"],
                "category": check["category"],
                "next_action": check["next_action"],
            }
            for check in blockers
        ],
        "command_plan": _command_plan(benchmark_dir, selector_panel or [], ablation_output_dir),
        "claim_boundary": (
            "This bundle is a pre-run paper_workflow gate. It records what is ready to execute, "
            "but it never supports scientific_claim_supported=true without completed run artifacts."
        ),
    }


def write_paper_workflow_readiness_bundle(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    selector_panel: list[dict[str, object]] | None = None,
    selector_evidence_path: Path | None = None,
    resource_constraints: dict[str, object] | None = None,
    expert_blueprint_id: str | None = None,
    domain_approval_path: Path | None = None,
    ablation_output_dir: Path | None = None,
    expected_seeds: list[int] | None = None,
    expected_variants: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_paper_workflow_readiness_bundle(
        benchmark_dir=benchmark_dir,
        selector_panel=selector_panel,
        selector_evidence_path=selector_evidence_path,
        resource_constraints=resource_constraints,
        expert_blueprint_id=expert_blueprint_id,
        domain_approval_path=domain_approval_path,
        ablation_output_dir=ablation_output_dir,
        expected_seeds=expected_seeds,
        expected_variants=expected_variants,
        env=env,
    )
    plan_json = output_dir / "paper_workflow_readiness.json"
    plan_md = output_dir / "paper_workflow_readiness.md"
    domain_template = output_dir / "domain_approval_template.json"
    benchmark_template = output_dir / "paper_benchmark_manifest_template.json"
    commands_md = output_dir / "paper_workflow_commands.md"
    _atomic_write_text(plan_json, json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(plan_md, render_paper_workflow_readiness_markdown(bundle))
    _atomic_write_text(
        domain_template,
        json.dumps(domain_approval_template(), indent=2, sort_keys=True, allow_nan=False),
    )
    _atomic_write_text(
        benchmark_template,
        json.dumps(paper_benchmark_manifest_template(bundle["benchmark"]), indent=2, sort_keys=True, allow_nan=False),
    )
    _atomic_write_text(commands_md, render_command_plan_markdown(bundle))
    return {
        "bundle": bundle,
        "paths": {
            "plan_json": str(plan_json),
            "plan_md": str(plan_md),
            "domain_approval_template": str(domain_template),
            "paper_benchmark_manifest_template": str(benchmark_template),
            "commands_md": str(commands_md),
        },
    }


def domain_approval_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "approved": False,
        "reviewer": "",
        "review_notes": "",
        "approved_claim_level": "paper_workflow",
        "checklist": {check_id: False for check_id in DOMAIN_CHECKLIST_IDS},
        "claim_boundary": "Approval records domain review only; it does not override failing evidence gates.",
    }


def paper_benchmark_manifest_template(benchmark: dict[str, object]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_name": benchmark.get("name"),
        "paper_task_name": benchmark.get("paper_task_name"),
        "paper_reference": {
            "title": "",
            "version_or_date": "",
            "url_or_doi": "",
        },
        "data_provenance": {
            "source": "",
            "license_or_terms": "",
            "train_digest": "",
            "validation_digest": "",
            "private_label_protocol": "validation labels are evaluator-only",
        },
        "evaluator_provenance": {
            "metric": benchmark.get("metric"),
            "paper_metric_match": False,
            "baseline_reproduced": False,
            "evaluator_digest": "",
        },
        "resource_budget": {
            "paper_budget": "",
            "local_budget": "",
            "expected_runtime_s": benchmark.get("expected_runtime_s"),
            "requires_gpu": benchmark.get("requires_gpu"),
        },
        "approved": False,
        "claim_boundary": "This manifest must be completed before a benchmark can be treated as paper-like.",
    }


def render_paper_workflow_readiness_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Paper Workflow Readiness",
        "",
        f"- status: {bundle.get('status')}",
        f"- ready_to_execute_real_run: {bundle.get('ready_to_execute_real_run')}",
        f"- scientific_claim_supported: {bundle.get('scientific_claim_supported')}",
        "",
        "## Checks",
        "",
    ]
    for check in bundle.get("checks", []):
        if isinstance(check, dict):
            status = "pass" if check.get("passed") else "blocked"
            lines.append(f"- {check.get('check_id')}: {status} - {check.get('message')}")
    lines.extend(["", "## Claim Boundary", "", str(bundle.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def render_command_plan_markdown(bundle: dict[str, Any]) -> str:
    lines = ["# Paper Workflow Command Plan", ""]
    for command in bundle.get("command_plan", []):
        lines.extend(["```bash", str(command), "```", ""])
    return "\n".join(lines)


def _provider_readiness(env: Mapping[str, str]) -> dict[str, Any]:
    base_url = env.get("OPENAI_BASE_URL") or None
    capabilities = capabilities_for_openai_compatible(base_url).to_dict()
    budget_keys = (
        "AGENTICSCIML_MAX_LLM_CALLS",
        "AGENTICSCIML_MAX_PROMPT_TOKENS",
        "AGENTICSCIML_MAX_OUTPUT_TOKENS",
        "AGENTICSCIML_MAX_TOTAL_TOKENS",
        "AGENTICSCIML_MAX_COST_USD",
    )
    budget = {key: "set" if env.get(key) else "missing" for key in budget_keys}
    return {
        "api_key_present": bool(env.get("OPENAI_API_KEY")),
        "model": env.get("OPENAI_MODEL", "gpt-5-mini"),
        "base_url_present": bool(base_url),
        "provider_hint": capabilities["provider"],
        "capabilities": capabilities,
        "budget": budget,
        "budget_configured": any(value == "set" for value in budget.values()),
        "secret_policy": "credential values are never written to readiness artifacts",
    }


def _selector_readiness(
    selector_panel: list[dict[str, object]],
    *,
    provider_api_key_present: bool,
    selector_evidence_path: Path | None = None,
) -> dict[str, Any]:
    members: list[dict[str, object]] = []
    for index, member in enumerate(selector_panel, start=1):
        base_url = str(member.get("base_url") or "").strip() or None
        capabilities = capabilities_for_openai_compatible(base_url).to_dict()
        model = str(member.get("model") or "").strip()
        members.append(
            {
                "role": str(member.get("role") or f"selector_{index:03d}"),
                "model": model,
                "provider": capabilities["provider"],
                "adapter_type": capabilities["adapter_type"],
                "configured_base_url_present": bool(base_url),
                "credential_present": provider_api_key_present,
            }
        )
    real_members = [member for member in members if member["credential_present"] and member["model"]]
    providers = sorted({str(member["provider"]) for member in real_members})
    models = sorted({str(member["model"]) for member in real_members})
    provider_model_pairs = sorted({f"{member['provider']}::{member['model']}" for member in real_members})
    evidence_packet = _selector_evidence_packet_readiness(selector_evidence_path)
    return {
        "configured_member_count": len(members),
        "real_member_count": len(real_members),
        "unique_providers": providers,
        "unique_models": models,
        "unique_provider_model_pairs": provider_model_pairs,
        "heterogeneous_selector_candidate": len(real_members) >= 2 and len(provider_model_pairs) >= 2,
        "runtime_selector_evidence_ready": evidence_packet["paper_workflow_selector_ready"] is True,
        "selector_evidence_packet": evidence_packet,
        "members": members,
        "claim_boundary": (
            "Configuration is only a candidate; completed runtime selector_votes.json or "
            "selector_evidence_packet.json remains the evidence source."
        ),
    }


def _selector_evidence_packet_readiness(selector_evidence_path: Path | None) -> dict[str, Any]:
    if selector_evidence_path is None:
        return {
            "path": None,
            "exists": False,
            "status": "not_provided",
            "paper_workflow_selector_ready": False,
            "blockers": ["selector_evidence_path not provided"],
        }
    path = Path(selector_evidence_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "path": str(path),
            "exists": False,
            "status": "missing",
            "paper_workflow_selector_ready": False,
            "blockers": ["selector evidence packet is missing"],
        }
    except json.JSONDecodeError as exc:
        return {
            "path": str(path),
            "exists": True,
            "status": "invalid_json",
            "paper_workflow_selector_ready": False,
            "blockers": [f"selector evidence packet is invalid JSON: {exc}"],
        }
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "exists": True,
            "status": "invalid_schema",
            "paper_workflow_selector_ready": False,
            "blockers": ["selector evidence packet must be a JSON object"],
        }
    scientific_claim_supported = payload.get("scientific_claim_supported") is True
    ready = (
        payload.get("status") == "ready"
        and payload.get("paper_workflow_selector_ready") is True
        and scientific_claim_supported is False
    )
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    if scientific_claim_supported:
        blockers = [*blockers, "selector evidence packet must not claim scientific support"]
    return {
        "path": str(path),
        "exists": True,
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "paper_workflow_selector_ready": ready,
        "scientific_claim_supported": scientific_claim_supported,
        "runtime_vote_summary": payload.get("runtime_vote_summary")
        if isinstance(payload.get("runtime_vote_summary"), dict)
        else {},
        "blockers": blockers,
    }


def _paper_benchmark_readiness(benchmark_dir: Path, benchmark: dict[str, object]) -> dict[str, Any]:
    manifest_path = benchmark_dir / PAPER_BENCHMARK_MANIFEST
    manifest: dict[str, Any] = {}
    issues: list[str] = []
    if manifest_path.is_file():
        try:
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError as exc:
            issues.append(f"{PAPER_BENCHMARK_MANIFEST} is invalid JSON: {exc}")
    else:
        issues.append(f"{PAPER_BENCHMARK_MANIFEST} is missing")
    required_manifest_flags = (
        ("approved", manifest.get("approved") is True),
        (
            "paper_metric_match",
            isinstance(manifest.get("evaluator_provenance"), dict)
            and manifest["evaluator_provenance"].get("paper_metric_match") is True,
        ),
        (
            "baseline_reproduced",
            isinstance(manifest.get("evaluator_provenance"), dict)
            and manifest["evaluator_provenance"].get("baseline_reproduced") is True,
        ),
    )
    for label, passed in required_manifest_flags:
        if not passed:
            issues.append(f"{label} is not confirmed")
    paper_like_ready = benchmark.get("fidelity_level") == "paper-like" and not issues
    return {
        "paper_like_ready": paper_like_ready,
        "fidelity_level": benchmark.get("fidelity_level"),
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_path.is_file(),
        "issues": issues,
        "manifest": manifest,
    }


def _domain_approval_readiness(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "approved": False,
            "path": None,
            "issues": ["domain approval packet is missing"],
            "required_checklist_ids": list(DOMAIN_CHECKLIST_IDS),
        }
    if not path.is_file():
        return {
            "approved": False,
            "path": str(path),
            "issues": ["domain approval packet path does not exist"],
            "required_checklist_ids": list(DOMAIN_CHECKLIST_IDS),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "approved": False,
            "path": str(path),
            "issues": [f"domain approval packet is invalid JSON: {exc}"],
            "required_checklist_ids": list(DOMAIN_CHECKLIST_IDS),
        }
    packet = dict(payload) if isinstance(payload, dict) else {}
    checklist = packet.get("checklist") if isinstance(packet.get("checklist"), dict) else {}
    issues = []
    if packet.get("approved") is not True:
        issues.append("approved must be true")
    if not str(packet.get("reviewer") or "").strip():
        issues.append("reviewer is required")
    if not str(packet.get("review_notes") or "").strip():
        issues.append("review_notes is required")
    for check_id in DOMAIN_CHECKLIST_IDS:
        if checklist.get(check_id) is not True:
            issues.append(f"checklist.{check_id} must be true")
    return {
        "approved": not issues,
        "path": str(path),
        "issues": issues,
        "reviewer_present": bool(str(packet.get("reviewer") or "").strip()),
        "review_notes_present": bool(str(packet.get("review_notes") or "").strip()),
        "required_checklist_ids": list(DOMAIN_CHECKLIST_IDS),
    }


def _ablation_readiness(
    *,
    ablation_output_dir: Path | None,
    expected_seeds: list[int],
    expected_variants: list[str],
) -> dict[str, Any]:
    if ablation_output_dir is None:
        return {
            "verified_multi_seed_ablation": False,
            "issues": ["ablation_output_dir is missing"],
        }
    manifest = build_multi_seed_ablation_verified_manifest(
        {
            "ablation_output_dir": str(ablation_output_dir),
            "verified_by": "paper-workflow-readiness",
            "expected_seeds": expected_seeds,
            "expected_variants": expected_variants,
        }
    )
    return {
        "verified_multi_seed_ablation": manifest.get("verified") is True,
        "manifest": manifest,
        "issues": list(manifest.get("blockers", [])),
    }


def _resource_readiness(resource_constraints: dict[str, object], expert_blueprint_id: str | None) -> dict[str, Any]:
    required = ("cpu", "gpu", "timeout_s", "dependency_limits", "data_limits")
    missing = [key for key in required if key not in resource_constraints]
    return {
        "ready": bool(expert_blueprint_id and not missing),
        "expert_blueprint_id": expert_blueprint_id,
        "resource_constraints": dict(resource_constraints),
        "missing_resource_keys": missing,
    }


def _check(
    check_id: str,
    category: str,
    passed: bool,
    message: str,
    next_action: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "category": category,
        "passed": bool(passed),
        "severity": "info" if passed else "blocker",
        "message": message,
        "next_action": next_action,
        "evidence": evidence,
    }


def _command_plan(
    benchmark_dir: Path,
    selector_panel: list[dict[str, object]],
    ablation_output_dir: Path | None,
) -> list[str]:
    selector_json = json.dumps(selector_panel, separators=(",", ":"), sort_keys=True)
    commands = [
        "export OPENAI_API_KEY=<redacted>",
        "export AGENTICSCIML_MAX_LLM_CALLS=20",
        (
            "PYTHONPATH=src uv run --python 3.11 --extra real-llm python -m agenticsciml.cli run "
            f"{benchmark_dir} --max-iterations 1 --parallel-mutations 2 --visual-audit-mode real "
            f"--selector-panel-json '{selector_json}'"
        ),
    ]
    if ablation_output_dir is not None:
        commands.append(
            "PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli "
            f"verify-ablation-evidence {ablation_output_dir} --verified-by domain-reviewer"
        )
    commands.append("PYTHONPATH=src uv run --python 3.11 --extra dev python -m agenticsciml.cli trace-summary <run_dir>")
    return commands
