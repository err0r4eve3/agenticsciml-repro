from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from agenticsciml.evidence import CLAIM_LEVEL_WORKFLOW_PROXY
from agenticsciml.storage import _atomic_write_text


SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = "method_substrate.v1"


@dataclass(frozen=True, slots=True)
class MethodAction:
    action_id: str
    family: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source_scope: str = "local_method_contract"

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("method action requires action_id")
        if not self.family.strip():
            raise ValueError("method action requires family")
        for key in self.parameters:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("method action parameter keys must be non-empty strings")
        _canonical_json(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "action_id": self.action_id,
            "family": self.family,
            "parameters": self.parameters,
            "source_scope": self.source_scope,
        }

    def identity_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "family": self.family,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MethodAction":
        _require_schema(payload, "method action")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("method action parameters must be an object")
        return cls(
            action_id=_required_str(payload, "action_id"),
            family=_required_str(payload, "family"),
            parameters=parameters,
            source_scope=_required_str(payload, "source_scope"),
        )


@dataclass(frozen=True, slots=True)
class ExpertBlueprint:
    blueprint_id: str
    allowed_families: tuple[str, ...]
    required_parameters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    forbidden_parameters: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.blueprint_id.strip():
            raise ValueError("expert blueprint requires blueprint_id")
        if not self.allowed_families:
            raise ValueError("expert blueprint requires at least one allowed family")
        for family in self.allowed_families:
            if not family.strip():
                raise ValueError("allowed families must be non-empty strings")
        _validate_parameter_contract(self.required_parameters, "required")
        _validate_parameter_contract(self.forbidden_parameters, "forbidden")

    def validate_action(self, action: MethodAction) -> MethodAction:
        if action.family not in self.allowed_families:
            raise ValueError(f"action family {action.family!r} is not allowed by blueprint {self.blueprint_id!r}")
        required = set(self.required_parameters.get("*", ())) | set(
            self.required_parameters.get(action.family, ())
        )
        forbidden = set(self.forbidden_parameters.get("*", ())) | set(
            self.forbidden_parameters.get(action.family, ())
        )
        missing = sorted(required - set(action.parameters))
        if missing:
            raise ValueError(f"action {action.action_id!r} is missing required parameters: {missing}")
        present_forbidden = sorted(forbidden & set(action.parameters))
        if present_forbidden:
            raise ValueError(
                f"action {action.action_id!r} contains forbidden parameters: {present_forbidden}"
            )
        return action


@dataclass(frozen=True, slots=True)
class MethodPath:
    actions: tuple[MethodAction, ...]
    source_scope: str = "athena_graft_local_contract"

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("method path requires at least one action")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_scope": self.source_scope,
            "actions": [action.to_dict() for action in self.actions],
        }

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "actions": [action.identity_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MethodPath":
        _require_schema(payload, "method path")
        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise ValueError("method path actions must be a list")
        return cls(
            actions=tuple(MethodAction.from_dict(action) for action in actions),
            source_scope=_required_str(payload, "source_scope"),
        )

    def fingerprint(self) -> str:
        digest = sha256(_canonical_json(self.identity_dict()).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def method_tags(self) -> tuple[str, ...]:
        tags: list[str] = []
        for action in self.actions:
            tags.extend((action.family, action.action_id))
        return tuple(dict.fromkeys(tags))


@dataclass(frozen=True, slots=True)
class ScientificReward:
    metric: str
    value: float
    higher_is_better: bool
    source_artifact: str
    evidence_mode: str = CLAIM_LEVEL_WORKFLOW_PROXY

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("scientific reward requires metric")
        if not isinstance(self.value, int | float) or isinstance(self.value, bool):
            raise ValueError("scientific reward value must be numeric")
        if not isfinite(float(self.value)):
            raise ValueError("scientific reward value must be finite")
        if not self.source_artifact.strip():
            raise ValueError("scientific reward requires source_artifact")
        artifact_path = Path(self.source_artifact)
        if artifact_path.is_absolute() or ".." in artifact_path.parts or self.source_artifact.startswith("~"):
            raise ValueError("scientific reward source_artifact must be a repo-relative path")
        if not self.evidence_mode.strip():
            raise ValueError("scientific reward requires evidence_mode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metric": self.metric,
            "value": float(self.value),
            "higher_is_better": self.higher_is_better,
            "source_artifact": self.source_artifact,
            "evidence_mode": self.evidence_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScientificReward":
        _require_schema(payload, "scientific reward")
        higher_is_better = payload.get("higher_is_better")
        if not isinstance(higher_is_better, bool):
            raise ValueError("scientific reward higher_is_better must be boolean")
        return cls(
            metric=_required_str(payload, "metric"),
            value=payload.get("value"),  # type: ignore[arg-type]
            higher_is_better=higher_is_better,
            source_artifact=_required_str(payload, "source_artifact"),
            evidence_mode=_required_str(payload, "evidence_mode"),
        )


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    method_path: MethodPath
    benchmark_name: str
    reward: ScientificReward
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.benchmark_name.strip():
            raise ValueError("experience record requires benchmark_name")

    @property
    def method_fingerprint(self) -> str:
        return self.method_path.fingerprint()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "method_fingerprint": self.method_fingerprint,
            "method_path": self.method_path.to_dict(),
            "benchmark_name": self.benchmark_name,
            "reward": self.reward.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceRecord":
        _require_schema(payload, "experience record")
        method_path = MethodPath.from_dict(_required_dict(payload, "method_path"))
        expected_fingerprint = method_path.fingerprint()
        recorded_fingerprint = _required_str(payload, "method_fingerprint")
        if recorded_fingerprint != expected_fingerprint:
            raise ValueError("experience record fingerprint does not match method path")
        return cls(
            method_path=method_path,
            benchmark_name=_required_str(payload, "benchmark_name"),
            reward=ScientificReward.from_dict(_required_dict(payload, "reward")),
            notes=str(payload.get("notes", "")),
        )


class ExperienceSubstrate:
    def __init__(self, path: Path):
        self.path = path

    def save(self, record: ExperienceRecord) -> None:
        payload = self._load_payload()
        records = payload["records"].setdefault(record.method_fingerprint, [])
        if not isinstance(records, list):
            raise ValueError("experience substrate fingerprint records must be a list")
        records.append(record.to_dict())
        _atomic_write_text(self.path, _canonical_json(payload, indent=2))

    def get(self, method_fingerprint: str) -> ExperienceRecord | None:
        records = self.records_for_fingerprint(method_fingerprint)
        if not records:
            return None
        return records[-1]

    def get_for_path(self, method_path: MethodPath) -> ExperienceRecord | None:
        return self.get(method_path.fingerprint())

    def records_for_fingerprint(self, method_fingerprint: str) -> list[ExperienceRecord]:
        records = self._load_payload()["records"].get(method_fingerprint, [])
        if not isinstance(records, list):
            raise ValueError("experience substrate fingerprint records must be a list")
        return [ExperienceRecord.from_dict(record) for record in records]

    def _load_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "records": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        _require_schema(payload, "experience substrate")
        records = payload.get("records")
        if not isinstance(records, dict):
            raise ValueError("experience substrate records must be an object")
        return payload


def _canonical_json(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, allow_nan=False, indent=indent, sort_keys=True, separators=(",", ":"))


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _require_schema(payload: dict[str, Any], label: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{label} schema_version must be {SCHEMA_VERSION}")


def _validate_parameter_contract(contract: dict[str, tuple[str, ...]], label: str) -> None:
    for family, parameters in contract.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"{label} parameter contract families must be non-empty strings")
        for parameter in parameters:
            if not isinstance(parameter, str) or not parameter.strip():
                raise ValueError(f"{label} parameter names must be non-empty strings")
