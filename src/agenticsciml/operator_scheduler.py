from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticsciml.algorithm_catalog import (
    OPERATOR_CLAIM_BOUNDARY,
    AlgorithmSpec,
    list_algorithms,
    operator_metadata_for_algorithm,
)
from agenticsciml.state import SolutionNode, solution_id_index


OPERATOR_ASSIGNMENT_SCHEMA_VERSION = 1
OPERATOR_SCHEDULER_VERSION = "operator_scheduler.v1"
OPERATOR_SCHEDULER_MODE = "auto-audited"

_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "function_approximation": ("function_approx",),
    "function_approx": ("function_approx",),
    "pinn": ("poisson", "burgers_pinn", "reaction_diffusion"),
    "physics_informed": ("poisson", "burgers_pinn", "reaction_diffusion"),
    "operator_learning": ("operator_learning", "reaction_diffusion"),
    "custom_operator_learning": ("operator_learning", "reaction_diffusion"),
    "inverse_reconstruction": ("sensor_reconstruction",),
    "custom_inverse_reconstruction": ("sensor_reconstruction",),
    "sensor_reconstruction": ("sensor_reconstruction",),
    "custom_pinn": ("poisson", "burgers_pinn", "reaction_diffusion"),
    "custom_temporal_regression": ("operator_learning", "reaction_diffusion", "function_approx"),
    "custom_regression": ("function_approx",),
}


@dataclass(frozen=True, slots=True)
class OperatorAssignment:
    solution_id: str
    parent_id: str
    operator_id: str
    operator_name: str
    operator_family: str
    mutation_axis: str
    selection_source: str
    compatibility_reason: str
    expected_static_terms: tuple[str, ...]
    risk_notes: tuple[str, ...]
    selected_algorithm_ids: tuple[str, ...]
    candidate_operator_ids: tuple[str, ...]
    penalized_operator_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPERATOR_ASSIGNMENT_SCHEMA_VERSION,
            "scheduler": OPERATOR_SCHEDULER_VERSION,
            "scheduler_mode": OPERATOR_SCHEDULER_MODE,
            "solution_id": self.solution_id,
            "parent_id": self.parent_id,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "operator_family": self.operator_family,
            "mutation_axis": self.mutation_axis,
            "selection_source": self.selection_source,
            "compatibility_reason": self.compatibility_reason,
            "operator_expected_terms": list(self.expected_static_terms),
            "risk_notes": list(self.risk_notes),
            "selected_algorithm_ids": list(self.selected_algorithm_ids),
            "candidate_operator_ids": list(self.candidate_operator_ids),
            "penalized_operator_ids": list(self.penalized_operator_ids),
            "warnings": list(self.warnings),
            "claim_boundary": OPERATOR_CLAIM_BOUNDARY,
        }


class OperatorScheduler:
    def __init__(
        self,
        *,
        benchmark_name: str,
        benchmark_family: str,
        selected_algorithm_ids: list[str],
        nodes: list[SolutionNode],
        run_dir: Path,
    ):
        self.benchmark_name = benchmark_name
        self.benchmark_family = benchmark_family
        self.selected_algorithm_ids = _dedupe(selected_algorithm_ids)
        self.nodes = list(nodes)
        self.run_dir = run_dir
        self.algorithms = list_algorithms()
        self.by_id = {algorithm.algorithm_id: algorithm for algorithm in self.algorithms}
        self.history = operator_history(nodes, run_dir)

    def assign(
        self,
        *,
        solution_id: str,
        parent: SolutionNode,
        branch_context: dict[str, object],
        used_axes_for_parent: set[str] | None = None,
    ) -> OperatorAssignment:
        used_axes = set(used_axes_for_parent or set())
        candidates = self._candidate_algorithms()
        warnings: list[str] = []
        if not candidates:
            candidates = self._fallback_algorithms()
            warnings.append(
                "No benchmark-compatible operator was found; using deterministic fallback catalog operator."
            )
        selected_known = [
            algorithm_id
            for algorithm_id in self.selected_algorithm_ids
            if algorithm_id in self.by_id
        ]
        incompatible_selected = [
            algorithm_id
            for algorithm_id in selected_known
            if not self._compatible(self.by_id[algorithm_id])
        ]
        if incompatible_selected:
            warnings.append(
                "Selected algorithm(s) incompatible with benchmark family were not scheduled: "
                + ", ".join(incompatible_selected)
            )

        ranked = self._rank_candidates(candidates, solution_id=solution_id, used_axes=used_axes)
        chosen = ranked[0]
        metadata = operator_metadata_for_algorithm(chosen)
        axes = [str(axis) for axis in metadata["mutation_axes"]]
        mutation_axis = next((axis for axis in axes if axis not in used_axes), axes[0])
        selected_source = (
            "manual_selected"
            if chosen.algorithm_id in self.selected_algorithm_ids and self._compatible(chosen)
            else "auto_compatible"
            if self._compatible(chosen)
            else "fallback_catalog"
        )
        reason = self._compatibility_reason(chosen)
        if branch_context.get("branch_intent"):
            reason += f"; branch_intent={branch_context['branch_intent']}"
        return OperatorAssignment(
            solution_id=solution_id,
            parent_id=parent.node_id,
            operator_id=chosen.algorithm_id,
            operator_name=chosen.name,
            operator_family=chosen.family,
            mutation_axis=mutation_axis,
            selection_source=selected_source,
            compatibility_reason=reason,
            expected_static_terms=tuple(str(item) for item in metadata["expected_static_terms"]),
            risk_notes=(chosen.safety_notes,),
            selected_algorithm_ids=tuple(self.selected_algorithm_ids),
            candidate_operator_ids=tuple(algorithm.algorithm_id for algorithm in candidates),
            penalized_operator_ids=tuple(self._penalized_operator_ids()),
            warnings=tuple(warnings),
        )

    def _candidate_algorithms(self) -> list[AlgorithmSpec]:
        if self.selected_algorithm_ids:
            selected = [
                self.by_id[algorithm_id]
                for algorithm_id in self.selected_algorithm_ids
                if algorithm_id in self.by_id and self._compatible(self.by_id[algorithm_id])
            ]
            if selected:
                return selected
        compatible = [algorithm for algorithm in self.algorithms if self._compatible(algorithm)]
        return compatible

    def _fallback_algorithms(self) -> list[AlgorithmSpec]:
        for fallback_id in ("baseline_mlp_regressor", "kernel_surrogate_regression"):
            algorithm = self.by_id.get(fallback_id)
            if algorithm:
                return [algorithm]
        return self.algorithms[:1]

    def _rank_candidates(
        self,
        candidates: list[AlgorithmSpec],
        *,
        solution_id: str,
        used_axes: set[str],
    ) -> list[AlgorithmSpec]:
        if not candidates:
            return []
        ranked = sorted(candidates, key=lambda algorithm: self._rank_key(algorithm, used_axes))
        best_prefix = self._rank_key(ranked[0], used_axes)[:4]
        pool = [algorithm for algorithm in ranked if self._rank_key(algorithm, used_axes)[:4] == best_prefix]
        rest = [algorithm for algorithm in ranked if self._rank_key(algorithm, used_axes)[:4] != best_prefix]
        offset = solution_id_index(solution_id) or 0
        rotated = pool[offset % len(pool) :] + pool[: offset % len(pool)]
        return rotated + rest

    def _rank_key(
        self,
        algorithm: AlgorithmSpec,
        used_axes: set[str],
    ) -> tuple[int, int, int, int, str]:
        metadata = operator_metadata_for_algorithm(algorithm)
        axes = [str(axis) for axis in metadata["mutation_axes"]]
        source_rank = 0 if algorithm.algorithm_id in self.selected_algorithm_ids else 1
        axis_penalty = 0 if any(axis not in used_axes for axis in axes) else 1
        history_penalty = self._history_penalty(algorithm.algorithm_id)
        compatibility_rank = 0 if self._compatible(algorithm) else 1
        return (source_rank, axis_penalty, history_penalty, compatibility_rank, algorithm.algorithm_id)

    def _history_penalty(self, operator_id: str) -> int:
        stats = self.history.get(operator_id, {})
        return int(stats.get("duplicate", 0)) * 3 + int(stats.get("plateau", 0)) * 2

    def _penalized_operator_ids(self) -> list[str]:
        return sorted(
            operator_id
            for operator_id, stats in self.history.items()
            if int(stats.get("duplicate", 0)) > 0 or int(stats.get("plateau", 0)) > 0
        )

    def _compatible(self, algorithm: AlgorithmSpec) -> bool:
        if self.benchmark_name in algorithm.benchmark_examples:
            return True
        target_keys = _benchmark_keys(self.benchmark_name, self.benchmark_family)
        compatible = {_normalize_key(item) for item in algorithm.compatible_benchmark_families}
        return bool(target_keys & compatible)

    def _compatibility_reason(self, algorithm: AlgorithmSpec) -> str:
        if self.benchmark_name in algorithm.benchmark_examples:
            return f"benchmark example match: {self.benchmark_name}"
        matched = sorted(
            _benchmark_keys(self.benchmark_name, self.benchmark_family)
            & {_normalize_key(item) for item in algorithm.compatible_benchmark_families}
        )
        if matched:
            return "benchmark family match: " + ", ".join(matched)
        return "fallback; no direct family match"


def operator_history(nodes: list[SolutionNode], run_dir: Path) -> dict[str, dict[str, float | int]]:
    history: dict[str, dict[str, float | int]] = {}
    for node in nodes:
        assignment = _read_json_object(Path(node.workspace) / "operator_assignment.json")
        operator_id = assignment.get("operator_id")
        if not isinstance(operator_id, str) or not operator_id:
            continue
        stats = history.setdefault(
            operator_id,
            {
                "assigned": 0,
                "evaluated": 0,
                "duplicate": 0,
                "plateau": 0,
                "improved": 0,
                "best_improvement": 0.0,
            },
        )
        stats["assigned"] = int(stats["assigned"]) + 1
        if node.status == "evaluated":
            stats["evaluated"] = int(stats["evaluated"]) + 1
        mutation = _read_json_object(Path(node.workspace) / "mutation_effect_report.json")
        status = mutation.get("status")
        if status == "duplicate_parent":
            stats["duplicate"] = int(stats["duplicate"]) + 1
        if status == "changed_but_score_plateau":
            stats["plateau"] = int(stats["plateau"]) + 1
        if node.score and node.score_delta_from_parent is not None:
            improvement = (
                node.score_delta_from_parent
                if node.score.higher_is_better
                else -node.score_delta_from_parent
            )
            if improvement > 0:
                stats["improved"] = int(stats["improved"]) + 1
                stats["best_improvement"] = max(float(stats["best_improvement"]), improvement)
    return history


def operator_assignment_context(assignment: dict[str, object] | OperatorAssignment | None) -> str:
    payload = assignment.to_dict() if isinstance(assignment, OperatorAssignment) else assignment
    if not isinstance(payload, dict) or not payload:
        return "No operator assignment."
    return (
        "Operator assignment is workflow guidance only. EvaluationContract, benchmark guidelines, "
        "sandbox rules, and private-label boundaries override it.\n"
        f"{_json_dumps(payload)}"
    )


def _benchmark_keys(benchmark_name: str, benchmark_family: str) -> set[str]:
    keys = {_normalize_key(benchmark_name), _normalize_key(benchmark_family)}
    for key in list(keys):
        keys.update(_FAMILY_ALIASES.get(key, ()))
        if key.startswith("custom_"):
            keys.add(key.removeprefix("custom_"))
            keys.update(_FAMILY_ALIASES.get(key.removeprefix("custom_"), ()))
    lowered_name = benchmark_name.lower()
    if "poisson" in lowered_name:
        keys.add("poisson")
    if "burgers" in lowered_name:
        keys.add("burgers_pinn")
    if "kuramoto" in lowered_name or "sivashinsky" in lowered_name:
        keys.update({"burgers_pinn", "reaction_diffusion", "operator_learning"})
    if "reaction_diffusion" in lowered_name or "reaction-diffusion" in lowered_name:
        keys.add("reaction_diffusion")
    if "operator" in lowered_name:
        keys.add("operator_learning")
    if "cylinder" in lowered_name or "wake" in lowered_name:
        keys.add("sensor_reconstruction")
    return keys


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(payload)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
