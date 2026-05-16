from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agenticsciml.algorithm_catalog import list_algorithms
from agenticsciml.benchmarks import list_benchmarks


PAPER_TASK_CLAIM_BOUNDARY = (
    "This page aligns local benchmark evidence with AgenticSciML paper sections. "
    "Only run artifacts, evaluator outputs, trace summaries, and repository tests "
    "support local claims; mock runs and faithful-small benchmarks are not paper-score evidence."
)


@dataclass(frozen=True, slots=True)
class PaperTaskSpec:
    paper_section: str
    title: str
    summary: str
    benchmark_names: tuple[str, ...]
    algorithm_ids: tuple[str, ...]
    reference_primitives: tuple[str, ...]
    figure_labels: tuple[str, ...]
    claim_boundary: str = PAPER_TASK_CLAIM_BOUNDARY

    def to_dict(self, benchmark_by_name: dict[str, Any], algorithm_by_id: dict[str, Any]) -> dict[str, object]:
        benchmarks = [
            benchmark_by_name[name].to_dict()
            for name in self.benchmark_names
            if name in benchmark_by_name
        ]
        algorithms = [
            algorithm_by_id[algorithm_id].to_dict()
            for algorithm_id in self.algorithm_ids
            if algorithm_id in algorithm_by_id
        ]
        return {
            "paper_section": self.paper_section,
            "title": self.title,
            "summary": self.summary,
            "benchmarks": benchmarks,
            "algorithms": algorithms,
            "reference_primitives": list(self.reference_primitives),
            "figure_labels": list(self.figure_labels),
            "local_artifact_figures": [
                "reports/data_overview.svg",
                "solutions/<solution_id>/prediction_overview.svg",
            ],
            "claim_boundary": self.claim_boundary,
        }


PAPER_TASKS: tuple[PaperTaskSpec, ...] = (
    PaperTaskSpec(
        paper_section="S1.1",
        title="Discontinuous Function Fitting",
        summary="One-dimensional discontinuous function approximation with local or gated basis strategies.",
        benchmark_names=("function_approx", "function_approx_faithful_small"),
        algorithm_ids=("paper_sigmoid_moe_gate",),
        reference_primitives=("sigmoid-gated two-expert composition", "piecewise/local basis regression"),
        figure_labels=("paper S1.1 task panel", "local data/prediction overview artifacts"),
    ),
    PaperTaskSpec(
        paper_section="S1.2",
        title="L-Shaped Poisson PINN",
        summary="Irregular-domain Poisson task with residual, boundary, and decomposition strategies.",
        benchmark_names=("poisson_lshape", "poisson_lshape_faithful_small"),
        algorithm_ids=("paper_poisson_decomposition_sampler",),
        reference_primitives=("particular-plus-residual decomposition", "corner-biased collocation sampling"),
        figure_labels=("paper S1.2 task panel", "local data/prediction overview artifacts"),
    ),
    PaperTaskSpec(
        paper_section="S1.3",
        title="Burgers PINN",
        summary="Time-dependent Burgers equation task with staged residual and boundary treatment.",
        benchmark_names=("burgers_pinn", "burgers_pinn_faithful_small"),
        algorithm_ids=("paper_burgers_staged_pinn_schedule",),
        reference_primitives=("staged PINN schedule", "residual-weight helper"),
        figure_labels=("paper S1.3 task panel", "local data/prediction overview artifacts"),
    ),
    PaperTaskSpec(
        paper_section="S1.4",
        title="Antiderivative Operator Learning",
        summary="Function-to-function antiderivative operator learning with DeepONet-style primitives.",
        benchmark_names=("antiderivative_operator", "antiderivative_operator_faithful_small"),
        algorithm_ids=("paper_linear_bias_free_deeponet",),
        reference_primitives=("linear bias-free branch map", "branch/trunk operator factorization"),
        figure_labels=("paper S1.4 task panel", "local data/prediction overview artifacts"),
    ),
    PaperTaskSpec(
        paper_section="S1.5",
        title="Reaction-Diffusion Operator Learning",
        summary="Multiple-input reaction-diffusion operator task with spectral and derivative-enhanced helpers.",
        benchmark_names=("reaction_diffusion_operator", "reaction_diffusion_operator_faithful_small"),
        algorithm_ids=("paper_reaction_diffusion_fno_helpers",),
        reference_primitives=("derivative-enhanced loss", "spectral smoothing and boundary enforcement"),
        figure_labels=("paper S1.5 task panel", "local data/prediction overview artifacts"),
    ),
    PaperTaskSpec(
        paper_section="S1.6",
        title="Cylinder Wake Reconstruction",
        summary="Sparse-sensor two-dimensional wake reconstruction with bandlimited decoder/filter primitives.",
        benchmark_names=("cylinder_wake_reconstruction", "cylinder_wake_reconstruction_faithful_small"),
        algorithm_ids=("paper_cylinder_bandlimited_filter",),
        reference_primitives=("bandlimit-preserving activation", "Gaussian low-pass decoder filtering"),
        figure_labels=("paper S1.6 task panel", "local data/prediction overview artifacts"),
    ),
)


def list_paper_tasks() -> list[dict[str, object]]:
    benchmark_by_name = {benchmark.name: benchmark for benchmark in list_benchmarks()}
    algorithm_by_id = {algorithm.algorithm_id: algorithm for algorithm in list_algorithms()}
    return [task.to_dict(benchmark_by_name, algorithm_by_id) for task in PAPER_TASKS]
