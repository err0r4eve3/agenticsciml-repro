from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    path: Path
    paper_section: str
    family: str
    metric: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": str(self.path),
            "paper_section": self.paper_section,
            "family": self.family,
            "metric": self.metric,
            "description": self.description,
        }


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "function_approx": BenchmarkSpec(
        name="function_approx",
        path=EXAMPLES_DIR / "function_approx",
        paper_section="S1.1",
        family="function approximation",
        metric="validation_mse",
        description="Discontinuous oscillatory one-dimensional function approximation.",
    ),
    "poisson_lshape": BenchmarkSpec(
        name="poisson_lshape",
        path=EXAMPLES_DIR / "poisson_lshape",
        paper_section="S1.2",
        family="PINN",
        metric="relative_l2",
        description="Poisson equation surrogate on an L-shaped domain.",
    ),
    "burgers_pinn": BenchmarkSpec(
        name="burgers_pinn",
        path=EXAMPLES_DIR / "burgers_pinn",
        paper_section="S1.3",
        family="PINN",
        metric="relative_l2",
        description="Time-dependent viscous Burgers equation surrogate.",
    ),
    "antiderivative_operator": BenchmarkSpec(
        name="antiderivative_operator",
        path=EXAMPLES_DIR / "antiderivative_operator",
        paper_section="S1.4",
        family="operator learning",
        metric="relative_l2",
        description="Operator learning from input functions to antiderivatives.",
    ),
    "reaction_diffusion_operator": BenchmarkSpec(
        name="reaction_diffusion_operator",
        path=EXAMPLES_DIR / "reaction_diffusion_operator",
        paper_section="S1.5",
        family="operator learning",
        metric="relative_l2",
        description="Multiple-input reaction-diffusion operator surrogate.",
    ),
    "cylinder_wake_reconstruction": BenchmarkSpec(
        name="cylinder_wake_reconstruction",
        path=EXAMPLES_DIR / "cylinder_wake_reconstruction",
        paper_section="S1.6",
        family="inverse reconstruction",
        metric="relative_l2",
        description="2D cylinder wake vorticity reconstruction from sparse noisy sensors.",
    ),
}


def list_benchmarks() -> list[BenchmarkSpec]:
    return list(BENCHMARKS.values())
