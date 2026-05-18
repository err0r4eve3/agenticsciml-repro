from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CUSTOM_BENCHMARK_SPEC = "Benchmark_spec.json"
CUSTOM_BENCHMARK_SCHEMA_VERSION = 1

CUSTOM_BENCHMARK_CLAIM_BOUNDARY = (
    "This auto-generated evaluator is a deterministic workflow proxy derived from "
    "the user's problem text. It is useful for exercising the AgenticSciML loop, "
    "artifact plumbing, and private-label evaluation boundary. It is not a "
    "scientific validation of the real problem, not a paper-like benchmark, and "
    "not evidence for paper-score reproduction."
)


@dataclass(frozen=True, slots=True)
class CustomBenchmarkBundle:
    benchmark: str
    benchmark_dir: Path
    files: list[str]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "benchmark_dir": str(self.benchmark_dir),
            "files": list(self.files),
            "claim_boundary": self.claim_boundary,
        }


def create_custom_benchmark_bundle(
    root_dir: Path,
    *,
    problem_statement: str,
    requirements: str = "",
    evaluation_criteria: str = "",
    data_description: str = "",
) -> CustomBenchmarkBundle:
    root_dir = root_dir.resolve(strict=False)
    root_dir.mkdir(parents=True, exist_ok=True)
    source_text = "\n".join(
        item.strip()
        for item in (problem_statement, requirements, evaluation_criteria, data_description)
        if item.strip()
    )
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    benchmark = f"custom_{_slugify(problem_statement)}_{digest[:10]}"
    benchmark_dir = (root_dir / benchmark).resolve(strict=False)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "schema_version": CUSTOM_BENCHMARK_SCHEMA_VERSION,
        "name": benchmark,
        "paper_section": "custom",
        "paper_task_name": "User-defined proxy evaluator",
        "family": _family_from_text(source_text),
        "metric": "custom_proxy_relative_l2",
        "description": _one_line(problem_statement),
        "fidelity_level": "proxy",
        "expected_runtime_s": 20,
        "requires_torch": False,
        "requires_gpu": False,
        "paper_gap_notes": CUSTOM_BENCHMARK_CLAIM_BOUNDARY,
        "source_digest": digest,
        "generator": "agenticsciml.custom_benchmarks.v1",
    }
    files = {
        CUSTOM_BENCHMARK_SPEC: json.dumps(spec, indent=2, sort_keys=True) + "\n",
        "Problem.md": _problem_md(problem_statement, data_description),
        "Requirements.md": _requirements_md(requirements),
        "Evaluation.md": _evaluation_md(evaluation_criteria),
        "Data_config.json": json.dumps(
            {
                "train_path": "train_data.npz",
                "validation_path": "val_data.npz",
                "description": (
                    "Auto-generated deterministic proxy dataset. "
                    "Private validation labels are evaluator-only."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "guidelines.md": _guidelines_md(),
        "generate_data.py": _generate_data_py(digest),
        "evaluate.py": _evaluate_py(),
    }
    for relative_path, text in files.items():
        _atomic_write_text(benchmark_dir / relative_path, text)
    return CustomBenchmarkBundle(
        benchmark=benchmark,
        benchmark_dir=benchmark_dir,
        files=sorted(files),
        claim_boundary=CUSTOM_BENCHMARK_CLAIM_BOUNDARY,
    )


def _slugify(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    slug = "_".join(words[:5]) or "problem"
    return slug[:48].strip("_") or "problem"


def _one_line(text: str, *, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _family_from_text(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("operator", "deeponet", "function-to-function")):
        return "custom operator learning"
    if any(token in lower for token in ("pde", "pinn", "poisson", "burgers", "diffusion")):
        return "custom PINN"
    if any(token in lower for token in ("inverse", "sensor", "reconstruct", "scattering")):
        return "custom inverse reconstruction"
    return "custom regression"


def _problem_md(problem_statement: str, data_description: str) -> str:
    return (
        "# Custom Problem\n\n"
        "This benchmark was auto-generated from a user problem description.\n\n"
        "## User Problem Statement\n\n"
        f"{problem_statement.strip()}\n\n"
        "## Data Description\n\n"
        f"{data_description.strip() or 'No structured data description was provided.'}\n\n"
        "## Claim Boundary\n\n"
        f"{CUSTOM_BENCHMARK_CLAIM_BOUNDARY}\n"
    )


def _requirements_md(requirements: str) -> str:
    return (
        "# Requirements\n\n"
        f"{requirements.strip() or 'No additional user requirements were provided.'}\n\n"
        "Generated solutions must keep the evaluator contract fixed and must not "
        "read validation labels or evaluator-private files.\n"
    )


def _evaluation_md(evaluation_criteria: str) -> str:
    return (
        "# Evaluation\n\n"
        f"{evaluation_criteria.strip() or 'No domain-specific metric was provided.'}\n\n"
        "The generated proxy evaluator uses private-label relative L2 on a deterministic "
        "one-dimensional surrogate target derived from the problem text hash. This metric "
        "is only workflow evidence.\n"
    )


def _guidelines_md() -> str:
    return (
        "# Custom Proxy Benchmark Guidelines\n\n"
        "- Implement `solution.py` with `--mode=validate`, `--mode=train`, and `--mode=predict`.\n"
        "- `train` may read only `train_data.npz`.\n"
        "- `predict` receives `predict_input.npz` with only `x_val` and must write `predictions.npz`.\n"
        "- Do not read `val_data.npz`, evaluator-private paths, network resources, or host secrets.\n"
        "- Treat this benchmark as workflow proxy evidence only.\n"
    )


def _generate_data_py(problem_digest: str) -> str:
    return f'''from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROBLEM_DIGEST = "{problem_digest}"


def _coefficients() -> tuple[float, float, float, float, float]:
    values = [int(PROBLEM_DIGEST[i:i + 8], 16) / 0xFFFFFFFF for i in range(0, 40, 8)]
    amplitude = 0.7 + 0.8 * values[0]
    frequency = 1.5 + 5.0 * values[1]
    phase = 2.0 * np.pi * values[2]
    slope = -0.8 + 1.6 * values[3]
    jump = -0.35 + 0.7 * values[4]
    return amplitude, frequency, phase, slope, jump


def target_function(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    amplitude, frequency, phase, slope, jump = _coefficients()
    base = amplitude * np.sin(frequency * np.pi * x[:, 0] + phase)
    trend = slope * x[:, 0] + 0.25 * x[:, 0] ** 2
    discontinuity = jump * (x[:, 0] > 0.05).astype(float)
    return (base + trend + discontinuity).reshape(-1, 1)


def generate(seed: int, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = rng.uniform(-1.0, 1.0, size=(160, 1))
    u_train = target_function(x_train) + rng.normal(0.0, 0.02, size=(160, 1))
    x_val = np.linspace(-1.0, 1.0, 320, dtype=float).reshape(-1, 1)
    u_val = target_function(x_val)
    np.savez(output_dir / "train_data.npz", x_train=x_train, u_train=u_train)
    np.savez(output_dir / "val_data.npz", x_val=x_val, u_val=u_val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
'''


def _evaluate_py() -> str:
    return '''from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _align_predictions(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    preds = _to_numpy(preds)
    if preds.shape == target.shape:
        return preds
    if preds.ndim == 1 and preds.shape[0] == target.shape[0]:
        return preds.reshape(-1, 1)
    if preds.ndim == 2 and preds.shape[0] == target.shape[0] and preds.shape[1] == 1:
        return np.repeat(preds, target.shape[1], axis=1)
    if preds.size == target.size:
        return preds.reshape(target.shape)
    raise ValueError(f"Prediction shape {{preds.shape}} cannot align with target {{target.shape}}.")


def _load_predictions(path: Path) -> np.ndarray:
    data = np.load(path)
    if "predictions" not in data:
        raise ValueError("predictions.npz must contain a 'predictions' array.")
    return _to_numpy(data["predictions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=os.environ.get("AGENTICSCIML_PREDICTIONS_DATA", "predictions.npz"))
    args = parser.parse_args()
    validation_path = Path(os.environ.get("AGENTICSCIML_VALIDATION_DATA", "val_data.npz"))
    data = np.load(validation_path)
    u_val = data["u_val"]
    preds = _align_predictions(_load_predictions(Path(args.predictions)), u_val)
    denom = float(np.linalg.norm(u_val)) + 1e-12
    score = float(np.linalg.norm(preds - u_val) / denom)
    payload = {
        "metric": "custom_proxy_relative_l2",
        "score": score,
        "higher_is_better": False,
    }
    Path("eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
