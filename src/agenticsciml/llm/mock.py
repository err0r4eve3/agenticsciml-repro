from __future__ import annotations

import json
import re
from typing import Any

from agenticsciml.llm.base import LLMClient
from agenticsciml.patching import make_unified_patch, solution_digest


ROOT_SOLUTION = r'''
from __future__ import annotations

import argparse
import pickle

import numpy as np

MODEL_CHECKPOINT = "model.pkl"


class MODEL:
    def __init__(self):
        self.mean = 0.0

    def fit(self, x, y):
        self.mean = float(np.mean(y))

    def predict(self, x):
        return np.full((len(x), 1), self.mean, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    if args.mode == "validate":
        model = MODEL()
        probe = model.predict(np.zeros((3, 1)))
        assert probe.shape == (3, 1)
        return
    if args.mode == "predict":
        with open(MODEL_CHECKPOINT, "rb") as f:
            model = pickle.load(f)
        data = np.load(args.input)
        np.savez(args.output, predictions=model.predict(data["x_val"]))
        return

    data = np.load("train_data.npz")
    model = MODEL()
    model.fit(data["x_train"], data["u_train"])
    with open(MODEL_CHECKPOINT, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
'''.strip()


FOURIER_RIDGE_SOLUTION = r'''
from __future__ import annotations

import argparse
import pickle

import numpy as np

MODEL_CHECKPOINT = "model.pkl"


class MODEL:
    def __init__(self, order=4, ridge=1e-5):
        self.order = order
        self.ridge = ridge
        self.coef = None
        self.x_mean = None
        self.x_scale = None

    def _features(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.x_mean is None or self.x_scale is None:
            z = x
        else:
            z = (x - self.x_mean) / self.x_scale
        cols = [np.ones((len(z), 1)), z, z ** 2]
        cols.append((z[:, :1] > 0.15).astype(float))
        for k in range(1, self.order + 1):
            cols.append(np.sin(k * np.pi * z))
            cols.append(np.cos(k * np.pi * z))
        return np.concatenate(cols, axis=1)

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.x_mean = np.mean(x, axis=0, keepdims=True)
        self.x_scale = np.std(x, axis=0, keepdims=True) + 1e-6
        phi = self._features(x)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        gram = phi.T @ phi + self.ridge * np.eye(phi.shape[1])
        self.coef = np.linalg.solve(gram, phi.T @ y)

    def predict(self, x):
        if self.coef is None:
            raise RuntimeError("Model has not been trained.")
        return self._features(x) @ self.coef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    if args.mode == "validate":
        model = MODEL()
        model.x_mean = np.zeros((1, 1))
        model.x_scale = np.ones((1, 1))
        model.coef = np.zeros((model._features(np.zeros((2, 1))).shape[1], 1))
        probe = model.predict(np.zeros((2, 1)))
        assert probe.shape == (2, 1)
        return
    if args.mode == "predict":
        with open(MODEL_CHECKPOINT, "rb") as f:
            model = pickle.load(f)
        data = np.load(args.input)
        np.savez(args.output, predictions=model.predict(data["x_val"]))
        return

    data = np.load("train_data.npz")
    model = MODEL()
    model.fit(data["x_train"], data["u_train"])
    with open(MODEL_CHECKPOINT, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
'''.strip()


class MockLLMClient(LLMClient):
    def complete_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        if "data analyst" in prompt.lower():
            return (
                "The data describes a one-dimensional target with oscillation, "
                "a clear jump discontinuity, and likely underfitting risk for "
                "plain smooth baselines."
            )
        if "critic" in prompt.lower():
            return "The plan is feasible if it keeps the evaluation contract fixed and avoids overfitting."
        if "debug" in prompt.lower():
            return "No deterministic patch is required in mock mode."
        return "Mock response."

    def complete_json(
        self,
        prompt: str,
        schema_name: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        name = schema_name.lower()
        if name == "selector":
            return {
                "selected_parent_ids": ["solution_000"],
                "rationale": "Best known root is selected for exploitation.",
            }
        if name == "proposal":
            return {
                "title": "Add Fourier ridge features",
                "diagnosis": "The parent constant baseline underfits oscillation and the discontinuous jump.",
                "mutation_plan": [
                    "Replace the constant predictor with a deterministic Fourier feature map.",
                    "Add an explicit jump indicator around the observed discontinuity.",
                    "Fit coefficients with ridge regression for stable local execution.",
                ],
                "expected_effect": "The child should reduce validation MSE on oscillatory regions and the jump.",
                "risks": ["The fixed jump location may not generalize to other datasets."],
            }
        if name == "root_engineer":
            return {
                "proposal": "Use a deliberately simple single-agent baseline.",
                "code": ROOT_SOLUTION,
            }
        if name == "engineer":
            match = re.search(r"parent_digest:\s*([a-f0-9]{64})", prompt)
            parent_digest = match.group(1) if match else solution_digest(ROOT_SOLUTION + "\n")
            return {
                "mutation_summary": "Mutated the baseline into Fourier ridge regression.",
                "expected_effect": "Lower validation error on smooth, oscillatory, and multi-output proxy tasks.",
                "risks": ["The fixed feature basis may underfit high-dimensional targets."],
                "parent_digest": parent_digest,
                "patch": make_unified_patch(
                    prompt.split("Parent code:\n", 1)[1] if "Parent code:\n" in prompt else ROOT_SOLUTION + "\n",
                    FOURIER_RIDGE_SOLUTION + "\n",
                ),
                "files_changed": ["solution.py"],
            }
        if name == "analysis":
            return {
                "summary": "The solution completed validation, training, and evaluation under the fixed contract.",
                "strengths": ["Artifacts are reproducible.", "Evaluation uses the shared metric."],
                "weaknesses": ["Mock analysis is not a scientific interpretation."],
                "next_steps": ["Compare neighboring mutations under the same budget."],
            }
        if name == "evaluator":
            return {
                "metric_name": "validation_mse",
                "higher_is_better": False,
                "checkpoint_path": "model.pkl",
            }
        if name == "debugger":
            match = re.search(r"parent_digest:\s*([a-f0-9]{64})", prompt)
            parent_digest = match.group(1) if match else ""
            return {
                "summary": "No deterministic patch produced in mock mode.",
                "failure_kind": "runtime_error",
                "minimal_fix": True,
                "parent_digest": parent_digest,
                "patch": "",
                "files_changed": ["solution.py"],
                "risks": [],
            }
        return {"text": json.dumps({"schema_name": schema_name, "mock": True})}
