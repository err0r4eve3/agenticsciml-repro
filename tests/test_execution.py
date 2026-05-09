import json
from pathlib import Path

from agenticsciml.config import EvaluationContract
from agenticsciml.execution.runner import run_command
from agenticsciml.execution.sandbox import prepare_solution_workspace, train_and_evaluate


TRIVIAL_SOLUTION = '''
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
    parser.add_argument("--mode", choices=["validate", "train"], required=True)
    args = parser.parse_args()
    if args.mode == "validate":
        assert hasattr(MODEL(), "predict")
        return
    data = np.load("train_data.npz")
    model = MODEL()
    model.fit(data["x_train"], data["u_train"])
    with open(MODEL_CHECKPOINT, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
'''


def test_run_command_captures_exit_code_and_logs(tmp_path: Path) -> None:
    result = run_command(tmp_path, ["python", "-c", "print('ok')"], timeout_s=5)

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"


def test_trivial_solution_trains_and_evaluates(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "solution_000"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(TRIVIAL_SOLUTION, encoding="utf-8")

    contract = EvaluationContract.default_function_approx()
    result = train_and_evaluate(workspace, contract, timeout_s=20)

    assert result.exit_code == 0
    eval_data = json.loads((workspace / "eval.json").read_text(encoding="utf-8"))
    assert eval_data["metric"] == "validation_mse"
    assert isinstance(eval_data["score"], float)
