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


MALICIOUS_VAL_LOAD_SOLUTION = '''
import argparse
import pickle
import numpy as np

MODEL_CHECKPOINT = "model.pkl"

class MODEL:
    def predict(self, x):
        return np.zeros((len(x), 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train"], required=True)
    args = parser.parse_args()
    if args.mode == "train":
        np.load("val_data.npz")
        with open(MODEL_CHECKPOINT, "wb") as f:
            pickle.dump(MODEL(), f)

if __name__ == "__main__":
    main()
'''


MALICIOUS_VAL_OPEN_SOLUTION = '''
import argparse
import pickle
import numpy as np

MODEL_CHECKPOINT = "model.pkl"

class MODEL:
    def predict(self, x):
        return np.zeros((len(x), 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train"], required=True)
    args = parser.parse_args()
    if args.mode == "train":
        open("val_data.npz", "rb").read(1)
        with open(MODEL_CHECKPOINT, "wb") as f:
            pickle.dump(MODEL(), f)

if __name__ == "__main__":
    main()
'''


DISCOVERY_RESISTANT_SOLUTION = '''
import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np

MODEL_CHECKPOINT = "model.pkl"

class MODEL:
    def __init__(self):
        self.mean = 0.0

    def fit(self, x, y):
        self.mean = float(np.mean(y))

    def predict(self, x):
        return np.full((len(x), 1), self.mean, dtype=float)

def assert_no_private_data_visible():
    dynamic_name = "val" + "_data.npz"
    discovered = [str(path) for path in Path(".").rglob("*.npz")]
    discovered.extend(glob.glob("**/*.npz", recursive=True))
    if any(Path(path).name == dynamic_name for path in discovered):
        raise RuntimeError("validation data was discoverable")
    if Path(dynamic_name).exists():
        raise RuntimeError("dynamic validation path exists")
    if os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("host secret leaked into generated process")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train"], required=True)
    args = parser.parse_args()
    assert_no_private_data_visible()
    if args.mode == "validate":
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


def test_run_command_uses_clean_environment_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")

    result = run_command(
        tmp_path,
        ["python", "-c", "import os; raise SystemExit(1 if os.environ.get('OPENAI_API_KEY') else 0)"],
        timeout_s=5,
    )

    assert result.exit_code == 0


def test_trivial_solution_trains_and_evaluates(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "solution_000"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(TRIVIAL_SOLUTION, encoding="utf-8")

    contract = EvaluationContract.default_function_approx()
    assert not (workspace / "val_data.npz").exists()
    assert not (workspace / ".evaluator").exists()
    assert (tmp_path / "private_eval" / "solution_000" / "val_data.npz").exists()
    result = train_and_evaluate(workspace, contract, timeout_s=20)

    assert result.exit_code == 0
    eval_data = json.loads((workspace / "eval.json").read_text(encoding="utf-8"))
    assert eval_data["metric"] == "validation_mse"
    assert isinstance(eval_data["score"], float)


def test_solution_cannot_load_validation_data_during_train(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "malicious_load"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(MALICIOUS_VAL_LOAD_SOLUTION, encoding="utf-8")

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code != 0
    assert "validation data" in result.stderr.lower()


def test_solution_cannot_discover_validation_data_or_host_secret_during_train(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "discovery_resistant"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(DISCOVERY_RESISTANT_SOLUTION, encoding="utf-8")

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code == 0
    assert not (workspace / ".evaluator").exists()
    assert not any(path.name == "val_data.npz" for path in workspace.rglob("*.npz"))
    eval_data = json.loads((workspace / "eval.json").read_text(encoding="utf-8"))
    assert eval_data["metric"] == "validation_mse"


def test_solution_cannot_open_validation_data_during_train(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "malicious_open"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(MALICIOUS_VAL_OPEN_SOLUTION, encoding="utf-8")

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code != 0
    assert "validation data" in result.stderr.lower()
