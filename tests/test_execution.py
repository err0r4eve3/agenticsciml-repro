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
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    if args.mode == "validate":
        assert hasattr(MODEL(), "predict")
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
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
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
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
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
    validation_key = "AGENTICSCIML_" + "VALIDATION_DATA"
    if os.environ.get(validation_key):
        raise RuntimeError("validation path leaked into generated process env")
    if os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("host secret leaked into generated process")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    assert_no_private_data_visible()
    if args.mode == "validate":
        return
    if args.mode == "predict":
        assert_no_private_data_visible()
        with open(MODEL_CHECKPOINT, "rb") as f:
            model = pickle.load(f)
        data = np.load(args.input)
        if "u_val" in data:
            raise RuntimeError("validation labels leaked into prediction input")
        np.savez(args.output, predictions=model.predict(data["x_val"]))
        return
    data = np.load("train_data.npz")
    model = MODEL()
    model.fit(data["x_train"], data["u_train"])
    with open(MODEL_CHECKPOINT, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
'''


PARENT_TRAVERSAL_SOLUTION = '''
import argparse
import pickle
from pathlib import Path

import numpy as np

MODEL_CHECKPOINT = "model.pkl"

class MODEL:
    def predict(self, x):
        return np.zeros((len(x), 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    if args.mode == "train":
        discovered = [path for path in Path("../..").rglob("*.npz")]
        if discovered:
            raise RuntimeError("parent traversal discovered npz files")
        with open(MODEL_CHECKPOINT, "wb") as f:
            pickle.dump(MODEL(), f)
    elif args.mode == "predict":
        data = np.load(args.input)
        np.savez(args.output, predictions=np.zeros((len(data["x_val"]), 1)))

if __name__ == "__main__":
    main()
'''


PREDICT_LEAK_PROBE_SOLUTION = '''
import argparse
import json
import os
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
        return
    if args.mode == "train":
        data = np.load("train_data.npz")
        model = MODEL()
        model.fit(data["x_train"], data["u_train"])
        with open(MODEL_CHECKPOINT, "wb") as f:
            pickle.dump(model, f)
        return
    with open(MODEL_CHECKPOINT, "rb") as f:
        model = pickle.load(f)
    data = np.load(args.input)
    validation_key = "AGENTICSCIML_" + "VALIDATION_DATA"
    leak_report = {
        "input_keys": sorted(data.files),
        "has_validation_env": validation_key in os.environ,
        "validation_env_values": [
            value for key, value in os.environ.items()
            if "VALIDATION" in key or "val_data" in value
        ],
    }
    open("predict_leak_report.json", "w", encoding="utf-8").write(json.dumps(leak_report, sort_keys=True))
    if "u_val" in data or leak_report["has_validation_env"] or leak_report["validation_env_values"]:
        raise RuntimeError("validation labels or paths leaked into predict")
    np.savez(args.output, predictions=model.predict(data["x_val"]))

if __name__ == "__main__":
    main()
'''


SYNTHETIC_FALLBACK_SOLUTION = '''
import argparse
import pickle

import numpy as np

MODEL_CHECKPOINT = "model.pkl"

class MODEL:
    def predict(self, x):
        return np.zeros((len(x), 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "train", "predict"], required=True)
    parser.add_argument("--input", default="predict_input.npz")
    parser.add_argument("--output", default="predictions.npz")
    args = parser.parse_args()
    if args.mode == "validate":
        print("Error loading data: Cannot infer features/targets.")
        return
    if args.mode == "train":
        print("Falling back to synthetic piecewise data.")
        with open(MODEL_CHECKPOINT, "wb") as f:
            pickle.dump(MODEL(), f)
        return
    data = np.load(args.input)
    np.savez(args.output, predictions=MODEL().predict(data["x_val"]))

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


def test_solution_cannot_parent_traverse_to_validation_data(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "parent_traversal"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(PARENT_TRAVERSAL_SOLUTION, encoding="utf-8")

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code != 0
    assert "parent traversal" in result.stderr.lower()


def test_predict_phase_sees_features_but_not_validation_labels_or_path(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "predict_probe"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(PREDICT_LEAK_PROBE_SOLUTION, encoding="utf-8")

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code == 0, result.stderr
    leak_report = json.loads((workspace / "predict_leak_report.json").read_text(encoding="utf-8"))
    assert leak_report["input_keys"] == ["x_val"]
    assert leak_report["has_validation_env"] is False
    assert leak_report["validation_env_values"] == []


def test_solution_cannot_open_validation_data_during_train(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx").resolve()
    workspace = tmp_path / "malicious_open"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(MALICIOUS_VAL_OPEN_SOLUTION, encoding="utf-8")

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code != 0
    assert "validation data" in result.stderr.lower()


def test_solution_cannot_import_benchmark_generator_source(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx_faithful_small").resolve()
    workspace = tmp_path / "generator_leak"
    prepare_solution_workspace(benchmark, workspace)
    assert not (workspace / "generate_data.py").exists()
    (workspace / "solution.py").write_text(
        "import generate_data\n",
        encoding="utf-8",
    )

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code != 0
    assert "generate_data" in result.stderr


def test_solution_cannot_silently_fallback_to_synthetic_training_data(tmp_path: Path) -> None:
    benchmark = Path("examples/function_approx_faithful_small").resolve()
    workspace = tmp_path / "synthetic_fallback"
    prepare_solution_workspace(benchmark, workspace)
    (workspace / "solution.py").write_text(SYNTHETIC_FALLBACK_SOLUTION, encoding="utf-8")

    result = train_and_evaluate(workspace, EvaluationContract.default_function_approx(), timeout_s=20)

    assert result.exit_code == 125
    assert "synthetic-data fallback" in result.stderr
