# Requirements

- Python 3.11.
- Linux target runtime.
- `solution.py` must define class `MODEL`.
- `solution.py --mode=validate` must run without training.
- `solution.py --mode=train` must write `model.pkl`.
- `solution.py --mode=predict --input predict_input.npz --output predictions.npz`
  must write a `predictions` array for the provided `x_val` features.
- Do not modify `evaluate.py`, `val_data.npz`, or other evaluator artifacts.
- Tests must not require network access.
