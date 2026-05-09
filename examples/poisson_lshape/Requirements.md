# Requirements

- Python 3.11.
- Linux target runtime.
- `solution.py` must define class `MODEL`.
- `solution.py --mode=validate` must run without training.
- `solution.py --mode=train` must write `model.pkl`.
- Do not modify `evaluate.py`, `val_data.npz`, or other evaluator artifacts.
- Tests must not require network access.
