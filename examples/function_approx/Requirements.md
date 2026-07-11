# Requirements

- Implement `solution.py`.
- Define a class named `MODEL`.
- `python solution.py --mode=validate` must run without training.
- Support `python solution.py --mode=train`.
- Support `python solution.py --mode=predict --input predict_input.npz --output predictions.npz`.
- Training must write `model.pkl`.
- Predict mode receives only `x_val` features and must write `predictions.npz`
  with a `predictions` array.
- Prediction shape must be `(n, 1)` or `(n,)` for an input batch `x_val` with shape `(n, 1)`.
- Real SciML submissions may use PyTorch, NumPy, or the Python standard library.
- Do not require internet access.
- Keep runtime short enough for local iteration.
