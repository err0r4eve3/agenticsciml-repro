# Guidelines

- `x_train` has shape `(n, 64)` and stores input function samples.
- `u_train` has shape `(n, 64)` and stores antiderivative samples.
- `MODEL.predict(x)` should return `(n, 64)`.
- The metric is relative L2 error across all validation functions.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
