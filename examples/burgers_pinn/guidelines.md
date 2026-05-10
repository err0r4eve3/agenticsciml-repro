# Guidelines

- `x_train` has shape `(n, 2)` with columns `x` and `t`.
- `u_train` has shape `(n, 1)`.
- `MODEL.predict(x)` must accept `(n, 2)` arrays.
- The score is relative L2 error.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
