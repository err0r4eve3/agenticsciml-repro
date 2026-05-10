# Guidelines

- `x_train` has shape `(n, 96)`: diffusion, source, and initial-condition blocks over a 32-point grid.
- `u_train` has shape `(n, 32)`.
- `MODEL.predict(x)` should return `(n, 32)`.
- The metric is relative L2 error.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
