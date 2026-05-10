# Guidelines

- `x_train` has shape `(n, 5)`: four sensor readings plus normalized time.
- `u_train` has shape `(n, 256)` for a flattened `16 x 16` vorticity grid.
- `MODEL.predict(x)` should return `(n, 256)`.
- The metric is relative L2 error.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
