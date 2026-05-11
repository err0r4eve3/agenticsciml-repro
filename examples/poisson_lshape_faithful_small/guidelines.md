# Guidelines

- Load `train_data.npz` with supervised arrays `x_train` shape `(n, 2)` and
  `u_train` shape `(n, 1)`.
- Optional PDE/PINN-style arrays are also present:
  - `x_boundary`, `u_boundary`
  - `x_residual`, `f_residual`
- Implement `MODEL.predict(x)` for `x` shape `(n, 2)`.
- Save the trained model to `model.pkl` using `pickle` or another
  evaluator-compatible object.
- The validation metric is relative L2 error.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
