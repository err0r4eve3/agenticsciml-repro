# Guidelines

- Load `train_data.npz` with `x_train` shape `(n, 2)` and `u_train` shape `(n, 1)`.
- Implement `MODEL.predict(x)` for `x` shape `(n, 2)`.
- Save the trained model to `model.pkl` using `pickle` or another evaluator-compatible object.
- The validation metric is relative L2 error.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
