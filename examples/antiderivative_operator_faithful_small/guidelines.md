# Antiderivative Operator Faithful-Small Guidelines

- `x_train` has shape `(n, 100)` and stores input function samples.
- `u_train` has shape `(n, 100)` and stores antiderivative samples.
- `x_grid` stores the shared 100-point grid for training-time use.
- `MODEL.predict(x)` should return `(n, 100)`.
- The metric is mean per-sample relative L2 error across validation functions.
- Useful strategies include DeepONet-style branch/trunk decompositions, linear
  or bias-free branch maps, spectral features, and direct operator regression.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
- Do not infer hidden validation regimes by reading benchmark generator source.
