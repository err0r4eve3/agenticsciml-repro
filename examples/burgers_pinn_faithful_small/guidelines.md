# Burgers PINN Faithful-Small Guidelines

- `x_train` has shape `(n, 2)` with columns `x` and `t`.
- `u_train` has shape `(n, 1)`.
- `x_initial` / `u_initial` provide initial-condition anchors.
- `x_boundary` / `u_boundary` provide periodic boundary anchors.
- `x_collocation` provides unlabeled interior points for optional residual or
  curriculum strategies.
- `MODEL.predict(x)` must accept `(n, 2)` arrays.
- The score combines dense solution relative L2 with initial and boundary
  relative L2 components.
- Useful strategies include periodic input features, staged IC/BC pretraining,
  smooth residual-inspired regularization, and adaptive sampling over time.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
- Do not infer the manufactured target by reading benchmark generator source.
