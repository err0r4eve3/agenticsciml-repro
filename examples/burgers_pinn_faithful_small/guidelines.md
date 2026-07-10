# Burgers PINN Faithful-Small Guidelines

- `x_train` has shape `(n, 2)` with columns `x` and `t`.
- `u_train` has shape `(n, 1)`.
- `x_initial` / `u_initial` provide initial-condition anchors.
- `x_boundary` / `u_boundary` provide periodic boundary anchors.
- `x_collocation` provides unlabeled interior points for optional residual or
  curriculum strategies.
- `viscosity` is the scalar `nu = 0.04`.
- The governing equation is `u_t + u*u_x - nu*u_xx = 0`; the residual is
  `R = u_t + u*u_x - nu*u_xx`.
- `MODEL.predict(x)` must accept `(n, 2)` arrays.
- The score combines dense solution relative L2 with initial and boundary
  components and a centered-finite-difference PDE residual term. Because the
  boundary labels are zero, boundary RMSE is normalized by dense-solution RMS
  rather than by the near-zero boundary norm.
- Useful strategies include periodic input features, staged IC/BC pretraining,
  automatic-differentiation residual regularization, and adaptive sampling over
  time.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
- Do not infer the manufactured target by reading benchmark generator source.
