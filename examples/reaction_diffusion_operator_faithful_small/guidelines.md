# Reaction-Diffusion Operator Faithful-Small Guidelines

- `x_train` has shape `(n, 150)`: diffusion, source, and initial-condition
  blocks over a 50-point grid.
- `u_train` has shape `(n, 2000)`: a time-major flattening of 40 time steps by
  50 spatial points.
- `x_grid` and `t_grid` store the shared spatial and temporal grids for
  training-time interpretation.
- `MODEL.predict(x)` should return `(n, 2000)`.
- The official metric is mean per-sample relative L2 error over the full
  spatiotemporal response.
- Useful strategies include multiple branch encoders, Fourier or spectral
  features, low-rank operator regression, and channel-aware normalization.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
- Do not claim full PDE-solving ability or paper-score reproduction from this
  local deterministic task.
