# Cylinder Wake Faithful-Small Guidelines

- `x_train` has shape `(n, 41)`: five time steps of eight sparse sensor readings
  followed by the current normalized time.
- `u_train` has shape `(n, 576)` for a flattened `24 x 24` vorticity grid.
- `MODEL.predict(x)` should return `(n, 576)`.
- Useful strategies include lagged sensor-history encoders, shallow decoders,
  low-rank/POD-style bases derived from training fields, temporal features, and
  band-limited smoothing.
- This benchmark is SHRED-style, not a literal SHRED-ROM or PySHRED run: no
  LSTM/RNN implementation is required by the benchmark contract.
- The metric is mean per-sample relative L2 error.
- `solution.py` must support `--mode=validate`, `--mode=train`, and
  `--mode=predict --input predict_input.npz --output predictions.npz`.
  Predict mode receives only `x_val` and must write `predictions.npz` with a
  `predictions` array.
