# SHRED-style Sensor History Decoder

Applies to: `cylinder_wake_reconstruction_faithful_small`.

Public grounding:

- Nature Communications 2025, "Reduced order modeling with shallow recurrent
  decoder networks", describes SHRED-ROM as sparse sensor sequences encoded by
  a temporal model and decoded by a shallow decoder, with optional POD
  compression.
- Public code: `https://github.com/MatteoTomasetto/SHRED-ROM`.
- PySHRED describes the same sparse-sensor to high-dimensional reconstruction
  pattern with a sequence model and decoder.

Local benchmark boundary:

- This benchmark uses lagged sensor-history features and deterministic
  synthetic cylinder-wake-like fields.
- It does not run the SHRED-ROM codebase, train an LSTM, use the original data,
  or reproduce paper scores.

Useful mutation ideas:

- normalize each sensor channel using training-set statistics;
- fit a low-rank basis to `u_train`, then regress latent coefficients from
  lagged sensor history and current time;
- add sinusoidal time features before a shallow decoder;
- smooth decoded fields only with train-visible assumptions, never private
  validation labels.
