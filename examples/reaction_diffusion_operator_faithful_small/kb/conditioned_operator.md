# Conditioned Operator

Applies to: `reaction_diffusion_operator_faithful_small`.

Multiple-input reaction-diffusion operators often improve when each input field
has its own encoder before fusion. Other useful ideas include normalization per
field type, Fourier features over the shared grid, constraint-conditioned
branches, and low-rank decoders over the time-major output grid.

Good proposals:

- split the 150 input features into diffusion, source, and initial-condition
  channels before feature construction;
- normalize each channel separately instead of using one global scale;
- predict the flattened 40x50 response with a low-rank or basis decoder.

Boundaries:

- do not claim full PDE solver correctness from the prediction-only score;
- do not read `val_data.npz` or evaluator-private artifacts;
- keep the official score as relative L2 and treat gradient/time diagnostics as
  secondary evidence only.
