# Function Approximation Faithful-Small Guidelines

The dataset is small, one-dimensional, piecewise, and discontinuous. Effective
solutions usually need to combine local expressivity near the interface with
oscillatory basis capacity on one side.

Useful strategies include mixture-of-experts, gated piecewise models, Fourier
features, robust training schedules, and localized loss weighting. Keep the
fixed evaluator contract unchanged.

Do not infer the target by reading benchmark generator source. Generated
solutions should learn from `train_data.npz` and predict on `predict_input.npz`.
