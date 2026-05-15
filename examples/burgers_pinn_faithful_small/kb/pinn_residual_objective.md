# PINN Residual Objective

Applies to: `burgers_pinn_faithful_small`.

Sources:

- Raissi, Perdikaris, and Karniadakis, Physics-informed neural networks, Journal of Computational Physics 378 (2019), DOI `10.1016/j.jcp.2018.10.045`.
- Public code path: `maziarraissi/PINNs/appendix/continuous_time_inference (Burgers)/Burgers.py`.

The useful mutation pattern is to keep two separate pressures in the model:
fit observed initial/boundary samples and reduce a Burgers-style PDE residual.
For this benchmark, generated code does not receive private labels during
training, so residual-inspired ideas must be implemented from public training
features or model regularization rather than evaluator data.

Good proposals:

- add a public residual proxy term computed from model predictions on sampled
  `(x, t)` points;
- keep supervised fit and residual penalty separately weighted;
- report expected effects and risks for each weight or sampling change.

Boundaries:

- Do not read validation labels or evaluator-private paths.
- Do not copy TensorFlow 1 code into the generated solution.
- Keep changes compatible with the repository's timeout and dependency budget.
