# Collocation and Scaling

Applies to: `burgers_pinn`.

Sources:

- Raissi, Perdikaris, and Karniadakis, Physics-informed neural networks, Journal of Computational Physics 378 (2019), DOI `10.1016/j.jcp.2018.10.045`.
- Public code paths: `maziarraissi/PINNs/appendix/continuous_time_inference (Burgers)/Burgers.py` and `Burgers_systematic.py`.

The public Burgers example combines sparse supervised samples with many
collocation points and normalizes the two input coordinates before the neural
network. In this local proxy, the same idea should be budgeted: favor stable
feature scaling and targeted sampling over large model or data expansion.

Good proposals:

- normalize `x` and `t` to a stable range before nonlinear features;
- add deterministic collocation points around early time, boundaries, or steep
  gradient regions;
- vary collocation count, depth, or width in one controlled step and record the
  expected score effect.

Boundaries:

- Keep data generation deterministic under the benchmark seed.
- Do not create collocation files outside the solution workspace.
- Treat larger collocation grids as a timeout risk.

