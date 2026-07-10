# Faithful-Small Burgers PINN Benchmark

Approximate the one-dimensional unforced viscous Burgers equation

```text
u_t + u * u_x - nu * u_xx = 0,   x in [-1, 1], t in [0, 1], nu = 0.04,
u(-1, t) = u(1, t),   u_x(-1, t) = u_x(1, t).
```

The initial condition is

```text
u(x, 0) = 2 * nu * 0.9 * pi * sin(pi*x) / (1 + 0.9*cos(pi*x)).
```

Training data includes initial-condition anchors, periodic boundary anchors,
and unlabeled collocation coordinates. Agents may use the explicit PDE residual
`R = u_t + u*u_x - nu*u_xx`, periodic features, curriculum sampling, or
data-driven surrogates.

This is a low-budget faithful-small benchmark for the paper's Burgers PINN
task. It preserves the time-dependent coordinate interface, IC/BC data shape,
and collocation interface. The private deterministic reference is generated
from the corresponding exact Cole-Hopf solution, but this low-budget task does
not claim paper-score parity.
