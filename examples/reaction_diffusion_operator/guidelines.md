# Guidelines

- `x_train` has shape `(n, 96)`: diffusion, source, and initial-condition blocks over a 32-point grid.
- `u_train` has shape `(n, 32)`.
- `MODEL.predict(x)` should return `(n, 32)`.
- The metric is relative L2 error.
