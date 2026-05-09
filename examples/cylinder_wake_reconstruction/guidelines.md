# Guidelines

- `x_train` has shape `(n, 5)`: four sensor readings plus normalized time.
- `u_train` has shape `(n, 256)` for a flattened `16 x 16` vorticity grid.
- `MODEL.predict(x)` should return `(n, 256)`.
- The metric is relative L2 error.
