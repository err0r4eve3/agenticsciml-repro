# Guidelines

- `x_train` has shape `(n, 64)` and stores input function samples.
- `u_train` has shape `(n, 64)` and stores antiderivative samples.
- `MODEL.predict(x)` should return `(n, 64)`.
- The metric is relative L2 error across all validation functions.
