# Faithful-Small Burgers PINN Benchmark

Approximate a time-dependent viscous Burgers-style solution with periodic
spatial boundaries, initial-condition anchors, and unlabeled collocation
coordinates. Agents may use PINN residual ideas, periodic features, curriculum
sampling, or data-driven surrogates.

This is a low-budget faithful-small benchmark for the paper's Burgers PINN
task. It preserves the time-dependent coordinate interface, IC/BC data shape,
and collocation interface, but it uses a small deterministic manufactured
solution and does not claim paper-score parity.
