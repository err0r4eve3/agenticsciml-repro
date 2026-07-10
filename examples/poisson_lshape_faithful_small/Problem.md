# Faithful-Small L-shaped Poisson PINN Benchmark

Approximate a Poisson equation on an L-shaped domain with a re-entrant corner.
The generated data includes supervised anchors, boundary samples, and interior
collocation points with forcing values from an analytic singular solution. The
singular term is measured inside the `3*pi/2` wedge, vanishes continuously on
both positive-axis notch boundaries, and is harmonic away from the corner; the
reported forcing is therefore the forcing of the added smooth term.

This is a low-budget faithful-small benchmark for the paper's L-shaped Poisson
PINN task. It preserves the irregular geometry, singular corner behavior, and
PDE-residual data interface, but it is still much smaller than the paper-scale
experiment and does not claim reported-score parity.
