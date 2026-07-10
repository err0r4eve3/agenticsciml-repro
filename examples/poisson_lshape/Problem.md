# L-shaped Poisson PINN Benchmark

Approximate the solution of a Poisson-style elliptic problem on an L-shaped
domain. The data generator provides reference values on the domain, including a
corner singularity near the re-entrant corner. The singular basis uses the
interior wedge angle in `[0, 3*pi/2]` and is continuous on both positive-axis
notch boundaries. Generated solutions may use PINN ideas, domain decomposition,
singular basis functions, or data-driven surrogates.

This benchmark is an engineering proxy for the paper's L-shaped Poisson task:
it preserves the irregular-domain and singular-corner pressure without claiming
paper-score parity.
