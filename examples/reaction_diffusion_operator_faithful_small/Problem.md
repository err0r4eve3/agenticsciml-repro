# Faithful-Small Reaction-Diffusion Operator Benchmark

Learn a multiple-input operator that maps diffusion coefficient samples, source
term samples, and initial-condition samples to a full spatiotemporal
reaction-diffusion response on a shared local grid.

This is a low-budget faithful-small benchmark for the paper's multiple-input
reaction-diffusion operator learning task. It preserves the multi-input
function-to-field shape, fixed grid, and relative-L2 scoring, but uses a small
deterministic local generator and does not claim paper-score reproduction.
