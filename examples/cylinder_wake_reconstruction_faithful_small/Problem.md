# Faithful-Small Cylinder Wake Reconstruction Benchmark

Reconstruct a 2D vorticity field from sparse temporal sensor history. Inputs are
lagged readings from eight fixed wake sensors plus the current normalized time;
outputs are flattened `24 x 24` vorticity snapshots.

This is a low-budget faithful-small benchmark for the paper's cylinder wake
sparse-reconstruction task. It is inspired by SHRED-style sparse temporal
sensing and shallow decoding, but it uses a deterministic synthetic
cylinder-wake-like generator rather than the original SHRED-ROM datasets,
training budget, or reported scores.
