# Faithful-Small Antiderivative Operator Learning Benchmark

Learn an operator that maps a discretized input function `a(s)` to its
antiderivative on the same 100-point grid. The model must generalize across
unseen input functions rather than fit a single curve.

This is a low-budget faithful-small benchmark for the paper's antiderivative
operator learning task. It preserves the function-to-function operator shape
and relative-L2 scoring, but it is still smaller than the paper-scale
experiment and does not claim reported-score parity.
