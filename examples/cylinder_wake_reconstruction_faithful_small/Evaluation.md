# Evaluation

The trusted runner writes only lagged sparse-sensor validation features to
`predict_input.npz`. `solution.py --mode=predict` writes `predictions.npz`; the
evaluator reads private vorticity labels and computes mean per-sample relative
L2 error without importing generated `solution.py`.

The official score is the mean per-sample relative L2 error. Diagnostics such as
mean absolute error and maximum absolute error are reported for audit only and
do not change the official score.

Lower score is better.
