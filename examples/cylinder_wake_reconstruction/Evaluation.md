# Evaluation

The trusted runner writes only sparse-sensor validation features to
`predict_input.npz`. `solution.py --mode=predict` writes `predictions.npz`;
the evaluator reads private vorticity labels and computes relative L2 error
without importing generated `solution.py`.

Lower score is better.
