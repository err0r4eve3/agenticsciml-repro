# Evaluation

The trusted runner writes only validation features to `predict_input.npz`.
`solution.py --mode=predict` writes `predictions.npz`; the evaluator reads
private `u_val` labels and computes validation mean squared error without
importing generated `solution.py`.

Lower `validation_mse` is better.
