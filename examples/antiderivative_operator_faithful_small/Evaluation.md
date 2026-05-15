# Evaluation

The trusted runner writes only validation input-function features to
`predict_input.npz`. `solution.py --mode=predict` writes `predictions.npz`;
the evaluator reads private antiderivative labels and computes the mean
per-sample relative L2 error without importing generated `solution.py`.

Lower score is better.

This benchmark can support low-budget scientific smoke evidence only after a
real LLM run, multi-seed repeats, and ablation review.
