# Evaluation

The trusted runner writes only validation features to `predict_input.npz`.
`solution.py --mode=predict` writes `predictions.npz`; the evaluator reads
private spatiotemporal labels and computes mean per-sample relative L2 error
without importing generated `solution.py`.

The official score is:

```text
mean_i ||u_pred_i - u_true_i||_2 / (||u_true_i||_2 + eps)
```

The evaluator also reports diagnostic early-time, late-time, spatial-gradient,
and temporal-difference relative L2 values. These diagnostics are not separate
leaderboard objectives.

Lower score is better.

This benchmark can support low-budget scientific smoke evidence only after a
real LLM run, multi-seed repeats, and ablation review.
