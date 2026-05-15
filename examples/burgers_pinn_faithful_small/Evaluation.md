# Evaluation

The trusted runner writes only validation `(x, t)` features to
`predict_input.npz`. `solution.py --mode=predict` writes `predictions.npz`;
the evaluator reads private labels and computes a low-budget composite without
importing generated `solution.py`:

```text
solution_relative_l2 + 0.1 * initial_relative_l2 + 0.1 * boundary_relative_l2
```

Lower score is better.

This benchmark can support low-budget scientific smoke evidence only after a
real LLM run, multi-seed repeats, and ablation review.
