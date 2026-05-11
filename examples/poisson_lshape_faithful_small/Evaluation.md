# Evaluation

The trusted runner writes only validation coordinates to `predict_input.npz`.
`solution.py --mode=predict` writes `predictions.npz`; the evaluator reads
private labels and residual metadata without importing generated `solution.py`.
The score is a low-budget composite:

```text
solution_relative_l2 + 0.1 * boundary_relative_l2 + 0.01 * residual_relative_l2
```

Lower score is better. Validation data is fixed by `generate_data.py --seed 0`.
This benchmark can support low-budget scientific smoke evidence only after a
real LLM run, multi-seed repeats, and ablation review.
