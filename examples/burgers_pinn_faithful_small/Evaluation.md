# Evaluation

The trusted runner writes only validation `(x, t)` features to
`predict_input.npz`. `solution.py --mode=predict` writes `predictions.npz`;
the evaluator reads private labels and computes a low-budget composite without
importing generated `solution.py`:

```text
solution_relative_l2
+ 0.1 * initial_relative_l2
+ 0.1 * boundary_scaled_rmse
+ 0.01 * residual_relative_l2
```

The exact periodic solution is zero at both sampled spatial endpoints, so the
boundary component uses boundary RMSE normalized by the RMS of the private
dense solution. It deliberately does not divide by a near-zero boundary-label
norm.

The residual component evaluates `u_t + u*u_x - 0.04*u_xx` using centered
finite differences on the private `83 x 40` space-time validation grid. It is
normalized by the sum of the private reference term norms, so it remains
dimensionless and finite without dividing by the near-zero exact residual.
Only predicted values at validation coordinates are used; generated code is
never imported by the evaluator.

Lower score is better.

This benchmark can support low-budget scientific smoke evidence only after a
real LLM run, multi-seed repeats, and ablation review.
