# Evaluation

The trusted runner writes only validation coordinates to `predict_input.npz`.
`solution.py --mode=predict` writes `predictions.npz`; the evaluator reads
private `u_val` labels and computes relative L2 error without importing
generated `solution.py`.

Lower score is better. Validation data is fixed by `generate_data.py --seed 0`.
This benchmark can support low-budget scientific smoke evidence only after a
real LLM run, multi-seed repeats, and ablation review.
