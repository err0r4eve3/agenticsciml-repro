# Evaluation

The evaluator loads `model.pkl`, calls `MODEL.predict(x_val)`, and computes
relative L2 error against `u_val`.

Lower score is better. Validation data is fixed by `generate_data.py --seed 0`
inside each isolated solution workspace.
