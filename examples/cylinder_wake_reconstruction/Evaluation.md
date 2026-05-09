# Evaluation

The evaluator loads `model.pkl`, calls `MODEL.predict(x_val)`, and computes
relative L2 error over flattened vorticity fields.

Lower score is better.
