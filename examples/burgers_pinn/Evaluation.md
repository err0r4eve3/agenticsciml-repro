# Evaluation

The evaluator loads `model.pkl`, calls `MODEL.predict(x_val)`, and computes
relative L2 error over validation `(x, t)` points.

Lower score is better.
