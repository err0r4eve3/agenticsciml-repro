# Evaluation

The evaluator loads `model.pkl`, calls `MODEL.predict(x_val)`, and computes
relative L2 error for the predicted final field.

Lower score is better.
