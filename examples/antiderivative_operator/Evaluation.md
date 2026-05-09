# Evaluation

The evaluator loads `model.pkl`, calls `MODEL.predict(x_val)`, and compares the
predicted antiderivative grid values with `u_val` using relative L2 error.

Lower score is better.
