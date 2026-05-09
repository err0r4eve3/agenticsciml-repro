# Evaluation

The evaluator loads `val_data.npz`, restores `model.pkl`, calls
`MODEL.predict(x_val)`, and computes validation mean squared error.

Lower `validation_mse` is better.
