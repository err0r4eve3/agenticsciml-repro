# Function Approximation Guidelines

Solutions should focus on validation MSE. Useful strategies include feature
expansion, piecewise modeling, weighted losses around discontinuities, stable
optimization, and regularization against overfitting.

The evaluator is fixed. Do not modify `evaluate.py` inside a solution workspace.
`solution.py` must support `--mode=validate`, `--mode=train`, and
`--mode=predict --input predict_input.npz --output predictions.npz`. Predict
mode receives only `x_val` and must write `predictions.npz` with a
`predictions` array.
