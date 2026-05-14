# Requirements

- Python 3.11.
- Use PyTorch for real submitted solutions when available.
- `solution.py` must define class `MODEL`.
- `solution.py --mode=validate` must run without training.
- `solution.py --mode=train` must write `model.pkl`.
- `solution.py --mode=predict --input predict_input.npz --output predictions.npz`
  must write a `predictions` array for the provided `x_val` features.
- Training may read only `train_data.npz`.
- Predict mode receives only validation features through `predict_input.npz`.
- Do not read `val_data.npz`, evaluator-private artifacts, or benchmark data
  generator source from generated solution code.
- Do not require internet access.
