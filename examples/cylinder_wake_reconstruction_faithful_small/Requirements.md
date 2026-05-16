# Requirements

- Python 3.11.
- `solution.py` must define class `MODEL`.
- `solution.py --mode=validate` must run without training.
- `solution.py --mode=train` must save `model.pkl`.
- `solution.py --mode=predict --input predict_input.npz --output predictions.npz`
  must write a `predictions` array for the provided `x_val` features.
- Do not modify evaluator or validation artifacts.
- Do not assume access to future sensor frames during prediction.
- Do not claim SHRED-ROM, PySHRED, real-fluid, or paper-score reproduction.
- No network access in tests.
