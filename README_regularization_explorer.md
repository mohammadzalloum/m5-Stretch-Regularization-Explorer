# Regularization Explorer

This project trains `LogisticRegression` on the telecom churn dataset using 20 logarithmically spaced values of `C` from `0.001` to `100`, then visualizes how each feature coefficient changes under **L1** and **L2** regularization.

## Files
- `regularization_explorer.py` - main script
- `telecom_churn(1).csv` - dataset
- `regularization_paths.png` - output plot
- `interpretation_paragraph.txt` - 1-paragraph interpretation

## How to run
```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scikit-learn
python regularization_explorer.py
```

## What the plot shows
- **L1** pushes weaker coefficients to exactly zero, producing a sparse model.
- **L2** shrinks coefficients smoothly and usually keeps them non-zero.
- In this dataset, contract-related features remain among the strongest signals across the path.
