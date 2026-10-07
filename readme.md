# Loan Approval Prediction

Predicts whether a loan application is approved or rejected using **Decision Tree** and **Random Forest** classifiers, trained on the Kaggle *Loan Prediction Problem Dataset* (614 applicant records).

## Features

- Data cleaning: missing values filled with mode (text) / median (numeric)
- Label encoding of categorical features
- 80/20 stratified train-test split
- Two models compared: Decision Tree (depth 4) and Random Forest (500 trees, depth 5)
- Accuracy, classification reports, confusion matrices, feature importance
- Interactive web app with a live prediction form

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | 84.55% |
| Random Forest | 85.37% |

Top features (Random Forest importance): **Credit_History** (49.0%), **ApplicantIncome** (12.6%), **LoanAmount** (11.7%).

## Files

| File | Description |
|---|---|
| `app.py` | Interactive Streamlit web app (sidebar form, live prediction, tabs for results/visuals/data) |
| `mini.py` | Standalone script: trains models, prints metrics, saves PNG charts |
| `mini-backup.py` | Backup copy of the standalone script |
| `loan_data.csv` | Dataset (614 rows x 13 columns) |
| `*.png` | Generated charts: accuracy comparison, decision tree, confusion matrices, feature importance |

## Requirements

- Python 3.10+
- pandas, matplotlib, scikit-learn, streamlit

```bash
pip install pandas matplotlib scikit-learn streamlit
```

## Usage

**Interactive web app (recommended):**

```bash
streamlit run app.py
```

Then open http://localhost:8501. Use the sidebar to enter applicant details — the prediction (APPROVED/REJECTED with probability) updates live. Tabs show model performance, visualisations, and the dataset.

**Standalone script:**

```bash
python mini.py
```

Prints metrics to the console and saves `accuracy_comparison.png`, `decision_tree.png`, `confusion_matrices.png`, `feature_importance.png` in the current folder.

## Dataset

Kaggle *Loan Prediction Problem Dataset* (Analytics Vidhya). Columns:

`Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status` (target: Y = approved, N = rejected).

`Loan_ID` is dropped before training since it is only an identifier.

## Notes

- The app caches data loading and model training, so interactions are instant after first load.
- Predictions use the model trained on the 80% training split; for production, retrain on the full dataset.
