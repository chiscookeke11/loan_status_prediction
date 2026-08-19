# Loan Status Prediction

A machine learning project that predicts whether a loan application is likely to be **approved** or **not approved** from applicant, loan, credit-history, and property-area information.

The repository contains a Jupyter notebook that walks through the complete workflow: loading the training data, cleaning missing values, encoding categorical features, training a Support Vector Machine classifier, evaluating accuracy, saving the trained model, and running a sample prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Artifacts](#model-artifacts)
- [Getting Started](#getting-started)
- [Running the Notebook](#running-the-notebook)
- [Using the Saved Model](#using-the-saved-model)
- [Input Schema](#input-schema)
- [Important Notes and Limitations](#important-notes-and-limitations)
- [Future Improvements](#future-improvements)

## Project Overview

Loan approval decisions are commonly influenced by demographic details, income, loan amount, repayment term, credit history, and property location. This project uses these fields to train a binary classification model that predicts the target variable `Loan_Status`:

- `Y` / `1`: loan is approved
- `N` / `0`: loan is not approved

The model is trained in `loan_status_prediction.ipynb` using a linear Support Vector Machine (`sklearn.svm.SVC`). The notebook also includes exploratory data analysis and simple visualizations that compare loan status against categorical features such as education and marital status.

## Repository Structure

```text
loan_status_prediction/
├── README.md
├── loan_status_prediction.ipynb
├── loan_status_model.pkl
├── loan_status_model_columns.json
└── sample_data/
    └── train_u6lujuX_CVtuZ9i (1).csv
```

| Path | Description |
| --- | --- |
| `loan_status_prediction.ipynb` | Main notebook containing data preparation, model training, evaluation, persistence, and sample inference code. |
| `sample_data/train_u6lujuX_CVtuZ9i (1).csv` | Training dataset used by the notebook. |
| `loan_status_model.pkl` | Serialized trained Support Vector Machine model. |
| `loan_status_model_columns.json` | Serialized feature-column order expected by the trained model. Despite the `.json` extension, this file is saved with `joblib.dump`, so load it with `joblib.load`. |
| `README.md` | Project documentation. |

## Dataset

The included dataset contains **614 loan applications** with **13 columns**. The prediction target is `Loan_Status`.

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `Loan_ID` | Identifier | Unique loan application ID. Dropped before model training. |
| `Gender` | Categorical | Applicant gender: `Male` or `Female`. |
| `Married` | Categorical | Whether the applicant is married: `Yes` or `No`. |
| `Dependents` | Categorical / numeric-like | Number of dependents: `0`, `1`, `2`, or `3+`. |
| `Education` | Categorical | Applicant education level: `Graduate` or `Not Graduate`. |
| `Self_Employed` | Categorical | Whether the applicant is self-employed: `Yes` or `No`. |
| `ApplicantIncome` | Numeric | Applicant income. |
| `CoapplicantIncome` | Numeric | Co-applicant income. |
| `LoanAmount` | Numeric | Requested loan amount. |
| `Loan_Amount_Term` | Numeric | Loan repayment term. |
| `Credit_History` | Numeric / binary | Whether the applicant has a qualifying credit history, commonly `1.0` or `0.0`. |
| `Property_Area` | Categorical | Property location category: `Rural`, `Semiurban`, or `Urban`. |
| `Loan_Status` | Target | Approval outcome: `Y` or `N`. |

### Target Distribution

The raw dataset contains:

- `Y`: 422 approved applications
- `N`: 192 rejected applications

## Machine Learning Workflow

The notebook follows these steps:

1. **Import dependencies**
   - Uses `pandas`, `seaborn`, and `scikit-learn`.
2. **Load the dataset**
   - Reads `sample_data/train_u6lujuX_CVtuZ9i (1).csv` into a pandas DataFrame.
3. **Explore the data**
   - Checks shape, summary statistics, missing values, and sample rows.
4. **Handle missing values**
   - Drops rows containing missing values with `dropna(inplace=True)`.
5. **Encode the target**
   - Maps `Loan_Status` values from `N`/`Y` to `0`/`1`.
6. **Prepare categorical features**
   - Converts categorical values to numeric representations, for example:
     - `Gender`: `Female = 0`, `Male = 1`
     - `Married`: `No = 0`, `Yes = 1`
     - `Education`: `Not Graduate = 0`, `Graduate = 1`
     - `Self_Employed`: `No = 0`, `Yes = 1`
     - `Property_Area`: `Rural = 0`, `Semiurban = 1`, `Urban = 2`
     - `Dependents`: replaces `3+` with `4`
7. **Split data**
   - Splits features and labels into training and test sets using stratified sampling.
8. **Train model**
   - Trains a linear Support Vector Machine classifier.
9. **Evaluate model**
   - Calculates accuracy on the training and test data.
10. **Persist model artifacts**
    - Saves the trained classifier and expected feature-column order.
11. **Run sample prediction**
    - Defines `predict_loan_status` for single-applicant prediction.

## Model Artifacts

The repository includes two generated artifacts:

- `loan_status_model.pkl`: the trained `sklearn.svm.SVC` classifier.
- `loan_status_model_columns.json`: the feature order used during training.

> **Note:** `loan_status_model_columns.json` is a binary joblib artifact, not plain JSON. Load it with `joblib.load("loan_status_model_columns.json")`.

The feature-column order matters because the model expects inference data to be provided in the same order used during training.

## Getting Started

### Prerequisites

Use Python 3.9+ and install the required packages:

```bash
pip install pandas seaborn scikit-learn joblib notebook
```

If you prefer an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas seaborn scikit-learn joblib notebook
```

## Running the Notebook

Start Jupyter Notebook from the repository root:

```bash
jupyter notebook loan_status_prediction.ipynb
```

Then run the notebook cells from top to bottom. The notebook will:

1. Load the CSV from `sample_data/`.
2. Clean and transform the data.
3. Train the SVM model.
4. Print training and test accuracy.
5. Save the model artifacts.
6. Demonstrate a sample prediction.

## Using the Saved Model

The notebook includes a `predict_loan_status` helper. The same approach can be used in a standalone script after installing the dependencies.

```python
import joblib
import pandas as pd

model = joblib.load("loan_status_model.pkl")
columns = joblib.load("loan_status_model_columns.json")

applicant = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 4583,
    "CoapplicantIncome": 1508.0,
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban",
}

gender_map = {"Female": 0, "Male": 1}
married_map = {"No": 0, "Yes": 1}
education_map = {"Not Graduate": 0, "Graduate": 1}
employed_map = {"No": 0, "Yes": 1}
property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 4}

row = {
    "Gender": gender_map[applicant["Gender"]],
    "Married": married_map[applicant["Married"]],
    "Dependents": dependents_map.get(str(applicant["Dependents"]), applicant["Dependents"]),
    "Education": education_map[applicant["Education"]],
    "Self_Employed": employed_map[applicant["Self_Employed"]],
    "ApplicantIncome": applicant["ApplicantIncome"],
    "CoapplicantIncome": applicant["CoapplicantIncome"],
    "LoanAmount": applicant["LoanAmount"],
    "Loan_Amount_Term": applicant["Loan_Amount_Term"],
    "Credit_History": applicant["Credit_History"],
    "Property_Area": property_map[applicant["Property_Area"]],
}

input_df = pd.DataFrame([row])[columns]
prediction = model.predict(input_df)[0]

print("Approved" if prediction == 1 else "Not Approved")
```

## Input Schema

When using the trained model, provide all required features except `Loan_ID` and `Loan_Status`.

| Field | Accepted Example Values |
| --- | --- |
| `Gender` | `Male`, `Female` |
| `Married` | `Yes`, `No` |
| `Dependents` | `0`, `1`, `2`, `3+` |
| `Education` | `Graduate`, `Not Graduate` |
| `Self_Employed` | `Yes`, `No` |
| `ApplicantIncome` | Numeric value, for example `4583` |
| `CoapplicantIncome` | Numeric value, for example `1508.0` |
| `LoanAmount` | Numeric value, for example `128.0` |
| `Loan_Amount_Term` | Numeric value, for example `360.0` |
| `Credit_History` | `1.0` or `0.0` |
| `Property_Area` | `Rural`, `Semiurban`, `Urban` |

## Important Notes and Limitations

- This project is intended for learning and demonstration purposes.
- The model is trained on a small sample dataset and should not be used as-is for real lending decisions.
- The notebook drops rows with missing values, which is simple but may discard useful information.
- Categorical values are manually label-encoded. For production systems, use a reproducible preprocessing pipeline such as `sklearn.pipeline.Pipeline` and `ColumnTransformer`.
- Fairness, explainability, regulatory compliance, data privacy, and model monitoring are essential for real-world credit or lending applications and are outside the scope of this notebook.
- The saved column artifact has a `.json` extension but is joblib-serialized binary data. Rename or regenerate it as true JSON if human-readable metadata is required.

## Future Improvements

Potential enhancements include:

- Add a `requirements.txt` or `pyproject.toml` for reproducible dependency installation.
- Replace manual preprocessing with a scikit-learn pipeline.
- Add imputation instead of dropping all rows with missing values.
- Compare multiple algorithms such as logistic regression, random forest, gradient boosting, and calibrated SVMs.
- Add cross-validation and richer metrics such as precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Save model metadata, training metrics, and preprocessing configuration in a clearly versioned format.
- Add automated tests for the prediction helper.
- Package the model behind a simple API or web application for easier inference.

## License

No license file is currently included. Add a license before distributing or reusing this project publicly.
