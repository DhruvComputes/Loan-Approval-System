"""
=====================================================================
 MINOR PROJECT IN AI/ML : Loan Approval Prediction System
 File        : Project_ML_pipeline.py
 Description : Cleans the supplied CSV dataset, trains a Gaussian
               Naive Bayes classifier, evaluates the fitted pipeline,
               and saves preprocessing plus the model as 'loan_model.pkl'.

 Dataset (20 columns):
   Applicant_ID, Applicant_Income, Coapplicant_Income,
   Employment_Status, Age, Marital_Status, Dependents, Credit_Score,
   Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount,
   Loan_Term, Loan_Purpose, Property_Area, Education_Level, Gender,
   Employer_Category, Loan_Approved (target: Yes/No)

 The dataset contains many MISSING values (visible in the PDF), so
 the pipeline imputes them: median for numeric, mode for categorical.
=====================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
CSV_PATH   = "loan_approval_dataset.csv"
MODEL_PATH = "loan_model.pkl"
TARGET     = "Loan_Approved"

NUMERIC_FEATURES = [
    "Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
    "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
    "Collateral_Value", "Loan_Amount", "Loan_Term",
]
CATEGORICAL_FEATURES = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Education_Level", "Gender", "Employer_Category",
]

# Normalize category spellings that can differ between source data and the CSV.
CATEGORY_FIXES = {
    "Property_Area":     {"Semi-Urban": "Semiurban", "Semi Urban": "Semiurban"},
    "Employer_Category": {"Govt": "Government", "Gov": "Government"},
    "Employment_Status": {"Self Employed": "Self-employed", "Selfemployed": "Self-employed"},
}


def load_and_clean():
    """Load the dataset and perform the initial data-cleaning operations."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"'{CSV_PATH}' not found!\n"
            f"Create it first:  python pdf_to_csv.py\n"
            f"(or export the PDF table to Excel and save it as loan_data.csv)"
        )

    # Load the source dataset into a pandas DataFrame for inspection and cleaning.
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded '{CSV_PATH}' -> {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove Applicant_ID because it identifies a row and is not used as a predictive feature.
    if "Applicant_ID" in df.columns:
        df = df.drop(columns=["Applicant_ID"])

    missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {missing_cols}")

    # Convert the target labels into numeric classes: approved is 1 and not approved is 0.
    df[TARGET] = (df[TARGET].astype("string").str.strip().str.title()
                  .map({"Yes": 1, "No": 0, "Y": 1, "N": 0, "1": 1, "0": 0}))
    before = len(df)
    df = df.dropna(subset=[TARGET]).astype({TARGET: int})
    print(f"[INFO] Target cleaned: {before - len(df)} unusable rows dropped | "
          f"approval rate = {df[TARGET].mean():.2%}")

    # Convert numeric feature columns to numbers and turn invalid text into missing values.
    # Missing numeric values are retained here and filled by the median imputer in the pipeline.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical text by trimming whitespace, standardizing known spellings, and preserving missing values.
    for col in CATEGORICAL_FEATURES:
        s = df[col].astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
        if col in CATEGORY_FIXES:
            s = s.replace(CATEGORY_FIXES[col])
        df[col] = s.astype(object).where(s.notna(), np.nan)

    print("[INFO] Missing values the pipeline will impute (top 8):")
    print(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().sum()
          .sort_values(ascending=False).head(8).to_string())
    return df


def build_pipeline():
    """Build the preprocessing pipeline and Gaussian Naive Bayes classifier."""
    # Prepare numeric features by filling missing values with the median and standardizing their scales.
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    # Prepare categorical features by filling missing values with the mode and applying one-hot encoding.
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    # Apply the numeric and categorical preprocessing steps to their respective feature columns.
    preprocessor = ColumnTransformer([
        ("num", numeric,     NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    # Combine preprocessing and classification so the same transformations are used during training and prediction.
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   GaussianNB()),
    ])


def main():
    # Load the cleaned dataset and separate the input features from the target labels.
    df = load_and_clean()
    X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]

    # Split the data into training and testing sets while preserving the target-class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # Build the complete preprocessing and classification pipeline, then fit it using the training data.
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Generate predictions for the held-out test data to evaluate model performance.
    y_pred = model.predict(X_test)
    print("\n" + "=" * 55)
    print(" MODEL EVALUATION — Gaussian Naive Bayes")
    print("=" * 55)
    print(f" Accuracy : {accuracy_score(y_test, y_pred):.4f}\n")
    print(" Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Approved", "Approved"]))

    # Retrieve the transformed feature names to show the representation received by Gaussian Naive Bayes.
    encoded = model.named_steps["preprocessor"].get_feature_names_out()
    print(f"[INFO] GaussianNB receives {len(encoded)} encoded features "
          f"(11 numeric + one-hot expanded categoricals). Sample:")
    for name in list(encoded)[:12]:
        print(f"        - {name}")

    # Save the fitted preprocessing and classification pipeline for reuse by the Flask application.
    joblib.dump(model, MODEL_PATH)
    print(f"\n[SUCCESS] Trained pipeline saved to '{MODEL_PATH}'")
    print("[INFO] Start the web app with:  python app.py")


if __name__ == "__main__":
    main()