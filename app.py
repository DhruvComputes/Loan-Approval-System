import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "loan_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found!\nTrain the model first:\n    python Project_ML_pipeline.py"
    )
model = joblib.load(MODEL_PATH)

# Define validation limits using the exact feature names expected by the trained pipeline.
# Each numeric rule stores its display label, minimum value, and optional maximum value.
NUMERIC_RULES = {
    "Applicant_Income":   ("Applicant income", 1, None),
    "Coapplicant_Income": ("Coapplicant income", 0, None),
    "Age":                ("Age", 18, 80),
    "Dependents":         ("Dependents", 0, 20),
    "Credit_Score":       ("Credit score", 300, 900),
    "Existing_Loans":     ("Existing loans", 0, 20),
    "DTI_Ratio":          ("DTI ratio", 0.0, 1.0),
    "Savings":            ("Savings", 0, None),
    "Collateral_Value":   ("Collateral value", 0, None),
    "Loan_Amount":        ("Loan amount", 1, None),
}
INT_FIELDS  = {"Age", "Dependents", "Credit_Score", "Existing_Loans"}
LOAN_TERMS  = [12, 24, 36, 48, 60, 72, 84]  # Valid loan durations present in the dataset.
CATEGORICAL_RULES = {
    "Employment_Status": ["Salaried", "Self-employed", "Business", "Contract", "Unemployed"],
    "Marital_Status":    ["Married", "Single"],
    "Loan_Purpose":      ["Home", "Education", "Personal", "Business", "Car"],
    "Property_Area":     ["Urban", "Semiurban", "Rural"],
    "Education_Level":   ["Graduate", "Not Graduate"],
    "Gender":            ["Male", "Female"],
    "Employer_Category": ["Government", "Private", "MNC", "Business", "Unemployed"],
}


def parse_and_validate(form):
    """
    Converts raw HTML form fields into a clean, model-ready dict whose
    keys exactly match the training column names. The pipeline's
    ColumnTransformer selects columns BY NAME, so this dict is all the
    model needs. Returns (features | None, errors).
    """
    errors, features = {}, {}

    # Validate and convert all numeric form fields before creating the model input.
    for field, (label, lo, hi) in NUMERIC_RULES.items():
        raw = (form.get(field) or "").strip()
        if raw == "":
            errors[field] = f"{label} is required."
            continue
        try:
            val = float(raw)
        except ValueError:
            errors[field] = f"Enter a valid number for {label.lower()}."
            continue
        if lo is not None and val < lo:
            errors[field] = f"{label} must be at least {lo}."
        elif hi is not None and val > hi:
            errors[field] = f"{label} must be at most {hi}."
        else:
            features[field] = int(val) if field in INT_FIELDS else val

    # Validate the selected loan duration against the supported dataset values.
    try:
        term = int((form.get("Loan_Term") or "").strip())
        if term not in LOAN_TERMS:
            raise ValueError
        features["Loan_Term"] = term
    except ValueError:
        errors["Loan_Term"] = "Please choose a valid loan term."

    # Check each categorical value against the options used during model training.
    for field, options in CATEGORICAL_RULES.items():
        val = (form.get(field) or "").strip()
        if val not in options:
            errors[field] = "Please select a valid option."
        else:
            features[field] = val

    return (features if not errors else None), errors


# Render the HTML form when the user opens the application homepage.
@app.route("/")
def home():
    return render_template("index.html")


# Validate submitted form data and return the model prediction as JSON.
@app.route("/predict", methods=["POST"])
def predict():
    features, errors = parse_and_validate(request.form)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    input_df = pd.DataFrame([features])  # Create one applicant record; the pipeline handles imputation and encoding.

    try:
        pred = int(model.predict(input_df)[0])
    except Exception as exc:
        return jsonify({"success": False, "message": f"Prediction failed: {exc}"}), 500

    return jsonify({
        "success":    True,
        "prediction": pred,
        "result":     "Congrats! Loan Approved" if pred == 1 else "Sorry! Loan Not Approved",
    })


# Start the Flask development server when this file is executed directly.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)