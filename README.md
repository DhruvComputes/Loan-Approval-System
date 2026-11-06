# Loan Approval Prediction System

## Project Overview

The Loan Approval Prediction System is a minor project in Artificial Intelligence and Machine Learning. It uses a **Gaussian Naive Bayes classifier** to predict whether a loan application is likely to be approved or not approved based on applicant, financial, employment, loan, and property information.

The project includes a Flask web application, an HTML form, a trained scikit-learn pipeline, and a CSV dataset.

## Objectives

- Prepare applicant data for machine learning.
- Train and evaluate a Gaussian Naive Bayes classification model.
- Save the fitted preprocessing and classification pipeline.
- Provide a web form for submitting applicant information.
- Return a loan approval prediction through the Flask application.

## Technology Stack

- Python
- Flask
- pandas and NumPy
- scikit-learn
- joblib
- HTML, CSS, JavaScript
- Bootstrap 5.3.3 and Bootstrap Icons

The Python package requirements are listed in `requirements.txt`.

## Project Structure

```text
Loan-Approval-System/
│
├── app.py                       # Flask application and prediction endpoint
├── Project_ML_pipeline.py       # Data preparation, model training, and evaluation
├── templates/
│   └── index.html                # Applicant form and prediction-result interface
├── 2) loan approval data.csv    # Dataset used to train the model
├── loan_model.pkl               # Saved fitted preprocessing and model pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

**Flask template note:** `app.py` calls `render_template("index.html")`, so the HTML file is included inside the `templates` directory alongside `app.py`.

## Dataset Description

The supplied CSV contains **1,000 records and 20 columns**. The columns include:

- Applicant information: `Applicant_ID`, `Age`, `Gender`, `Marital_Status`, `Dependents`, and `Education_Level`.
- Employment information: `Employment_Status` and `Employer_Category`.
- Financial information: `Applicant_Income`, `Coapplicant_Income`, `Credit_Score`, `Existing_Loans`, `DTI_Ratio`, and `Savings`.
- Loan and property information: `Collateral_Value`, `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, and `Property_Area`.
- Target: `Loan_Approved`, containing the approval outcome.

`Applicant_ID` is removed before model training because it is an identifier rather than a predictive feature.

## Machine Learning Workflow

The training workflow is implemented in `Project_ML_pipeline.py`.

1. **Load data:** Read the CSV dataset into a pandas DataFrame.
2. **Remove identifier:** Drop `Applicant_ID` when it is present.
3. **Validate columns:** Check that all expected numeric, categorical, and target columns exist.
4. **Clean target labels:** Map approval labels such as `Yes`/`No` to `1`/`0` and remove rows with unusable target values.
5. **Clean numeric features:** Convert numeric columns to numeric data types; invalid values become missing values.
6. **Clean categorical features:** Trim whitespace, normalize selected category spellings, and preserve missing values.
7. **Split data:** Divide the data into training and testing subsets using an 80:20 split, `random_state=42`, and stratification.
8. **Preprocess numeric features:** Fill missing values using the median and standardize the values with `StandardScaler`.
9. **Preprocess categorical features:** Fill missing values using the most frequent category and apply `OneHotEncoder(handle_unknown="ignore")`.
10. **Train classifier:** Fit a `GaussianNB` classifier using the preprocessed training data.
11. **Evaluate model:** Print accuracy, a confusion matrix, and a classification report for the test data.
12. **Save pipeline:** Store the fitted preprocessing and classifier pipeline in `loan_model.pkl`.

The preprocessing operations are part of the scikit-learn pipeline. This allows the same fitted transformations to be applied when the Flask application receives a new applicant record.

## Web Application

`app.py` provides two routes:

- `GET /` — renders the applicant form.
- `POST /predict` — validates the submitted fields, creates a one-row pandas DataFrame, obtains a prediction from the saved pipeline, and returns a JSON response.

The form is organized into applicant, financial, and loan/property sections. Client-side JavaScript sends the form data asynchronously and displays the returned approval decision.

The prediction output is encoded as:

- `1` — Loan Approved
- `0` — Loan Not Approved

## Setup and Installation

### 1. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Check the project files

Ensure that the following files are in the project directory:

- `2) loan approval data.csv`
- `Project_ML_pipeline.py`
- `app.py`
- `requirements.txt`

Also ensure that `index.html` is inside the `templates` folder for Flask.

### 4. Train the model

Run:

```bash
python Project_ML_pipeline.py
```

The script trains and evaluates the model, then saves the fitted pipeline as `loan_model.pkl`.

### 5. Start the web application

Run:

```bash
python app.py
```

Open the local address shown by Flask. The current application configuration uses `http://127.0.0.1:5000`.

If a compatible `loan_model.pkl` is already present, model training can be skipped and the web application can load that saved file directly.

## Evaluation

When `Project_ML_pipeline.py` runs, it reports:

- Accuracy
- Confusion matrix
- Classification report, including precision, recall, and F1-score

The README does not state a fixed performance score because the actual evaluation output depends on the run and environment. Use the metrics printed by the training script when reporting experimental results.

## Important Implementation Notes

- Keep the feature names and categorical values consistent between the HTML form, Flask validation rules, dataset, and training pipeline.
- The model file `loan_model.pkl` must be available in the working directory when `app.py` starts.
- The current Flask configuration uses `debug=True` and binds to `127.0.0.1`; it is intended for local development, not direct production deployment.
- The supplied `requirements.txt` also lists `pdfplumber`; the current training and web application scripts do not import it directly.

## Future Scope

Possible future improvements, subject to project requirements and approval, could include model comparison, additional evaluation techniques, deployment configuration, and expanded user-facing explanations. These are suggestions only and are not implemented as part of the current project.

## Project Files

The source code, dataset, saved model, and dependency list are included with this project. Review the setup instructions above before running the application.