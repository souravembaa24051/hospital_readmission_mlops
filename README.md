# Hospital Readmission Prediction – End-to-End MLOps Project

## Project Overview

Hospital readmissions within 30 days are a major challenge for healthcare providers, impacting patient outcomes and increasing healthcare costs. This project develops a machine learning solution to predict whether a diabetic patient is likely to be readmitted within 30 days of discharge.

The project demonstrates a complete MLOps workflow, including data preprocessing, model development, experiment tracking, explainability, API deployment, and containerization.

---

## Business Problem

Early identification of patients at high risk of readmission enables hospitals to:

* Improve patient care and outcomes
* Reduce avoidable readmissions
* Optimize resource allocation
* Lower healthcare costs
* Support clinical decision-making

The objective is to predict whether a patient will be readmitted within 30 days using historical hospital encounter data.

---

## Dataset

**Source:** Diabetes 130-US Hospitals Dataset

* 101,766 patient encounters
* 50 original features
* Multiple encounters per patient
* Readmission outcomes:

  * `<30` : Readmitted within 30 days
  * `>30` : Readmitted after 30 days
  * `NO` : No readmission

### Target Variable

Binary classification target:

* 1 = Readmitted within 30 days (`<30`)
* 0 = Not readmitted within 30 days

Class distribution:

| Class |  Count |
| ----- | -----: |
| 0     | 90,409 |
| 1     | 11,357 |

The dataset is highly imbalanced, with approximately 11% positive cases.

---

## Project Architecture

```text
Data Ingestion
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Patient-Level Train/Test Split
      ↓
Model Training
      ↓
MLflow Tracking
      ↓
SHAP Explainability
      ↓
FastAPI Deployment
      ↓
Docker Containerization
```

---

## Data Preprocessing

### Missing Value Handling

The dataset contains missing values represented by `?`.

Actions performed:

* Replaced `?` with null values
* Dropped high-missing columns:

  * weight
  * payer_code
  * medical_specialty
* Applied median imputation for numerical features
* Applied constant imputation for categorical features

### Data Leakage Prevention

The dataset contains multiple encounters for the same patient.

To prevent data leakage:

* Used `GroupShuffleSplit`
* Split performed using `patient_nbr`
* Ensured the same patient never appears in both train and test sets

---

## Feature Engineering

Features used include:

* Demographics
* Admission details
* Laboratory procedures
* Medication information
* Diagnosis codes
* Hospital utilization metrics

Categorical features were encoded using One-Hot Encoding.

---

## Models Implemented

### Logistic Regression Baseline

Configuration:

* Class Weight = Balanced
* Max Iterations = 1000

Results:

| Metric  |  Value |
| ------- | -----: |
| PR-AUC  | 0.1825 |
| ROC-AUC | 0.6333 |

---

### XGBoost Model

Configuration:

* n_estimators = 300
* max_depth = 5
* learning_rate = 0.05
* subsample = 0.8
* colsample_bytree = 0.8
* scale_pos_weight = 8

Results:

| Metric  |  Value |
| ------- | -----: |
| PR-AUC  | 0.2100 |
| ROC-AUC | 0.6728 |

### Improvement over Baseline

| Metric  | Logistic Regression | XGBoost |
| ------- | ------------------: | ------: |
| PR-AUC  |              0.1825 |  0.2100 |
| ROC-AUC |              0.6333 |  0.6728 |

The XGBoost model achieved superior performance and was selected as the final production model.

---

## Experiment Tracking with MLflow

MLflow was used for:

* Experiment tracking
* Parameter logging
* Metric tracking
* Model artifact storage

Tracked Metrics:

* PR-AUC
* ROC-AUC

Tracked Parameters:

* Model type
* Number of estimators
* Tree depth
* Learning rate

---

## Model Explainability

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

Generated Artifacts:

* SHAP Summary Plot
* Global Feature Importance Analysis

Benefits:

* Improved model transparency
* Better understanding of prediction drivers
* Support for stakeholder communication

---

## API Deployment

The trained model is deployed using FastAPI.

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "running"
}
```

### Prediction Endpoint

```http
POST /predict
```

Response:

```json
{
  "risk_score": 0.4901,
  "prediction": 0
}
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Docker Deployment

Build Image:

```bash
docker build -t hospital-readmission .
```

Run Container:

```bash
docker run -p 8000:8000 hospital-readmission
```

API Access:

```text
http://127.0.0.1:8000/docs
```

---

## Project Structure

```text
hospital_readmission_mlops/
│
├── api/
│   └── app.py
│
├── data/
│   └── raw/
│
├── src/
│   ├── train.py
│   ├── train_xgboost.py
│   ├── train_xgboost_mlflow.py
│   └── shap_explain.py
│
├── models/
│
├── reports/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* MLflow
* SHAP
* FastAPI
* Docker
* Joblib

---

## Future Improvements

* Hyperparameter optimization
* Feature store integration
* CI/CD pipeline implementation
* Model monitoring with Prometheus
* Drift detection using Evidently AI
* Cloud deployment (AWS/Azure/GCP)

---

## Conclusion

This project demonstrates an end-to-end machine learning and MLOps workflow for predicting hospital readmissions. The final XGBoost model achieved improved predictive performance over the baseline model while maintaining explainability and deployability through SHAP, MLflow, FastAPI, and Docker.
