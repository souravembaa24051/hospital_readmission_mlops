# Hospital Readmission Prediction - End-to-End MLOps Project

## Overview

This project develops a machine learning solution to predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge. The solution demonstrates a complete Machine Learning Operations (MLOps) workflow, including data preprocessing, model development, experiment tracking, explainability, API deployment, and containerization.

The project uses the Diabetes 130-US Hospitals dataset and implements both baseline and advanced machine learning models to identify patients at risk of readmission.

---

## Business Problem

Hospital readmissions are a significant challenge for healthcare systems because they:

* Increase healthcare costs
* Reduce operational efficiency
* Indicate potential gaps in patient care
* Impact patient outcomes

Predicting readmissions allows hospitals to proactively identify high-risk patients and implement intervention strategies before discharge.

---

## Project Objectives

* Predict whether a patient will be readmitted within 30 days
* Prevent data leakage through patient-level train/test splitting
* Compare baseline and advanced machine learning models
* Track experiments using MLflow
* Explain model predictions using SHAP
* Deploy the model through a FastAPI service
* Containerize the application using Docker

---

## Dataset

### Source

Diabetes 130-US Hospitals Dataset

### Dataset Characteristics

| Metric                                |   Value |
| ------------------------------------- | ------: |
| Total Records                         | 101,766 |
| Original Features                     |      50 |
| Positive Class (<30 Days Readmission) |  11,357 |
| Negative Class                        |  90,409 |

### Target Variable

The original target column is:

```text
readmitted
```

Converted into:

| Value | Meaning                       |
| ----- | ----------------------------- |
| 1     | Readmitted within 30 days     |
| 0     | Not readmitted within 30 days |

---

## Data Processing

### Missing Value Handling

The dataset contains missing values represented by:

```text
?
```

These were converted to:

```python
np.nan
```

### High Missing Columns Removed

The following columns were removed due to excessive missing values:

* weight
* payer_code
* medical_specialty

### Data Leakage Prevention

Multiple encounters may belong to the same patient.

To prevent leakage:

```python
GroupShuffleSplit(
    groups=df["patient_nbr"]
)
```

was used to ensure that the same patient never appears in both training and testing datasets.

---

## Exploratory Data Analysis (EDA)

The following visualizations were generated:

### 1. Target Distribution

* Demonstrates severe class imbalance
* Approximately 11% positive cases

### 2. Age vs Readmission

* Analyzes readmission patterns across age groups

### 3. Time in Hospital Distribution

* Shows patient stay duration distribution

### 4. Gender vs Readmission

* Compares readmission patterns by gender

### 5. Correlation Heatmap

* Examines relationships among numerical features

### 6. Top Diagnoses

* Displays the most frequent primary diagnosis codes

Generated files:

```text
reports/eda/
├── target_distribution.png
├── age_vs_readmission.png
├── time_in_hospital_distribution.png
├── gender_vs_readmission.png
├── correlation_heatmap.png
└── top_diagnoses.png
```

---

## Machine Learning Models

### Logistic Regression (Baseline)

Configuration:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

Results:

| Metric  |  Value |
| ------- | -----: |
| PR-AUC  | 0.1825 |
| ROC-AUC | 0.6333 |

---

### XGBoost (Final Model)

Configuration:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=8
)
```

Results:

| Metric  |  Value |
| ------- | -----: |
| PR-AUC  | 0.2100 |
| ROC-AUC | 0.6728 |

### Performance Improvement

| Metric  | Logistic Regression | XGBoost |
| ------- | ------------------: | ------: |
| PR-AUC  |              0.1825 |  0.2100 |
| ROC-AUC |              0.6333 |  0.6728 |

XGBoost was selected as the production model.

---

## Experiment Tracking with MLflow

MLflow was used to:

* Track experiments
* Store parameters
* Log evaluation metrics
* Save model artifacts

Tracked metrics include:

* PR-AUC
* ROC-AUC

Experiment Name:

```text
Hospital_Readmission
```

---

## Model Explainability

SHAP (SHapley Additive exPlanations) was used to explain model behavior.

Generated Artifact:

```text
reports/shap_summary.png
```

Benefits:

* Global feature importance analysis
* Transparent model interpretation
* Better understanding of prediction drivers

---

## API Deployment

The trained XGBoost model is deployed using FastAPI.

### Health Check Endpoint

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

Sample Response:

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

### Build Image

```bash
docker build -t hospital-readmission .
```

### Run Container

```bash
docker run -p 8000:8000 hospital-readmission
```

### Access API

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
├── reports/
│   ├── eda/
│   └── shap_summary.png
│
├── src/
│   ├── test_load.py
│   ├── preprocessing.py
│   ├── clean_data.py
│   ├── split_data.py
│   ├── train.py
│   ├── train_xgboost.py
│   ├── train_xgboost_mlflow.py
│   ├── shap_explain.py
│   └── eda_visualization.py
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
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* MLflow
* SHAP
* FastAPI
* Uvicorn
* Docker
* Joblib

---

## Key Learnings

* Handling healthcare data with class imbalance
* Preventing patient-level data leakage
* Building reusable machine learning pipelines
* Experiment tracking with MLflow
* Model explainability with SHAP
* Serving models through FastAPI
* Containerization using Docker

---

## Future Improvements

* Hyperparameter tuning using Optuna
* CI/CD pipeline integration
* Prometheus monitoring
* Grafana dashboards
* Data drift detection with Evidently AI
* Cloud deployment (Azure/AWS/GCP)

---

## Conclusion

This project demonstrates a complete end-to-end MLOps workflow for hospital readmission prediction. Starting from raw healthcare data, the project progresses through data preparation, machine learning modeling, experiment tracking, explainability, deployment, and containerization. The final XGBoost model outperformed the Logistic Regression baseline and was successfully deployed through a FastAPI service.
