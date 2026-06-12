# Hospital Readmission Prediction Project Report

## 1. Project Summary
This project builds an end-to-end MLOps pipeline to predict 30-day hospital readmissions for diabetic patients using the Diabetes 130-US Hospitals Dataset.
The workflow includes data cleaning, preprocessing, patient-level train/test splitting, XGBoost model training, MLflow tracking, SHAP explainability, and model deployment support.

## 2. Data and Preprocessing
- Dataset: `data/raw/diabetic_data.csv`
- Target: `readmitted == "<30"` mapped to binary label `target`
- Missing values: replaced `?` with `NaN`
- Dropped columns with high missing or leakage risk:
  - `weight`
  - `payer_code`
  - `medical_specialty`
  - `readmitted`
  - `encounter_id`
  - `patient_nbr`
- Patient-level split ensures the same `patient_nbr` does not appear in both train and test sets.
- Train/test split: 80% train, 20% test.

## 3. Model Training
### XGBoost configuration
- `n_estimators = 300`
- `max_depth = 5`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `scale_pos_weight = 8`
- `random_state = 42`
- `eval_metric = "logloss"`

### Results from `src/train_xgboost.py`
- PR-AUC: `0.2100`
- ROC-AUC: `0.6728`

### Classification report on test set:
- Class 0 (not readmitted within 30 days):
  - Precision: `0.93`
  - Recall: `0.66`
  - F1-score: `0.78`
- Class 1 (readmitted within 30 days):
  - Precision: `0.17`
  - Recall: `0.59`
  - F1-score: `0.27`
- Overall accuracy: `0.66`

## 4. Model Artifacts
- Trained model saved as `models/xgboost_model.pkl`
- MLflow model artifact saved as `models/xgboost_mlflow.pkl`
- SHAP summary plot generated as `reports/shap_summary.png`

## 5. Explainability
- SHAP was used to compute feature attributions for the trained XGBoost model.
- Generated artifact: `reports/shap_summary.png`
- This visualization identifies the most important features influencing model predictions.

## 6. Notes
- The model successfully completed training and explainability steps using the available dataset.
- The current pipeline outputs confirm the trained model and SHAP artifact were produced successfully.

## 7. Recommended Next Steps
1. Add cross-validation and hyperparameter tuning to strengthen model performance.
2. Evaluate calibration and fairness across patient subgroups.
3. Deploy the model through `api/app.py` and test API inference.
4. Extend documentation with summary plots, feature importance tables, and MLflow experiment details.
