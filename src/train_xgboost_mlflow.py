import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

from xgboost import XGBClassifier

# =====================================================
# MLflow Setup
# =====================================================

mlflow.set_experiment(
    "Hospital_Readmission"
)

# =====================================================
# Load Data
# =====================================================

df = pd.read_csv(
    "data/raw/diabetic_data.csv"
)

df["target"] = (
    df["readmitted"] == "<30"
).astype(int)

df.replace(
    "?",
    np.nan,
    inplace=True
)

# =====================================================
# Split
# =====================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["patient_nbr"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

drop_cols = [
    "weight",
    "payer_code",
    "medical_specialty",
    "readmitted",
    "encounter_id",
    "patient_nbr"
]

train_df.drop(
    columns=drop_cols,
    inplace=True
)

test_df.drop(
    columns=drop_cols,
    inplace=True
)

X_train = train_df.drop(
    columns=["target"]
)

X_test = test_df.drop(
    columns=["target"]
)

y_train = train_df["target"]
y_test = test_df["target"]

# =====================================================
# Features
# =====================================================

numeric_features = (
    X_train
    .select_dtypes(
        include=["int64", "float64"]
    )
    .columns
)

categorical_features = (
    X_train
    .select_dtypes(
        include=["object"]
    )
    .columns
)

# =====================================================
# Pipeline
# =====================================================

numeric_transformer = Pipeline(
[
(
"imputer",
SimpleImputer(
strategy="median"
)
)
]
)

categorical_transformer = Pipeline(
[
(
"imputer",
SimpleImputer(
strategy="constant",
fill_value="Missing"
)
),
(
"encoder",
OneHotEncoder(
handle_unknown="ignore"
)
)
]
)

preprocessor = ColumnTransformer(
[
(
"num",
numeric_transformer,
numeric_features
),
(
"cat",
categorical_transformer,
categorical_features
)
]
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=8,
    random_state=42,
    eval_metric="logloss"
)

pipeline = Pipeline(
[
("preprocessor", preprocessor),
("model", xgb)
]
)

# =====================================================
# MLflow Run
# =====================================================

with mlflow.start_run():

    pipeline.fit(
        X_train,
        y_train
    )

    pred_probs = (
        pipeline
        .predict_proba(X_test)
    )[:,1]

    pr_auc = average_precision_score(
        y_test,
        pred_probs
    )

    roc_auc = roc_auc_score(
        y_test,
        pred_probs
    )

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "n_estimators",
        300
    )

    mlflow.log_param(
        "max_depth",
        5
    )

    mlflow.log_metric(
        "pr_auc",
        pr_auc
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.sklearn.log_model(
        pipeline,
        "model"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

joblib.dump(
    pipeline,
    "models/xgboost_mlflow.pkl"
)

print("Done.")