import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier

print("Loading data...")

df = pd.read_csv("data/raw/diabetic_data.csv")

# Target
df["target"] = (
    df["readmitted"] == "<30"
).astype(int)

# Missing values
df.replace("?", np.nan, inplace=True)

# Split by patient
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

# Drop columns
drop_cols = [
    "weight",
    "payer_code",
    "medical_specialty",
    "readmitted",
    "encounter_id",
    "patient_nbr"
]

train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

X_train = train_df.drop(columns=["target"])
X_test = test_df.drop(columns=["target"])

y_train = train_df["target"]
y_test = test_df["target"]

# Features
numeric_features = (
    X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
)

categorical_features = (
    X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()
)

# Pipeline
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
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
    transformers=[
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

# XGBoost
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
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb)
    ]
)

print("Training XGBoost...")

pipeline.fit(
    X_train,
    y_train
)

pred_probs = pipeline.predict_proba(
    X_test
)[:, 1]

preds = pipeline.predict(X_test)

pr_auc = average_precision_score(
    y_test,
    pred_probs
)

roc_auc = roc_auc_score(
    y_test,
    pred_probs
)

print("\nResults")
print("=" * 50)

print(f"PR-AUC  : {pr_auc:.4f}")
print(f"ROC-AUC : {roc_auc:.4f}")

print(
    classification_report(
        y_test,
        preds
    )
)

joblib.dump(
    pipeline,
    "models/xgboost_model.pkl"
)

print(
    "\nSaved: models/xgboost_model.pkl"
)