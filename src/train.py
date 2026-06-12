import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
)

# =====================================================
# Load Data
# =====================================================

print("Loading data...")

df = pd.read_csv("data/raw/diabetic_data.csv")

print(f"Dataset Shape: {df.shape}")

# =====================================================
# Create Target
# =====================================================

df["target"] = (df["readmitted"] == "<30").astype(int)

print("\nTarget Distribution:")
print(df["target"].value_counts())

# =====================================================
# Replace ? with np.nan
# =====================================================

df.replace("?", np.nan, inplace=True)

print("\nTop Missing Columns:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# =====================================================
# Split Before Dropping patient_nbr
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

print("\nTrain Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# =====================================================
# Drop Columns
# =====================================================

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

# =====================================================
# Features / Target
# =====================================================

X_train = train_df.drop(columns=["target"])
X_test = test_df.drop(columns=["target"])

y_train = train_df["target"]
y_test = test_df["target"]

# =====================================================
# Feature Types
# =====================================================

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

print("\nNumeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# =====================================================
# Preprocessing
# =====================================================

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
                missing_values=np.nan,
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

# =====================================================
# Model
# =====================================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# =====================================================
# Train
# =====================================================

print("\nTraining Logistic Regression...")

pipeline.fit(
    X_train,
    y_train
)

print("Training Complete")

# =====================================================
# Predict
# =====================================================

pred_probs = pipeline.predict_proba(X_test)[:, 1]
preds = pipeline.predict(X_test)

# =====================================================
# Metrics
# =====================================================

pr_auc = average_precision_score(
    y_test,
    pred_probs
)

roc_auc = roc_auc_score(
    y_test,
    pred_probs
)

print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)

print(f"PR-AUC  : {pr_auc:.4f}")
print(f"ROC-AUC : {roc_auc:.4f}")

print("\nClassification Report")
print(
    classification_report(
        y_test,
        preds
    )
)

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    pipeline,
    "models/logistic_baseline.pkl"
)

print("\nModel Saved:")
print("models/logistic_baseline.pkl")