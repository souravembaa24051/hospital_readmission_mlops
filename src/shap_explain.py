import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

print("Loading model...")

pipeline = joblib.load(
    "models/xgboost_model.pkl"
)

print("Loading data...")

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

# -----------------------------------
# Same split as training
# -----------------------------------

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

test_df = df.iloc[test_idx].copy()

drop_cols = [
    "weight",
    "payer_code",
    "medical_specialty",
    "readmitted",
    "encounter_id",
    "patient_nbr"
]

test_df.drop(
    columns=drop_cols,
    inplace=True
)

X_test = test_df.drop(
    columns=["target"]
)

# -----------------------------------
# Get trained components
# -----------------------------------

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]

print("Transforming features...")

X_transformed = preprocessor.transform(
    X_test
)

# Limit sample size for speed
X_sample = X_transformed[:1000]

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_sample
)

# -----------------------------------
# Summary Plot
# -----------------------------------

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "reports/shap_summary.png",
    bbox_inches="tight"
)

print(
    "Saved: reports/shap_summary.png"
)