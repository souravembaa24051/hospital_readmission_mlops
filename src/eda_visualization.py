import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Setup
# =====================================================

os.makedirs("reports/eda", exist_ok=True)

plt.style.use("default")
sns.set_theme(style="whitegrid")

print("Loading dataset...")

df = pd.read_csv("data/raw/diabetic_data.csv")

# Create target
df["target"] = (
    df["readmitted"] == "<30"
).astype(int)

# Replace missing markers
df.replace("?", np.nan, inplace=True)

print(f"Dataset Shape: {df.shape}")

# =====================================================
# 1. Target Distribution
# =====================================================

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="target"
)

plt.title(
    "Readmission Target Distribution",
    fontsize=14
)

plt.xlabel("Target")
plt.ylabel("Count")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    "reports/eda/target_distribution.png"
)

plt.close()

print("Saved: target_distribution.png")

# =====================================================
# 2. Age vs Readmission
# =====================================================

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="age",
    hue="target"
)

plt.title(
    "Readmission by Age Group",
    fontsize=14
)

plt.xlabel("Age Group")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "reports/eda/age_vs_readmission.png"
)

plt.close()

print("Saved: age_vs_readmission.png")

# =====================================================
# 3. Time in Hospital Distribution
# =====================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["time_in_hospital"],
    bins=14,
    kde=True
)

plt.title(
    "Distribution of Time in Hospital",
    fontsize=14
)

plt.xlabel("Days in Hospital")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "reports/eda/time_in_hospital_distribution.png"
)

plt.close()

print("Saved: time_in_hospital_distribution.png")

# =====================================================
# 4. Readmission by Gender
# =====================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="target"
)

plt.title(
    "Readmission by Gender",
    fontsize=14
)

plt.xlabel("Gender")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "reports/eda/gender_vs_readmission.png"
)

plt.close()

print("Saved: gender_vs_readmission.png")

# =====================================================
# 5. Correlation Heatmap
# =====================================================

corr_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "target"
]

corr_matrix = df[corr_features].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "reports/eda/correlation_heatmap.png"
)

plt.close()

print("Saved: correlation_heatmap.png")

# =====================================================
# 6. Top Diagnoses
# =====================================================

plt.figure(figsize=(12, 6))

top_diag = (
    df["diag_1"]
    .value_counts()
    .head(10)
)

sns.barplot(
    x=top_diag.index,
    y=top_diag.values
)

plt.title(
    "Top 10 Primary Diagnoses",
    fontsize=14
)

plt.xlabel("Diagnosis Code")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "reports/eda/top_diagnoses.png"
)

plt.close()

print("Saved: top_diagnoses.png")

print("\nEDA Visualization Complete!")
print("Files saved in reports/eda/")