import pandas as pd

df = pd.read_csv(
    "data/raw/diabetic_data.csv"
)

print("Original Shape:")
print(df.shape)

df.replace(
    "?",
    pd.NA,
    inplace=True
)

drop_cols = [
    "weight",
    "payer_code",
    "medical_specialty"
]

df.drop(
    columns=drop_cols,
    inplace=True
)

print("\nCleaned Shape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isna().sum().sort_values(
    ascending=False
).head(10))