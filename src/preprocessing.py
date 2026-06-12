import pandas as pd

df = pd.read_csv("data/raw/diabetic_data.csv")

df["target"] = (
    df["readmitted"] == "<30"
).astype(int)

print("\nTarget Distribution")
print(df["target"].value_counts())

print("\nTarget Percentage")
print(df["target"].value_counts(normalize=True))

print("\nColumns with '?' values")

missing = (
    (df == "?")
    .sum()
    .sort_values(ascending=False)
)

print(
    missing[missing > 0]
)