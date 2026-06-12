import pandas as pd

df = pd.read_csv("data/raw/diabetic_data.csv")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["readmitted"].value_counts())

print("\nMissing Values:")
print((df == "?").sum().sort_values(ascending=False).head(20))