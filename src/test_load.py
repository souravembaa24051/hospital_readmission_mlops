import pandas as pd

df = pd.read_csv(
    "data/raw/diabetic_data.csv"
)

print(df.shape)
print(df.head())