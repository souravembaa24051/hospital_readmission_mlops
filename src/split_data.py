import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv(
    "data/raw/diabetic_data.csv"
)

df["target"] = (
    df["readmitted"] == "<30"
).astype(int)

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

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

print("Train Shape:")
print(train_df.shape)

print("\nTest Shape:")
print(test_df.shape)

print("\nUnique Patients Train:")
print(train_df["patient_nbr"].nunique())

print("\nUnique Patients Test:")
print(test_df["patient_nbr"].nunique())