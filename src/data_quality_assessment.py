import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = (
    Path(__file__).parent.parent
    / "data" 
    / "processed.cleveland.data"
)


column_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

df = pd.read_csv(
    DATA_PATH, 
    header=None, 
    names = column_names
)

print(df.isnull().sum())
print(df=="?")

mask = df == "?"
print(mask.sum())

patient_with_missing = mask.any(axis=1)
print("Patients with at least one missing value = ", patient_with_missing.sum())

print(df.duplicated().sum())
print(df[df.duplicated()])

print(df["sex"].value_counts(dropna=False))

categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
    "target"
]

for column in categorical_columns:
    print("\nColumn:", column)
    print([df[column].value_counts(dropna=False)])


numerical_ranges = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

for column in numerical_ranges:
    print("\nColumn: ", column, 
          "| Min: ", np.min(df[column]), 
          "| Max: ", np.max(df[column]), 
          "| Mean: ", np.mean(df[column]), 
          "| Median: ", np.median(df[column])
          )
