import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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

for column in numerical_ranges:
    plt.hist(df[column], bins=20)
    plt.title(f"Distribution of {column}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

for column in numerical_ranges:
    plt.boxplot(df[column], tick_labels=[column])
    plt.title(f"Boxplot of {column}")
    plt.ylabel("Value")
    plt.show()

for column in numerical_ranges:
    q1 = np.percentile(df[column], 25)
    q3 = np.percentile(df[column], 75)
    iqr = q3-q1
    lower_bound = q1-(1.5*iqr)
    upper_bound = q3+(1.5*iqr)

    print("\nColumn:", column)
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)

    mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers = df.loc[mask, column]

    print("Candidates for outlier:", mask.sum())
    print(outliers.to_list())