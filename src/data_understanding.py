import pandas as pd
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

print(df.head())
print(df.shape)
print(df.columns)
df.info()
print(df.describe())

print(df["ca"].unique())
print(df["thal"].unique())

print(df["ca"].value_counts())
print(df["thal"].value_counts())

print(df["target"].unique())
print(df["target"].value_counts().sort_index())
