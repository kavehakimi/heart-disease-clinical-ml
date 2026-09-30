import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).parent.parent
    / "data/raw" 
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
    names = column_names,
    na_values=["?"]
)

print(df.isna().sum())
df.info()

assert len(df) == 303, "Expected 303 rows"
assert len(df.columns) == 14, "Expected 14 columns"
assert df["ca"].isna().sum() == 4, "Expected 4 missing values in ca"
assert df["thal"].isna().sum() == 2, "Expected 2 missing values in thal"

# همه مقادیر مجاز باشند
assert df["target"].isin([0,1,2,3,4]).all(), "Unexpected value found in target"

# همه مقادیر مورد انتظار واقعا حضور داشته باشند
assert set(df["target"].unique()) == {0, 1, 2, 3, 4}, "Expected values: 0,1,2,3,4"
