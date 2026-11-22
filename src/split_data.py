import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

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

#------------------------------------
# Function for loading and split data
def load_and_split_data():
    df = pd.read_csv(
        DATA_PATH, 
        header=None, 
        names = column_names,
        na_values=["?"]
    )

    y = (df["target"]>0).astype(int)

    X = df.drop(columns=["target"])

    assert len(X) == 303, "Expected 303 rows"
    assert len(X.columns) == 13, "Expected 13 columns"
    assert len(y) == 303, "Expected 303 target values"
    assert set(y.unique()) == {0,1}, "Expected binary target values 0 and 1"
    assert len(X) == len(y), "X and y must have the same number of observations"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
#------------------------------------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()

    print("Training:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
