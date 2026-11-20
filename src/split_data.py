import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

y = (df["target"]>0).astype(int)
print(y.value_counts())

X = df.drop(columns=["target"])
print(X.shape)
print(y.shape)

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

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print("Full dataset:")
print(y.value_counts(normalize=True))
print(y.value_counts())

print("\nTraining set:")
print(y_train.value_counts(normalize=True))
print(y_train.value_counts())

print("\nTest set:")
print(y_test.value_counts(normalize=True))
print(y_test.value_counts())

print(X_train["ca"].median())
print(X_train["thal"].mode())

print(X_train[["ca", "thal"]].isna().sum())
print(X_test[["ca", "thal"]].isna().sum())

ca_imputer = SimpleImputer(strategy="median")
thal_imputer = SimpleImputer(strategy="most_frequent")

# "---------------"
# "Learn from data"
# [["ca"]] = Because Scikit learn works with 2D feature
ca_imputer.fit(X_train[["ca"]])
thal_imputer.fit(X_train[["thal"]])
# "----------------------"

print(ca_imputer.statistics_)
print(thal_imputer.statistics_)

print(X_train["ca"].shape)
print(X_train[["ca"]].shape)

# "----------------------"
# "Apply what was learned"
# ravel = Because pandas works with 1D feature
X_train["ca"] = ca_imputer.transform(X_train[["ca"]]).ravel()
X_test["ca"] = ca_imputer.transform(X_test[["ca"]]).ravel()

X_train["thal"] = thal_imputer.transform(X_train[["thal"]]).ravel()
X_test["thal"] = thal_imputer.transform(X_test[["thal"]]).ravel()
# "----------------------"

print(X_train[["ca", "thal"]].isna().sum())
print(X_test[["ca", "thal"]].isna().sum())

assert X_train["ca"].isna().sum() == 0, "Expected 0 missing values in ca"
assert X_test["ca"].isna().sum() == 0, "Expected 0 missing values in ca"
assert X_train["thal"].isna().sum() == 0, "Expected 0 missing values in thal"
assert X_test["thal"].isna().sum() == 0, "Expected 0 missing values in thal"

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import OneHotEncoder

categorical_features = ["cp", "restecg", "thal"]

categorical_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Fit only on training data
categorical_encoder.fit(X_train[categorical_features])
print(categorical_encoder.categories_)
print(categorical_encoder.get_feature_names_out())

# Transform both datasets
train_encoded = categorical_encoder.transform(X_train[categorical_features])
test_encoded = categorical_encoder.transform(X_test[categorical_features])

# Convert NumPy arrays to DataFrames
train_encoded_df = pd.DataFrame(
    train_encoded,
    columns=categorical_encoder.get_feature_names_out(),
    index=X_train.index
)

test_encoded_df = pd.DataFrame(
    test_encoded,
    columns=categorical_encoder.get_feature_names_out(),
    index=X_test.index
)

X_train_encoded = pd.concat(
    [
        X_train.drop(columns=categorical_features), 
        train_encoded_df
    ],
    axis=1
)

X_test_encoded = pd.concat(
    [
        X_test.drop(columns=categorical_features), 
        test_encoded_df
    ],
    axis=1
)

print("Train shape:", X_train_encoded.shape)
print("Test shape:", X_test_encoded.shape)

print("\nEncoded columns:")
print(X_train_encoded.columns.tolist())

# -----------
# Validation:
assert X_train_encoded.columns.equals(X_test_encoded.columns), "Expected same columns for both train and test dataframes"
assert X_train_encoded.isna().sum().sum() == 0, "Expected 0 missing value in train dataframe"
assert X_test_encoded.isna().sum().sum() == 0, "Expected 0 missing value in test dataframe"
assert X_train_encoded.shape == (242, 20), "Expected 20 features in train dataframe"
assert X_test_encoded.shape == (61, 20), "Expected 20 features in test dataframe"
assert not any(
    col in X_train_encoded.columns
    for col in categorical_features
), "Expected cp, thal, restecg are not available in dataframe"


