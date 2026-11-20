import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

median_features = ["ca"]
categorical_features = [
    "thal", "cp", "restecg"
    ]
passthrough_features = [
    "age", "sex", "trestbps", "chol",
    "fbs", "thalach", "exang",
    "oldpeak", "slope"
    ]

thal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    ))
])

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "ca_imputer",
            SimpleImputer(strategy="median"),
            median_features
        ),
        (
            "thal_processor",
            thal_pipeline,
            ["thal"]
        ),
        (
            "categorical_encoder",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            ),
            ["cp", "restecg"]
        )
    ],
    remainder="passthrough"
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Train shape:", X_train_processed.shape)
print("Test shape:", X_test_processed.shape)

print("Train type:", type(X_train_processed))
print("Feature names:", preprocessor.get_feature_names_out())

X_train_final = pd.DataFrame(
    data=X_train_processed,
    columns=preprocessor.get_feature_names_out(),
    index=X_train.index
)

print(X_train_final.shape)
print(X_train_final.head())
print(X_train_final.columns)

X_test_final = pd.DataFrame(
    data=X_test_processed,
    columns=preprocessor.get_feature_names_out(),
    index=X_test.index
)

print("Train:", X_train_final.shape)
print("Test:", X_test_final.shape)

# 1. Validate row counts
assert len(X_train_final) == len(X_train), "Train row count has changed"
assert len(X_test_final) == len(X_test), "Test row count has changed"

# 2. Validate feature counts
assert X_train_final.shape[1] == 20, "Expected 20 features in Train"
assert X_test_final.shape[1] == 20, "Expected 20 features in Test"

# 3. Validate feature names and order
assert X_train_final.columns.equals(X_test_final.columns), \
    "Train and Test feature columns do not match"

# 4. Validate missing values
assert X_train_final.isna().sum().sum() == 0, "Missing values found in Train"
assert X_test_final.isna().sum().sum() == 0, "Missing values found in Test"

# 5. Validate patient indexes
assert X_train_final.index.equals(X_train.index), "Train indexes have changed"
assert X_test_final.index.equals(X_test.index), "Test indexes have changed"

#  Data Leakage
ca_imputer = preprocessor.named_transformers_["ca_imputer"]
print("CA median:", ca_imputer.statistics_)

fitted_thal_pipeline = preprocessor.named_transformers_["thal_processor"]
thal_imputer = fitted_thal_pipeline.named_steps["imputer"]
print("Thal mode:", thal_imputer.statistics_)


missing_ca_test = X_test["ca"].isna()
print("Missing CA in Test:", missing_ca_test.sum())
print(
    X_test_final.loc[
        missing_ca_test,
        "ca_imputer__ca"
    ]
)