import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from split_data import load_and_split_data

def preprocess_data(X_train, X_test):
    median_features = ["ca"]

    thal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        ))
    ])

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

    X_train_final = pd.DataFrame(
        data=X_train_processed,
        columns=preprocessor.get_feature_names_out(),
        index=X_train.index
    )

    X_test_final = pd.DataFrame(
        data=X_test_processed,
        columns=preprocessor.get_feature_names_out(),
        index=X_test.index
    )

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

    return X_train_final, X_test_final, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_final, X_test_final, preprocessor = preprocess_data(
        X_train, 
        X_test
        )

    #  Inspect fitted imputers and verify test transformations
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

    missing_thal_test = X_test["thal"].isna()
    print("Missing THAL in Test:", missing_thal_test.sum())

    print(
        X_test_final.loc[
            missing_thal_test,
            [
                "thal_processor__thal_3.0", 
                "thal_processor__thal_6.0", 
                "thal_processor__thal_7.0"
            ]
        ]
    )

    #-------------------------------------------------
    # Validate alignment between Features and Targets:
    # 1. Validate Train & Test count with targets
    assert len(X_train_final) == len(y_train), "Final Train row count does not match target count"
    assert len(X_test_final) == len(y_test), "Final Test row count does not match target count"

    # 2. Validate patient indexes
    assert X_train_final.index.equals(y_train.index), "Train indexes have changed"
    assert X_test_final.index.equals(y_test.index), "Test indexex have changed"
