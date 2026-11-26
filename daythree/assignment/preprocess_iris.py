"""Week 11 Class 3 student lab: implement a leakage-safe feature pipeline.

Run: python preprocess_iris.py --input iris_preprocessing_case_PUBLIC_DERIVED.csv --output-dir outputs
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
CATEGORICAL = ["collection_site", "instrument_type"]
LABEL = "species"

def build_preprocessor():
    """Return a ColumnTransformer with numeric imputation+scaling and category imputation+one-hot.

    Make unseen categories transformable without an exception.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])

def prepare_dataset(input_csv):
    """Load CSV, split features/target before fitting, then return (transformer, train_df, test_df, report_dict).

    Split must be deterministic: test_size=0.2, random_state=42, stratify=species.

    Output DataFrames must have matching columns, numeric finite values, and no species/sample_id.

    Report keys: source_rows, training_rows, test_rows, transformed_features, missing_source_cells.
    """
    df = pd.read_csv(input_csv)

    source_rows = len(df)
    missing_source_cells = int(df.isna().sum().sum())

    # X = predictors only. Drop LABEL (species) and sample_id (ID only)
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df[LABEL].copy()

    # Split FIRST - before any fitting, to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit ONLY on train, transform test with same fitted transformer
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())

    train_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    test_df = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    report = {
        "source_rows": source_rows,
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "transformed_features": len(feature_names),
        "missing_source_cells": missing_source_cells,
    }

    return preprocessor, train_df, test_df, report

def write_outputs(train_df, test_df, report, output_dir):
    """Write train_features.csv, test_features.csv, data_quality_summary.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train_features.csv", index=False)
    test_df.to_csv(out / "test_features.csv", index=False)
    pd.DataFrame([report]).to_csv(out / "data_quality_summary.csv", index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='iris_preprocessing_case_PUBLIC_DERIVED.csv')
    p.add_argument('--output-dir', default='outputs')
    args = p.parse_args()

    transformer, train, test, report = prepare_dataset(args.input)
    write_outputs(train, test, report, args.output_dir)
    print('Preprocessing completed')
    print(report)
    print('train:', train.shape)
    print('test:', test.shape)

if __name__ == '__main__':
    main()