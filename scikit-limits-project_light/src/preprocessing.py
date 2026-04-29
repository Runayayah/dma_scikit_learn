from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def infer_feature_types(df) -> Tuple[List[str], List[str], str]:
    numeric_features = [col for col in df.columns if col.startswith("num_")]
    categorical_features = [col for col in df.columns if col.startswith("cat_")]
    text_feature = "text_feature"
    return numeric_features, categorical_features, text_feature


def split_features_target(df, target_col: str = "target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


class TextColumnFlattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "ravel"):
            arr = X.ravel()
        else:
            arr = np.array([row[0] for row in X], dtype=object)

        return np.array(["" if x is None else str(x) for x in arr], dtype=object)


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            return X.toarray()
        return X


def build_standard_preprocessor(df) -> Pipeline:
    numeric_features, categorical_features, text_feature = infer_feature_types(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    text_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", TextColumnFlattener()),
            ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2))),
        ]
    )

    transformers = [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]

    if text_feature in df.columns:
        transformers.append(("txt", text_pipeline, [text_feature]))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", column_transformer),
        ]
    )


def build_dense_preprocessor(df) -> Pipeline:
    numeric_features, categorical_features, text_feature = infer_feature_types(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    text_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", TextColumnFlattener()),
            ("tfidf", TfidfVectorizer(max_features=100)),
        ]
    )

    transformers = [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]

    if text_feature in df.columns:
        transformers.append(("txt", text_pipeline, [text_feature]))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", column_transformer),
            ("to_dense", DenseTransformer()),
        ]
    )


def build_preprocessor(df, preprocessor_name: str = "standard") -> Pipeline:
    if preprocessor_name == "dense":
        return build_dense_preprocessor(df)

    if preprocessor_name == "standard":
        return build_standard_preprocessor(df)

    raise ValueError(f"Préprocessor inconnu : {preprocessor_name}")