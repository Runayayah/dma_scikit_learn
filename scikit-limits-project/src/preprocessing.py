from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from src.config import PREPROCESSOR_CONFIG


# =========================================================
# HELPERS DE COLONNES
# =========================================================

def infer_feature_types(df) -> Tuple[List[str], List[str], str]:
    """
    Détecte les colonnes numériques, catégorielles et texte
    à partir des conventions de nommage.
    """
    numeric_features = [col for col in df.columns if col.startswith("num_")]
    categorical_features = [col for col in df.columns if col.startswith("cat_")]
    text_feature = "text_feature"

    return numeric_features, categorical_features, text_feature


def split_features_target(df, target_col: str = "target"):
    """
    Sépare les features et la cible.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# =========================================================
# TRANSFORMERS CUSTOM
# =========================================================

class TextColumnFlattener(BaseEstimator, TransformerMixin):
    """
    Transforme une colonne texte 2D en tableau 1D de chaînes.
    Compatible avec CountVectorizer / TfidfVectorizer / HashingVectorizer.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "ravel"):
            arr = X.ravel()
        else:
            arr = np.array([row[0] for row in X], dtype=object)

        return np.array(["" if x is None else str(x) for x in arr], dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Regroupe les catégories rares en '__RARE__'.
    Utile avant OneHot ou OrdinalEncoder.
    """

    def __init__(self, min_freq: float = 0.01):
        self.min_freq = min_freq
        self.frequent_categories_: dict[str, set] = {}

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=object)
        n_rows, n_cols = X_arr.shape

        self.frequent_categories_ = {}

        for j in range(n_cols):
            col_values = np.array(
                ["__MISSING__" if v is None or str(v) == "nan" else str(v) for v in X_arr[:, j]],
                dtype=object,
            )
            values, counts = np.unique(col_values, return_counts=True)
            freqs = counts / n_rows
            allowed = set(values[freqs >= self.min_freq])
            self.frequent_categories_[f"col_{j}"] = allowed

        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=object).copy()
        n_rows, n_cols = X_arr.shape

        for j in range(n_cols):
            allowed = self.frequent_categories_.get(f"col_{j}", set())
            cleaned = []
            for v in X_arr[:, j]:
                s = "__MISSING__" if v is None or str(v) == "nan" else str(v)
                if s not in allowed:
                    s = "__RARE__"
                cleaned.append(s)
            X_arr[:, j] = cleaned

        return X_arr


class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Convertit une matrice sparse en dense si nécessaire.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            return X.toarray()
        return X


class NonNegativeClipper(BaseEstimator, TransformerMixin):
    """
    Force les features à être >= 0.
    Utile pour MultinomialNB.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            X = X.copy()
            X.data = np.clip(X.data, 0, None)
            return X
        return np.clip(X, 0, None)


# =========================================================
# BLOCS DE PREPROCESSING
# =========================================================

def build_numeric_pipeline(
    scaler_name: str = "robust",
    dense_output: bool = False,
) -> Pipeline:
    """
    Construit le pipeline numérique.
    """
    scaler_map = {
        "standard": StandardScaler(with_mean=not dense_output is False),
        "minmax": MinMaxScaler(),
        "maxabs": MaxAbsScaler(),
        "robust": RobustScaler(with_centering=not dense_output is False),
        "power": PowerTransformer(),
        "quantile": QuantileTransformer(output_distribution="normal"),
        "none": "passthrough",
    }

    scaler = scaler_map.get(scaler_name, RobustScaler())

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scaler != "passthrough":
        # StandardScaler / RobustScaler sur sparse avec centrage posent problème
        if scaler_name == "standard":
            scaler = StandardScaler(with_mean=False if not dense_output else True)
        elif scaler_name == "robust":
            scaler = RobustScaler(with_centering=True if dense_output else False)
        steps.append(("scaler", scaler))

    return Pipeline(steps=steps)


def build_categorical_pipeline(
    encoder_name: str = "onehot",
    min_freq: float = 0.01,
) -> Pipeline:
    """
    Construit le pipeline catégoriel.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare_grouper", RareCategoryGrouper(min_freq=min_freq)),
    ]

    if encoder_name == "onehot":
        steps.append(
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        )
    elif encoder_name == "ordinal":
        steps.append(
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        )
    else:
        raise ValueError(f"Encoder inconnu: {encoder_name}")

    return Pipeline(steps=steps)


def build_text_pipeline(
    vectorizer_name: str = "tfidf",
    max_features: int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    use_stop_words: bool = False,
    apply_svd: bool = False,
    svd_n_components: int = 100,
) -> Pipeline:
    """
    Construit le pipeline texte.
    """
    stop_words = "english" if use_stop_words else None
    max_features = max_features or PREPROCESSOR_CONFIG["text_max_features"]

    if vectorizer_name == "count":
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
        )
    elif vectorizer_name == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
        )
    elif vectorizer_name == "hashing":
        vectorizer = HashingVectorizer(
            n_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            alternate_sign=False,
        )
    else:
        raise ValueError(f"Vectorizer inconnu: {vectorizer_name}")

    steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("flatten", TextColumnFlattener()),
        ("vectorizer", vectorizer),
    ]

    if apply_svd:
        steps.append(("svd", TruncatedSVD(n_components=svd_n_components, random_state=42)))

    return Pipeline(steps=steps)


# =========================================================
# FEATURE SELECTION / REDUCTION
# =========================================================

def build_feature_selector(
    method: str | None = None,
    k: int = 100,
    percentile: int = 30,
) -> BaseEstimator | str:
    """
    Retourne un sélecteur de variables.
    """
    if method is None or method == "none":
        return "passthrough"

    if method == "variance_threshold":
        return VarianceThreshold()

    if method == "select_k_best":
        return SelectKBest(score_func=f_classif, k=k)

    if method == "select_percentile":
        return SelectPercentile(score_func=f_classif, percentile=percentile)

    raise ValueError(f"Méthode de sélection inconnue: {method}")


def build_dimensionality_reducer(
    method: str | None = None,
    n_components: int = 50,
) -> BaseEstimator | str:
    """
    Retourne un réducteur de dimension.
    """
    if method is None or method == "none":
        return "passthrough"

    if method == "pca":
        return PCA(n_components=n_components, random_state=42)

    if method == "truncated_svd":
        return TruncatedSVD(n_components=n_components, random_state=42)

    raise ValueError(f"Méthode de réduction inconnue: {method}")


# =========================================================
# CONSTRUCTEURS PRINCIPAUX
# =========================================================

def build_column_transformer(
    df,
    numeric_scaler: str = "robust",
    categorical_encoder: str = "onehot",
    text_vectorizer: str = "tfidf",
    include_text: bool = True,
    text_only: bool = False,
    tabular_only: bool = False,
    dense_output: bool = False,
    use_stop_words: bool = False,
    ngram_range: tuple[int, int] = (1, 1),
    text_max_features: int | None = None,
    rare_category_min_freq: float | None = None,
    apply_text_svd: bool = False,
) -> ColumnTransformer:
    """
    Construit le ColumnTransformer de base.
    """
    numeric_features, categorical_features, text_feature = infer_feature_types(df)
    rare_category_min_freq = (
        rare_category_min_freq
        if rare_category_min_freq is not None
        else PREPROCESSOR_CONFIG["rare_category_min_freq"]
    )

    transformers = []

    if not text_only:
        if numeric_features:
            transformers.append(
                (
                    "num",
                    build_numeric_pipeline(
                        scaler_name=numeric_scaler,
                        dense_output=dense_output,
                    ),
                    numeric_features,
                )
            )

        if categorical_features:
            transformers.append(
                (
                    "cat",
                    build_categorical_pipeline(
                        encoder_name=categorical_encoder,
                        min_freq=rare_category_min_freq,
                    ),
                    categorical_features,
                )
            )

    if include_text and not tabular_only and text_feature in df.columns:
        transformers.append(
            (
                "txt",
                build_text_pipeline(
                    vectorizer_name=text_vectorizer,
                    max_features=text_max_features,
                    ngram_range=ngram_range,
                    use_stop_words=use_stop_words,
                    apply_svd=apply_text_svd,
                ),
                [text_feature],
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_preprocessor_basic(df) -> Pipeline:
    """
    Préprocessor standard :
    - numérique : RobustScaler
    - catégoriel : OneHot
    - texte : TF-IDF
    """
    column_transformer = build_column_transformer(
        df=df,
        numeric_scaler="robust",
        categorical_encoder="onehot",
        text_vectorizer="tfidf",
        include_text=True,
        dense_output=False,
        ngram_range=(1, 1),
    )

    return Pipeline([
        ("preprocessor", column_transformer),
    ])


def build_preprocessor_no_text(df) -> Pipeline:
    """
    Pipeline tabulaire seulement.
    """
    column_transformer = build_column_transformer(
        df=df,
        numeric_scaler="robust",
        categorical_encoder="onehot",
        include_text=False,
        tabular_only=True,
        dense_output=False,
    )

    return Pipeline([
        ("preprocessor", column_transformer),
    ])


def build_preprocessor_text_only(df) -> Pipeline:
    """
    Pipeline texte seul.
    """
    column_transformer = build_column_transformer(
        df=df,
        text_vectorizer="tfidf",
        include_text=True,
        text_only=True,
        dense_output=False,
        ngram_range=(1, 2),
    )

    return Pipeline([
        ("preprocessor", column_transformer),
    ])


def build_preprocessor_dense(df) -> Pipeline:
    """
    Pipeline dense :
    utile pour GaussianNB / HistGradientBoosting / certains benchmarks.
    """
    column_transformer = build_column_transformer(
        df=df,
        numeric_scaler="standard",
        categorical_encoder="ordinal",
        text_vectorizer="tfidf",
        include_text=True,
        dense_output=True,
        ngram_range=(1, 1),
        apply_text_svd=True,
    )

    return Pipeline([
        ("preprocessor", column_transformer),
        ("to_dense", DenseTransformer()),
    ])


def build_preprocessor_sparse(df) -> Pipeline:
    """
    Pipeline sparse :
    utile pour SVM linéaires, SGD, logreg, grands vocabulaires.
    """
    column_transformer = build_column_transformer(
        df=df,
        numeric_scaler="maxabs",
        categorical_encoder="onehot",
        text_vectorizer="tfidf",
        include_text=True,
        dense_output=False,
        ngram_range=(1, 2),
    )

    return Pipeline([
        ("preprocessor", column_transformer),
    ])


def build_preprocessor_non_negative(df) -> Pipeline:
    """
    Pipeline avec sortie non négative :
    utile pour MultinomialNB.
    """
    column_transformer = build_column_transformer(
        df=df,
        numeric_scaler="minmax",
        categorical_encoder="onehot",
        text_vectorizer="count",
        include_text=True,
        dense_output=False,
        ngram_range=(1, 2),
    )

    return Pipeline([
        ("preprocessor", column_transformer),
        ("non_negative", NonNegativeClipper()),
    ])


def build_preprocessor_from_name(df, preprocessor_name: str) -> Pipeline:
    """
    Sélectionne un préprocessor par nom.
    """
    mapping = {
        "basic_tfidf_robust": lambda x: build_preprocessor_basic(x),
        "basic_count_standard": lambda x: Pipeline([
            ("preprocessor", build_column_transformer(
                df=x,
                numeric_scaler="standard",
                categorical_encoder="onehot",
                text_vectorizer="count",
                include_text=True,
                dense_output=False,
                ngram_range=(1, 1),
            ))
        ]),
        "no_text_robust": lambda x: build_preprocessor_no_text(x),
        "text_only_tfidf": lambda x: build_preprocessor_text_only(x),
        "dense_standard_ordinal": lambda x: build_preprocessor_dense(x),
        "sparse_tfidf_maxabs": lambda x: build_preprocessor_sparse(x),
        "non_negative_count_minmax": lambda x: build_preprocessor_non_negative(x),
    }

    if preprocessor_name not in mapping:
        raise ValueError(f"Préprocessor inconnu: {preprocessor_name}")

    return mapping[preprocessor_name](df)


def build_preprocessor(
    df,
    preprocessor_name: str = "basic_tfidf_robust",
) -> Pipeline:
    """
    Fonction par défaut utilisée par le reste du projet.
    """
    return build_preprocessor_from_name(df, preprocessor_name)


# =========================================================
# PIPELINES AVEC SÉLECTION / RÉDUCTION
# =========================================================

def attach_feature_processing_steps(
    base_pipeline: Pipeline,
    feature_selection_method: str | None = None,
    dimensionality_reduction_method: str | None = None,
    feature_selection_k: int = 100,
    feature_selection_percentile: int = 30,
    reduction_n_components: int = 50,
) -> Pipeline:
    """
    Ajoute éventuellement :
    - feature selection
    - réduction de dimension
    à un pipeline existant.
    """
    steps = list(base_pipeline.steps)

    selector = build_feature_selector(
        method=feature_selection_method,
        k=feature_selection_k,
        percentile=feature_selection_percentile,
    )
    reducer = build_dimensionality_reducer(
        method=dimensionality_reduction_method,
        n_components=reduction_n_components,
    )

    if selector != "passthrough":
        steps.append(("feature_selection", selector))

    if reducer != "passthrough":
        steps.append(("dimensionality_reduction", reducer))

    return Pipeline(steps=steps)


# =========================================================
# MÉTADONNÉES UTILES
# =========================================================

def get_preprocessor_registry() -> dict:
    """
    Retourne les preprocessors disponibles.
    """
    return {
        "basic_tfidf_robust": {
            "dense": False,
            "non_negative": False,
            "with_text": True,
            "text_only": False,
        },
        "basic_count_standard": {
            "dense": False,
            "non_negative": False,
            "with_text": True,
            "text_only": False,
        },
        "no_text_robust": {
            "dense": False,
            "non_negative": False,
            "with_text": False,
            "text_only": False,
        },
        "text_only_tfidf": {
            "dense": False,
            "non_negative": False,
            "with_text": True,
            "text_only": True,
        },
        "dense_standard_ordinal": {
            "dense": True,
            "non_negative": False,
            "with_text": True,
            "text_only": False,
        },
        "sparse_tfidf_maxabs": {
            "dense": False,
            "non_negative": False,
            "with_text": True,
            "text_only": False,
        },
        "non_negative_count_minmax": {
            "dense": False,
            "non_negative": True,
            "with_text": True,
            "text_only": False,
        },
    }


def is_dense_preprocessor(preprocessor_name: str) -> bool:
    registry = get_preprocessor_registry()
    return registry.get(preprocessor_name, {}).get("dense", False)


def is_non_negative_preprocessor(preprocessor_name: str) -> bool:
    registry = get_preprocessor_registry()
    return registry.get(preprocessor_name, {}).get("non_negative", False)


def uses_text(preprocessor_name: str) -> bool:
    registry = get_preprocessor_registry()
    return registry.get(preprocessor_name, {}).get("with_text", False)


if __name__ == "__main__":
    print("Préprocessors disponibles :")
    for name, meta in get_preprocessor_registry().items():
        print(f"- {name}: {meta}")