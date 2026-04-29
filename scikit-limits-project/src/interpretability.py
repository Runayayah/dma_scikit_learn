from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


# =========================================================
# HELPERS GÉNÉRAUX
# =========================================================

def _to_1d_importance_array(values) -> np.ndarray:
    """
    Convertit une structure de coefficients / importances en vecteur 1D.
    """
    arr = np.asarray(values)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        # binaire ou multiclasse : moyenne absolue par feature
        return np.mean(np.abs(arr), axis=0)

    raise ValueError("Format d'importance non supporté.")


def _safe_get_pipeline_step(pipeline: Pipeline, step_name: str):
    """
    Récupère un step du pipeline s'il existe.
    """
    if not hasattr(pipeline, "named_steps"):
        return None
    return pipeline.named_steps.get(step_name)


def _is_text_feature_name(name: str) -> bool:
    return str(name).startswith("txt__") or "text_feature" in str(name)


def _is_numeric_feature_name(name: str) -> bool:
    return str(name).startswith("num__") or "num_" in str(name)


def _is_categorical_feature_name(name: str) -> bool:
    return str(name).startswith("cat__") or "cat_" in str(name)


# =========================================================
# FEATURE NAMES
# =========================================================

def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Retourne les noms de features après preprocessing.
    Suppose un step 'preprocessor' dans le pipeline global.
    """
    preprocessor = _safe_get_pipeline_step(pipeline, "preprocessor")
    if preprocessor is None:
        return []

    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return []


# =========================================================
# COEFFICIENTS / FEATURE IMPORTANCES
# =========================================================

def extract_model_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extrait les coefficients d'un modèle linéaire.
    Retourne un DataFrame trié par importance absolue.
    """
    model = _safe_get_pipeline_step(pipeline, "model")
    if model is None or not hasattr(model, "coef_"):
        return pd.DataFrame()

    feature_names = get_transformed_feature_names(pipeline)
    coefficients = _to_1d_importance_array(model.coef_)

    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return coef_df


def extract_tree_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extrait les feature importances d'un modèle d'arbre / ensemble.
    """
    model = _safe_get_pipeline_step(pipeline, "model")
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    feature_names = get_transformed_feature_names(pipeline)
    importances = _to_1d_importance_array(model.feature_importances_)

    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "abs_importance": np.abs(importances),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return imp_df


def extract_model_importances(pipeline: Pipeline) -> pd.DataFrame:
    """
    Route automatiquement vers :
    - coefficients si modèle linéaire
    - feature_importances si modèle arbre
    """
    coef_df = extract_model_coefficients(pipeline)
    if not coef_df.empty:
        result = coef_df.copy()
        result["importance_type"] = "coefficient"
        return result

    imp_df = extract_tree_feature_importances(pipeline)
    if not imp_df.empty:
        result = imp_df.copy()
        result["importance_type"] = "feature_importance"
        return result

    return pd.DataFrame()


# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================

def compute_permutation_importance_df(
    pipeline: Pipeline,
    X,
    y,
    scoring: str = "f1",
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: float | int = 1.0,
) -> pd.DataFrame:
    """
    Calcule la permutation importance directement sur le pipeline complet.
    """
    result = permutation_importance(
        estimator=pipeline,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        max_samples=max_samples,
    )

    feature_names = list(X.columns) if hasattr(X, "columns") else [f"feature_{i}" for i in range(len(result.importances_mean))]

    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "abs_importance_mean": np.abs(result.importances_mean),
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return perm_df


# =========================================================
# AGRÉGATION PAR BLOCS
# =========================================================

def summarize_importance_by_feature_group(
    importance_df: pd.DataFrame,
    feature_col: str = "feature",
    value_col: str = "importance",
) -> pd.DataFrame:
    """
    Regroupe les importances en blocs :
    - numeric
    - categorical
    - text
    - other
    """
    if importance_df.empty:
        return pd.DataFrame()

    df = importance_df.copy()

    def detect_group(feature_name: str) -> str:
        if _is_text_feature_name(feature_name):
            return "text"
        if _is_numeric_feature_name(feature_name):
            return "numeric"
        if _is_categorical_feature_name(feature_name):
            return "categorical"
        return "other"

    df["feature_group"] = df[feature_col].map(detect_group)
    df["abs_value"] = np.abs(pd.to_numeric(df[value_col], errors="coerce"))

    grouped = (
        df.groupby("feature_group", dropna=False)["abs_value"]
        .agg(["count", "sum", "mean", "max"])
        .reset_index()
        .sort_values("sum", ascending=False)
        .reset_index(drop=True)
    )

    grouped.rename(
        columns={
            "count": "n_features",
            "sum": "total_importance",
            "mean": "mean_importance",
            "max": "max_importance",
        },
        inplace=True,
    )

    return grouped


# =========================================================
# TOP FEATURES
# =========================================================

def get_top_features(
    importance_df: pd.DataFrame,
    n: int = 20,
    value_col: str | None = None,
) -> pd.DataFrame:
    """
    Retourne les top features.
    """
    if importance_df.empty:
        return pd.DataFrame()

    df = importance_df.copy()

    if value_col is None:
        if "abs_coefficient" in df.columns:
            value_col = "abs_coefficient"
        elif "abs_importance" in df.columns:
            value_col = "abs_importance"
        elif "abs_importance_mean" in df.columns:
            value_col = "abs_importance_mean"
        elif "importance" in df.columns:
            value_col = "importance"
        else:
            value_col = df.columns[-1]

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.sort_values(value_col, ascending=False).head(n).reset_index(drop=True)

    return df


def get_top_text_features(
    importance_df: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """
    Filtre les top features texte.
    """
    if importance_df.empty or "feature" not in importance_df.columns:
        return pd.DataFrame()

    df = importance_df[importance_df["feature"].map(_is_text_feature_name)].copy()
    if df.empty:
        return df

    return get_top_features(df, n=n)


def get_top_numeric_features(
    importance_df: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """
    Filtre les top features numériques.
    """
    if importance_df.empty or "feature" not in importance_df.columns:
        return pd.DataFrame()

    df = importance_df[importance_df["feature"].map(_is_numeric_feature_name)].copy()
    if df.empty:
        return df

    return get_top_features(df, n=n)


def get_top_categorical_features(
    importance_df: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """
    Filtre les top features catégorielles.
    """
    if importance_df.empty or "feature" not in importance_df.columns:
        return pd.DataFrame()

    df = importance_df[importance_df["feature"].map(_is_categorical_feature_name)].copy()
    if df.empty:
        return df

    return get_top_features(df, n=n)


# =========================================================
# RÉSUMÉ GLOBAL
# =========================================================

def build_interpretability_summary(
    pipeline: Pipeline,
    X=None,
    y=None,
    permutation_scoring: str = "f1",
    permutation_n_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Produit un résumé d'interprétabilité complet.
    """
    model_importances = extract_model_importances(pipeline)
    grouped_model_importances = pd.DataFrame()
    permutation_df = pd.DataFrame()
    grouped_permutation = pd.DataFrame()

    if not model_importances.empty:
        if "coefficient" in model_importances.columns:
            grouped_model_importances = summarize_importance_by_feature_group(
                model_importances,
                feature_col="feature",
                value_col="coefficient",
            )
        elif "importance" in model_importances.columns:
            grouped_model_importances = summarize_importance_by_feature_group(
                model_importances,
                feature_col="feature",
                value_col="importance",
            )

    if X is not None and y is not None:
        try:
            permutation_df = compute_permutation_importance_df(
                pipeline=pipeline,
                X=X,
                y=y,
                scoring=permutation_scoring,
                n_repeats=permutation_n_repeats,
                random_state=random_state,
            )
            grouped_permutation = summarize_importance_by_feature_group(
                permutation_df,
                feature_col="feature",
                value_col="importance_mean",
            )
        except Exception:
            permutation_df = pd.DataFrame()
            grouped_permutation = pd.DataFrame()

    return {
        "feature_names": get_transformed_feature_names(pipeline),
        "model_importances": model_importances,
        "model_importances_by_group": grouped_model_importances,
        "top_model_features": get_top_features(model_importances, n=20) if not model_importances.empty else pd.DataFrame(),
        "top_text_features": get_top_text_features(model_importances, n=20) if not model_importances.empty else pd.DataFrame(),
        "top_numeric_features": get_top_numeric_features(model_importances, n=20) if not model_importances.empty else pd.DataFrame(),
        "top_categorical_features": get_top_categorical_features(model_importances, n=20) if not model_importances.empty else pd.DataFrame(),
        "permutation_importances": permutation_df,
        "permutation_importances_by_group": grouped_permutation,
        "top_permutation_features": get_top_features(permutation_df, n=20) if not permutation_df.empty else pd.DataFrame(),
    }


if __name__ == "__main__":
    from sklearn.pipeline import Pipeline

    from src.config import DATASET_CONFIG, RANDOM_STATE
    from src.data_generation import generate_dataset
    from src.models import get_models
    from src.preprocessing import build_preprocessor, split_features_target

    df = generate_dataset(**DATASET_CONFIG)
    X, y = split_features_target(df)

    model_name = "logistic_regression"
    model = get_models(random_state=RANDOM_STATE, selected_models=[model_name])[model_name]
    preprocessor = build_preprocessor(X, preprocessor_name="basic_tfidf_robust")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    summary = build_interpretability_summary(
        pipeline=pipeline,
        X=X,
        y=y,
        permutation_scoring="f1",
        permutation_n_repeats=3,
        random_state=RANDOM_STATE,
    )

    print("Top model features:")
    print(summary["top_model_features"].head(10))

    if not summary["permutation_importances"].empty:
        print("\nTop permutation features:")
        print(summary["top_permutation_features"].head(10))

    if not summary["model_importances_by_group"].empty:
        print("\nImportance par groupe de features:")
        print(summary["model_importances_by_group"])