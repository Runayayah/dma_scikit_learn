from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, DENSE_ONLY_MODELS, RANDOM_STATE, TARGET_COLUMN
from src.models import get_model_family, get_models
from src.preprocessing import build_preprocessor, split_features_target


def get_continuous_scores(model: Any, X):
    """
    Récupère un score continu pour ROC-AUC si possible.
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass

    return None


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """
    Calcule les métriques principales de classification binaire.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            results["roc_auc"] = np.nan
    else:
        results["roc_auc"] = np.nan

    return results


def choose_preprocessor_name(model_name: str) -> str:
    """
    Choisit un preprocessing adapté au modèle.
    """
    if model_name in DENSE_ONLY_MODELS:
        return "dense"

    return "standard"


def evaluate_models_cv(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    selected_models: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Évalue les modèles avec une cross-validation stratifiée.

    Retourne :
    - raw_results : une ligne par fold / modèle
    - aggregated_results : moyenne et écart-type par modèle
    """
    X, y = split_features_target(df, target_col=target_col)
    models = get_models(random_state=seed, selected_models=selected_models)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    raw_rows = []

    for model_name, estimator in models.items():
        preprocessor_name = choose_preprocessor_name(model_name)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            row = {
                "model": model_name,
                "family": get_model_family(model_name),
                "preprocessor": preprocessor_name,
                "fold": fold_idx,
                "seed": seed,
                "status": "ok",
                "error": "",
                "warnings": "",
            }

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", category=ConvergenceWarning)
                warnings.simplefilter("always", category=UserWarning)

                try:
                    preprocessor = build_preprocessor(
                        X_train,
                        preprocessor_name=preprocessor_name,
                    )

                    pipeline = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("model", clone(estimator)),
                        ]
                    )

                    start_fit = time.perf_counter()
                    pipeline.fit(X_train, y_train)
                    fit_time = time.perf_counter() - start_fit

                    start_predict = time.perf_counter()
                    y_pred = pipeline.predict(X_test)
                    predict_time = time.perf_counter() - start_predict

                    y_score = get_continuous_scores(pipeline, X_test)
                    metrics = compute_metrics(y_test, y_pred, y_score)

                    row.update(metrics)
                    row["fit_time_sec"] = fit_time
                    row["predict_time_sec"] = predict_time

                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)
                    row["accuracy"] = np.nan
                    row["balanced_accuracy"] = np.nan
                    row["precision"] = np.nan
                    row["recall"] = np.nan
                    row["f1"] = np.nan
                    row["roc_auc"] = np.nan
                    row["fit_time_sec"] = np.nan
                    row["predict_time_sec"] = np.nan

                if caught_warnings:
                    row["warnings"] = " | ".join(
                        f"{w.category.__name__}: {w.message}"
                        for w in caught_warnings
                    )

            raw_rows.append(row)

    raw_results = pd.DataFrame(raw_rows)
    aggregated_results = aggregate_results(raw_results)

    return raw_results, aggregated_results


def aggregate_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les résultats par modèle.
    """
    if raw_results.empty:
        return pd.DataFrame()

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fit_time_sec",
        "predict_time_sec",
    ]

    rows = []

    group_cols = ["model", "family", "preprocessor"]

    for keys, group in raw_results.groupby(group_cols, dropna=False):
        row = {
            "model": keys[0],
            "family": keys[1],
            "preprocessor": keys[2],
            "n_folds": len(group),
            "n_errors": int((group["status"] == "error").sum()),
        }

        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        rows.append(row)

    aggregated = pd.DataFrame(rows)

    if "f1_mean" in aggregated.columns:
        aggregated = aggregated.sort_values(
            by="f1_mean",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return aggregated


if __name__ == "__main__":
    from src.config import DATASET_CONFIG
    from src.data_generation import generate_dataset

    df = generate_dataset(**DATASET_CONFIG)

    raw, agg = evaluate_models_cv(df)

    print("\nRésultats bruts :")
    print(raw.head())

    print("\nRésultats agrégés :")
    print(agg)