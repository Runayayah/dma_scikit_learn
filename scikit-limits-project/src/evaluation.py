from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ACTIVE_METRICS,
    CALIBRATION_CONFIG,
    CV_FOLDS,
    N_JOBS,
    PRIMARY_RANKING_METRIC,
    RANDOM_STATE,
    THRESHOLD_ANALYSIS_CONFIG,
)
from src.models import (
    DENSE_ONLY_MODELS,
    NON_NEGATIVE_FEATURE_MODELS,
    get_models,
    get_model_family_map,
)
from src.preprocessing import (
    build_preprocessor,
    is_dense_preprocessor,
    is_non_negative_preprocessor,
)


# =========================================================
# OUTILS DE SÉLECTION DE PIPELINE
# =========================================================

def choose_preprocessor_for_model(
    model_name: str,
    preferred_preprocessor: str | None = None,
) -> str:
    """
    Choisit automatiquement un preprocessing compatible avec le modèle.
    """
    if preferred_preprocessor is not None:
        if model_name in DENSE_ONLY_MODELS and not is_dense_preprocessor(preferred_preprocessor):
            if model_name == "hist_gradient_boosting" or model_name == "gaussian_nb":
                return "dense_standard_ordinal"

        if model_name in NON_NEGATIVE_FEATURE_MODELS and not is_non_negative_preprocessor(preferred_preprocessor):
            return "non_negative_count_minmax"

        return preferred_preprocessor

    if model_name in DENSE_ONLY_MODELS:
        return "dense_standard_ordinal"

    if model_name in NON_NEGATIVE_FEATURE_MODELS:
        return "non_negative_count_minmax"

    return "basic_tfidf_robust"


# =========================================================
# OUTILS DE SCORES / PROBA
# =========================================================

def safe_predict_scores(model: Any, X):
    """
    Retourne un score continu si possible :
    - predict_proba[:, 1]
    - decision_function
    - sinon None
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
            scores = model.decision_function(X)
            return scores
        except Exception:
            pass

    return None


def safe_predict_proba_positive_class(model: Any, X):
    """
    Retourne la probabilité de la classe positive si disponible.
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
        except Exception:
            pass
    return None


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applique un seuil à un score continu.
    """
    return (scores >= threshold).astype(int)


# =========================================================
# MÉTRIQUES
# =========================================================

def compute_classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    y_proba=None,
) -> dict[str, Any]:
    """
    Calcule un ensemble riche de métriques de classification binaire.
    """
    metrics: dict[str, Any] = {}

    if ACTIVE_METRICS.get("accuracy", True):
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if ACTIVE_METRICS.get("balanced_accuracy", True):
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    if ACTIVE_METRICS.get("precision", True):
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)

    if ACTIVE_METRICS.get("recall", True):
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    if ACTIVE_METRICS.get("f1", True):
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if ACTIVE_METRICS.get("mcc", True):
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    if ACTIVE_METRICS.get("cohen_kappa", True):
        metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    if y_score is not None:
        if ACTIVE_METRICS.get("roc_auc", True):
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            except Exception:
                metrics["roc_auc"] = np.nan

        if ACTIVE_METRICS.get("average_precision", True):
            try:
                metrics["average_precision"] = average_precision_score(y_true, y_score)
            except Exception:
                metrics["average_precision"] = np.nan

    if y_proba is not None:
        if ACTIVE_METRICS.get("log_loss", True):
            try:
                p = np.clip(y_proba, 1e-8, 1 - 1e-8)
                metrics["log_loss"] = log_loss(y_true, np.column_stack([1 - p, p]))
            except Exception:
                metrics["log_loss"] = np.nan

        if ACTIVE_METRICS.get("brier_score", True):
            try:
                metrics["brier_score"] = brier_score_loss(y_true, y_proba)
            except Exception:
                metrics["brier_score"] = np.nan

    return metrics


def aggregate_fold_results(raw_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les résultats fold-level en moyenne / std / min / max.
    """
    if raw_results_df.empty:
        return pd.DataFrame()

    metadata_cols = [
        "model",
        "family",
        "preprocessor",
        "seed",
        "status",
    ]

    metric_cols = [
        col for col in raw_results_df.columns
        if col not in metadata_cols + ["fold", "warnings", "error"]
    ]

    grouped = raw_results_df.groupby(
        ["model", "family", "preprocessor", "seed", "status"],
        dropna=False,
    )

    rows = []
    for keys, group in grouped:
        row = {
            "model": keys[0],
            "family": keys[1],
            "preprocessor": keys[2],
            "seed": keys[3],
            "status": keys[4],
            "n_folds": len(group),
        }

        for col in metric_cols:
            series = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = series.mean()
            row[f"{col}_std"] = series.std()
            row[f"{col}_min"] = series.min()
            row[f"{col}_max"] = series.max()

        rows.append(row)

    aggregated_df = pd.DataFrame(rows)

    ranking_col = f"{PRIMARY_RANKING_METRIC}_mean"
    if ranking_col in aggregated_df.columns:
        aggregated_df = aggregated_df.sort_values(
            by=ranking_col,
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return aggregated_df


# =========================================================
# THRESHOLD / COÛT MÉTIER
# =========================================================

def find_best_threshold(
    y_true,
    y_score,
    metric: str = "f1",
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    grid_size: int = 101,
) -> dict[str, float]:
    """
    Cherche le meilleur seuil selon une métrique simple.
    """
    thresholds = np.linspace(0.0, 1.0, grid_size)

    best_threshold = 0.5
    best_value = -np.inf
    best_cost = np.inf

    for threshold in thresholds:
        y_pred = apply_threshold(y_score, threshold)

        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, y_pred)
        else:
            value = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cost = fp_cost * fp + fn_cost * fn

        if value > best_value or (np.isclose(value, best_value) and cost < best_cost):
            best_value = value
            best_threshold = threshold
            best_cost = cost

    return {
        "best_threshold": best_threshold,
        "best_metric_value": best_value,
        "business_cost": best_cost,
    }


# =========================================================
# CALIBRATION
# =========================================================

def maybe_calibrate_model(model, method: str = "sigmoid", cv: int = 3):
    """
    Calibre un modèle si possible.
    """
    try:
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=cv,
        )
        return calibrated
    except Exception:
        return model


# =========================================================
# MESURES TECHNIQUES
# =========================================================

def estimate_matrix_characteristics(X_transformed) -> dict[str, Any]:
    """
    Donne quelques infos sur la matrice transformée.
    """
    result = {
        "n_rows_transformed": None,
        "n_cols_transformed": None,
        "matrix_density": None,
        "is_sparse": False,
    }

    try:
        result["n_rows_transformed"] = X_transformed.shape[0]
        result["n_cols_transformed"] = X_transformed.shape[1]

        from scipy import sparse

        if sparse.issparse(X_transformed):
            result["is_sparse"] = True
            total = X_transformed.shape[0] * X_transformed.shape[1]
            nnz = X_transformed.nnz
            result["matrix_density"] = nnz / total if total > 0 else np.nan
        else:
            result["is_sparse"] = False
            total = X_transformed.size
            nnz = np.count_nonzero(X_transformed)
            result["matrix_density"] = nnz / total if total > 0 else np.nan
    except Exception:
        pass

    return result


# =========================================================
# ÉVALUATION D'UN MODÈLE SUR 1 FOLD
# =========================================================

def evaluate_single_fold(
    model_name: str,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor_name: str,
    calibrate: bool = False,
) -> dict[str, Any]:
    """
    Évalue un modèle sur un fold.
    """
    result: dict[str, Any] = {
        "model": model_name,
        "family": get_model_family_map().get(model_name, "unknown"),
        "preprocessor": preprocessor_name,
        "status": "ok",
        "warnings": "",
        "error": "",
    }

    caught_warning_messages = []

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        try:
            preprocessor = build_preprocessor(X_train, preprocessor_name=preprocessor_name)
            model = clone(estimator)

            if calibrate and CALIBRATION_CONFIG["enabled"]:
                model = maybe_calibrate_model(
                    model,
                    method=CALIBRATION_CONFIG["method"],
                    cv=CALIBRATION_CONFIG["cv"],
                )

            preprocess_start = time.perf_counter()
            X_train_transformed = preprocessor.fit_transform(X_train, y_train)
            preprocess_fit_transform_time = time.perf_counter() - preprocess_start

            X_test_transform_start = time.perf_counter()
            X_test_transformed = preprocessor.transform(X_test)
            preprocess_test_transform_time = time.perf_counter() - X_test_transform_start

            matrix_info = estimate_matrix_characteristics(X_train_transformed)

            fit_start = time.perf_counter()
            model.fit(X_train_transformed, y_train)
            fit_time = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            y_pred = model.predict(X_test_transformed)
            predict_time = time.perf_counter() - predict_start

            y_score = safe_predict_scores(model, X_test_transformed)
            y_proba = safe_predict_proba_positive_class(model, X_test_transformed)

            metrics = compute_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                y_proba=y_proba,
            )

            result.update(metrics)
            result["fit_time_sec"] = fit_time
            result["predict_time_sec"] = predict_time
            result["preprocess_fit_transform_time_sec"] = preprocess_fit_transform_time
            result["preprocess_test_transform_time_sec"] = preprocess_test_transform_time
            result["n_rows_transformed"] = matrix_info["n_rows_transformed"]
            result["n_cols_transformed"] = matrix_info["n_cols_transformed"]
            result["matrix_density"] = matrix_info["matrix_density"]
            result["is_sparse"] = matrix_info["is_sparse"]

            # threshold analysis
            if THRESHOLD_ANALYSIS_CONFIG["enabled"] and y_score is not None:
                threshold_info = find_best_threshold(
                    y_true=y_test,
                    y_score=np.asarray(y_score),
                    metric=PRIMARY_RANKING_METRIC,
                    fp_cost=THRESHOLD_ANALYSIS_CONFIG["fp_cost"],
                    fn_cost=THRESHOLD_ANALYSIS_CONFIG["fn_cost"],
                    grid_size=THRESHOLD_ANALYSIS_CONFIG["threshold_grid_size"],
                )
                result.update(threshold_info)

            # rapports textuels simples
            try:
                result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
            except Exception:
                result["confusion_matrix"] = None

            try:
                result["classification_report"] = classification_report(
                    y_test,
                    y_pred,
                    zero_division=0,
                    output_dict=False,
                )
            except Exception:
                result["classification_report"] = None

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        for w in caught_warnings:
            warning_message = f"{w.category.__name__}: {str(w.message)}"
            caught_warning_messages.append(warning_message)

    result["warnings"] = " | ".join(caught_warning_messages)

    return result


# =========================================================
# ÉVALUATION GLOBALE
# =========================================================

def evaluate_models_cv(
    df: pd.DataFrame,
    target_col: str = "target",
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
    calibrate_probabilities: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Évalue plusieurs modèles avec validation croisée stratifiée.

    Retourne :
    - raw_results_df : un résultat par fold et par modèle
    - aggregated_results_df : résultats agrégés
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    models = get_models(random_state=seed, selected_models=selected_models)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    raw_results: list[dict[str, Any]] = []

    for model_name, estimator in models.items():
        if preferred_preprocessors:
            preprocessors_to_try = preferred_preprocessors
        else:
            preprocessors_to_try = [choose_preprocessor_for_model(model_name)]

        for preprocessor_name in preprocessors_to_try:
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                X_train = X.iloc[train_idx].copy()
                X_test = X.iloc[test_idx].copy()
                y_train = y.iloc[train_idx].copy()
                y_test = y.iloc[test_idx].copy()

                fold_result = evaluate_single_fold(
                    model_name=model_name,
                    estimator=estimator,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    preprocessor_name=preprocessor_name,
                    calibrate=calibrate_probabilities,
                )
                fold_result["fold"] = fold_idx
                fold_result["seed"] = seed

                raw_results.append(fold_result)

    raw_results_df = pd.DataFrame(raw_results)
    aggregated_results_df = aggregate_fold_results(raw_results_df)

    return raw_results_df, aggregated_results_df


def summarize_evaluation_results(aggregated_results_df: pd.DataFrame) -> dict[str, Any]:
    """
    Résumé compact des meilleurs résultats.
    """
    if aggregated_results_df.empty:
        return {
            "best_model": None,
            "best_preprocessor": None,
            "best_score": None,
        }

    ranking_col = f"{PRIMARY_RANKING_METRIC}_mean"
    if ranking_col not in aggregated_results_df.columns:
        ranking_col = "f1_mean"

    best_row = aggregated_results_df.iloc[0]

    return {
        "best_model": best_row.get("model"),
        "best_family": best_row.get("family"),
        "best_preprocessor": best_row.get("preprocessor"),
        "best_score": best_row.get(ranking_col),
        "ranking_metric": ranking_col,
    }


if __name__ == "__main__":
    from src.data_generation import generate_dataset
    from src.config import DATASET_CONFIG

    df = generate_dataset(**DATASET_CONFIG)

    raw_df, agg_df = evaluate_models_cv(
        df=df,
        target_col="target",
        seed=RANDOM_STATE,
    )

    print("\n=== RAW RESULTS ===")
    print(raw_df.head())

    print("\n=== AGGREGATED RESULTS ===")
    print(agg_df.head())

    print("\n=== SUMMARY ===")
    print(summarize_evaluation_results(agg_df))