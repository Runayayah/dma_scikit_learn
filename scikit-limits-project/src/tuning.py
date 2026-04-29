from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    PRIMARY_RANKING_METRIC,
    RANDOM_STATE,
    TUNING_CONFIG,
)
from src.evaluation import choose_preprocessor_for_model
from src.models import get_models
from src.preprocessing import build_preprocessor


# =========================================================
# ESPACES DE RECHERCHE
# =========================================================

def get_param_grids() -> dict[str, dict[str, list[Any]]]:
    """
    Grilles de recherche pour GridSearchCV.
    Les noms de paramètres ciblent le step 'model' du Pipeline.
    """
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "svc_rbf": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"],
        },
        "mlp": {
            "model__hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01],
        },
        "ridge_classifier": {
            "model__alpha": [0.1, 1.0, 10.0],
        },
        "decision_tree": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "extra_trees": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
        },
        "hist_gradient_boosting": {
            "model__max_iter": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [None, 5, 10],
        },
    }


def get_param_distributions() -> dict[str, dict[str, list[Any]]]:
    """
    Distributions simples pour RandomizedSearchCV.
    Ici on utilise des listes discrètes pour rester sans scipy.stats.
    """
    return {
        "logistic_regression": {
            "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": [100, 150, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": [2, 3, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        "svc_rbf": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__gamma": ["scale", "auto"],
        },
        "mlp": {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            "model__alpha": [0.0001, 0.001, 0.01, 0.1],
            "model__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
        },
        "ridge_classifier": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "decision_tree": {
            "model__max_depth": [None, 3, 5, 10, 20, 30],
            "model__min_samples_split": [2, 3, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
        },
        "extra_trees": {
            "model__n_estimators": [100, 150, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": [2, 3, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        },
        "gradient_boosting": {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
        },
        "hist_gradient_boosting": {
            "model__max_iter": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [None, 3, 5, 10],
        },
    }


# =========================================================
# HELPERS
# =========================================================

def get_cv(seed: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=TUNING_CONFIG["cv_folds"],
        shuffle=True,
        random_state=seed,
    )


def get_scoring_metric() -> str:
    """
    Convertit la métrique primaire du projet vers un scoring sklearn standard.
    """
    mapping = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    return mapping.get(TUNING_CONFIG["refit_metric"], "f1")


def build_model_pipeline(
    df: pd.DataFrame,
    model_name: str,
    estimator,
    preferred_preprocessor: str | None = None,
) -> tuple[Pipeline, str]:
    """
    Construit le pipeline sklearn complet pour un modèle donné.
    """
    preprocessor_name = choose_preprocessor_for_model(
        model_name=model_name,
        preferred_preprocessor=preferred_preprocessor,
    )

    preprocessor = build_preprocessor(
        df=df,
        preprocessor_name=preprocessor_name,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(estimator)),
        ]
    )

    return pipeline, preprocessor_name


# =========================================================
# TUNING PAR MODÈLE
# =========================================================

def run_grid_search_for_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    estimator,
    preferred_preprocessor: str | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Lance un GridSearchCV pour un modèle.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    param_grids = get_param_grids()
    if model_name not in param_grids:
        return {
            "model": model_name,
            "status": "skipped",
            "reason": "no_grid_defined",
        }

    pipeline, preprocessor_name = build_model_pipeline(
        df=X,
        model_name=model_name,
        estimator=estimator,
        preferred_preprocessor=preferred_preprocessor,
    )

    cv = get_cv(seed=seed)
    scoring = get_scoring_metric()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=True,
    )

    start = time.perf_counter()
    search.fit(X, y)
    duration = time.perf_counter() - start

    return {
        "model": model_name,
        "preprocessor": preprocessor_name,
        "search_type": "grid",
        "status": "ok",
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "duration_sec": duration,
        "n_candidates": len(search.cv_results_["params"]),
        "refit_metric": scoring,
    }


def run_random_search_for_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    estimator,
    preferred_preprocessor: str | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Lance un RandomizedSearchCV pour un modèle.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    param_distributions = get_param_distributions()
    if model_name not in param_distributions:
        return {
            "model": model_name,
            "status": "skipped",
            "reason": "no_distribution_defined",
        }

    pipeline, preprocessor_name = build_model_pipeline(
        df=X,
        model_name=model_name,
        estimator=estimator,
        preferred_preprocessor=preferred_preprocessor,
    )

    cv = get_cv(seed=seed)
    scoring = get_scoring_metric()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions[model_name],
        n_iter=TUNING_CONFIG["n_iter_random_search"],
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=1,
        random_state=seed,
        return_train_score=True,
    )

    start = time.perf_counter()
    search.fit(X, y)
    duration = time.perf_counter() - start

    return {
        "model": model_name,
        "preprocessor": preprocessor_name,
        "search_type": "random",
        "status": "ok",
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "duration_sec": duration,
        "n_candidates": len(search.cv_results_["params"]),
        "refit_metric": scoring,
    }


# =========================================================
# COMPARAISON AVANT / APRÈS
# =========================================================

def compare_tuning_results(tuning_results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Construit un tableau de synthèse des résultats de tuning.
    """
    rows = []

    for result in tuning_results:
        row = {
            "model": result.get("model"),
            "preprocessor": result.get("preprocessor"),
            "search_type": result.get("search_type"),
            "status": result.get("status"),
            "best_score": result.get("best_score"),
            "duration_sec": result.get("duration_sec"),
            "n_candidates": result.get("n_candidates"),
            "refit_metric": result.get("refit_metric"),
            "best_params": str(result.get("best_params")),
            "reason": result.get("reason"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty and "best_score" in df.columns:
        df = df.sort_values("best_score", ascending=False, na_position="last").reset_index(drop=True)

    return df


# =========================================================
# ORCHESTRATEUR PRINCIPAL
# =========================================================

def run_tuning(
    df: pd.DataFrame,
    target_col: str = "target",
    model_names: list[str] | None = None,
    search_methods: list[str] | None = None,
    preferred_preprocessor: str | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Lance le tuning sur une liste de modèles et de méthodes.
    """
    all_models = get_models(random_state=seed)

    if model_names is None:
        model_names = TUNING_CONFIG["models_to_tune"]

    if search_methods is None:
        search_methods = TUNING_CONFIG["search_methods"]

    results: list[dict[str, Any]] = []

    for model_name in model_names:
        if model_name not in all_models:
            results.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "reason": "model_not_found",
                }
            )
            continue

        estimator = all_models[model_name]

        for search_method in search_methods:
            try:
                if search_method == "grid":
                    result = run_grid_search_for_model(
                        df=df,
                        target_col=target_col,
                        model_name=model_name,
                        estimator=estimator,
                        preferred_preprocessor=preferred_preprocessor,
                        seed=seed,
                    )
                elif search_method == "random":
                    result = run_random_search_for_model(
                        df=df,
                        target_col=target_col,
                        model_name=model_name,
                        estimator=estimator,
                        preferred_preprocessor=preferred_preprocessor,
                        seed=seed,
                    )
                else:
                    result = {
                        "model": model_name,
                        "search_type": search_method,
                        "status": "skipped",
                        "reason": "unknown_search_method",
                    }
            except Exception as e:
                result = {
                    "model": model_name,
                    "search_type": search_method,
                    "status": "error",
                    "reason": str(e),
                }

            results.append(result)

    summary_df = compare_tuning_results(results)
    return results, summary_df


# =========================================================
# EXPORT
# =========================================================

def export_tuning_results(
    tuning_results: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    output_dir,
    prefix: str = "tuning",
) -> None:
    """
    Exporte le résumé et les CV results détaillés.
    """
    output_dir = pd.io.common.stringify_path(output_dir)

    summary_df.to_csv(f"{output_dir}/{prefix}_summary.csv", index=False)

    for result in tuning_results:
        if result.get("status") != "ok":
            continue

        model = result.get("model", "unknown")
        search_type = result.get("search_type", "unknown")

        cv_results = result.get("cv_results")
        if isinstance(cv_results, pd.DataFrame):
            cv_results.to_csv(
                f"{output_dir}/{prefix}_{model}_{search_type}_cv_results.csv",
                index=False,
            )


if __name__ == "__main__":
    from src.config import DATASET_CONFIG
    from src.data_generation import generate_dataset

    df = generate_dataset(**DATASET_CONFIG)

    tuning_results, summary_df = run_tuning(
        df=df,
        target_col="target",
        model_names=["logistic_regression", "random_forest"],
        search_methods=["grid", "random"],
        seed=RANDOM_STATE,
    )

    print("Résumé tuning :")
    print(summary_df.head())

    for result in tuning_results:
        if result.get("status") == "ok":
            print(
                f"\nModel={result['model']} | search={result['search_type']} | "
                f"best_score={result['best_score']:.4f}"
            )
            print("Best params:", result["best_params"])