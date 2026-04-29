from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.pipeline import Pipeline

from src.config import DATASET_CONFIG, FIGURES_DIR, RANDOM_STATE, TARGET_COLUMN
from src.data_generation import generate_dataset
from src.models import get_models
from src.preprocessing import build_preprocessor, split_features_target


def run_light_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    X, y = split_features_target(df, TARGET_COLUMN)

    models = get_models(
        selected_models=["logistic_regression", "random_forest"],
        random_state=RANDOM_STATE,
    )

    param_grids = {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "random_forest": {
            "model__max_depth": [5, 10, None],
            "model__n_estimators": [100, 200],
        },
    }

    rows = []

    for model_name, model in models.items():
        preprocessor = build_preprocessor(X, preprocessor_name="standard")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )

        search.fit(X, y)

        rows.append(
            {
                "model": model_name,
                "best_score_f1": search.best_score_,
                "best_params": search.best_params_,
            }
        )

    return pd.DataFrame(rows)


def run_calibration_analysis(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = get_models(
        selected_models=["logistic_regression"],
        random_state=RANDOM_STATE,
    )["logistic_regression"]

    preprocessor = build_preprocessor(X_train, preprocessor_name="standard")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=model,
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Modèle calibré")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Calibration parfaite")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence observée")
    plt.title("Courbe de calibration - risque client")
    plt.legend()

    path = output_dir / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path


def run_permutation_importance(df: pd.DataFrame) -> pd.DataFrame:
    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = get_models(
        selected_models=["random_forest"],
        random_state=RANDOM_STATE,
    )["random_forest"]

    preprocessor = build_preprocessor(X_train, preprocessor_name="standard")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    result = permutation_importance(
        estimator=pipeline,
        X=X_test,
        y=y_test,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df


def run_learning_curve_analysis(
    df: pd.DataFrame,
    output_dir: str | Path = FIGURES_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = split_features_target(df, TARGET_COLUMN)

    model = get_models(
        selected_models=["logistic_regression"],
        random_state=RANDOM_STATE,
    )["logistic_regression"]

    preprocessor = build_preprocessor(X, preprocessor_name="standard")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        cv=3,
        scoring="f1",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Train")
    plt.plot(train_sizes, test_mean, marker="o", label="Validation")
    plt.xlabel("Nombre d'exemples d'entraînement")
    plt.ylabel("F1")
    plt.title("Learning curve - Logistic Regression")
    plt.legend()

    path = output_dir / "learning_curve_logistic_regression.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path


def run_roc_pr_curves(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = get_models(
        selected_models=["logistic_regression"],
        random_state=RANDOM_STATE,
    )["logistic_regression"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train, preprocessor_name="standard")),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    paths = []

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("Courbe ROC - Logistic Regression")
    roc_path = output_dir / "roc_curve_logistic_regression.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(roc_path)

    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("Courbe Precision-Recall - Logistic Regression")
    pr_path = output_dir / "precision_recall_curve_logistic_regression.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(pr_path)

    return paths


def run_advanced_analysis() -> dict:
    df = generate_dataset(**DATASET_CONFIG)

    grid_results = run_light_grid_search(df)
    permutation_df = run_permutation_importance(df)

    calibration_path = run_calibration_analysis(df)
    learning_curve_path = run_learning_curve_analysis(df)
    curve_paths = run_roc_pr_curves(df)

    return {
        "grid_search_results": grid_results,
        "permutation_importance": permutation_df,
        "calibration_curve": calibration_path,
        "learning_curve": learning_curve_path,
        "roc_pr_curves": curve_paths,
    }


if __name__ == "__main__":
    results = run_advanced_analysis()

    print("\nGridSearch léger :")
    print(results["grid_search_results"])

    print("\nPermutation importance :")
    print(results["permutation_importance"].head(15))

    print("\nFigures générées :")
    print("-", results["calibration_curve"])
    print("-", results["learning_curve"])

    for path in results["roc_pr_curves"]:
        print("-", path)