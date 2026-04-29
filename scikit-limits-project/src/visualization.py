from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    roc_curve,
)


# =========================================================
# HELPERS
# =========================================================

def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _safe_savefig(output_dir: str | Path, filename: str) -> Path:
    output_path = ensure_output_dir(output_dir)
    full_path = output_path / filename
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches="tight", dpi=150)
    plt.close()
    return full_path


def _prepare_top_n(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=[metric])
    plot_df = plot_df.sort_values(metric, ascending=False).head(top_n)
    return plot_df


# =========================================================
# LEADERBOARD / BARPLOTS
# =========================================================

def plot_leaderboard_bar(
    leaderboard_df: pd.DataFrame,
    metric: str = "f1_mean",
    top_n: int = 15,
    output_dir: str | Path = "data/outputs/figures",
    filename: str | None = None,
) -> Path | None:
    """
    Barplot des meilleurs modèles sur une métrique.
    """
    plot_df = _prepare_top_n(leaderboard_df, metric, top_n)
    if plot_df.empty:
        return None

    labels = [
        f"{row['model']}\n({row['preprocessor']})"
        if "preprocessor" in plot_df.columns else str(row["model"])
        for _, row in plot_df.iterrows()
    ]
    values = plot_df[metric].to_numpy()

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Top {len(values)} modèles selon {metric}")

    filename = filename or f"leaderboard_{metric}.png"
    return _safe_savefig(output_dir, filename)


def plot_metric_histogram(
    aggregated_df: pd.DataFrame,
    metric: str = "f1_mean",
    output_dir: str | Path = "data/outputs/figures",
    filename: str | None = None,
) -> Path | None:
    """
    Histogramme d'une métrique.
    """
    if aggregated_df.empty or metric not in aggregated_df.columns:
        return None

    values = pd.to_numeric(aggregated_df[metric], errors="coerce").dropna()
    if values.empty:
        return None

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=20)
    plt.xlabel(metric)
    plt.ylabel("Fréquence")
    plt.title(f"Distribution de {metric}")

    filename = filename or f"hist_{metric}.png"
    return _safe_savefig(output_dir, filename)


# =========================================================
# HEATMAPS
# =========================================================

def plot_heatmap_model_preprocessor(
    aggregated_df: pd.DataFrame,
    metric: str = "f1_mean",
    output_dir: str | Path = "data/outputs/figures",
    filename: str | None = None,
) -> Path | None:
    """
    Heatmap modèle x preprocessor.
    """
    required = {"model", "preprocessor", metric}
    if aggregated_df.empty or not required.issubset(aggregated_df.columns):
        return None

    pivot_df = aggregated_df.pivot_table(
        index="model",
        columns="preprocessor",
        values=metric,
        aggfunc="mean",
    )

    if pivot_df.empty:
        return None

    plt.figure(figsize=(12, 8))
    plt.imshow(pivot_df, aspect="auto")
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.colorbar(label=metric)
    plt.title(f"Heatmap modèle × preprocessor ({metric})")

    filename = filename or f"heatmap_model_preprocessor_{metric}.png"
    return _safe_savefig(output_dir, filename)


def plot_heatmap_model_scenario(
    raw_results_df: pd.DataFrame,
    scenario_col: str = "n_samples",
    metric: str = "f1",
    output_dir: str | Path = "data/outputs/figures",
    filename: str | None = None,
) -> Path | None:
    """
    Heatmap modèle x scénario.
    """
    required = {"model", scenario_col, metric}
    if raw_results_df.empty or not required.issubset(raw_results_df.columns):
        return None

    pivot_df = raw_results_df.pivot_table(
        index="model",
        columns=scenario_col,
        values=metric,
        aggfunc="mean",
    )

    if pivot_df.empty:
        return None

    plt.figure(figsize=(12, 8))
    plt.imshow(pivot_df, aspect="auto")
    plt.xticks(range(len(pivot_df.columns)), [str(c) for c in pivot_df.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.colorbar(label=metric)
    plt.title(f"Heatmap modèle × {scenario_col} ({metric})")

    filename = filename or f"heatmap_model_{scenario_col}_{metric}.png"
    return _safe_savefig(output_dir, filename)


# =========================================================
# SCALABILITÉ
# =========================================================

def plot_scalability_curve(
    raw_results_df: pd.DataFrame,
    x_col: str = "n_samples",
    y_col: str = "fit_time_sec",
    group_col: str = "model",
    top_n_models: int = 8,
    metric_for_selection: str = "f1",
    output_dir: str | Path = "data/outputs/figures",
    filename: str | None = None,
) -> Path | None:
    """
    Courbes de scalabilité : y en fonction de x, une courbe par modèle.
    """
    required = {x_col, y_col, group_col}
    if raw_results_df.empty or not required.issubset(raw_results_df.columns):
        return None

    selection = (
        raw_results_df.groupby(group_col)[metric_for_selection]
        .mean()
        .sort_values(ascending=False)
        .head(top_n_models)
        .index.tolist()
        if metric_for_selection in raw_results_df.columns
        else raw_results_df[group_col].dropna().unique()[:top_n_models]
    )

    plot_df = raw_results_df[raw_results_df[group_col].isin(selection)].copy()
    if plot_df.empty:
        return None

    plt.figure(figsize=(10, 6))
    for model_name, sub_df in plot_df.groupby(group_col):
        curve_df = sub_df.groupby(x_col)[y_col].mean().reset_index().sort_values(x_col)
        plt.plot(curve_df[x_col], curve_df[y_col], marker="o", label=model_name)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Scalabilité : {y_col} selon {x_col}")
    plt.legend()

    filename = filename or f"scalability_{y_col}_by_{x_col}.png"
    return _safe_savefig(output_dir, filename)


# =========================================================
# COURBES ROC / PR / CALIBRATION
# =========================================================

def plot_roc_curve_from_arrays(
    y_true,
    y_score,
    title: str = "ROC Curve",
    output_dir: str | Path = "data/outputs/figures",
    filename: str = "roc_curve.png",
) -> Path:
    """
    Courbe ROC à partir de y_true et y_score.
    """
    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    return _safe_savefig(output_dir, filename)


def plot_precision_recall_curve_from_arrays(
    y_true,
    y_score,
    title: str = "Precision-Recall Curve",
    output_dir: str | Path = "data/outputs/figures",
    filename: str = "pr_curve.png",
) -> Path:
    """
    Courbe PR à partir de y_true et y_score.
    """
    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    return _safe_savefig(output_dir, filename)


def plot_calibration_curve_from_arrays(
    y_true,
    y_proba,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    output_dir: str | Path = "data/outputs/figures",
    filename: str = "calibration_curve.png",
) -> Path:
    """
    Courbe de calibration à partir de y_true et y_proba.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence observée")
    plt.title(title)
    plt.legend()

    return _safe_savefig(output_dir, filename)


# =========================================================
# IMPORTANCE / COEFFS
# =========================================================

def plot_top_feature_importances(
    feature_names: Iterable[str],
    importances: Iterable[float],
    top_n: int = 20,
    title: str = "Top Feature Importances",
    output_dir: str | Path = "data/outputs/figures",
    filename: str = "feature_importances.png",
) -> Path | None:
    """
    Affiche les top importances.
    """
    feature_names = list(feature_names)
    importances = np.asarray(list(importances), dtype=float)

    if len(feature_names) == 0 or len(importances) == 0:
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(title)

    return _safe_savefig(output_dir, filename)


def plot_top_coefficients(
    feature_names: Iterable[str],
    coefficients: Iterable[float],
    top_n: int = 20,
    title: str = "Top Coefficients",
    output_dir: str | Path = "data/outputs/figures",
    filename: str = "top_coefficients.png",
) -> Path | None:
    """
    Affiche les plus grands coefficients positifs / négatifs.
    """
    feature_names = list(feature_names)
    coefficients = np.asarray(list(coefficients), dtype=float)

    if len(feature_names) == 0 or len(coefficients) == 0:
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    top_pos = df.sort_values("coefficient", ascending=False).head(top_n // 2)
    top_neg = df.sort_values("coefficient", ascending=True).head(top_n // 2)
    plot_df = pd.concat([top_neg, top_pos])

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.xlabel("Coefficient")
    plt.title(title)

    return _safe_savefig(output_dir, filename)


# =========================================================
# GÉNÉRATION AUTOMATIQUE DE FIGURES DEPUIS LES CSV
# =========================================================

def generate_standard_report_figures(
    raw_results_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    output_dir: str | Path = "data/outputs/figures",
) -> list[Path]:
    """
    Génère un set standard de figures à partir des résultats.
    """
    output_paths: list[Path] = []

    figure = plot_leaderboard_bar(
        leaderboard_df=leaderboard_df,
        metric="f1_mean",
        output_dir=output_dir,
        filename="leaderboard_f1.png",
    )
    if figure is not None:
        output_paths.append(figure)

    figure = plot_leaderboard_bar(
        leaderboard_df=leaderboard_df,
        metric="roc_auc_mean",
        output_dir=output_dir,
        filename="leaderboard_roc_auc.png",
    )
    if figure is not None:
        output_paths.append(figure)

    figure = plot_metric_histogram(
        aggregated_df=aggregated_df,
        metric="f1_mean",
        output_dir=output_dir,
        filename="hist_f1.png",
    )
    if figure is not None:
        output_paths.append(figure)

    figure = plot_heatmap_model_preprocessor(
        aggregated_df=aggregated_df,
        metric="f1_mean",
        output_dir=output_dir,
        filename="heatmap_model_preprocessor_f1.png",
    )
    if figure is not None:
        output_paths.append(figure)

    if "n_samples" in raw_results_df.columns:
        figure = plot_heatmap_model_scenario(
            raw_results_df=raw_results_df,
            scenario_col="n_samples",
            metric="f1",
            output_dir=output_dir,
            filename="heatmap_model_n_samples_f1.png",
        )
        if figure is not None:
            output_paths.append(figure)

        figure = plot_scalability_curve(
            raw_results_df=raw_results_df,
            x_col="n_samples",
            y_col="fit_time_sec",
            output_dir=output_dir,
            filename="scalability_fit_time_by_n_samples.png",
        )
        if figure is not None:
            output_paths.append(figure)

    if "vocab_size" in raw_results_df.columns:
        figure = plot_scalability_curve(
            raw_results_df=raw_results_df,
            x_col="vocab_size",
            y_col="f1",
            output_dir=output_dir,
            filename="scalability_f1_by_vocab_size.png",
        )
        if figure is not None:
            output_paths.append(figure)

    return output_paths


if __name__ == "__main__":
    # Exemple minimal de test
    dummy_leaderboard = pd.DataFrame(
        {
            "model": ["logreg", "random_forest", "linear_svc"],
            "preprocessor": ["basic_tfidf_robust", "dense_standard_ordinal", "sparse_tfidf_maxabs"],
            "f1_mean": [0.71, 0.76, 0.74],
            "roc_auc_mean": [0.80, 0.85, 0.83],
        }
    )

    dummy_raw = pd.DataFrame(
        {
            "model": ["logreg", "logreg", "random_forest", "random_forest"],
            "n_samples": [1000, 5000, 1000, 5000],
            "vocab_size": [50, 150, 50, 150],
            "f1": [0.68, 0.72, 0.75, 0.77],
            "fit_time_sec": [0.4, 1.2, 0.8, 2.0],
        }
    )

    dummy_agg = pd.DataFrame(
        {
            "model": ["logreg", "random_forest", "linear_svc"],
            "preprocessor": ["basic_tfidf_robust", "dense_standard_ordinal", "sparse_tfidf_maxabs"],
            "f1_mean": [0.71, 0.76, 0.74],
            "roc_auc_mean": [0.80, 0.85, 0.83],
        }
    )

    paths = generate_standard_report_figures(
        raw_results_df=dummy_raw,
        aggregated_df=dummy_agg,
        leaderboard_df=dummy_leaderboard,
    )

    print("Figures générées :")
    for path in paths:
        print("-", path)