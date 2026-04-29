from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================================================
# STYLE GLOBAL
# =========================================================

def _prepare_output_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _savefig(output_dir: str | Path, filename: str) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    path = output_dir / filename

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    return path


def _clean_model_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def _clean_scenario_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def _annotate_bars(ax, decimals: int = 3) -> None:
    for bar in ax.patches:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        if pd.notna(width):
            ax.text(
                width,
                y,
                f" {width:.{decimals}f}",
                va="center",
                fontsize=9,
            )


# =========================================================
# GRAPHIQUES GLOBAUX
# =========================================================

def plot_global_f1_leaderboard(
    leaderboard: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    if leaderboard.empty or "f1_mean" not in leaderboard.columns:
        return None

    plot_df = leaderboard.sort_values("f1_mean", ascending=True).copy()
    plot_df["model_label"] = plot_df["model"].map(_clean_model_name)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["model_label"], plot_df["f1_mean"])

    ax.set_xlabel("F1 moyen")
    ax.set_ylabel("Modèle")
    ax.set_title("Classement global des modèles selon le F1-score")
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax)

    return _savefig(output_dir, "01_global_f1_leaderboard.png")


def plot_global_balanced_accuracy(
    leaderboard: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    if leaderboard.empty or "balanced_accuracy_mean" not in leaderboard.columns:
        return None

    plot_df = leaderboard.sort_values("balanced_accuracy_mean", ascending=True).copy()
    plot_df["model_label"] = plot_df["model"].map(_clean_model_name)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["model_label"], plot_df["balanced_accuracy_mean"])

    ax.set_xlabel("Balanced accuracy moyenne")
    ax.set_ylabel("Modèle")
    ax.set_title("Comparaison globale des modèles - Balanced accuracy")
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax)

    return _savefig(output_dir, "02_global_balanced_accuracy.png")


def plot_global_training_time(
    leaderboard: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    if leaderboard.empty or "fit_time_sec_mean" not in leaderboard.columns:
        return None

    plot_df = leaderboard.sort_values("fit_time_sec_mean", ascending=True).copy()
    plot_df["model_label"] = plot_df["model"].map(_clean_model_name)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["model_label"], plot_df["fit_time_sec_mean"])

    ax.set_xlabel("Temps moyen d'entraînement par fold (secondes)")
    ax.set_ylabel("Modèle")
    ax.set_title("Coût d'entraînement moyen par modèle")
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax, decimals=2)

    return _savefig(output_dir, "03_global_training_time.png")


def plot_precision_recall_tradeoff(
    leaderboard: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    required = {"precision_mean", "recall_mean", "model"}
    if leaderboard.empty or not required.issubset(leaderboard.columns):
        return None

    plot_df = leaderboard.dropna(subset=["precision_mean", "recall_mean"]).copy()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(plot_df["recall_mean"], plot_df["precision_mean"], s=70)

    for _, row in plot_df.iterrows():
        ax.annotate(
            _clean_model_name(row["model"]),
            (row["recall_mean"], row["precision_mean"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Recall moyen")
    ax.set_ylabel("Precision moyenne")
    ax.set_title("Compromis precision / recall par modèle")
    ax.grid(alpha=0.25)

    return _savefig(output_dir, "04_global_precision_recall_tradeoff.png")


# =========================================================
# GRAPHIQUES PAR SCÉNARIO
# =========================================================

def plot_scenario_heatmap_readable(
    aggregated_results: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    required = {"model", "scenario", "f1_mean"}
    if aggregated_results.empty or not required.issubset(aggregated_results.columns):
        return None

    pivot = aggregated_results.pivot_table(
        index="model",
        columns="scenario",
        values="f1_mean",
        aggfunc="mean",
    )

    if pivot.empty:
        return None

    pivot = pivot.rename(index=_clean_model_name, columns=_clean_scenario_name)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("F1 moyen par modèle et par scénario")
    ax.set_xlabel("Scénario")
    ax.set_ylabel("Modèle")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="F1 moyen")

    return _savefig(output_dir, "05_scenario_heatmap_f1.png")


def plot_best_model_by_scenario(
    aggregated_results: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    required = {"model", "scenario", "f1_mean"}
    if aggregated_results.empty or not required.issubset(aggregated_results.columns):
        return None

    rows = []

    for scenario, group in aggregated_results.groupby("scenario"):
        best = group.sort_values("f1_mean", ascending=False).iloc[0]
        rows.append(
            {
                "scenario": _clean_scenario_name(scenario),
                "model": _clean_model_name(best["model"]),
                "f1_mean": best["f1_mean"],
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("f1_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = plot_df["scenario"] + "\n" + plot_df["model"]

    ax.barh(labels, plot_df["f1_mean"])
    ax.set_xlabel("Meilleur F1 moyen")
    ax.set_ylabel("Scénario / meilleur modèle")
    ax.set_title("Meilleur modèle pour chaque scénario")
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax)

    return _savefig(output_dir, "06_best_model_by_scenario.png")


# =========================================================
# UN GRAPHIQUE PAR MODÈLE
# =========================================================

def plot_model_f1_by_scenario(
    aggregated_results: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
) -> Path | None:
    required = {"model", "scenario", "f1_mean"}
    if aggregated_results.empty or not required.issubset(aggregated_results.columns):
        return None

    model_df = aggregated_results[aggregated_results["model"] == model_name].copy()
    if model_df.empty:
        return None

    plot_df = (
        model_df
        .groupby("scenario")["f1_mean"]
        .mean()
        .reset_index()
    )
    plot_df["scenario_label"] = plot_df["scenario"].map(_clean_scenario_name)
    plot_df = plot_df.sort_values("f1_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["scenario_label"], plot_df["f1_mean"])

    ax.set_xlabel("F1 moyen")
    ax.set_ylabel("Scénario")
    ax.set_title(f"{_clean_model_name(model_name)} - performance par scénario")
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax)

    filename = f"model_{model_name}_f1_by_scenario.png"
    return _savefig(output_dir, filename)


def plot_model_metrics_profile(
    leaderboard: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
) -> Path | None:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return None

    model_df = leaderboard[leaderboard["model"] == model_name]
    if model_df.empty:
        return None

    row = model_df.iloc[0]

    metrics = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    ]

    available_metrics = [m for m in metrics if m in leaderboard.columns and pd.notna(row.get(m))]
    if not available_metrics:
        return None

    values = [row[m] for m in available_metrics]
    labels = [
        m.replace("_mean", "").replace("_", " ").title()
        for m in available_metrics
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], values[::-1])

    ax.set_xlabel("Score")
    ax.set_title(f"{_clean_model_name(model_name)} - profil de métriques")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax)

    filename = f"model_{model_name}_metrics_profile.png"
    return _savefig(output_dir, filename)


def generate_per_model_figures(
    leaderboard: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    if leaderboard.empty or "model" not in leaderboard.columns:
        return paths

    model_names = leaderboard["model"].dropna().unique().tolist()

    for model_name in model_names:
        for path in [
            plot_model_f1_by_scenario(aggregated_results, model_name, model_dir),
            plot_model_metrics_profile(leaderboard, model_name, model_dir),
        ]:
            if path is not None:
                paths.append(path)

    return paths


# =========================================================
# GRAPHIQUES AVANCÉS DEPUIS advanced_analysis
# =========================================================

def plot_grid_search_results(
    grid_results: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    if grid_results is None or grid_results.empty or "best_score_f1" not in grid_results.columns:
        return None

    plot_df = grid_results.sort_values("best_score_f1", ascending=True).copy()
    plot_df["model_label"] = plot_df["model"].map(_clean_model_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["model_label"], plot_df["best_score_f1"])

    ax.set_xlabel("Meilleur F1 en GridSearchCV")
    ax.set_ylabel("Modèle")
    ax.set_title("Résultats du GridSearchCV léger")
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax)

    return _savefig(output_dir, "07_advanced_grid_search.png")


def plot_permutation_importance(
    permutation_df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 15,
) -> Path | None:
    if permutation_df is None or permutation_df.empty:
        return None

    required = {"feature", "importance_mean"}
    if not required.issubset(permutation_df.columns):
        return None

    plot_df = (
        permutation_df
        .dropna(subset=["importance_mean"])
        .sort_values("importance_mean", ascending=False)
        .head(top_n)
        .sort_values("importance_mean", ascending=True)
    )

    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["feature"], plot_df["importance_mean"])

    ax.set_xlabel("Importance moyenne")
    ax.set_ylabel("Variable")
    ax.set_title("Top variables selon permutation importance")
    ax.grid(axis="x", alpha=0.25)

    _annotate_bars(ax, decimals=4)

    return _savefig(output_dir, "08_advanced_permutation_importance.png")


def generate_advanced_figures(
    advanced_results: dict | None,
    output_dir: str | Path,
) -> list[Path]:
    if advanced_results is None:
        return []

    paths: list[Path] = []

    advanced_dir = Path(output_dir) / "advanced"
    advanced_dir.mkdir(parents=True, exist_ok=True)

    grid_df = advanced_results.get("grid_search_results")
    perm_df = advanced_results.get("permutation_importance")

    for path in [
        plot_grid_search_results(grid_df, advanced_dir),
        plot_permutation_importance(perm_df, advanced_dir),
    ]:
        if path is not None:
            paths.append(path)

    return paths


# =========================================================
# GÉNÉRATION COMPLÈTE
# =========================================================

def generate_figures(
    leaderboard: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    output_dir: str | Path,
    advanced_results: dict | None = None,
) -> list[Path]:
    """
    Génère des figures lisibles :
    - graphiques globaux
    - graphiques par scénario
    - graphiques par modèle
    - graphiques avancés optionnels
    """
    output_dir = _prepare_output_dir(output_dir)

    paths: list[Path] = []

    global_paths = [
        plot_global_f1_leaderboard(leaderboard, output_dir),
        plot_global_balanced_accuracy(leaderboard, output_dir),
        plot_global_training_time(leaderboard, output_dir),
        plot_precision_recall_tradeoff(leaderboard, output_dir),
        plot_scenario_heatmap_readable(aggregated_results, output_dir),
        plot_best_model_by_scenario(aggregated_results, output_dir),
    ]

    paths.extend([p for p in global_paths if p is not None])

    paths.extend(
        generate_per_model_figures(
            leaderboard=leaderboard,
            aggregated_results=aggregated_results,
            output_dir=output_dir,
        )
    )

    paths.extend(
        generate_advanced_figures(
            advanced_results=advanced_results,
            output_dir=output_dir,
        )
    )

    print("Figures générées :")
    for path in paths:
        print(f"- {path}")

    return paths