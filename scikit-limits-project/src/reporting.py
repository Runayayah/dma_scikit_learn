from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    AGGREGATED_RESULTS_FILENAME,
    FIGURES_DIR,
    RAW_RESULTS_FILENAME,
    REPORTING_CONFIG,
    REPORTS_DIR,
    SUMMARY_JSON_FILENAME,
    SUMMARY_TEXT_FILENAME,
)
from src.visualization import generate_standard_report_figures


# =========================================================
# HELPERS I/O
# =========================================================

def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def save_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# =========================================================
# LEADERBOARD
# =========================================================

def build_leaderboard(
    aggregated_df: pd.DataFrame,
    sort_metric: str = "f1_mean",
) -> pd.DataFrame:
    """
    Construit un leaderboard propre à partir des résultats agrégés.
    """
    if aggregated_df.empty:
        return pd.DataFrame()

    leaderboard_cols = [
        col for col in [
            "model",
            "family",
            "preprocessor",
            "status",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
            "average_precision_mean",
            "log_loss_mean",
            "brier_score_mean",
            "mcc_mean",
            "cohen_kappa_mean",
            "fit_time_sec_mean",
            "predict_time_sec_mean",
            "preprocess_fit_transform_time_sec_mean",
            "matrix_density_mean",
            "is_sparse_mean",
        ]
        if col in aggregated_df.columns
    ]

    leaderboard = aggregated_df[leaderboard_cols].copy()

    if sort_metric in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            by=sort_metric,
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    leaderboard["rank"] = range(1, len(leaderboard) + 1)

    ordered_cols = ["rank"] + [col for col in leaderboard.columns if col != "rank"]
    leaderboard = leaderboard[ordered_cols]

    return leaderboard


# =========================================================
# RÉSUMÉS
# =========================================================

def compute_report_summary(
    raw_results_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    primary_metric: str = "f1_mean",
) -> dict[str, Any]:
    """
    Calcule un résumé global du benchmark.
    """
    summary: dict[str, Any] = {
        "n_raw_rows": int(len(raw_results_df)) if not raw_results_df.empty else 0,
        "n_aggregated_rows": int(len(aggregated_df)) if not aggregated_df.empty else 0,
        "n_leaderboard_rows": int(len(leaderboard_df)) if not leaderboard_df.empty else 0,
        "primary_metric": primary_metric,
        "best_model": None,
        "best_family": None,
        "best_preprocessor": None,
        "best_score": None,
        "fastest_model": None,
        "fastest_fit_time_sec": None,
        "most_stable_model": None,
        "best_roc_auc_model": None,
        "best_average_precision_model": None,
    }

    if not leaderboard_df.empty:
        best_row = leaderboard_df.iloc[0]
        summary["best_model"] = best_row.get("model")
        summary["best_family"] = best_row.get("family")
        summary["best_preprocessor"] = best_row.get("preprocessor")
        summary["best_score"] = best_row.get(primary_metric)

    if not leaderboard_df.empty and "fit_time_sec_mean" in leaderboard_df.columns:
        fastest_df = leaderboard_df.dropna(subset=["fit_time_sec_mean"])
        if not fastest_df.empty:
            fastest_row = fastest_df.sort_values("fit_time_sec_mean", ascending=True).iloc[0]
            summary["fastest_model"] = fastest_row.get("model")
            summary["fastest_fit_time_sec"] = fastest_row.get("fit_time_sec_mean")

    if not aggregated_df.empty and "f1_std" in aggregated_df.columns and "f1_mean" in aggregated_df.columns:
        stable_df = aggregated_df.dropna(subset=["f1_std", "f1_mean"]).copy()
        stable_df = stable_df[stable_df["f1_mean"] > 0]
        if not stable_df.empty:
            stable_df["stability_score"] = stable_df["f1_mean"] / (stable_df["f1_std"] + 1e-8)
            stable_row = stable_df.sort_values("stability_score", ascending=False).iloc[0]
            summary["most_stable_model"] = stable_row.get("model")

    if not leaderboard_df.empty and "roc_auc_mean" in leaderboard_df.columns:
        roc_df = leaderboard_df.dropna(subset=["roc_auc_mean"])
        if not roc_df.empty:
            summary["best_roc_auc_model"] = roc_df.sort_values("roc_auc_mean", ascending=False).iloc[0].get("model")

    if not leaderboard_df.empty and "average_precision_mean" in leaderboard_df.columns:
        ap_df = leaderboard_df.dropna(subset=["average_precision_mean"])
        if not ap_df.empty:
            summary["best_average_precision_model"] = ap_df.sort_values(
                "average_precision_mean",
                ascending=False,
            ).iloc[0].get("model")

    return summary


def build_text_summary(summary: dict[str, Any], leaderboard_df: pd.DataFrame | None = None) -> str:
    """
    Génère un résumé texte du benchmark.
    """
    lines = [
        "Résumé du benchmark scikit-learn",
        "================================",
        "",
        f"Nombre de résultats bruts : {summary.get('n_raw_rows')}",
        f"Nombre de résultats agrégés : {summary.get('n_aggregated_rows')}",
        f"Nombre de lignes dans le leaderboard : {summary.get('n_leaderboard_rows')}",
        f"Métrique principale : {summary.get('primary_metric')}",
        "",
        "Meilleur modèle global",
        "----------------------",
        f"Modèle : {summary.get('best_model')}",
        f"Famille : {summary.get('best_family')}",
        f"Préprocesseur : {summary.get('best_preprocessor')}",
        f"Score principal : {summary.get('best_score')}",
        "",
        "Autres points notables",
        "----------------------",
        f"Modèle le plus rapide : {summary.get('fastest_model')}",
        f"Temps de fit moyen le plus rapide : {summary.get('fastest_fit_time_sec')}",
        f"Modèle le plus stable : {summary.get('most_stable_model')}",
        f"Meilleur ROC-AUC : {summary.get('best_roc_auc_model')}",
        f"Meilleure Average Precision : {summary.get('best_average_precision_model')}",
    ]

    if leaderboard_df is not None and not leaderboard_df.empty:
        lines.extend([
            "",
            "Top 10 leaderboard",
            "------------------",
        ])

        top_n = min(10, len(leaderboard_df))
        for i in range(top_n):
            row = leaderboard_df.iloc[i]
            lines.append(
                f"{i + 1}. {row.get('model')} | "
                f"famille={row.get('family')} | "
                f"preprocessor={row.get('preprocessor')} | "
                f"f1_mean={row.get('f1_mean')}"
            )

    return "\n".join(lines)


# =========================================================
# REPORTING COMPLET
# =========================================================

def generate_report_from_dataframes(
    raw_results_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    reports_dir: str | Path = REPORTS_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    sort_metric: str = "f1_mean",
) -> dict[str, Any]:
    """
    Génère un reporting complet à partir de DataFrames déjà chargés.
    """
    reports_dir = ensure_directory(reports_dir)
    figures_dir = ensure_directory(figures_dir)

    leaderboard_df = build_leaderboard(
        aggregated_df=aggregated_df,
        sort_metric=sort_metric,
    )

    summary = compute_report_summary(
        raw_results_df=raw_results_df,
        aggregated_df=aggregated_df,
        leaderboard_df=leaderboard_df,
        primary_metric=sort_metric,
    )

    text_summary = build_text_summary(
        summary=summary,
        leaderboard_df=leaderboard_df,
    )

    saved_files: dict[str, Any] = {}

    if REPORTING_CONFIG.get("save_leaderboard_csv", True):
        leaderboard_path = reports_dir / "leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        saved_files["leaderboard_csv"] = str(leaderboard_path)

    if REPORTING_CONFIG.get("save_json_summary", True):
        summary_json_path = reports_dir / SUMMARY_JSON_FILENAME
        save_json(summary, summary_json_path)
        saved_files["summary_json"] = str(summary_json_path)

    if REPORTING_CONFIG.get("save_text_summary", True):
        summary_text_path = reports_dir / SUMMARY_TEXT_FILENAME
        save_text(text_summary, summary_text_path)
        saved_files["summary_text"] = str(summary_text_path)

    figure_paths = []
    if REPORTING_CONFIG.get("save_figures", True):
        figure_paths = generate_standard_report_figures(
            raw_results_df=raw_results_df,
            aggregated_df=aggregated_df,
            leaderboard_df=leaderboard_df,
            output_dir=figures_dir,
        )
        saved_files["figures"] = [str(path) for path in figure_paths]

    return {
        "leaderboard": leaderboard_df,
        "summary": summary,
        "text_summary": text_summary,
        "saved_files": saved_files,
    }


def generate_report_from_files(
    reports_dir: str | Path = REPORTS_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    raw_results_filename: str = RAW_RESULTS_FILENAME,
    aggregated_results_filename: str = AGGREGATED_RESULTS_FILENAME,
    sort_metric: str = "f1_mean",
) -> dict[str, Any]:
    """
    Génère un reporting complet à partir des CSV sauvegardés.
    """
    reports_dir = Path(reports_dir)
    raw_results_path = reports_dir / raw_results_filename
    aggregated_results_path = reports_dir / aggregated_results_filename

    raw_results_df = load_csv_if_exists(raw_results_path)
    aggregated_df = load_csv_if_exists(aggregated_results_path)

    return generate_report_from_dataframes(
        raw_results_df=raw_results_df,
        aggregated_df=aggregated_df,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        sort_metric=sort_metric,
    )


if __name__ == "__main__":
    report = generate_report_from_files()

    print("Reporting généré.")
    print("\nRésumé :")
    print(report["text_summary"])

    print("\nFichiers sauvegardés :")
    for key, value in report["saved_files"].items():
        print(f"- {key}: {value}")