from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LEADERBOARD_FILENAME, SUMMARY_FILENAME


def build_text_report(
    leaderboard: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    advanced_results: dict | None = None,
) -> str:
    lines = [
        "Rapport synthétique - Benchmark scikit-learn",
        "===========================================",
        "",
        "Objectif :",
        "Évaluer les capacités et limites de scikit-learn sur un problème de prédiction du risque client en assurance.",
        "",
        "Données :",
        "- variables numériques : âge, revenu, sinistres, ancienneté, kilométrage...",
        "- variables catégorielles : région, type de véhicule, couverture, profession...",
        "- variable textuelle : note métier synthétique liée au profil client.",
        "",
        "Méthodologie :",
        "- génération de données synthétiques assurantielles",
        "- preprocessing avec Pipeline et ColumnTransformer",
        "- vectorisation texte avec TF-IDF",
        "- validation croisée stratifiée",
        "- comparaison de plusieurs familles de modèles",
        "- analyse avancée : GridSearchCV, calibration, courbes ROC/PR, learning curve et permutation importance",
        "",
    ]

    if not leaderboard.empty:
        best = leaderboard.iloc[0]

        lines.extend([
            "Meilleur modèle global :",
            f"- modèle : {best.get('model')}",
            f"- famille : {best.get('family')}",
            f"- F1 moyen : {best.get('f1_mean'):.4f}",
            f"- balanced accuracy moyenne : {best.get('balanced_accuracy_mean'):.4f}",
            f"- recall moyen : {best.get('recall_mean'):.4f}",
            f"- ROC-AUC moyen : {best.get('roc_auc_mean'):.4f}",
            f"- temps d'entraînement moyen : {best.get('fit_time_sec_mean'):.4f} sec",
            "",
            "Top 5 modèles :",
        ])

        for _, row in leaderboard.head(5).iterrows():
            lines.append(
                f"- {row.get('model')} | "
                f"F1={row.get('f1_mean'):.4f} | "
                f"Recall={row.get('recall_mean'):.4f} | "
                f"ROC-AUC={row.get('roc_auc_mean'):.4f} | "
                f"Fit={row.get('fit_time_sec_mean'):.3f}s"
            )

    if not aggregated_results.empty and "scenario" in aggregated_results.columns:
        lines.extend([
            "",
            "Analyse par scénario :",
        ])

        scenario_summary = (
            aggregated_results
            .groupby("scenario")
            .agg(
                f1_mean=("f1_mean", "mean"),
                balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
                recall_mean=("recall_mean", "mean"),
            )
            .reset_index()
        )

        for _, row in scenario_summary.iterrows():
            lines.append(
                f"- {row['scenario']} : "
                f"F1 moyen={row['f1_mean']:.4f}, "
                f"balanced accuracy={row['balanced_accuracy_mean']:.4f}, "
                f"recall={row['recall_mean']:.4f}"
            )

    if advanced_results is not None:
        lines.extend([
            "",
            "Analyse avancée sklearn :",
            "--------------------------",
        ])

        grid_df = advanced_results.get("grid_search_results")
        if isinstance(grid_df, pd.DataFrame) and not grid_df.empty:
            lines.append("")
            lines.append("Résultats GridSearchCV léger :")
            for _, row in grid_df.iterrows():
                lines.append(
                    f"- {row['model']} : best F1={row['best_score_f1']:.4f}, "
                    f"params={row['best_params']}"
                )

        perm_df = advanced_results.get("permutation_importance")
        if isinstance(perm_df, pd.DataFrame) and not perm_df.empty:
            lines.append("")
            lines.append("Top variables selon permutation importance :")
            for _, row in perm_df.head(10).iterrows():
                lines.append(
                    f"- {row['feature']} : importance={row['importance_mean']:.4f}"
                )

        calibration_path = advanced_results.get("calibration_curve")
        learning_curve_path = advanced_results.get("learning_curve")
        roc_pr_paths = advanced_results.get("roc_pr_curves", [])

        lines.extend([
            "",
            "Figures avancées générées :",
            f"- calibration : {calibration_path}",
            f"- learning curve : {learning_curve_path}",
        ])

        for path in roc_pr_paths:
            lines.append(f"- courbe ROC/PR : {path}")

    lines.extend([
        "",
        "Limites observables de scikit-learn dans ce projet :",
        "- les pipelines classiques sont très efficaces pour des données tabulaires hétérogènes ;",
        "- le texte est exploitable via TF-IDF, mais sans compréhension sémantique profonde ;",
        "- certains modèles nécessitent des preprocessings spécifiques dense/sparse ;",
        "- les performances dépendent fortement du bruit, du déséquilibre et de la qualité des données ;",
        "- les modèles comme MLPClassifier restent limités par rapport à des frameworks deep learning spécialisés ;",
        "- scikit-learn reste très adapté au machine learning classique, mais moins au très gros volume ou au multimodal profond.",
    ])

    return "\n".join(lines)


def generate_report(
    leaderboard: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    reports_dir: str | Path,
    advanced_results: dict | None = None,
) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = reports_dir / LEADERBOARD_FILENAME
    summary_path = reports_dir / SUMMARY_FILENAME

    leaderboard.to_csv(leaderboard_path, index=False)

    if advanced_results is not None:
        grid_df = advanced_results.get("grid_search_results")
        if isinstance(grid_df, pd.DataFrame):
            grid_df.to_csv(reports_dir / "advanced_grid_search.csv", index=False)

        perm_df = advanced_results.get("permutation_importance")
        if isinstance(perm_df, pd.DataFrame):
            perm_df.to_csv(reports_dir / "advanced_permutation_importance.csv", index=False)

    report_text = build_text_report(
        leaderboard=leaderboard,
        aggregated_results=aggregated_results,
        advanced_results=advanced_results,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Leaderboard sauvegardé : {leaderboard_path}")
    print(f"Résumé sauvegardé : {summary_path}")

    return summary_path