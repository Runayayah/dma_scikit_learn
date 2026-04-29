from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from src.config import (
    ACTIVE_MODELS,
    ACTIVE_SCENARIOS,
    AGGREGATED_RESULTS_FILENAME,
    DATASET_CONFIG,
    LEADERBOARD_FILENAME,
    RAW_DATA_DIR,
    RAW_RESULTS_FILENAME,
    REPORTS_DIR,
    SCENARIO_CONFIGS,
    SEEDS,
    SUMMARY_FILENAME,
    TARGET_COLUMN,
    ensure_directories,
    get_mode_summary,
)
from src.data_generation import generate_dataset
from src.evaluation import evaluate_models_cv


def apply_scenario(base_config: dict, scenario_name: str, seed: int) -> dict:
    """
    Applique les overrides d'un scénario à la config dataset.
    On retire les paramètres qui ne sont pas destinés à generate_dataset().
    """
    scenario = SCENARIO_CONFIGS[scenario_name]

    config = base_config.copy()
    config.update(scenario.get("overrides", {}))
    config["random_state"] = seed

    # Paramètres utilisés après la génération, pas par generate_dataset()
    config.pop("remove_text", None)

    return config

def postprocess_dataset_for_scenario(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """
    Applique les modifications qui ne passent pas directement par generate_dataset.
    Exemple : supprimer / neutraliser le texte.
    """
    df = df.copy()

    if scenario_name == "text_removed" and "text_feature" in df.columns:
        df["text_feature"] = ""

    return df


def build_leaderboard(aggregated_results: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un leaderboard global à partir des résultats agrégés.
    """
    if aggregated_results.empty:
        return pd.DataFrame()

    leaderboard = (
        aggregated_results
        .groupby(["model", "family", "preprocessor"], dropna=False)
        .agg(
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
            recall_mean=("recall_mean", "mean"),
            precision_mean=("precision_mean", "mean"),
            roc_auc_mean=("roc_auc_mean", "mean"),
            fit_time_sec_mean=("fit_time_sec_mean", "mean"),
            n_errors=("n_errors", "sum"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    return leaderboard


def build_summary_text(
    leaderboard: pd.DataFrame,
    raw_results: pd.DataFrame,
    aggregated_results: pd.DataFrame,
    duration_sec: float,
) -> str:
    """
    Génère un résumé texte simple.
    """
    lines = [
        "Résumé du benchmark scikit-learn",
        "================================",
        "",
        f"Durée totale : {duration_sec:.2f} secondes",
        "",
        "Configuration active :",
    ]

    for key, value in get_mode_summary().items():
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        f"Nombre de résultats bruts : {len(raw_results)}",
        f"Nombre de résultats agrégés : {len(aggregated_results)}",
        f"Nombre de modèles dans le leaderboard : {len(leaderboard)}",
        "",
    ])

    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        lines.extend([
            "Meilleur modèle global :",
            f"- Modèle : {best['model']}",
            f"- Famille : {best['family']}",
            f"- Préprocessing : {best['preprocessor']}",
            f"- F1 moyen : {best['f1_mean']:.4f}",
            f"- Balanced accuracy moyen : {best['balanced_accuracy_mean']:.4f}",
            f"- Temps fit moyen : {best['fit_time_sec_mean']:.4f} sec",
            "",
            "Top modèles :",
        ])

        for _, row in leaderboard.head(10).iterrows():
            lines.append(
                f"{int(row['rank'])}. {row['model']} "
                f"| F1={row['f1_mean']:.4f} "
                f"| Recall={row['recall_mean']:.4f} "
                f"| ROC-AUC={row['roc_auc_mean']:.4f} "
                f"| Fit={row['fit_time_sec_mean']:.3f}s"
            )

    return "\n".join(lines)


def run_experiments() -> dict[str, pd.DataFrame]:
    """
    Lance toutes les expériences prévues dans la config.
    """
    ensure_directories()
    start = time.perf_counter()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_raw_results = []
    all_aggregated_results = []

    print("=== Lancement benchmark M2 scikit-learn ===")
    print(f"Run ID : {run_id}")
    print(f"Scénarios : {ACTIVE_SCENARIOS}")
    print(f"Seeds : {SEEDS}")
    print(f"Modèles : {ACTIVE_MODELS}")

    for scenario_name in ACTIVE_SCENARIOS:
        scenario_description = SCENARIO_CONFIGS[scenario_name]["description"]

        for seed in SEEDS:
            print(f"\nScenario={scenario_name} | seed={seed}")
            print(f"Description : {scenario_description}")

            dataset_config = apply_scenario(
                base_config=DATASET_CONFIG,
                scenario_name=scenario_name,
                seed=seed,
            )

            df = generate_dataset(**dataset_config)
            df = postprocess_dataset_for_scenario(df, scenario_name)

            dataset_path = RAW_DATA_DIR / f"dataset_{scenario_name}_seed_{seed}.csv"
            df.to_csv(dataset_path, index=False)

            raw_df, agg_df = evaluate_models_cv(
                df=df,
                target_col=TARGET_COLUMN,
                selected_models=ACTIVE_MODELS,
                seed=seed,
            )

            for result_df in [raw_df, agg_df]:
                result_df["run_id"] = run_id
                result_df["scenario"] = scenario_name
                result_df["scenario_description"] = scenario_description
                result_df["seed"] = seed

            all_raw_results.append(raw_df)
            all_aggregated_results.append(agg_df)

    raw_results = (
        pd.concat(all_raw_results, ignore_index=True)
        if all_raw_results else pd.DataFrame()
    )

    aggregated_results = (
        pd.concat(all_aggregated_results, ignore_index=True)
        if all_aggregated_results else pd.DataFrame()
    )

    leaderboard = build_leaderboard(aggregated_results)

    duration_sec = time.perf_counter() - start
    summary_text = build_summary_text(
        leaderboard=leaderboard,
        raw_results=raw_results,
        aggregated_results=aggregated_results,
        duration_sec=duration_sec,
    )

    raw_results.to_csv(REPORTS_DIR / RAW_RESULTS_FILENAME, index=False)
    aggregated_results.to_csv(REPORTS_DIR / AGGREGATED_RESULTS_FILENAME, index=False)
    leaderboard.to_csv(REPORTS_DIR / LEADERBOARD_FILENAME, index=False)

    with open(REPORTS_DIR / SUMMARY_FILENAME, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n=== Benchmark terminé ===")
    print(summary_text)
    print(f"\nRésultats sauvegardés dans : {REPORTS_DIR}")

    return {
        "raw_results": raw_results,
        "aggregated_results": aggregated_results,
        "leaderboard": leaderboard,
    }


if __name__ == "__main__":
    run_experiments()