from __future__ import annotations

import itertools
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    AGGREGATED_RESULTS_FILENAME,
    BENCHMARK_MODE,
    DATASET_CONFIG,
    EXPERIMENT_CONFIG,
    LOGGING_CONFIG,
    RAW_DATA_DIR,
    RAW_RESULTS_FILENAME,
    REPORTING_CONFIG,
    REPORTS_DIR,
    RUN_LOG_FILENAME,
    SUMMARY_JSON_FILENAME,
    SUMMARY_TEXT_FILENAME,
    TARGET_COLUMN,
    ensure_directories,
    get_active_models,
    get_mode_summary,
)
from src.data_generation import generate_dataset
from src.evaluation import evaluate_models_cv, summarize_evaluation_results


# =========================================================
# HELPERS CONFIG / RUNS
# =========================================================

def build_experiment_grid() -> list[dict]:
    """
    Construit la grille d'expériences à partir de EXPERIMENT_CONFIG.
    """
    grid = []

    for (
        n_samples,
        noise_scale,
        class_imbalance,
        missing_rate,
        outlier_rate,
        label_noise_rate,
        cardinality_level,
        vocab_size,
        text_length,
        seed,
    ) in itertools.product(
        EXPERIMENT_CONFIG["sample_sizes"],
        EXPERIMENT_CONFIG["noise_levels"],
        EXPERIMENT_CONFIG["imbalance_levels"],
        EXPERIMENT_CONFIG["missing_rates"],
        EXPERIMENT_CONFIG["outlier_rates"],
        EXPERIMENT_CONFIG["label_noise_rates"],
        EXPERIMENT_CONFIG["cardinality_levels"],
        EXPERIMENT_CONFIG["vocab_sizes"],
        EXPERIMENT_CONFIG["text_lengths"],
        EXPERIMENT_CONFIG["seeds"],
    ):
        grid.append(
            {
                "n_samples": n_samples,
                "noise_scale": noise_scale,
                "class_imbalance": class_imbalance,
                "missing_rate": missing_rate,
                "outlier_rate": outlier_rate,
                "label_noise_rate": label_noise_rate,
                "cardinality_level": cardinality_level,
                "vocab_size": vocab_size,
                "text_length": text_length,
                "seed": seed,
            }
        )

    return grid


def build_dataset_config_from_scenario(base_config: dict, scenario: dict) -> dict:
    """
    Fusionne DATASET_CONFIG avec une configuration de scénario.
    Convertit aussi 'seed' en 'random_state' pour generate_dataset().
    """
    config = base_config.copy()

    for key, value in scenario.items():
        if key != "seed":
            config[key] = value

    config["random_state"] = scenario["seed"]
    return config

def make_run_id() -> str:
    """
    Génère un identifiant de run horodaté.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# AGRÉGATIONS GLOBALES
# =========================================================

def aggregate_global_results(raw_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège tous les résultats au niveau global
    sur les dimensions :
    - modèle
    - famille
    - preprocessor
    """
    if raw_results_df.empty:
        return pd.DataFrame()

    metadata_cols = [
        "model",
        "family",
        "preprocessor",
        "status",
    ]

    excluded_cols = metadata_cols + [
        "fold",
        "seed",
        "warnings",
        "error",
        "confusion_matrix",
        "classification_report",
        "scenario_id",
        "run_id",
        "n_samples",
        "noise_scale",
        "class_imbalance",
        "missing_rate",
        "outlier_rate",
        "label_noise_rate",
        "cardinality_level",
        "vocab_size",
        "text_length",
    ]

    metric_cols = [
        col for col in raw_results_df.columns
        if col not in excluded_cols
    ]

    rows = []
    grouped = raw_results_df.groupby(metadata_cols, dropna=False)

    for keys, group in grouped:
        row = {
            "model": keys[0],
            "family": keys[1],
            "preprocessor": keys[2],
            "status": keys[3],
            "n_runs": len(group),
        }

        for col in metric_cols:
            series = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = series.mean()
            row[f"{col}_std"] = series.std()
            row[f"{col}_min"] = series.min()
            row[f"{col}_max"] = series.max()

        rows.append(row)

    aggregated_df = pd.DataFrame(rows)

    sort_col = "f1_mean" if "f1_mean" in aggregated_df.columns else None
    if sort_col:
        aggregated_df = aggregated_df.sort_values(
            by=sort_col,
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return aggregated_df


def build_leaderboard(global_aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un leaderboard simplifié.
    """
    if global_aggregated_df.empty:
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
            "fit_time_sec_mean",
            "predict_time_sec_mean",
            "matrix_density_mean",
        ]
        if col in global_aggregated_df.columns
    ]

    leaderboard = global_aggregated_df[leaderboard_cols].copy()

    if "f1_mean" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            by="f1_mean",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return leaderboard


# =========================================================
# EXÉCUTION D'UN SCÉNARIO
# =========================================================

def run_single_scenario(
    scenario_id: int,
    scenario: dict,
    run_id: str,
    selected_models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exécute un scénario complet :
    - génération dataset
    - évaluation CV
    - enrichissement des résultats avec métadonnées scénario
    """
    dataset_config = build_dataset_config_from_scenario(DATASET_CONFIG, scenario)

    df = generate_dataset(**dataset_config)

    raw_df, agg_df = evaluate_models_cv(
        df=df,
        target_col=TARGET_COLUMN,
        selected_models=selected_models,
        seed=scenario["seed"],
        calibrate_probabilities=False,
    )

    scenario_meta = {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "n_samples": scenario["n_samples"],
        "noise_scale": scenario["noise_scale"],
        "class_imbalance": scenario["class_imbalance"],
        "missing_rate": scenario["missing_rate"],
        "outlier_rate": scenario["outlier_rate"],
        "label_noise_rate": scenario["label_noise_rate"],
        "cardinality_level": scenario["cardinality_level"],
        "vocab_size": scenario["vocab_size"],
        "text_length": scenario["text_length"],
        "seed": scenario["seed"],
    }

    for key, value in scenario_meta.items():
        raw_df[key] = value
        agg_df[key] = value

    # petit snapshot dataset
    dataset_snapshot = pd.DataFrame(
        [{
            "scenario_id": scenario_id,
            "run_id": run_id,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "target_positive_rate": df[TARGET_COLUMN].mean() if TARGET_COLUMN in df.columns else np.nan,
            "missing_global_rate": df.isna().mean().mean(),
        }]
    )

    return raw_df, agg_df, dataset_snapshot


# =========================================================
# RAPPORTS / RÉSUMÉS
# =========================================================

def build_run_summary(
    raw_results_df: pd.DataFrame,
    aggregated_results_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    started_at: float,
    finished_at: float,
    run_id: str,
) -> dict:
    """
    Génère un résumé de run.
    """
    duration_sec = finished_at - started_at

    evaluation_summary = summarize_evaluation_results(aggregated_results_df)

    summary = {
        "run_id": run_id,
        "benchmark_mode": BENCHMARK_MODE,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_sec": duration_sec,
        "mode_summary": get_mode_summary(),
        "n_raw_rows": int(len(raw_results_df)),
        "n_aggregated_rows": int(len(aggregated_results_df)),
        "n_leaderboard_rows": int(len(leaderboard_df)),
        "evaluation_summary": evaluation_summary,
    }

    if not leaderboard_df.empty:
        summary["top_5_models"] = leaderboard_df.head(5).to_dict(orient="records")
    else:
        summary["top_5_models"] = []

    return summary


def build_text_summary(summary: dict) -> str:
    """
    Génère un résumé texte simple.
    """
    lines = [
        f"Run ID: {summary['run_id']}",
        f"Mode: {summary['benchmark_mode']}",
        f"Durée totale (sec): {summary['duration_sec']:.2f}",
        f"Nombre de lignes raw: {summary['n_raw_rows']}",
        f"Nombre de lignes agrégées: {summary['n_aggregated_rows']}",
        f"Nombre de lignes leaderboard: {summary['n_leaderboard_rows']}",
        "",
        "Meilleur résultat global:",
        f"  - Modèle: {summary['evaluation_summary'].get('best_model')}",
        f"  - Famille: {summary['evaluation_summary'].get('best_family')}",
        f"  - Préprocessor: {summary['evaluation_summary'].get('best_preprocessor')}",
        f"  - Score: {summary['evaluation_summary'].get('best_score')}",
        f"  - Métrique: {summary['evaluation_summary'].get('ranking_metric')}",
        "",
        "Top 5 leaderboard:",
    ]

    for idx, item in enumerate(summary.get("top_5_models", []), start=1):
        lines.append(
            f"  {idx}. {item.get('model')} | "
            f"famille={item.get('family')} | "
            f"preprocessor={item.get('preprocessor')} | "
            f"f1_mean={item.get('f1_mean')}"
        )

    return "\n".join(lines)


# =========================================================
# ORCHESTRATEUR PRINCIPAL
# =========================================================

def run_full_benchmark(
    selected_models: list[str] | None = None,
    save_dataset_snapshots: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Lance le benchmark complet.
    """
    ensure_directories()

    started_at = time.time()
    run_id = make_run_id()

    if selected_models is None:
        selected_models = get_active_models()

    experiment_grid = build_experiment_grid()

    all_raw_results = []
    all_aggregated_results = []
    all_dataset_snapshots = []

    if LOGGING_CONFIG.get("verbose", True):
        print(f"Run ID: {run_id}")
        print(f"Mode: {BENCHMARK_MODE}")
        print(f"Nombre de scénarios: {len(experiment_grid)}")
        print(f"Modèles actifs: {selected_models}")

    for scenario_id, scenario in enumerate(experiment_grid, start=1):
        if LOGGING_CONFIG.get("verbose", True):
            print(
                f"[{scenario_id}/{len(experiment_grid)}] "
                f"samples={scenario['n_samples']} | "
                f"noise={scenario['noise_scale']} | "
                f"imbalance={scenario['class_imbalance']} | "
                f"missing={scenario['missing_rate']} | "
                f"outliers={scenario['outlier_rate']} | "
                f"label_noise={scenario['label_noise_rate']} | "
                f"cardinality={scenario['cardinality_level']} | "
                f"vocab={scenario['vocab_size']} | "
                f"text={scenario['text_length']} | "
                f"seed={scenario['seed']}"
            )

        raw_df, agg_df, dataset_snapshot = run_single_scenario(
            scenario_id=scenario_id,
            scenario=scenario,
            run_id=run_id,
            selected_models=selected_models,
        )

        all_raw_results.append(raw_df)
        all_aggregated_results.append(agg_df)
        all_dataset_snapshots.append(dataset_snapshot)

    raw_results_df = pd.concat(all_raw_results, ignore_index=True) if all_raw_results else pd.DataFrame()
    scenario_aggregated_df = pd.concat(all_aggregated_results, ignore_index=True) if all_aggregated_results else pd.DataFrame()
    dataset_snapshots_df = pd.concat(all_dataset_snapshots, ignore_index=True) if all_dataset_snapshots else pd.DataFrame()

    global_aggregated_df = aggregate_global_results(raw_results_df)
    leaderboard_df = build_leaderboard(global_aggregated_df)

    finished_at = time.time()

    run_summary = build_run_summary(
        raw_results_df=raw_results_df,
        aggregated_results_df=global_aggregated_df,
        leaderboard_df=leaderboard_df,
        started_at=started_at,
        finished_at=finished_at,
        run_id=run_id,
    )
    text_summary = build_text_summary(run_summary)

    # =========================
    # SAUVEGARDE
    # =========================
    if REPORTING_CONFIG.get("save_raw_results_csv", True):
        raw_results_df.to_csv(REPORTS_DIR / RAW_RESULTS_FILENAME, index=False)

    if REPORTING_CONFIG.get("save_aggregated_results_csv", True):
        global_aggregated_df.to_csv(REPORTS_DIR / AGGREGATED_RESULTS_FILENAME, index=False)

    if REPORTING_CONFIG.get("save_leaderboard_csv", True):
        leaderboard_df.to_csv(REPORTS_DIR / "leaderboard.csv", index=False)

    if REPORTING_CONFIG.get("save_json_summary", True):
        save_json(run_summary, REPORTS_DIR / SUMMARY_JSON_FILENAME)

    if REPORTING_CONFIG.get("save_text_summary", True):
        with open(REPORTS_DIR / SUMMARY_TEXT_FILENAME, "w", encoding="utf-8") as f:
            f.write(text_summary)

    if save_dataset_snapshots:
        dataset_snapshots_df.to_csv(REPORTS_DIR / "dataset_snapshots.csv", index=False)

    # log du run
    save_json(
        {
            "run_id": run_id,
            "mode": BENCHMARK_MODE,
            "mode_summary": get_mode_summary(),
            "n_scenarios": len(experiment_grid),
            "selected_models": selected_models,
        },
        REPORTS_DIR / RUN_LOG_FILENAME,
    )

    if LOGGING_CONFIG.get("verbose", True):
        print("\nBenchmark terminé.")
        print(f"Raw results: {REPORTS_DIR / RAW_RESULTS_FILENAME}")
        print(f"Aggregated results: {REPORTS_DIR / AGGREGATED_RESULTS_FILENAME}")
        print(f"Leaderboard: {REPORTS_DIR / 'leaderboard.csv'}")
        print(f"Summary JSON: {REPORTS_DIR / SUMMARY_JSON_FILENAME}")
        print(f"Summary TXT: {REPORTS_DIR / SUMMARY_TEXT_FILENAME}")

    return {
        "raw_results": raw_results_df,
        "scenario_aggregated_results": scenario_aggregated_df,
        "global_aggregated_results": global_aggregated_df,
        "leaderboard": leaderboard_df,
        "dataset_snapshots": dataset_snapshots_df,
    }


if __name__ == "__main__":
    run_full_benchmark()