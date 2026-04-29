from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATASET_CONFIG,
    PRIMARY_RANKING_METRIC,
    RANDOM_STATE,
    SCALABILITY_CONFIG,
)
from src.data_generation import generate_dataset
from src.evaluation import evaluate_models_cv


# =========================================================
# HELPERS
# =========================================================

def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce")
    if series.dropna().empty:
        return None
    return float(series.mean())


def _base_dataset_config() -> dict[str, Any]:
    return DATASET_CONFIG.copy()


def _build_scalability_row(
    axis_name: str,
    axis_value: Any,
    model_count: int,
    raw_results_df: pd.DataFrame,
    aggregated_results_df: pd.DataFrame,
    total_duration_sec: float,
) -> dict[str, Any]:
    ranking_col = f"{PRIMARY_RANKING_METRIC}_mean"

    row = {
        "axis_name": axis_name,
        "axis_value": axis_value,
        "n_models": model_count,
        "n_raw_rows": len(raw_results_df),
        "n_aggregated_rows": len(aggregated_results_df),
        "total_duration_sec": total_duration_sec,
        "fit_time_sec_mean": _safe_mean(raw_results_df, "fit_time_sec"),
        "predict_time_sec_mean": _safe_mean(raw_results_df, "predict_time_sec"),
        "preprocess_fit_transform_time_sec_mean": _safe_mean(raw_results_df, "preprocess_fit_transform_time_sec"),
        "preprocess_test_transform_time_sec_mean": _safe_mean(raw_results_df, "preprocess_test_transform_time_sec"),
        "n_cols_transformed_mean": _safe_mean(raw_results_df, "n_cols_transformed"),
        "matrix_density_mean": _safe_mean(raw_results_df, "matrix_density"),
        "primary_metric_mean": _safe_mean(aggregated_results_df, ranking_col),
        "accuracy_mean": _safe_mean(aggregated_results_df, "accuracy_mean"),
        "balanced_accuracy_mean": _safe_mean(aggregated_results_df, "balanced_accuracy_mean"),
        "f1_mean": _safe_mean(aggregated_results_df, "f1_mean"),
        "roc_auc_mean": _safe_mean(aggregated_results_df, "roc_auc_mean"),
        "average_precision_mean": _safe_mean(aggregated_results_df, "average_precision_mean"),
        "error_count": int((raw_results_df["status"] == "error").sum()) if "status" in raw_results_df.columns else 0,
    }

    return row


def _run_one_scalability_step(
    axis_name: str,
    axis_value: Any,
    dataset_overrides: dict[str, Any],
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Exécute une étape de benchmark de scalabilité.
    """
    dataset_config = _base_dataset_config()
    dataset_config.update(dataset_overrides)
    dataset_config["random_state"] = seed

    df = generate_dataset(**dataset_config)

    start = time.perf_counter()

    raw_results_df, aggregated_results_df = evaluate_models_cv(
        df=df,
        target_col="target",
        selected_models=selected_models,
        preferred_preprocessors=preferred_preprocessors,
        seed=seed,
        calibrate_probabilities=False,
    )

    total_duration_sec = time.perf_counter() - start

    summary_row = _build_scalability_row(
        axis_name=axis_name,
        axis_value=axis_value,
        model_count=len(selected_models) if selected_models is not None else (
            raw_results_df["model"].nunique() if "model" in raw_results_df.columns else 0
        ),
        raw_results_df=raw_results_df,
        aggregated_results_df=aggregated_results_df,
        total_duration_sec=total_duration_sec,
    )

    summary_row["n_rows_dataset"] = int(df.shape[0])
    summary_row["n_cols_dataset"] = int(df.shape[1])

    return raw_results_df, aggregated_results_df, summary_row


# =========================================================
# AXES DE SCALABILITÉ
# =========================================================

def benchmark_n_rows(
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Benchmark sur le nombre de lignes.
    """
    summary_rows = []
    details = {}

    for n_rows in SCALABILITY_CONFIG["n_rows_grid"]:
        raw_df, agg_df, row = _run_one_scalability_step(
            axis_name="n_rows",
            axis_value=n_rows,
            dataset_overrides={"n_samples": n_rows},
            selected_models=selected_models,
            preferred_preprocessors=preferred_preprocessors,
            seed=seed,
        )

        summary_rows.append(row)
        details[f"n_rows_{n_rows}_raw"] = raw_df
        details[f"n_rows_{n_rows}_aggregated"] = agg_df

    return pd.DataFrame(summary_rows), details


def benchmark_n_numeric_features(
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Benchmark sur le nombre de features numériques.
    """
    summary_rows = []
    details = {}

    for n_features in SCALABILITY_CONFIG["n_num_features_grid"]:
        raw_df, agg_df, row = _run_one_scalability_step(
            axis_name="n_num_features",
            axis_value=n_features,
            dataset_overrides={"n_num_features": n_features},
            selected_models=selected_models,
            preferred_preprocessors=preferred_preprocessors,
            seed=seed,
        )

        summary_rows.append(row)
        details[f"n_num_features_{n_features}_raw"] = raw_df
        details[f"n_num_features_{n_features}_aggregated"] = agg_df

    return pd.DataFrame(summary_rows), details


def benchmark_n_categorical_features(
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Benchmark sur le nombre de features catégorielles.
    """
    summary_rows = []
    details = {}

    for n_features in SCALABILITY_CONFIG["n_cat_features_grid"]:
        raw_df, agg_df, row = _run_one_scalability_step(
            axis_name="n_cat_features",
            axis_value=n_features,
            dataset_overrides={"n_cat_features": n_features},
            selected_models=selected_models,
            preferred_preprocessors=preferred_preprocessors,
            seed=seed,
        )

        summary_rows.append(row)
        details[f"n_cat_features_{n_features}_raw"] = raw_df
        details[f"n_cat_features_{n_features}_aggregated"] = agg_df

    return pd.DataFrame(summary_rows), details


def benchmark_cardinality(
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Benchmark sur la cardinalité catégorielle.
    """
    summary_rows = []
    details = {}

    for level in SCALABILITY_CONFIG["cardinality_grid"]:
        raw_df, agg_df, row = _run_one_scalability_step(
            axis_name="cardinality_level",
            axis_value=level,
            dataset_overrides={"cardinality_level": level},
            selected_models=selected_models,
            preferred_preprocessors=preferred_preprocessors,
            seed=seed,
        )

        summary_rows.append(row)
        details[f"cardinality_{level}_raw"] = raw_df
        details[f"cardinality_{level}_aggregated"] = agg_df

    return pd.DataFrame(summary_rows), details


def benchmark_vocab_size(
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Benchmark sur la taille du vocabulaire texte.
    """
    summary_rows = []
    details = {}

    for vocab_size in SCALABILITY_CONFIG["vocab_size_grid"]:
        raw_df, agg_df, row = _run_one_scalability_step(
            axis_name="vocab_size",
            axis_value=vocab_size,
            dataset_overrides={"vocab_size": vocab_size},
            selected_models=selected_models,
            preferred_preprocessors=preferred_preprocessors,
            seed=seed,
        )

        summary_rows.append(row)
        details[f"vocab_size_{vocab_size}_raw"] = raw_df
        details[f"vocab_size_{vocab_size}_aggregated"] = agg_df

    return pd.DataFrame(summary_rows), details


def benchmark_text_length(
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Benchmark sur la longueur du texte.
    """
    summary_rows = []
    details = {}

    for text_length in ["short", "medium", "long"]:
        raw_df, agg_df, row = _run_one_scalability_step(
            axis_name="text_length",
            axis_value=text_length,
            dataset_overrides={"text_length": text_length},
            selected_models=selected_models,
            preferred_preprocessors=preferred_preprocessors,
            seed=seed,
        )

        summary_rows.append(row)
        details[f"text_length_{text_length}_raw"] = raw_df
        details[f"text_length_{text_length}_aggregated"] = agg_df

    return pd.DataFrame(summary_rows), details


# =========================================================
# RUN GLOBAL
# =========================================================

def run_full_scalability_benchmark(
    selected_models: list[str] | None = None,
    preferred_preprocessors: list[str] | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Lance tous les benchmarks de scalabilité.
    """
    results: dict[str, Any] = {}

    rows_df, rows_details = benchmark_n_rows(
        selected_models=selected_models,
        preferred_preprocessors=preferred_preprocessors,
        seed=seed,
    )
    results["n_rows_summary"] = rows_df
    results["n_rows_details"] = rows_details

    num_df, num_details = benchmark_n_numeric_features(
        selected_models=selected_models,
        preferred_preprocessors=preferred_preprocessors,
        seed=seed,
    )
    results["n_numeric_features_summary"] = num_df
    results["n_numeric_features_details"] = num_details

    cat_df, cat_details = benchmark_n_categorical_features(
        selected_models=selected_models,
        preferred_preprocessors=preferred_preprocessors,
        seed=seed,
    )
    results["n_categorical_features_summary"] = cat_df
    results["n_categorical_features_details"] = cat_details

    card_df, card_details = benchmark_cardinality(
        selected_models=selected_models,
        preferred_preprocessors=preferred_preprocessors,
        seed=seed,
    )
    results["cardinality_summary"] = card_df
    results["cardinality_details"] = card_details

    vocab_df, vocab_details = benchmark_vocab_size(
        selected_models=selected_models,
        preferred_preprocessors=preferred_preprocessors,
        seed=seed,
    )
    results["vocab_size_summary"] = vocab_df
    results["vocab_size_details"] = vocab_details

    text_df, text_details = benchmark_text_length(
        selected_models=selected_models,
        preferred_preprocessors=preferred_preprocessors,
        seed=seed,
    )
    results["text_length_summary"] = text_df
    results["text_length_details"] = text_details

    all_summaries = []
    for key, value in results.items():
        if key.endswith("_summary") and isinstance(value, pd.DataFrame):
            temp = value.copy()
            temp["benchmark_name"] = key
            all_summaries.append(temp)

    results["combined_summary"] = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    return results


# =========================================================
# EXPORT
# =========================================================

def export_scalability_results(
    results: dict[str, Any],
    output_dir,
    prefix: str = "scalability",
) -> None:
    """
    Sauvegarde les résultats de scalabilité en CSV.
    """
    output_dir = pd.io.common.stringify_path(output_dir)

    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            path = f"{output_dir}/{prefix}_{key}.csv"
            value.to_csv(path, index=False)

        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, pd.DataFrame):
                    path = f"{output_dir}/{prefix}_{key}_{subkey}.csv"
                    subvalue.to_csv(path, index=False)


if __name__ == "__main__":
    from src.config import get_active_models

    selected_models = get_active_models()[:3]

    results = run_full_scalability_benchmark(
        selected_models=selected_models,
        preferred_preprocessors=None,
        seed=RANDOM_STATE,
    )

    print("Benchmarks de scalabilité terminés.\n")

    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            print(f"{key}: {value.shape}")
            print(value.head(), "\n")