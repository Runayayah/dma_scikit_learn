from __future__ import annotations

import platform
import sys
from pathlib import Path

# =========================
# CHEMINS DU PROJET
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = DATA_DIR / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"

# =========================
# PARAMÈTRES GLOBAUX
# =========================

RANDOM_STATE = 42
N_JOBS = -1

# Modes disponibles :
# - debug  : très petit, pour débugger vite
# - fast   : benchmark rapide
# - full   : benchmark complet
# - stress : benchmark lourd / scalabilité
BENCHMARK_MODE = "fast"

# =========================
# MODES D'EXÉCUTION
# =========================

MODE_CONFIGS = {
    "debug": {
        "cv_folds": 3,
        "seeds": [42],
        "sample_sizes": [300],
        "noise_levels": [0.5],
        "imbalance_levels": [0.30],
        "missing_rates": [0.05],
        "outlier_rates": [0.02],
        "label_noise_rates": [0.00],
        "cardinality_levels": [1],
        "vocab_sizes": [30],
        "text_lengths": ["short"],
        "enable_tuning": False,
    },
    "fast": {
        "cv_folds": 4,
        "seeds": [0, 1],
        "sample_sizes": [1000, 3000],
        "noise_levels": [0.5, 1.0],
        "imbalance_levels": [0.30, 0.10],
        "missing_rates": [0.05, 0.15],
        "outlier_rates": [0.02, 0.08],
        "label_noise_rates": [0.00, 0.05],
        "cardinality_levels": [1, 2],
        "vocab_sizes": [50, 150],
        "text_lengths": ["short", "medium"],
        "enable_tuning": False,
    },
    "full": {
        "cv_folds": 5,
        "seeds": [0, 1, 2],
        "sample_sizes": [1000, 5000, 10000],
        "noise_levels": [0.5, 1.0, 2.0],
        "imbalance_levels": [0.50, 0.20, 0.05],
        "missing_rates": [0.05, 0.15, 0.30],
        "outlier_rates": [0.02, 0.08, 0.15],
        "label_noise_rates": [0.00, 0.05, 0.10],
        "cardinality_levels": [1, 2, 3],
        "vocab_sizes": [50, 150, 500],
        "text_lengths": ["short", "medium", "long"],
        "enable_tuning": True,
    },
    "stress": {
        "cv_folds": 5,
        "seeds": [0, 1, 2],
        "sample_sizes": [1000, 5000, 10000, 30000],
        "noise_levels": [0.5, 1.0, 2.0, 3.0],
        "imbalance_levels": [0.50, 0.20, 0.05, 0.01],
        "missing_rates": [0.05, 0.15, 0.30, 0.50],
        "outlier_rates": [0.02, 0.08, 0.15, 0.25],
        "label_noise_rates": [0.00, 0.05, 0.10, 0.20],
        "cardinality_levels": [1, 2, 3, 4],
        "vocab_sizes": [50, 150, 500, 1500],
        "text_lengths": ["short", "medium", "long"],
        "enable_tuning": True,
    },
}

ACTIVE_MODE_CONFIG = MODE_CONFIGS[BENCHMARK_MODE]

CV_FOLDS = ACTIVE_MODE_CONFIG["cv_folds"]
SEEDS = ACTIVE_MODE_CONFIG["seeds"]

# =========================
# GÉNÉRATION DE DONNÉES
# =========================

TARGET_TYPE = "binary_classification"
TARGET_COLUMN = "target"

DATASET_CONFIG = {
    "n_samples": 5000,
    "n_num_features": 12,
    "n_cat_features": 8,
    "vocab_size": 150,
    "text_length": "medium",          # short | medium | long
    "missing_rate": 0.05,
    "outlier_rate": 0.02,
    "noise_scale": 1.0,
    "label_noise_rate": 0.00,
    "class_imbalance": 0.30,          # part de classe positive
    "cardinality_level": 2,           # 1 faible, 2 moyenne, 3 forte, 4 extrême
    "add_irrelevant_features": True,
    "n_irrelevant_features": 5,
    "add_duplicated_features": True,
    "n_duplicated_features": 2,
    "add_constant_features": True,
    "n_constant_features": 1,
    "add_collinearity": True,
    "collinearity_strength": 0.90,
    "simulate_distribution_shift": False,
    "simulate_temporal_drift": False,
    "random_state": RANDOM_STATE,
}

# =========================
# SCÉNARIOS D'EXPÉRIENCE
# =========================

EXPERIMENT_CONFIG = {
    "sample_sizes": ACTIVE_MODE_CONFIG["sample_sizes"],
    "noise_levels": ACTIVE_MODE_CONFIG["noise_levels"],
    "imbalance_levels": ACTIVE_MODE_CONFIG["imbalance_levels"],
    "missing_rates": ACTIVE_MODE_CONFIG["missing_rates"],
    "outlier_rates": ACTIVE_MODE_CONFIG["outlier_rates"],
    "label_noise_rates": ACTIVE_MODE_CONFIG["label_noise_rates"],
    "cardinality_levels": ACTIVE_MODE_CONFIG["cardinality_levels"],
    "vocab_sizes": ACTIVE_MODE_CONFIG["vocab_sizes"],
    "text_lengths": ACTIVE_MODE_CONFIG["text_lengths"],
    "seeds": SEEDS,
}

# =========================
# FAMILLES DE MODÈLES
# =========================

MODEL_FAMILIES = {
    "baseline": True,
    "linear": True,
    "tree": True,
    "ensemble": True,
    "kernel": True,
    "neural_network": True,
    "naive_bayes": True,
    "online": True,
}

ACTIVE_MODELS = [
    "dummy",
    "logistic_regression",
    "ridge_classifier",
    "sgd_classifier",
    "passive_aggressive",
    "perceptron",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "adaboost",
    "bagging",
    "gradient_boosting",
    "hist_gradient_boosting",
    "linear_svc",
    "svc_rbf",
    "mlp",
    "gaussian_nb",
    "multinomial_nb",
]

FAST_MODELS = [
    "dummy",
    "logistic_regression",
    "ridge_classifier",
    "sgd_classifier",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "linear_svc",
    "mlp",
]

ONLINE_MODELS = [
    "sgd_classifier",
    "passive_aggressive",
    "perceptron",
]

DENSE_ONLY_MODELS = [
    "gaussian_nb",
    "hist_gradient_boosting",
]

NON_NEGATIVE_FEATURE_MODELS = [
    "multinomial_nb",
]

SLOW_MODELS = [
    "svc_rbf",
    "mlp",
    "gaussian_process",
]

# =========================
# PREPROCESSING
# =========================

ACTIVE_PREPROCESSORS = [
    "basic_tfidf_robust",
    "basic_count_standard",
    "no_text_robust",
    "text_only_tfidf",
    "dense_standard_ordinal",
    "sparse_tfidf_maxabs",
]

PREPROCESSOR_CONFIG = {
    "numeric_scalers": [
        "standard",
        "minmax",
        "maxabs",
        "robust",
        "power",
        "quantile",
    ],
    "categorical_encoders": [
        "onehot",
        "ordinal",
    ],
    "text_vectorizers": [
        "count",
        "tfidf",
        "hashing",
    ],
    "text_max_features": 1000,
    "text_ngram_range": (1, 2),
    "text_use_stop_words": False,
    "rare_category_min_freq": 0.01,
    "enable_feature_selection": True,
    "feature_selection_methods": [
        "variance_threshold",
        "select_k_best",
        "select_percentile",
    ],
    "enable_dimensionality_reduction": True,
    "dimensionality_reduction_methods": [
        "pca",
        "truncated_svd",
    ],
}

PIPELINE_COMPARISON_CONFIG = {
    "with_text": True,
    "without_text": True,
    "tabular_only": True,
    "text_only": True,
    "with_scaling": True,
    "without_scaling": False,
}

# =========================
# MÉTRIQUES
# =========================

ACTIVE_METRICS = {
    "accuracy": True,
    "balanced_accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "average_precision": True,
    "log_loss": True,
    "brier_score": True,
    "mcc": True,
    "cohen_kappa": True,
}

PRIMARY_RANKING_METRIC = "f1"
SECONDARY_RANKING_METRIC = "roc_auc"

THRESHOLD_ANALYSIS_CONFIG = {
    "enabled": True,
    "threshold_grid_size": 101,
    "fp_cost": 1.0,
    "fn_cost": 5.0,
}

CALIBRATION_CONFIG = {
    "enabled": True,
    "method": "sigmoid",   # sigmoid | isotonic
    "cv": 3,
}

# =========================
# ROBUSTESSE
# =========================

ROBUSTNESS_CONFIG = {
    "enabled": True,
    "scenarios": [
        "baseline",
        "high_missing",
        "high_outliers",
        "label_noise",
        "distribution_shift",
        "temporal_drift",
        "irrelevant_features",
        "duplicated_features",
        "constant_features",
        "high_collinearity",
        "unseen_categories",
        "empty_text",
        "noisy_text",
        "long_text",
    ],
    "extreme_missing_rate": 0.50,
    "extreme_outlier_rate": 0.25,
    "extreme_label_noise_rate": 0.20,
}

# =========================
# SCALABILITÉ
# =========================

SCALABILITY_CONFIG = {
    "enabled": True,
    "n_rows_grid": [1000, 5000, 10000, 30000],
    "n_num_features_grid": [10, 25, 50, 100],
    "n_cat_features_grid": [5, 10, 20, 40],
    "cardinality_grid": [1, 2, 3, 4],
    "vocab_size_grid": [50, 150, 500, 1500],
    "measure_memory": True,
    "measure_matrix_size": True,
    "compare_sparse_dense": True,
}

# =========================
# TUNING
# =========================

TUNING_CONFIG = {
    "enabled": ACTIVE_MODE_CONFIG["enable_tuning"],
    "models_to_tune": [
        "logistic_regression",
        "random_forest",
        "svc_rbf",
        "mlp",
    ],
    "search_methods": [
        "grid",
        "random",
    ],
    "cv_folds": 3,
    "n_iter_random_search": 10,
    "refit_metric": PRIMARY_RANKING_METRIC,
}

# =========================
# LOGGING / REPORTING
# =========================

LOGGING_CONFIG = {
    "capture_warnings": True,
    "capture_tracebacks": True,
    "save_run_config": True,
    "save_environment_info": True,
    "verbose": True,
}

REPORTING_CONFIG = {
    "save_raw_results_csv": True,
    "save_aggregated_results_csv": True,
    "save_leaderboard_csv": True,
    "save_json_summary": True,
    "save_text_summary": True,
    "save_figures": True,
}

# =========================
# NOMS DE FICHIERS
# =========================

DATA_FILENAME = "dataset.csv"
RAW_RESULTS_FILENAME = "raw_results.csv"
AGGREGATED_RESULTS_FILENAME = "aggregated_results.csv"
LEADERBOARD_FILENAME = "leaderboard.csv"
SUMMARY_JSON_FILENAME = "summary.json"
SUMMARY_TEXT_FILENAME = "summary.txt"
RUN_LOG_FILENAME = "run_log.json"

# =========================
# ENVIRONNEMENT
# =========================

ENVIRONMENT_INFO = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "system": platform.system(),
    "machine": platform.machine(),
}

# =========================
# HELPERS
# =========================

def ensure_directories() -> None:
    """
    Crée automatiquement les dossiers nécessaires si ils n'existent pas.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def get_active_models() -> list[str]:
    """
    Retourne la liste des modèles actifs selon le mode.
    """
    if BENCHMARK_MODE == "debug":
        return FAST_MODELS
    if BENCHMARK_MODE == "fast":
        return FAST_MODELS
    return ACTIVE_MODELS


def get_mode_summary() -> dict:
    """
    Retourne un résumé compact de la configuration active.
    """
    return {
        "benchmark_mode": BENCHMARK_MODE,
        "cv_folds": CV_FOLDS,
        "seeds": SEEDS,
        "n_jobs": N_JOBS,
        "target_type": TARGET_TYPE,
        "primary_metric": PRIMARY_RANKING_METRIC,
        "active_models": get_active_models(),
        "active_preprocessors": ACTIVE_PREPROCESSORS,
    }