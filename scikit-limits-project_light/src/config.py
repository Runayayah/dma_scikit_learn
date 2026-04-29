from __future__ import annotations

from pathlib import Path

# =========================================================
# CHEMINS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = DATA_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# =========================================================
# PARAMÈTRES GLOBAUX
# =========================================================

RANDOM_STATE = 42
TARGET_COLUMN = "target"

# Mode recommandé pour un projet M2 :
# - debug : très rapide
# - standard : soutenance / rapport
# - extended : plus long
BENCHMARK_MODE = "standard"

MODE_CONFIGS = {
    "debug": {
        "n_samples": 800,
        "cv_folds": 3,
        "seeds": [42],
        "scenarios": ["baseline", "high_noise"],
    },
    "standard": {
        "n_samples": 3000,
        "cv_folds": 3,
        "seeds": [0, 1],
        "scenarios": [
            "baseline",
            "high_noise",
            "class_imbalance",
            "missing_values",
            "text_removed",
        ],
    },
    "extended": {
        "n_samples": 6000,
        "cv_folds": 5,
        "seeds": [0, 1, 2],
        "scenarios": [
            "baseline",
            "high_noise",
            "class_imbalance",
            "missing_values",
            "outliers",
            "text_removed",
            "high_cardinality",
        ],
    },
}

ACTIVE_MODE_CONFIG = MODE_CONFIGS[BENCHMARK_MODE]

CV_FOLDS = ACTIVE_MODE_CONFIG["cv_folds"]
SEEDS = ACTIVE_MODE_CONFIG["seeds"]
ACTIVE_SCENARIOS = ACTIVE_MODE_CONFIG["scenarios"]

# =========================================================
# CONFIG DATASET ASSURANCE
# =========================================================

DATASET_CONFIG = {
    "n_samples": ACTIVE_MODE_CONFIG["n_samples"],
    "n_num_features": 12,
    "n_cat_features": 8,
    "vocab_size": 120,
    "text_length": "medium",
    "missing_rate": 0.05,
    "outlier_rate": 0.02,
    "noise_scale": 1.0,
    "class_imbalance": 0.30,
    "cardinality_level": 2,
    "random_state": RANDOM_STATE,
}

SCENARIO_CONFIGS = {
    "baseline": {
        "description": "Dataset assurance standard.",
        "overrides": {},
    },
    "high_noise": {
        "description": "Signal plus bruité, séparation plus difficile.",
        "overrides": {
            "noise_scale": 2.5,
        },
    },
    "class_imbalance": {
        "description": "Classe positive rare.",
        "overrides": {
            "class_imbalance": 0.10,
        },
    },
    "missing_values": {
        "description": "Plus de valeurs manquantes.",
        "overrides": {
            "missing_rate": 0.20,
        },
    },
    "outliers": {
        "description": "Plus d'outliers numériques.",
        "overrides": {
            "outlier_rate": 0.10,
        },
    },
    "text_removed": {
        "description": "Colonne texte neutralisée pour mesurer son apport.",
        "overrides": {
            "remove_text": True,
        },
    },
    "high_cardinality": {
        "description": "Variables catégorielles plus cardinales.",
        "overrides": {
            "cardinality_level": 3,
        },
    },
}

# =========================================================
# MODÈLES
# =========================================================

ACTIVE_MODELS = [
    "dummy",
    "logistic_regression",
    "ridge_classifier",
    "sgd_classifier",
    "linear_svc",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
]

DENSE_ONLY_MODELS = [
    "hist_gradient_boosting",
]

# =========================================================
# MÉTRIQUES
# =========================================================

PRIMARY_METRIC = "f1"

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

# =========================================================
# FICHIERS DE SORTIE
# =========================================================

RAW_RESULTS_FILENAME = "raw_results.csv"
AGGREGATED_RESULTS_FILENAME = "aggregated_results.csv"
LEADERBOARD_FILENAME = "leaderboard.csv"
SUMMARY_FILENAME = "summary.txt"

# =========================================================
# HELPERS
# =========================================================

def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_mode_summary() -> dict:
    return {
        "benchmark_mode": BENCHMARK_MODE,
        "n_samples": ACTIVE_MODE_CONFIG["n_samples"],
        "cv_folds": CV_FOLDS,
        "seeds": SEEDS,
        "scenarios": ACTIVE_SCENARIOS,
        "models": ACTIVE_MODELS,
        "primary_metric": PRIMARY_METRIC,
    }