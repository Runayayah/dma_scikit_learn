from __future__ import annotations

import gc
import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any

import numpy as np


# =========================================================
# RANDOM / REPRODUCTIBILITÉ
# =========================================================

def set_global_seed(seed: int) -> None:
    """
    Fixe toutes les sources de hasard principales.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# TIMING
# =========================================================

@contextmanager
def timer(name: str):
    """
    Context manager pour mesurer le temps d'exécution.
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    duration = end - start
    print(f"[TIMER] {name}: {duration:.4f} sec")


def time_function(func, *args, **kwargs):
    """
    Mesure le temps d'exécution d'une fonction.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    return result, end - start


# =========================================================
# MÉMOIRE
# =========================================================

def get_array_memory_usage(arr: np.ndarray) -> float:
    """
    Retourne la taille mémoire en MB.
    """
    return arr.nbytes / (1024 ** 2)


def estimate_dataframe_memory(df) -> float:
    """
    Estime la mémoire d'un DataFrame en MB.
    """
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def force_garbage_collection():
    """
    Force un nettoyage mémoire.
    """
    gc.collect()


# =========================================================
# JSON / LOGGING SIMPLE
# =========================================================

def save_json(data: dict, path: str) -> None:
    """
    Sauvegarde un dict en JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """
    Charge un JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# SAFE EXECUTION
# =========================================================

def safe_run(func, *args, **kwargs) -> dict[str, Any]:
    """
    Exécute une fonction avec gestion d'erreur.
    """
    try:
        result = func(*args, **kwargs)
        return {
            "status": "ok",
            "result": result,
            "error": None,
        }
    except Exception as e:
        return {
            "status": "error",
            "result": None,
            "error": str(e),
        }


# =========================================================
# DIAGNOSTIC DATA
# =========================================================

def quick_df_diagnostics(df) -> dict:
    """
    Donne un résumé rapide d'un DataFrame.
    """
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "missing_rate": float(df.isna().mean().mean()),
        "n_numeric": len([c for c in df.columns if c.startswith("num_")]),
        "n_categorical": len([c for c in df.columns if c.startswith("cat_")]),
        "has_text": "text_feature" in df.columns,
    }


# =========================================================
# DEBUG PRINT
# =========================================================

def debug_print(title: str, data: Any) -> None:
    """
    Print structuré pour debug.
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(data)