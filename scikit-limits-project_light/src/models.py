from __future__ import annotations

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from src.config import ACTIVE_MODELS, RANDOM_STATE


def build_model_registry(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Registre des modèles utilisés dans le projet.

    Objectif :
    couvrir plusieurs familles sklearn sans rendre le benchmark trop lourd.
    """
    return {
        # Baseline naïve obligatoire
        "dummy": DummyClassifier(
            strategy="most_frequent",
        ),

        # Modèles linéaires
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "ridge_classifier": RidgeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "sgd_classifier": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),

        # SVM linéaire
        "linear_svc": LinearSVC(
            class_weight="balanced",
            random_state=random_state,
        ),

        # Ensemble d'arbres
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),

        # Boosting histogram-based
        # Modèle performant, mais nécessite un pipeline dense.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            random_state=random_state,
        ),

        # Réseau de neurones sklearn
        # Utile pour montrer que sklearn en propose un,
        # mais aussi ses limites face au deep learning spécialisé.
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            random_state=random_state,
        ),
    }


def get_models(
    random_state: int = RANDOM_STATE,
    selected_models: list[str] | None = None,
) -> Dict[str, BaseEstimator]:
    """
    Retourne les modèles actifs.
    """
    registry = build_model_registry(random_state=random_state)

    if selected_models is None:
        selected_models = ACTIVE_MODELS

    return {
        name: registry[name]
        for name in selected_models
        if name in registry
    }


def get_model_family(model_name: str) -> str:
    """
    Retourne la famille d'un modèle.
    """
    family_map = {
        "dummy": "baseline",
        "logistic_regression": "linear",
        "ridge_classifier": "linear",
        "sgd_classifier": "linear",
        "linear_svc": "svm",
        "random_forest": "ensemble_tree",
        "hist_gradient_boosting": "boosting",
        "mlp": "neural_network",
    }

    return family_map.get(model_name, "unknown")


def get_model_family_map() -> dict[str, str]:
    """
    Retourne la correspondance modèle -> famille.
    """
    return {
        name: get_model_family(name)
        for name in build_model_registry().keys()
    }


if __name__ == "__main__":
    models = get_models()

    print("Modèles actifs :")
    for name in models:
        print(f"- {name} ({get_model_family(name)})")