from __future__ import annotations

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    ACTIVE_MODELS,
    DENSE_ONLY_MODELS,
    FAST_MODELS,
    NON_NEGATIVE_FEATURE_MODELS,
    ONLINE_MODELS,
    RANDOM_STATE,
    SLOW_MODELS,
    get_active_models,
)


def build_model_registry(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Registre central de tous les modèles disponibles.
    """
    registry: Dict[str, BaseEstimator] = {
        # =========================
        # BASELINES
        # =========================
        "dummy": DummyClassifier(
            strategy="most_frequent"
        ),

        # =========================
        # LINÉAIRES
        # =========================
        "logistic_regression": LogisticRegression(
            max_iter=3000,
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
            max_iter=3000,
            tol=1e-3,
            random_state=random_state,
        ),
        "passive_aggressive": PassiveAggressiveClassifier(
            class_weight="balanced",
            max_iter=3000,
            tol=1e-3,
            random_state=random_state,
        ),
        "perceptron": Perceptron(
            class_weight="balanced",
            max_iter=3000,
            tol=1e-3,
            random_state=random_state,
        ),

        # =========================
        # ARBRES
        # =========================
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),

        # =========================
        # ENSEMBLES
        # =========================
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "bagging": BaggingClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            random_state=random_state,
        ),

        # =========================
        # NOYAUX / SVM
        # =========================
        "linear_svc": LinearSVC(
            class_weight="balanced",
            random_state=random_state,
        ),
        "svc_rbf": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),

        # =========================
        # DISTANCE
        # =========================
        "knn": KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
        ),

        # =========================
        # RÉSEAUX
        # =========================
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=random_state,
        ),

        # =========================
        # NAIVE BAYES
        # =========================
        "gaussian_nb": GaussianNB(),
        "multinomial_nb": MultinomialNB(),
    }

    return registry


def get_models(
    random_state: int = RANDOM_STATE,
    selected_models: list[str] | None = None,
) -> Dict[str, BaseEstimator]:
    """
    Retourne les modèles sélectionnés.
    Si selected_models est None, on utilise la config active.
    """
    registry = build_model_registry(random_state=random_state)

    if selected_models is None:
        selected_models = get_active_models()

    models = {
        name: registry[name]
        for name in selected_models
        if name in registry
    }

    return models


def get_all_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Retourne tous les modèles du registre.
    """
    return build_model_registry(random_state=random_state)


def get_fast_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Retourne les modèles rapides pour debug / benchmark court.
    """
    return get_models(random_state=random_state, selected_models=FAST_MODELS)


def get_online_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Retourne les modèles adaptés à l'apprentissage incrémental / online.
    """
    return get_models(random_state=random_state, selected_models=ONLINE_MODELS)


def get_dense_only_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Modèles nécessitant en pratique un pipeline dense.
    """
    return get_models(random_state=random_state, selected_models=DENSE_ONLY_MODELS)


def get_non_negative_feature_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Modèles nécessitant des features non négatives.
    """
    return get_models(random_state=random_state, selected_models=NON_NEGATIVE_FEATURE_MODELS)


def get_slow_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Modèles potentiellement lents.
    """
    available = build_model_registry(random_state=random_state)
    return {
        name: available[name]
        for name in SLOW_MODELS
        if name in available
    }


def get_sparse_compatible_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Retourne les modèles compatibles avec un pipeline majoritairement sparse.

    Exclut :
    - les modèles dense-only
    - les modèles exigeant des features non négatives
      si le preprocessing ne le garantit pas
    """
    excluded = set(DENSE_ONLY_MODELS) | set(NON_NEGATIVE_FEATURE_MODELS)

    models = get_models(random_state=random_state)
    return {
        name: model
        for name, model in models.items()
        if name not in excluded
    }


def get_model_family_map() -> dict[str, str]:
    """
    Associe chaque modèle à une famille.
    """
    return {
        "dummy": "baseline",
        "logistic_regression": "linear",
        "ridge_classifier": "linear",
        "sgd_classifier": "linear",
        "passive_aggressive": "online",
        "perceptron": "online",
        "decision_tree": "tree",
        "random_forest": "ensemble",
        "extra_trees": "ensemble",
        "adaboost": "ensemble",
        "bagging": "ensemble",
        "gradient_boosting": "ensemble",
        "hist_gradient_boosting": "ensemble",
        "linear_svc": "kernel",
        "svc_rbf": "kernel",
        "knn": "distance",
        "mlp": "neural_network",
        "gaussian_nb": "naive_bayes",
        "multinomial_nb": "naive_bayes",
    }


def get_models_by_family(
    family: str,
    random_state: int = RANDOM_STATE,
) -> Dict[str, BaseEstimator]:
    """
    Retourne les modèles d'une famille donnée.
    """
    models = get_all_models(random_state=random_state)
    family_map = get_model_family_map()

    return {
        name: model
        for name, model in models.items()
        if family_map.get(name) == family
    }


def supports_partial_fit(model_name: str) -> bool:
    """
    Indique si le modèle supporte partial_fit.
    """
    partial_fit_models = {
        "sgd_classifier",
        "passive_aggressive",
        "perceptron",
    }
    return model_name in partial_fit_models


def get_partial_fit_models(
    random_state: int = RANDOM_STATE,
) -> Dict[str, BaseEstimator]:
    """
    Retourne les modèles supportant partial_fit.
    """
    models = get_all_models(random_state=random_state)
    return {
        name: model
        for name, model in models.items()
        if supports_partial_fit(name)
    }


if __name__ == "__main__":
    models = get_models()

    print("Modèles actifs :")
    for name in models:
        print("-", name)

    print("\nFamilles :")
    for model_name, family in get_model_family_map().items():
        print(f"{model_name}: {family}")