from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)


# =========================================================
# HELPERS
# =========================================================

def _extract_scores_or_labels(estimator, X):
    """
    Récupère un score continu si possible, sinon des labels.
    """
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1], "score"
        except Exception:
            pass

    if hasattr(estimator, "decision_function"):
        try:
            scores = estimator.decision_function(X)
            return scores, "score"
        except Exception:
            pass

    return estimator.predict(X), "label"


def _scores_to_labels(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def business_cost_from_predictions(
    y_true,
    y_pred,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    tp_gain: float = 0.0,
    tn_gain: float = 0.0,
) -> float:
    """
    Coût métier total.
    Plus petit = mieux.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_cost = fp_cost * fp + fn_cost * fn - tp_gain * tp - tn_gain * tn
    return float(total_cost)


def business_gain_from_predictions(
    y_true,
    y_pred,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    tp_gain: float = 1.0,
    tn_gain: float = 0.2,
) -> float:
    """
    Gain métier total.
    Plus grand = mieux.
    """
    return -business_cost_from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tp_gain=tp_gain,
        tn_gain=tn_gain,
    )


# =========================================================
# SCORES MÉTIER
# =========================================================

def false_negative_penalty_score(
    y_true,
    y_pred,
    fn_weight: float = 5.0,
    fp_weight: float = 1.0,
) -> float:
    """
    Score pénalisant fortement les faux négatifs.
    Plus grand = mieux.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    penalty = fn_weight * fn + fp_weight * fp
    max_penalty = fn_weight * len(y_true) + fp_weight * len(y_true)

    score = 1.0 - (penalty / max_penalty)
    return float(score)


def insurance_risk_score(
    y_true,
    y_pred,
    recall_weight: float = 0.6,
    precision_weight: float = 0.4,
) -> float:
    """
    Score métier assurance :
    privilégie le recall, tout en gardant la précision.
    Plus grand = mieux.
    """
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    total_weight = recall_weight + precision_weight
    score = (
        recall_weight * recall + precision_weight * precision
    ) / total_weight

    return float(score)


def thresholded_business_gain_score(
    estimator,
    X,
    y_true,
    threshold: float = 0.5,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    tp_gain: float = 1.0,
    tn_gain: float = 0.2,
) -> float:
    """
    Scorer métier basé sur les scores continus + seuil.
    Plus grand = mieux.
    """
    outputs, output_type = _extract_scores_or_labels(estimator, X)

    if output_type == "score":
        y_pred = _scores_to_labels(np.asarray(outputs), threshold=threshold)
    else:
        y_pred = np.asarray(outputs)

    return business_gain_from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tp_gain=tp_gain,
        tn_gain=tn_gain,
    )


def thresholded_false_negative_penalty_score(
    estimator,
    X,
    y_true,
    threshold: float = 0.5,
    fn_weight: float = 5.0,
    fp_weight: float = 1.0,
) -> float:
    """
    Version estimator-aware du score de pénalisation FN.
    """
    outputs, output_type = _extract_scores_or_labels(estimator, X)

    if output_type == "score":
        y_pred = _scores_to_labels(np.asarray(outputs), threshold=threshold)
    else:
        y_pred = np.asarray(outputs)

    return false_negative_penalty_score(
        y_true=y_true,
        y_pred=y_pred,
        fn_weight=fn_weight,
        fp_weight=fp_weight,
    )


def thresholded_insurance_risk_score(
    estimator,
    X,
    y_true,
    threshold: float = 0.5,
    recall_weight: float = 0.6,
    precision_weight: float = 0.4,
) -> float:
    """
    Version estimator-aware du score assurance.
    """
    outputs, output_type = _extract_scores_or_labels(estimator, X)

    if output_type == "score":
        y_pred = _scores_to_labels(np.asarray(outputs), threshold=threshold)
    else:
        y_pred = np.asarray(outputs)

    return insurance_risk_score(
        y_true=y_true,
        y_pred=y_pred,
        recall_weight=recall_weight,
        precision_weight=precision_weight,
    )


# =========================================================
# SCORERS SKLEARN
# =========================================================

def get_custom_scorers() -> dict:
    """
    Retourne un dictionnaire de scorers custom compatibles sklearn.
    """
    scorers = {
        "insurance_risk_score": make_scorer(
            insurance_risk_score,
            greater_is_better=True,
        ),
        "false_negative_penalty_score": make_scorer(
            false_negative_penalty_score,
            greater_is_better=True,
        ),
    }

    return scorers


def get_estimator_aware_custom_scorers() -> dict:
    """
    Retourne des callables à utiliser directement dans GridSearchCV / cross_validate.
    """
    return {
        "business_gain": thresholded_business_gain_score,
        "fn_penalty": thresholded_false_negative_penalty_score,
        "insurance_risk": thresholded_insurance_risk_score,
    }


# =========================================================
# DIAGNOSTIC
# =========================================================

def evaluate_business_metrics(
    y_true,
    y_pred,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    tp_gain: float = 1.0,
    tn_gain: float = 0.2,
) -> dict:
    """
    Retourne un petit résumé métier lisible.
    """
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "business_cost": business_cost_from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_gain=tp_gain,
            tn_gain=tn_gain,
        ),
        "business_gain": business_gain_from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            tp_gain=tp_gain,
            tn_gain=tn_gain,
        ),
        "fn_penalty_score": false_negative_penalty_score(
            y_true=y_true,
            y_pred=y_pred,
            fn_weight=fn_cost,
            fp_weight=fp_cost,
        ),
        "insurance_risk_score": insurance_risk_score(
            y_true=y_true,
            y_pred=y_pred,
            recall_weight=0.6,
            precision_weight=0.4,
        ),
    }


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 0])

    print("Résumé métier :")
    print(evaluate_business_metrics(y_true, y_pred))

    print("\nScorers sklearn :")
    print(get_custom_scorers().keys())

    print("\nScorers estimator-aware :")
    print(get_estimator_aware_custom_scorers().keys())