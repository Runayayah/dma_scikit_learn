from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DATASET_CONFIG


# =========================================================
# HELPERS GÉNÉRAUX
# =========================================================

def _clip_array(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _safe_choice(rng: np.random.Generator, values: list[str], size: int, p=None) -> np.ndarray:
    return rng.choice(values, size=size, replace=True, p=p)


def _cardinality_to_n_categories(cardinality_level: int, feature_idx: int) -> int:
    """
    Convertit un niveau de cardinalité en nombre de catégories.
    """
    base_map = {
        1: 3,    # faible
        2: 8,    # moyenne
        3: 20,   # forte
        4: 60,   # extrême
    }
    base = base_map.get(cardinality_level, 8)
    return base + feature_idx * max(1, base // 4)


def _text_length_to_sentence_count(text_length: str) -> tuple[int, int]:
    """
    Retourne un intervalle de nombre de phrases.
    """
    mapping = {
        "short": (2, 4),
        "medium": (4, 7),
        "long": (7, 11),
    }
    return mapping.get(text_length, (4, 7))


# =========================================================
# BLOC NUMÉRIQUE ASSURANCE
# =========================================================

def make_numeric_block(
    rng: np.random.Generator,
    n_samples: int,
    n_num_features: int,
) -> pd.DataFrame:
    """
    Génère les variables numériques assurance.
    Si n_num_features > nombre de variables métier de base,
    on ajoute des variables numériques dérivées.
    """
    age = rng.integers(18, 86, size=n_samples).astype(float)
    annual_income = _clip_array(rng.normal(36000, 16000, size=n_samples), 8000, 180000)
    contract_tenure_years = rng.integers(0, 26, size=n_samples).astype(float)

    license_age_at_start = rng.integers(18, 30, size=n_samples)
    years_with_license = np.maximum(age - license_age_at_start, 0).astype(float)

    vehicle_age = rng.integers(0, 21, size=n_samples).astype(float)
    annual_mileage = _clip_array(rng.normal(15000, 7000, size=n_samples), 1000, 70000)
    previous_claims_count = rng.poisson(lam=0.8, size=n_samples).astype(float)
    late_payments_count = rng.poisson(lam=0.4, size=n_samples).astype(float)
    credit_score = _clip_array(rng.normal(650, 110, size=n_samples), 300, 900)
    household_size = rng.integers(1, 7, size=n_samples).astype(float)
    insured_value = _clip_array(rng.normal(18000, 10000, size=n_samples), 2000, 150000)
    monthly_premium = _clip_array(rng.normal(65, 25, size=n_samples), 10, 400)
    years_at_address = rng.integers(0, 31, size=n_samples).astype(float)
    payment_incident_ratio = _clip_array(
        late_payments_count / np.maximum(contract_tenure_years + 1, 1),
        0,
        5,
    )
    claim_cost_history = _clip_array(
        previous_claims_count * rng.normal(1500, 700, size=n_samples),
        0,
        20000,
    )

    base_features = {
        "num_age": age,
        "num_annual_income": annual_income,
        "num_contract_tenure_years": contract_tenure_years,
        "num_years_with_license": years_with_license,
        "num_vehicle_age": vehicle_age,
        "num_annual_mileage": annual_mileage,
        "num_previous_claims_count": previous_claims_count,
        "num_late_payments_count": late_payments_count,
        "num_credit_score": credit_score,
        "num_household_size": household_size,
        "num_insured_value": insured_value,
        "num_monthly_premium": monthly_premium,
        "num_years_at_address": years_at_address,
        "num_payment_incident_ratio": payment_incident_ratio,
        "num_claim_cost_history": claim_cost_history,
    }

    df = pd.DataFrame(base_features)

    if n_num_features <= 0:
        return pd.DataFrame(index=np.arange(n_samples))

    current_cols = list(df.columns)

    if n_num_features < len(current_cols):
        df = df[current_cols[:n_num_features]].copy()
        return df

    # Ajout de variables dérivées si besoin
    extra_idx = 0
    while df.shape[1] < n_num_features:
        source_a = df[current_cols[extra_idx % len(current_cols)]].to_numpy()
        source_b = df[current_cols[(extra_idx + 3) % len(current_cols)]].to_numpy()

        mode = extra_idx % 5
        if mode == 0:
            values = 0.6 * source_a + 0.4 * source_b + rng.normal(0, 0.05 * np.std(source_a), size=n_samples)
        elif mode == 1:
            values = np.log1p(np.abs(source_a))
        elif mode == 2:
            values = np.sqrt(np.abs(source_b) + 1.0)
        elif mode == 3:
            values = source_a * 0.01 + rng.normal(0, 1, size=n_samples)
        else:
            values = np.sin(source_a / (np.std(source_a) + 1e-6))

        df[f"num_extra_{extra_idx}"] = values.astype(float)
        extra_idx += 1

    return df


# =========================================================
# BLOC CATÉGORIEL ASSURANCE
# =========================================================

def make_categorical_block(
    rng: np.random.Generator,
    n_samples: int,
    n_cat_features: int,
    cardinality_level: int,
) -> pd.DataFrame:
    """
    Génère les variables catégorielles assurance.
    """
    if n_cat_features <= 0:
        return pd.DataFrame(index=np.arange(n_samples))

    feature_builders: list[tuple[str, callable]] = [
        (
            "cat_region",
            lambda n: _safe_choice(
                rng,
                ["urbain", "periurbain", "rural"],
                size=n,
                p=[0.5, 0.3, 0.2],
            ),
        ),
        (
            "cat_vehicle_type",
            lambda n: _safe_choice(
                rng,
                ["citadine", "berline", "suv", "utilitaire", "sport"],
                size=n,
                p=[0.28, 0.24, 0.2, 0.16, 0.12],
            ),
        ),
        (
            "cat_fuel_type",
            lambda n: _safe_choice(
                rng,
                ["essence", "diesel", "hybride", "electrique"],
                size=n,
                p=[0.35, 0.25, 0.25, 0.15],
            ),
        ),
        (
            "cat_coverage_level",
            lambda n: _safe_choice(
                rng,
                ["tiers", "tiers_plus", "tous_risques"],
                size=n,
                p=[0.35, 0.3, 0.35],
            ),
        ),
        (
            "cat_payment_frequency",
            lambda n: _safe_choice(
                rng,
                ["mensuel", "trimestriel", "annuel"],
                size=n,
                p=[0.6, 0.15, 0.25],
            ),
        ),
        (
            "cat_home_ownership",
            lambda n: _safe_choice(
                rng,
                ["locataire", "proprietaire"],
                size=n,
                p=[0.45, 0.55],
            ),
        ),
        (
            "cat_profession_segment",
            lambda n: _safe_choice(
                rng,
                ["cadre", "employe", "independant", "etudiant", "retraite", "sans_emploi"],
                size=n,
                p=[0.2, 0.3, 0.12, 0.08, 0.2, 0.1],
            ),
        ),
        (
            "cat_prior_fraud_flag",
            lambda n: _safe_choice(
                rng,
                ["non", "oui"],
                size=n,
                p=[0.965, 0.035],
            ),
        ),
        (
            "cat_channel",
            lambda n: _safe_choice(
                rng,
                ["agence", "web", "courtier", "telephone"],
                size=n,
                p=[0.3, 0.35, 0.2, 0.15],
            ),
        ),
        (
            "cat_contract_type",
            lambda n: _safe_choice(
                rng,
                ["auto", "habitation", "auto_habitation"],
                size=n,
                p=[0.45, 0.25, 0.30],
            ),
        ),
    ]

    data: dict[str, np.ndarray] = {}
    base_count = min(n_cat_features, len(feature_builders))

    for i in range(base_count):
        name, builder = feature_builders[i]
        data[name] = builder(n_samples)

    # Variables catégorielles synthétiques additionnelles pilotées par cardinalité
    extra_needed = n_cat_features - base_count
    for extra_idx in range(extra_needed):
        n_categories = _cardinality_to_n_categories(cardinality_level, extra_idx)
        categories = [f"segment_{extra_idx}_{j}" for j in range(n_categories)]

        # distribution non uniforme pour créer catégories rares
        raw_weights = np.exp(-np.linspace(0, 3, n_categories))
        raw_weights = raw_weights / raw_weights.sum()

        data[f"cat_extra_{extra_idx}"] = rng.choice(
            categories,
            size=n_samples,
            replace=True,
            p=raw_weights,
        )

    return pd.DataFrame(data)


# =========================================================
# TEXTE ASSURANCE RICHE
# =========================================================

def _build_text_vocabulary(vocab_size: int) -> list[str]:
    domain_terms = [
        "sinistre", "responsable", "non_responsable", "franchise", "garantie",
        "avenant", "cotisation", "tarification", "surprime", "souscription",
        "résiliation", "indemnisation", "déclaration", "expertise", "échéance",
        "gestionnaire", "historique", "dossier", "analyse", "exposition",
        "risque", "incident", "paiement", "retard", "stabilité",
        "ancienneté", "vigilance", "contrat", "client", "véhicule",
        "habitation", "usage", "professionnel", "trajet", "urbain",
        "kilométrage", "score", "profil", "sévérité", "fréquence",
    ]

    extra_tokens = [f"token_{i}" for i in range(max(0, vocab_size - len(domain_terms)))]
    vocab = domain_terms + extra_tokens
    return vocab[:vocab_size]


def make_text_block(
    rng: np.random.Generator,
    n_samples: int,
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    vocab_size: int,
    text_length: str,
) -> pd.DataFrame:
    """
    Génère une note textuelle assurance semi-rédigée.
    """
    vocabulary = _build_text_vocabulary(vocab_size)
    min_sentences, max_sentences = _text_length_to_sentence_count(text_length)

    low_risk_sentences = [
        "Le profil semble stable au regard de l'historique disponible.",
        "Aucun élément majeur de vigilance n'est identifié à ce stade.",
        "Le client présente un comportement de paiement régulier.",
        "Le dossier ne montre pas de fréquence élevée de sinistres.",
        "L'ancienneté contractuelle contribue à un profil plutôt maîtrisé.",
        "Le risque observé reste contenu dans les standards du portefeuille.",
    ]

    high_risk_sentences = [
        "Le dossier présente plusieurs facteurs aggravants nécessitant une vigilance renforcée.",
        "Une fréquence de sinistres supérieure à la moyenne est observée.",
        "Le comportement de paiement apparaît irrégulier sur la période récente.",
        "Le profil ressort comme plus exposé que la moyenne du portefeuille.",
        "Le niveau de risque estimé est supérieur au standard de souscription.",
        "La combinaison usage intensif et historique incidenté augmente l'exposition.",
    ]

    neutral_sentences = [
        "Les éléments déclaratifs ont été consolidés dans le dossier.",
        "Les informations contractuelles sont cohérentes avec la demande en cours.",
        "Le gestionnaire a intégré les mises à jour du profil assuré.",
        "Le dossier a fait l'objet d'une revue selon le processus standard.",
        "Les informations disponibles sont compatibles avec une analyse automatisée.",
        "La fiche client a été révisée lors du dernier contrôle qualité.",
    ]

    template_sentences = [
        "Le client réside principalement en zone {region}.",
        "Le véhicule couvert est de type {vehicle_type}.",
        "Le niveau de garantie demandé correspond à une formule {coverage}.",
        "Le kilométrage annuel déclaré est {mileage_desc}.",
        "L'ancienneté du contrat peut être qualifiée de {tenure_desc}.",
        "Le profil de conduite est rapproché d'un usage {usage_desc}.",
        "L'historique de sinistres est décrit comme {claims_desc}.",
        "Le niveau de paiement observé est {payment_desc}.",
        "Le contexte socioprofessionnel est associé à la catégorie {profession}.",
        "Le véhicule assuré présente une ancienneté {vehicle_age_desc}.",
        "Le dossier de souscription est associé à un canal {channel}.",
        "Le type de contrat principal ressort comme {contract_type}.",
    ]

    texts: list[str] = []
    latent_signal: list[int] = []

    for i in range(n_samples):
        # récupération sécurisée
        age = float(num_df.iloc[i]["num_age"]) if "num_age" in num_df.columns else 40.0
        mileage = float(num_df.iloc[i]["num_annual_mileage"]) if "num_annual_mileage" in num_df.columns else 12000.0
        claims = float(num_df.iloc[i]["num_previous_claims_count"]) if "num_previous_claims_count" in num_df.columns else 0.0
        late_payments = float(num_df.iloc[i]["num_late_payments_count"]) if "num_late_payments_count" in num_df.columns else 0.0
        tenure = float(num_df.iloc[i]["num_contract_tenure_years"]) if "num_contract_tenure_years" in num_df.columns else 3.0
        vehicle_age = float(num_df.iloc[i]["num_vehicle_age"]) if "num_vehicle_age" in num_df.columns else 5.0
        credit_score = float(num_df.iloc[i]["num_credit_score"]) if "num_credit_score" in num_df.columns else 650.0

        region = str(cat_df.iloc[i]["cat_region"]) if "cat_region" in cat_df.columns else "urbain"
        vehicle_type = str(cat_df.iloc[i]["cat_vehicle_type"]) if "cat_vehicle_type" in cat_df.columns else "berline"
        coverage = str(cat_df.iloc[i]["cat_coverage_level"]) if "cat_coverage_level" in cat_df.columns else "tiers_plus"
        profession = str(cat_df.iloc[i]["cat_profession_segment"]) if "cat_profession_segment" in cat_df.columns else "employe"
        fraud_flag = str(cat_df.iloc[i]["cat_prior_fraud_flag"]) if "cat_prior_fraud_flag" in cat_df.columns else "non"
        channel = str(cat_df.iloc[i]["cat_channel"]) if "cat_channel" in cat_df.columns else "web"
        contract_type = str(cat_df.iloc[i]["cat_contract_type"]) if "cat_contract_type" in cat_df.columns else "auto"

        mileage_desc = (
            "faible" if mileage < 8000 else
            "modéré" if mileage < 18000 else
            "élevé"
        )
        tenure_desc = (
            "récente" if tenure <= 1 else
            "intermédiaire" if tenure <= 7 else
            "ancienne"
        )
        usage_desc = (
            "occasionnel" if mileage < 7000 else
            "classique" if mileage < 18000 else
            "intensif"
        )
        claims_desc = (
            "sans antécédent récent" if claims == 0 else
            "ponctuel" if claims == 1 else
            "répété"
        )
        payment_desc = (
            "régulier" if late_payments == 0 else
            "globalement correct" if late_payments == 1 else
            "irrégulier"
        )
        vehicle_age_desc = (
            "faible" if vehicle_age <= 3 else
            "intermédiaire" if vehicle_age <= 10 else
            "élevée"
        )

        # score latent texte
        text_score = 0.0
        text_score += 1.2 if age < 25 else 0.0
        text_score += 0.7 if age > 75 else 0.0
        text_score += 0.8 if mileage > 20000 else 0.0
        text_score += 1.0 if claims >= 2 else 0.0
        text_score += 0.8 if late_payments >= 2 else 0.0
        text_score += 0.8 if vehicle_age > 12 else 0.0
        text_score += 2.0 if fraud_flag == "oui" else 0.0
        text_score += 0.4 if region == "urbain" else 0.0
        text_score += 0.5 if profession in {"etudiant", "sans_emploi"} else 0.0
        text_score -= 0.7 if tenure > 8 else 0.0
        text_score -= 0.7 if credit_score > 750 else 0.0

        if text_score >= 2.3:
            latent = 1
        elif text_score <= 0.0:
            latent = -1
        else:
            latent = 0

        n_sentences = int(rng.integers(min_sentences, max_sentences + 1))
        sentences: list[str] = []

        formatted_templates = [
            t.format(
                region=region,
                vehicle_type=vehicle_type,
                coverage=coverage,
                mileage_desc=mileage_desc,
                tenure_desc=tenure_desc,
                usage_desc=usage_desc,
                claims_desc=claims_desc,
                payment_desc=payment_desc,
                profession=profession,
                vehicle_age_desc=vehicle_age_desc,
                channel=channel,
                contract_type=contract_type,
            )
            for t in template_sentences
        ]

        # 1-2 phrases orientées risque
        if latent == 1:
            sentences.extend(list(rng.choice(high_risk_sentences, size=2, replace=True)))
        elif latent == -1:
            sentences.extend(list(rng.choice(low_risk_sentences, size=2, replace=True)))
        else:
            sentences.extend(list(rng.choice(neutral_sentences, size=2, replace=True)))

        # phrases semi-structurées
        extra_pool = formatted_templates + neutral_sentences
        remaining = max(0, n_sentences - len(sentences))
        if remaining > 0:
            sentences.extend(list(rng.choice(extra_pool, size=remaining, replace=True)))

        # insertion de tokens de bruit / jargon
        noise_token_count = max(1, vocab_size // 40)
        sampled_tokens = list(rng.choice(vocabulary, size=noise_token_count, replace=False if noise_token_count <= len(vocabulary) else True))
        sentences.append("Mots-clés dossier : " + ", ".join(sampled_tokens) + ".")

        rng.shuffle(sentences)
        text = " ".join(sentences)

        texts.append(text)
        latent_signal.append(latent)

    return pd.DataFrame(
        {
            "text_feature": texts,
            "_latent_text_signal": latent_signal,
        }
    )


# =========================================================
# CIBLES
# =========================================================

def build_target_binary(
    rng: np.random.Generator,
    num_df: pd.DataFrame,
    cat_df: pd.DataFrame,
    latent_text_signal: np.ndarray,
    noise_scale: float,
    class_imbalance: float,
) -> np.ndarray:
    """
    1 = client à risque élevé
    0 = client à risque faible
    """
    def get_num(col: str, default: float = 0.0) -> np.ndarray:
        if col in num_df.columns:
            return num_df[col].to_numpy(dtype=float)
        return np.full(len(num_df), default, dtype=float)

    def get_cat(col: str, default: str = "unknown") -> np.ndarray:
        if col in cat_df.columns:
            return cat_df[col].astype(str).to_numpy()
        return np.full(len(num_df), default, dtype=object)

    age = get_num("num_age", 40.0)
    income = get_num("num_annual_income", 35000.0)
    tenure = get_num("num_contract_tenure_years", 3.0)
    years_license = get_num("num_years_with_license", 10.0)
    vehicle_age = get_num("num_vehicle_age", 5.0)
    mileage = get_num("num_annual_mileage", 15000.0)
    claims = get_num("num_previous_claims_count", 0.0)
    late_payments = get_num("num_late_payments_count", 0.0)
    credit_score = get_num("num_credit_score", 650.0)
    insured_value = get_num("num_insured_value", 15000.0)
    monthly_premium = get_num("num_monthly_premium", 60.0)

    region = get_cat("cat_region", "urbain")
    vehicle_type = get_cat("cat_vehicle_type", "berline")
    coverage = get_cat("cat_coverage_level", "tiers_plus")
    payment_frequency = get_cat("cat_payment_frequency", "mensuel")
    profession = get_cat("cat_profession_segment", "employe")
    fraud_flag = get_cat("cat_prior_fraud_flag", "non")
    channel = get_cat("cat_channel", "web")

    score = np.zeros(len(num_df), dtype=float)

    # effets métier
    score += 1.25 * claims
    score += 1.05 * late_payments
    score += 0.000045 * mileage
    score += 0.07 * vehicle_age
    score += -0.0032 * credit_score
    score += -0.000014 * income
    score += -0.05 * tenure
    score += -0.025 * years_license
    score += 0.000004 * insured_value
    score += 0.006 * monthly_premium

    # non-linéarités
    score += np.where(age < 25, 1.25, 0.0)
    score += np.where(age > 75, 0.60, 0.0)
    score += np.where((vehicle_age > 12) & (mileage > 22000), 1.1, 0.0)
    score += np.where((claims >= 2) & (late_payments >= 1), 1.4, 0.0)
    score += np.where((credit_score < 500) & (late_payments >= 2), 1.1, 0.0)

    # catégoriel
    score += np.where(region == "urbain", 0.45, 0.0)
    score += np.where(vehicle_type == "sport", 1.0, 0.0)
    score += np.where(vehicle_type == "utilitaire", 0.35, 0.0)
    score += np.where(coverage == "tiers", 0.25, 0.0)
    score += np.where(payment_frequency == "mensuel", 0.20, 0.0)
    score += np.where(fraud_flag == "oui", 2.4, 0.0)
    score += np.where(profession == "etudiant", 0.45, 0.0)
    score += np.where(profession == "sans_emploi", 0.7, 0.0)
    score += np.where(channel == "courtier", 0.1, 0.0)

    # texte latent
    score += 0.85 * latent_text_signal

    # bruit
    score += rng.normal(0, noise_scale, size=len(num_df))

    threshold = np.quantile(score, 1 - class_imbalance)
    target = (score > threshold).astype(int)

    return target


# =========================================================
# PERTURBATIONS / ROBUSTESSE
# =========================================================

def inject_label_noise(
    rng: np.random.Generator,
    y: np.ndarray,
    label_noise_rate: float,
) -> np.ndarray:
    """
    Inverse aléatoirement une proportion de labels.
    """
    if label_noise_rate <= 0:
        return y

    y_noisy = y.copy()
    mask = rng.random(len(y_noisy)) < label_noise_rate
    y_noisy[mask] = 1 - y_noisy[mask]
    return y_noisy


def inject_missing_values(
    rng: np.random.Generator,
    df: pd.DataFrame,
    missing_rate: float,
) -> pd.DataFrame:
    """
    Injecte des valeurs manquantes dans les features seulement.
    """
    if missing_rate <= 0:
        return df

    df = df.copy()
    feature_cols = [col for col in df.columns if col != "target"]

    for col in feature_cols:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    return df


def inject_outliers(
    rng: np.random.Generator,
    df: pd.DataFrame,
    outlier_rate: float,
) -> pd.DataFrame:
    """
    Injecte des outliers dans les colonnes numériques.
    """
    if outlier_rate <= 0:
        return df

    df = df.copy()
    numeric_cols = [col for col in df.columns if col.startswith("num_")]

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].astype(float)

    for col in numeric_cols:
        mask = rng.random(len(df)) < outlier_rate
        if not mask.any():
            continue

        multiplier = rng.uniform(1.8, 4.0)
        col_values = df.loc[mask, col].to_numpy(dtype=float)

        # mélange multiplicatif + additif
        std = np.nanstd(df[col].to_numpy(dtype=float))
        std = std if std > 0 else 1.0
        col_values = col_values * multiplier + rng.normal(0, std, size=len(col_values))
        df.loc[mask, col] = col_values

    return df


def add_irrelevant_features(
    rng: np.random.Generator,
    df: pd.DataFrame,
    n_irrelevant_features: int,
) -> pd.DataFrame:
    if n_irrelevant_features <= 0:
        return df

    df = df.copy()
    for i in range(n_irrelevant_features):
        df[f"num_irrelevant_{i}"] = rng.normal(0, 1, size=len(df))

    return df


def add_duplicated_features(
    rng: np.random.Generator,
    df: pd.DataFrame,
    n_duplicated_features: int,
) -> pd.DataFrame:
    if n_duplicated_features <= 0:
        return df

    df = df.copy()
    source_cols = [col for col in df.columns if col.startswith("num_")]

    if not source_cols:
        return df

    for i in range(n_duplicated_features):
        source_col = source_cols[i % len(source_cols)]
        noise = rng.normal(0, 1e-3, size=len(df))
        df[f"num_duplicate_{i}"] = df[source_col].astype(float) + noise

    return df


def add_constant_features(
    df: pd.DataFrame,
    n_constant_features: int,
) -> pd.DataFrame:
    if n_constant_features <= 0:
        return df

    df = df.copy()
    for i in range(n_constant_features):
        df[f"num_constant_{i}"] = 1.0
    return df


def add_collinearity(
    rng: np.random.Generator,
    df: pd.DataFrame,
    strength: float,
) -> pd.DataFrame:
    """
    Ajoute des colonnes fortement corrélées à des colonnes existantes.
    """
    if strength <= 0:
        return df

    df = df.copy()
    source_cols = [col for col in df.columns if col.startswith("num_")]
    if len(source_cols) < 2:
        return df

    n_new = min(3, len(source_cols))
    for i in range(n_new):
        source_col = source_cols[i]
        source = df[source_col].to_numpy(dtype=float)
        noise = rng.normal(0, np.std(source) * max(1 - strength, 1e-3), size=len(df))
        new_values = strength * source + noise
        df[f"num_collinear_{i}"] = new_values

    return df


def simulate_distribution_shift(
    df: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """
    Introduit un décalage léger de distribution dans tout le dataset.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    numeric_cols = [col for col in df.columns if col.startswith("num_")]
    for col in numeric_cols:
        shift = rng.normal(0.2, 0.1)
        scale = rng.uniform(1.02, 1.15)
        values = df[col].to_numpy(dtype=float)
        df[col] = values * scale + shift

    if "text_feature" in df.columns:
        df["text_feature"] = df["text_feature"].astype(str) + " revue_shift portefeuille recalibrage."

    return df


def simulate_temporal_drift(
    df: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """
    Simule un drift temporel.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    n = len(df)
    time_index = np.linspace(0, 1, n)

    numeric_cols = [col for col in df.columns if col.startswith("num_")]
    for col in numeric_cols:
        values = df[col].to_numpy(dtype=float)
        drift = time_index * rng.uniform(-0.5, 0.5) * (np.nanstd(values) + 1e-6)
        df[col] = values + drift

    if "text_feature" in df.columns:
        suffix = np.where(
            time_index > 0.7,
            " mise_a_jour_tarifaire periode_recente",
            " historique_stable",
        )
        df["text_feature"] = df["text_feature"].astype(str) + suffix

    return df


# =========================================================
# API PRINCIPALE
# =========================================================

def generate_dataset(
    n_samples: int = 5000,
    n_num_features: int = 12,
    n_cat_features: int = 8,
    vocab_size: int = 150,
    text_length: str = "medium",
    missing_rate: float = 0.05,
    outlier_rate: float = 0.02,
    noise_scale: float = 1.0,
    label_noise_rate: float = 0.0,
    class_imbalance: float = 0.30,
    cardinality_level: int = 2,
    add_irrelevant_features: bool = True,
    n_irrelevant_features: int = 5,
    add_duplicated_features: bool = True,
    n_duplicated_features: int = 2,
    add_constant_features: bool = True,
    n_constant_features: int = 1,
    add_collinearity: bool = True,
    collinearity_strength: float = 0.90,
    simulate_distribution_shift: bool = False,
    simulate_temporal_drift: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Génère un dataset assurance de classification binaire.
    """
    rng = np.random.default_rng(random_state)

    num_df = make_numeric_block(
        rng=rng,
        n_samples=n_samples,
        n_num_features=n_num_features,
    )

    cat_df = make_categorical_block(
        rng=rng,
        n_samples=n_samples,
        n_cat_features=n_cat_features,
        cardinality_level=cardinality_level,
    )

    text_df = make_text_block(
        rng=rng,
        n_samples=n_samples,
        num_df=num_df,
        cat_df=cat_df,
        vocab_size=vocab_size,
        text_length=text_length,
    )

    target = build_target_binary(
        rng=rng,
        num_df=num_df,
        cat_df=cat_df,
        latent_text_signal=text_df["_latent_text_signal"].to_numpy(),
        noise_scale=noise_scale,
        class_imbalance=class_imbalance,
    )

    target = inject_label_noise(
        rng=rng,
        y=target,
        label_noise_rate=label_noise_rate,
    )

    df = pd.concat(
        [
            num_df,
            cat_df,
            text_df.drop(columns=["_latent_text_signal"]),
        ],
        axis=1,
    )

    df["target"] = target

    # enrichissements structurels
    if add_irrelevant_features:
        df = add_irrelevant_features_fn(
            rng=rng,
            df=df,
            n_irrelevant_features=n_irrelevant_features,
        )

    if add_duplicated_features:
        df = add_duplicated_features_fn(
            rng=rng,
            df=df,
            n_duplicated_features=n_duplicated_features,
        )

    if add_constant_features:
        df = add_constant_features_fn(
            df=df,
            n_constant_features=n_constant_features,
        )

    if add_collinearity:
        df = add_collinearity_fn(
            rng=rng,
            df=df,
            strength=collinearity_strength,
        )

    # perturbations
    df = inject_outliers(rng=rng, df=df, outlier_rate=outlier_rate)
    df = inject_missing_values(rng=rng, df=df, missing_rate=missing_rate)

    if simulate_distribution_shift:
        df = simulate_distribution_shift_fn(df=df, random_state=random_state + 100)

    if simulate_temporal_drift:
        df = simulate_temporal_drift_fn(df=df, random_state=random_state + 200)

    return df


# alias pour éviter les collisions de noms avec booléens de signature
add_irrelevant_features_fn = add_irrelevant_features
add_duplicated_features_fn = add_duplicated_features
add_constant_features_fn = add_constant_features
add_collinearity_fn = add_collinearity
simulate_distribution_shift_fn = simulate_distribution_shift
simulate_temporal_drift_fn = simulate_temporal_drift


def generate_default_dataset() -> pd.DataFrame:
    return generate_dataset(**DATASET_CONFIG)


if __name__ == "__main__":
    df = generate_default_dataset()

    print("Aperçu du dataset :")
    print(df.head())

    print("\nShape :", df.shape)

    print("\nRépartition de target :")
    if "target" in df.columns:
        print(df["target"].value_counts(normalize=True))

    print("\nColonnes :")
    print(df.columns.tolist())

    print("\nTaux de valeurs manquantes :")
    print(df.isna().mean().sort_values(ascending=False).head(15))