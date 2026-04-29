from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# HELPERS
# =========================================================

def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("num_")]


def _get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("cat_")]


def _copy_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


# =========================================================
# LABEL NOISE
# =========================================================

def inject_label_noise(
    df: pd.DataFrame,
    target_col: str = "target",
    noise_rate: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Inverse aléatoirement une proportion de labels binaires.
    """
    if target_col not in df.columns or noise_rate <= 0:
        return df.copy()

    rng = np.random.default_rng(random_state)
    out = df.copy()

    mask = rng.random(len(out)) < noise_rate
    out.loc[mask, target_col] = 1 - out.loc[mask, target_col].astype(int)

    return out


# =========================================================
# MISSING VALUES
# =========================================================

def inject_extreme_missing_values(
    df: pd.DataFrame,
    missing_rate: float = 0.5,
    exclude_columns: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Injecte un fort taux de valeurs manquantes.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    exclude_columns = exclude_columns or ["target"]
    candidate_cols = [col for col in out.columns if col not in exclude_columns]

    for col in candidate_cols:
        mask = rng.random(len(out)) < missing_rate
        out.loc[mask, col] = np.nan

    return out


def make_columns_almost_empty(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    keep_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Rend certaines colonnes quasi vides.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    if columns is None:
        columns = [col for col in out.columns if col != "target"][:3]

    for col in columns:
        mask = rng.random(len(out)) > keep_rate
        out.loc[mask, col] = np.nan

    return out


def make_columns_fully_null(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Met certaines colonnes entièrement à null.
    """
    out = df.copy()

    if columns is None:
        columns = [col for col in out.columns if col != "target"][:2]

    for col in columns:
        out[col] = np.nan

    return out


# =========================================================
# OUTLIERS
# =========================================================

def inject_extreme_outliers(
    df: pd.DataFrame,
    outlier_rate: float = 0.2,
    multiplier_range: tuple[float, float] = (5.0, 20.0),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Injecte des outliers extrêmes dans les colonnes numériques.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    numeric_cols = _get_numeric_columns(out)
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].astype(float)

    for col in numeric_cols:
        mask = rng.random(len(out)) < outlier_rate
        if not mask.any():
            continue

        multiplier = rng.uniform(*multiplier_range)
        std = np.nanstd(out[col].to_numpy(dtype=float))
        std = std if std > 0 else 1.0

        out.loc[mask, col] = (
            out.loc[mask, col].astype(float) * multiplier
            + rng.normal(0, std * multiplier, size=mask.sum())
        )

    return out


# =========================================================
# FEATURES ADDITIONNELLES
# =========================================================

def add_irrelevant_features(
    df: pd.DataFrame,
    n_features: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Ajoute des colonnes numériques parasites.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    for i in range(n_features):
        out[f"num_irrelevant_robust_{i}"] = rng.normal(0, 1, size=len(out))

    return out


def add_duplicated_features(
    df: pd.DataFrame,
    n_features: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Duplique des colonnes numériques existantes avec très peu de bruit.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    numeric_cols = _get_numeric_columns(out)
    if not numeric_cols:
        return out

    for i in range(n_features):
        source_col = numeric_cols[i % len(numeric_cols)]
        noise = rng.normal(0, 1e-4, size=len(out))
        out[f"num_duplicate_robust_{i}"] = out[source_col].astype(float) + noise

    return out


def add_constant_columns(
    df: pd.DataFrame,
    n_features: int = 3,
    constant_value: float = 1.0,
) -> pd.DataFrame:
    """
    Ajoute des colonnes constantes.
    """
    out = df.copy()

    for i in range(n_features):
        out[f"num_constant_robust_{i}"] = constant_value

    return out


def add_collinearity(
    df: pd.DataFrame,
    strength: float = 0.95,
    n_features: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Ajoute des colonnes fortement colinéaires à des colonnes numériques existantes.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    numeric_cols = _get_numeric_columns(out)
    if not numeric_cols:
        return out

    for i in range(min(n_features, len(numeric_cols))):
        source_col = numeric_cols[i]
        source = out[source_col].astype(float).to_numpy()
        std = np.nanstd(source)
        std = std if std > 0 else 1.0
        noise = rng.normal(0, std * max(1 - strength, 1e-3), size=len(out))
        out[f"num_collinear_robust_{i}"] = strength * source + noise

    return out


# =========================================================
# CATÉGORIES / UNSEEN TEST
# =========================================================

def simulate_unseen_categories_in_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str] | None = None,
    unseen_rate: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remplace dans le test une partie des catégories par des catégories jamais vues en train.
    """
    rng = np.random.default_rng(random_state)
    train = X_train.copy()
    test = X_test.copy()

    if columns is None:
        columns = [col for col in test.columns if col.startswith("cat_")]

    for col in columns:
        if col not in test.columns:
            continue

        mask = rng.random(len(test)) < unseen_rate
        unseen_values = [f"__UNSEEN_{col}_{i}__" for i in range(mask.sum())]
        if mask.sum() > 0:
            test.loc[mask, col] = unseen_values

    return train, test


# =========================================================
# TEXTE
# =========================================================

def make_empty_text(
    df: pd.DataFrame,
    empty_rate: float = 0.5,
    text_col: str = "text_feature",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Vide une proportion des textes.
    """
    if text_col not in df.columns:
        return df.copy()

    rng = np.random.default_rng(random_state)
    out = df.copy()

    mask = rng.random(len(out)) < empty_rate
    out.loc[mask, text_col] = ""

    return out


def make_noisy_text(
    df: pd.DataFrame,
    noise_rate: float = 0.5,
    text_col: str = "text_feature",
    vocab_size: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Ajoute du bruit textuel dans une proportion des lignes.
    """
    if text_col not in df.columns:
        return df.copy()

    rng = np.random.default_rng(random_state)
    out = df.copy()

    noise_vocab = [f"noise_token_{i}" for i in range(vocab_size)]
    mask = rng.random(len(out)) < noise_rate

    for idx in out.index[mask]:
        extra_len = int(rng.integers(5, 20))
        extra_tokens = rng.choice(noise_vocab, size=extra_len, replace=True)
        base_text = "" if pd.isna(out.at[idx, text_col]) else str(out.at[idx, text_col])
        out.at[idx, text_col] = base_text + " " + " ".join(extra_tokens)

    return out


def make_very_long_text(
    df: pd.DataFrame,
    repeat_min: int = 5,
    repeat_max: int = 20,
    text_col: str = "text_feature",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Allonge fortement les textes en répétant leur contenu.
    """
    if text_col not in df.columns:
        return df.copy()

    rng = np.random.default_rng(random_state)
    out = df.copy()

    for idx in out.index:
        text = "" if pd.isna(out.at[idx, text_col]) else str(out.at[idx, text_col])
        repeats = int(rng.integers(repeat_min, repeat_max + 1))
        out.at[idx, text_col] = " ".join([text] * repeats)

    return out


# =========================================================
# SHIFT / DRIFT
# =========================================================

def simulate_distribution_shift(
    df: pd.DataFrame,
    shift_strength: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Applique un shift global sur les colonnes numériques et un léger changement texte.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    numeric_cols = _get_numeric_columns(out)
    for col in numeric_cols:
        values = out[col].astype(float).to_numpy()
        std = np.nanstd(values)
        std = std if std > 0 else 1.0
        shift = rng.normal(shift_strength, shift_strength / 2) * std
        scale = rng.uniform(1.0, 1.0 + shift_strength)
        out[col] = values * scale + shift

    if "text_feature" in out.columns:
        out["text_feature"] = out["text_feature"].astype(str) + " shift_distribution portefeuille_recent."

    return out


def simulate_temporal_drift(
    df: pd.DataFrame,
    strength: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Applique un drift progressif selon l'ordre des lignes.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    n = len(out)
    if n == 0:
        return out

    time_index = np.linspace(0, 1, n)
    numeric_cols = _get_numeric_columns(out)

    for col in numeric_cols:
        values = out[col].astype(float).to_numpy()
        std = np.nanstd(values)
        std = std if std > 0 else 1.0
        drift = time_index * strength * std * rng.uniform(-1, 1)
        out[col] = values + drift

    if "text_feature" in out.columns:
        suffix = np.where(
            time_index > 0.7,
            " dérive_temporelle_tarifaire récente",
            " profil_historique_stable",
        )
        out["text_feature"] = out["text_feature"].astype(str) + suffix

    return out


# =========================================================
# DONNÉES SALES
# =========================================================

def corrupt_numeric_columns_with_strings(
    df: pd.DataFrame,
    corruption_rate: float = 0.05,
    columns: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Injecte des strings dans certaines colonnes numériques.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    if columns is None:
        columns = _get_numeric_columns(out)[:3]

    corrupt_tokens = ["unknown", "error", "missing", "N/A", "bad_value"]

    for col in columns:
        if col not in out.columns:
            continue

        mask = rng.random(len(out)) < corruption_rate
        if mask.any():
            out.loc[mask, col] = rng.choice(corrupt_tokens, size=mask.sum())

    return out


def mix_numeric_and_string_types(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    string_rate: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Mélange types numériques et chaînes dans des colonnes numériques.
    """
    return corrupt_numeric_columns_with_strings(
        df=df,
        corruption_rate=string_rate,
        columns=columns,
        random_state=random_state,
    )


# =========================================================
# SCÉNARIOS PRÊTS À L'EMPLOI
# =========================================================

def apply_robustness_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    target_col: str = "target",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Applique un scénario standard de robustesse.
    """
    scenario_name = scenario_name.lower().strip()

    if scenario_name == "baseline":
        return df.copy()

    if scenario_name == "high_missing":
        return inject_extreme_missing_values(
            df=df,
            missing_rate=0.50,
            exclude_columns=[target_col],
            random_state=random_state,
        )

    if scenario_name == "high_outliers":
        return inject_extreme_outliers(
            df=df,
            outlier_rate=0.25,
            random_state=random_state,
        )

    if scenario_name == "label_noise":
        return inject_label_noise(
            df=df,
            target_col=target_col,
            noise_rate=0.20,
            random_state=random_state,
        )

    if scenario_name == "distribution_shift":
        return simulate_distribution_shift(
            df=df,
            shift_strength=0.25,
            random_state=random_state,
        )

    if scenario_name == "temporal_drift":
        return simulate_temporal_drift(
            df=df,
            strength=0.60,
            random_state=random_state,
        )

    if scenario_name == "irrelevant_features":
        return add_irrelevant_features(
            df=df,
            n_features=20,
            random_state=random_state,
        )

    if scenario_name == "duplicated_features":
        return add_duplicated_features(
            df=df,
            n_features=10,
            random_state=random_state,
        )

    if scenario_name == "constant_features":
        return add_constant_columns(
            df=df,
            n_features=5,
        )

    if scenario_name == "high_collinearity":
        return add_collinearity(
            df=df,
            strength=0.98,
            n_features=5,
            random_state=random_state,
        )

    if scenario_name == "empty_text":
        return make_empty_text(
            df=df,
            empty_rate=0.80,
            random_state=random_state,
        )

    if scenario_name == "noisy_text":
        return make_noisy_text(
            df=df,
            noise_rate=0.80,
            random_state=random_state,
        )

    if scenario_name == "long_text":
        return make_very_long_text(
            df=df,
            repeat_min=10,
            repeat_max=25,
            random_state=random_state,
        )

    if scenario_name == "mixed_types":
        return mix_numeric_and_string_types(
            df=df,
            string_rate=0.10,
            random_state=random_state,
        )

    if scenario_name == "fully_null_columns":
        return make_columns_fully_null(df=df)

    if scenario_name == "almost_empty_columns":
        return make_columns_almost_empty(
            df=df,
            keep_rate=0.03,
            random_state=random_state,
        )

    raise ValueError(f"Scénario inconnu: {scenario_name}")


if __name__ == "__main__":
    from src.data_generation import generate_default_dataset

    df = generate_default_dataset()

    print("Dataset original :", df.shape)

    scenario_names = [
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
        "empty_text",
        "noisy_text",
        "long_text",
        "mixed_types",
        "fully_null_columns",
        "almost_empty_columns",
    ]

    for scenario in scenario_names:
        df_scenario = apply_robustness_scenario(df, scenario_name=scenario)
        print(
            f"{scenario:20s} -> shape={df_scenario.shape}, "
            f"missing={df_scenario.isna().mean().mean():.3f}"
        )