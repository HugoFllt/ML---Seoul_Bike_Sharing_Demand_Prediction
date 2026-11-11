"""Préparation et découpage communs des données pour tous les notebooks de modélisation.

Toute l'équipe utilise ces fonctions : les données, le jeu de test et les plis de
cross-validation sont donc exactement les mêmes pour tous les modèles. Les choix du
preprocessing sont expliqués dans introduction.ipynb, ceux du découpage dans
decoupage_donnees.ipynb.

Exemple :
    from decoupage import charger_donnees, decoupage_daily, cv_daily, variables, CIBLE

    train, test = decoupage_daily(charger_donnees())
    X_train, y_train = train[variables(train)], train[CIBLE]
    splits = cv_daily(train)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, TimeSeriesSplit

CHEMIN_DONNEES = "SeoulBikeData.csv"
CIBLE = "rented_bike_count"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Noms des colonnes du CSV d'origine, dans l'ordre
COLONNES = ["date", "rented_bike_count", "hour", "temperature", "humidity", "wind_speed", "visibility",
            "dew_point_temperature", "solar_radiation", "rainfall", "snowfall", "season", "holiday",
            "functioning_day"]
# Variables météo avec quelques valeurs extrêmes, bornées aux percentiles 1 % et 99 %
VARIABLES_BORNEES = ["wind_speed", "solar_radiation", "rainfall", "snowfall"]


# ---------------------------------------------------------------- preprocessing

def charger_brut(chemin=CHEMIN_DONNEES):
    # CSV d'origine, avec des noms de colonnes simples
    df = pd.read_csv(chemin, encoding="latin-1")
    df.columns = COLONNES
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df


def preprocessing(brut):
    # Jours où le service était fermé : 0 location, ce n'est pas de la demande réelle
    df = brut[brut["functioning_day"] == "Yes"].drop(columns=["functioning_day", "dew_point_temperature"])
    df["holiday"] = (df["holiday"] == "Holiday").astype(int)

    # Calendrier
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = ((df["is_weekend"] == 0) & df["hour"].isin([7, 8, 9, 17, 18, 19, 20])).astype(int)

    # Encodage cyclique : 23 h proche de 0 h, décembre proche de janvier
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_raining"] = (df["rainfall"] > 0).astype(int)
    df["is_snowing"] = (df["snowfall"] > 0).astype(int)
    df = pd.concat([df.drop(columns="season"), pd.get_dummies(df["season"], prefix="season", dtype=int)], axis=1)

    for col in VARIABLES_BORNEES:
        df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
    return df


def charger_donnees(chemin=CHEMIN_DONNEES):
    # Dataset nettoyé, une ligne par heure, trié chronologiquement
    df = preprocessing(charger_brut(chemin))
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def variables(df):
    # Toutes les colonnes sauf la date, la cible et la colonne technique "semaine"
    return [col for col in df.columns if col not in ("date", CIBLE, "semaine")]


# ---------------------------------------------------------------- protocole daily

def decoupage_daily(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    # Pour chaque mois, test_size des jours sont tirés au hasard pour le test
    days = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    test_days = []
    for _, month_days in days.groupby(days.dt.to_period("M")):
        n_test = round(test_size * len(month_days))
        test_days.extend(rng.choice(month_days.values, size=n_test, replace=False))

    is_test = df["date"].isin(test_days)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def cv_daily(train, n_splits=5, random_state=RANDOM_STATE):
    # Plis de jours entiers (groupe = date), chaque mois représenté dans chaque pli (stratification = mois)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(train, y=train["month"], groups=train["date"]))


# ---------------------------------------------------------------- protocole temporal

def decoupage_temporal(df, test_size=TEST_SIZE, jours_par_semaine=7):
    # Les premiers jours pour l'entraînement, les derniers pour le test ; coupure entre deux journées
    days = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
    cutoff = days.iloc[int((1 - test_size) * len(days))]
    train = df[df["date"] < cutoff].reset_index(drop=True)
    test = df[df["date"] >= cutoff].reset_index(drop=True)

    # Numéro de semaine de test pour le walk-forward ; les jours restants rejoignent la dernière semaine
    test_days = pd.Series(test["date"].unique())
    n_weeks = len(test_days) // jours_par_semaine
    week = np.minimum(np.arange(len(test_days)) // jours_par_semaine, n_weeks - 1) + 1
    test["semaine"] = test["date"].map(pd.Series(week, index=test_days))
    return train, test


def cv_temporal(train, debut_validation="2018-04-01", jours_par_pli=7):
    # Chaque pli entraîne sur tout le passé et valide sur les jours_par_pli jours suivants
    days = pd.Series(train["date"].unique())
    n_splits = int((days >= pd.Timestamp(debut_validation)).sum() // jours_par_pli)
    splits = []
    for days_train, days_val in TimeSeriesSplit(n_splits=n_splits, test_size=jours_par_pli).split(days):
        idx_train = np.where(train["date"].isin(days.iloc[days_train]))[0]
        idx_val = np.where(train["date"].isin(days.iloc[days_val]))[0]
        splits.append((idx_train, idx_val))
    return splits


def walk_forward(train, test):
    # Évaluation finale : pour chaque semaine de test, (passé disponible, semaine à prédire)
    for week in sorted(test["semaine"].unique()):
        past = pd.concat([train, test[test["semaine"] < week].drop(columns="semaine")], ignore_index=True)
        yield past, test[test["semaine"] == week]
