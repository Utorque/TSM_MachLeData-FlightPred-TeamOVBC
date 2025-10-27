#!/usr/bin/env python

import pandas as pd
import joblib
import argparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# === 1) Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--week", type=int, required=True, help="Semaine ISO à tester (1-53).")
parser.add_argument("--nbweek", type=int, default=2, help="Nb de semaines d'entraînement juste AVANT la semaine test.")
parser.add_argument("--year", type=int, default=None, help="(Optionnel) Année ISO de la semaine test pour lever l'ambiguïté.")
parser.add_argument("--csv", type=str, default="data/Flights.csv", help="Chemin du CSV.")
args = parser.parse_args()

# === 2) Charger ===
df = pd.read_csv(args.csv, parse_dates=["date"])

# Nettoyer price si format "25,612"
if df["price"].dtype == object:
    df["price"] = (
        df["price"].astype(str).str.replace(",", "", regex=False).astype(float)
    )

# Colonnes requises (selon ton schéma)
required = ["date","airline","from","to","dep_time","arr_time","stop","Class","dayofweek","price"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing}")

# === 3) ISO year/week + ordre temporel des semaines ===
iso = df["date"].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)

# Date représentative de la semaine (min date de la semaine) pour trier correctement
week_order = (
    df.groupby(["iso_year","iso_week"], as_index=False)
      .agg(week_start=("date","min"), n=("date","size"))
      .sort_values(["week_start"])
      .reset_index(drop=True)
)

# === 4) Choisir la semaine test (year+week) ===
if args.year is not None:
    # on cherche exactement (year, week)
    candidates = week_order[(week_order["iso_year"] == args.year) &
                            (week_order["iso_week"] == args.week)]
    if candidates.empty:
        raise ValueError(f"Aucune semaine {args.year}-W{args.week} dans les données.")
    test_idx = candidates.index[-1]  # si multiple (rare), on prend la dernière par sécurité
else:
    # pas d'année fournie: on prend la DERNIÈRE occurrence de ce n° de semaine (la plus récente)
    candidates = week_order[week_order["iso_week"] == args.week]
    if candidates.empty:
        raise ValueError(f"Aucune occurrence de la semaine W{args.week} dans les données.")
    test_idx = candidates.index[-1]

test_year = int(week_order.loc[test_idx, "iso_year"])
test_week = int(week_order.loc[test_idx, "iso_week"])

# Indices d'entraînement = les nbweek semaines juste avant test_idx
start_idx = max(0, test_idx - args.nbweek)
train_slice = week_order.iloc[start_idx:test_idx]  # [start_idx, test_idx)

if train_slice.empty:
    raise ValueError(
        f"Pas assez de semaines avant {test_year}-W{test_week} pour nbweek={args.nbweek}."
    )

train_keys = set(map(tuple, train_slice[["iso_year","iso_week"]].to_numpy()))
test_key = (test_year, test_week)

print(f"Semaine test : {test_year}-W{test_week}")
print(f"Semaines d'entraînement : " +
      ", ".join([f"{y}-W{w}" for (y,w) in train_keys]))

# === 5) Split temporel par (iso_year, iso_week) ===
df_train = df[df[["iso_year","iso_week"]].apply(tuple, axis=1).isin(train_keys)].copy()
df_test  = df[(df["iso_year"]==test_year) & (df["iso_week"]==test_week)].copy()

if df_train.empty or df_test.empty:
    raise ValueError("Train ou test vide après découpe temporelle. Vérifie tes filtres.")

print(f" Train: {len(df_train)} lignes | Test: {len(df_test)} lignes")

# === 6) Feature engineering simple (heures numériques) ===
def to_hour(s):
    # gère 'HH:MM' et valeurs manquantes
    t = pd.to_datetime(s, format="%H:%M", errors="coerce")
    return t.dt.hour

for split_df in (df_train, df_test):
    split_df["dep_hour"] = to_hour(split_df["dep_time"])
    split_df["arr_hour"] = to_hour(split_df["arr_time"])

# === 7) Features / target ===
drop_cols = ["date","iso_year","iso_week","dep_time","arr_time"]
X_train = df_train.drop(columns=drop_cols + ["price"], errors="ignore")
y_train = df_train["price"]

X_test  = df_test.drop(columns=drop_cols + ["price"], errors="ignore")
y_test  = df_test["price"]

# Colonnes catégorielles/numériques
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()

# Sécurité: aligner les colonnes entre train/test (au cas où)
X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

# === 8) Pipeline + modèle ===
preproc = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = Pipeline(steps=[
    ("prep", preproc),
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === 9) Entraînement + évaluation ===
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\nRésultats")
print(f" MAE : {mae:.2f}")
print(f" R²  : {r2:.4f}")

# === 10) Sauvegarde ===
#model_path = f"model_train_{len(train_keys)}w_{min(train_slice.week_start).date()}__to__{max(train_slice.week_start).date()}__test_{test_year}-W{test_week}.pkl"
#joblib.dump(model, model_path)
#print(f"Modèle sauvegardé : {model_path}")
