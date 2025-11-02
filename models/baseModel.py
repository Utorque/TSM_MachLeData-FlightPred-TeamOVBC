
## python .\baseModel.py --nbweek X
## where X is the number of training week

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ----------------- Helpers -----------------
def parse_price(x):
    # "25,612" -> 25612.0
    return float(str(x).replace(",", "").strip())

def hhmm_to_hour(series):
    t = pd.to_datetime(series, format="%H:%M", errors="coerce")
    return t.dt.hour

def parse_duration_minutes(x):
    # "02h 00m" -> 120 ; "1h35m" -> 95 ; "3h" -> 180 ; "45m" -> 45
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace(" ", "")
    m = re.match(r"(?:(\d+)h)?(?:(\d+)m)?", s)
    if not m:
        return np.nan
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    return h * 60 + m_

def parse_stops(x):
    # "non-stop", "1 stop", "2 stops"
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "non" in s:  # "non-stop"
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Train GradientBoostingRegressor on first N weeks and test on next week."
    )
    parser.add_argument("--csv", type=str, default="../data/Flights.csv",
                        help="Chemin du CSV.")
    parser.add_argument("--nbweek", type=int, required=True,
                        help="Nombre de semaines d'entraînement à partir de la première semaine disponible (ex: 2).")
    parser.add_argument("--model-out", type=str, default=None,
                        help="Chemin de sortie du modèle .pkl (auto si non défini).")
    args = parser.parse_args()

    # 1) Lecture
    df = pd.read_csv(args.csv, parse_dates=["date"])

    required_cols = [
        "date","airline","ch_code","num_code","dep_time","from","time_taken","stop",
        "arr_time","to","price","Class","dayofweek","week"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    # 2) Nettoyages / features de base
    df["price"] = df["price"].apply(parse_price)
    df["dep_hour"] = hhmm_to_hour(df["dep_time"])
    df["arr_hour"] = hhmm_to_hour(df["arr_time"])
    df["duration_min"] = df["time_taken"].apply(parse_duration_minutes)
    df["stops_n"] = df["stop"].apply(parse_stops)

    # 3) Définition des semaines train / test
    weeks_sorted = sorted(df["week"].dropna().astype(int).unique())
    if len(weeks_sorted) < args.nbweek + 1:
        raise ValueError(
            f"Il faut au moins {args.nbweek + 1} semaines distinctes dans les données "
            f"(trouvé: {len(weeks_sorted)})."
        )

    # Train = les nbweek premières semaines disponibles ; Test = la suivante
    train_weeks = weeks_sorted[:args.nbweek]
    test_week = weeks_sorted[args.nbweek]

    df_train = df[df["week"].isin(train_weeks)].copy()
    df_test = df[df["week"] == test_week].copy()

    if df_train.empty or df_test.empty:
        raise ValueError("Jeu d'entraînement ou de test vide après découpe par semaines.")

    print(f"📆 Semaines d'entraînement: {train_weeks}")
    print(f"🧪 Semaine de test: {test_week}")
    print(f"📊 Lignes: train={len(df_train)} | test={len(df_test)}")

    # 4) Sélection des features / cible
    target = "price"
    feature_cols = [
        # Catégorielles
        "airline", "ch_code", "from", "to", "Class", "dayofweek",
        # Numériques
        "num_code", "dep_hour", "arr_hour", "duration_min", "stops_n"
    ]

    X_train = df_train[feature_cols]
    y_train = df_train[target]
    X_test = df_test[feature_cols]
    y_test = df_test[target]

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # 5) Modèle : Gradient Boosting (réglages sobres pour rester raisonnable en temps)
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",  # rapide et adapté CPU
    )

    model = Pipeline(steps=[
        ("prep", preproc),
        ("reg", xgb),
    ])

    # 6) Entraînement
    model.fit(X_train, y_train)

    # 7) Évaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    from math import sqrt
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Résultats (semaine de test = {})".format(test_week))
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # 8) Sauvegarde du modèle
    out_path = args.model_out or f"xgb_model_train_w{train_weeks[0]}-w{train_weeks[-1]}_test_w{test_week}.pkl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"💾 Modèle sauvegardé : {out_path}")

    # 9) (Optionnel) Sauvegarde prédictions
    preds_path = f"predictions_test_w{test_week}.csv"
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(preds_path, index=False)
    print(f"📝 Prédictions sauvegardées : {preds_path}")


if __name__ == "__main__":
    main()
