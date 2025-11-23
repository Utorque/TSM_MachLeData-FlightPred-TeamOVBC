#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from math import sqrt


# ---------- HELPER FUNCTIONS ----------
def parse_price(x):
    return float(str(x).replace(",", "").strip())

def hhmm_to_hour(series):
    t = pd.to_datetime(series, format="%H:%M", errors="coerce")
    return t.dt.hour

def parse_duration_minutes(x):
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
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "non" in s:
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan


# ---------- CORE TRAINING FUNCTION ----------
def train_model(df, nbweek, model_out=None):
    """
    Train XGBoost weekly model
    df      : pandas DataFrame
    nbweek  : nombre de semaines d'entraînement
    return  : (model, {"mae":..,"rmse":..,"r2":..}, train_weeks, test_week)
    """

    required_cols = [
        "date","airline","ch_code","num_code","dep_time","from",
        "time_taken","stop","arr_time","to","price","Class","dayofweek","week"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataframe: {missing}")

    # Convert / enrich
    df["price"] = df["price"].apply(parse_price)
    df["dep_hour"] = hhmm_to_hour(df["dep_time"])
    df["arr_hour"] = hhmm_to_hour(df["arr_time"])
    df["time_taken_minutes"] = df["time_taken"].apply(parse_duration_minutes)
    df["stops_n"] = df["stop"].apply(parse_stops)

    # Split weeks
    weeks_sorted = sorted(df["week"].dropna().astype(int).unique())
    if len(weeks_sorted) < nbweek - 6:
        raise ValueError(f"Pas assez de semaines dans les données (besoin: {nbweek+1}).")

    train_weeks = weeks_sorted[:nbweek-6]
    print(train_weeks)
    test_week = weeks_sorted[nbweek-6]
    print(test_week)

    df_train = df[df["week"].isin(train_weeks)].copy()
    df_test = df[df["week"] == test_week].copy()

    # Feature / target
    target = "price"
    feature_cols = [
        "airline","ch_code","from","to","Class","dayofweek",
        "num_code","dep_hour","arr_hour","time_taken_minutes","stops_n"
    ]

    X_train, y_train = df_train[feature_cols], df_train[target]
    X_test, y_test   = df_test[feature_cols], df_test[target]

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        tree_method="hist"
    )

    model = Pipeline([
        ("prep", preproc),
        ("reg", xgb),
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict & metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    # Save model if requested
    if model_out:
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
    
    
    print(f"\nSemaines d'entraînement : {train_weeks}")
    print(f"Semaine de test         : {test_week}")
    print(f"MAE   : {metrics['mae']:.2f}")
    print(f"RMSE  : {metrics['rmse']:.2f}")
    print(f"R²    : {metrics['r2']:.4f}")

    return model, metrics, train_weeks, test_week


# ---------- CLI ENTRY POINT ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../data/Flights.csv")
    parser.add_argument("--nbweek", type=int, required=True)
    parser.add_argument("--model-out", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"])
    model, metrics, train_weeks, test_week = train_model(
        df, args.nbweek, model_out=args.model_out
    )

    print(f"\nSemaines d'entraînement : {train_weeks}")
    print(f"Semaine de test         : {test_week}")
    print(f"MAE   : {metrics['mae']:.2f}")
    print(f"RMSE  : {metrics['rmse']:.2f}")
    print(f"R²    : {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
