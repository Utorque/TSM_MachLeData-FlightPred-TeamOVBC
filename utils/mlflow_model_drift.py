import mlflow
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ks_2samp, wasserstein_distance

API_URL = "http://127.0.0.1:52001/predict"

def predict_fastapi(df):
    """Send features to FastAPI model server for prediction."""
    payload = df.to_dict(orient="records")
    resp = requests.post(API_URL, json={"data": payload})
    resp.raise_for_status()
    return pd.Series(resp.json()["predictions"])

def check_and_log_model_drift(df, current_week):

    prev_week = current_week - 1
    if prev_week not in df["week"].unique():
        print("No previous week available for model drift comparison.")
        return None
    
    # Load model
    model = mlflow.sklearn.load_model("models:/flight_price_model/Production")

    # Extract data
    df_prev = df[df["week"] == prev_week]
    df_curr = df[df["week"] == current_week]

    target_col = "price"

    y_prev = df_prev[target_col].astype(float)
    X_prev = df_prev.drop(columns=[target_col, "week"], errors="ignore")

    y_curr = df_curr[target_col].astype(float)
    X_curr = df_curr.drop(columns=[target_col, "week"], errors="ignore")

    # Prediction
    y_pred_prev = model.predict(X_prev)
    y_pred_curr = model.predict(X_curr)

    # Errors
    e_prev = y_prev - y_pred_prev
    e_curr = y_curr - y_pred_curr
    
    # KS and Wasserstein Distance
    ks_stat, ks_p = ks_2samp(e_prev, e_curr)
    wd = wasserstein_distance(e_prev, e_curr) / (np.std(e_prev) + 1e-6)

    out_dir = f"report/model_drift/week_{prev_week}_vs_{current_week}"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(7, 4))
    sns.kdeplot(e_prev, fill=True, alpha=0.4, label=f"Week {prev_week}")
    sns.kdeplot(e_curr, fill=True, alpha=0.4, label=f"Week {current_week}")
    plt.title(f"Error Distribution - Week {prev_week} vs {current_week}")
    plt.xlabel("Prediction Error")
    plt.legend()
    drift_plot_path = f"{out_dir}/error_kde.png"
    plt.tight_layout()
    plt.savefig(drift_plot_path)
    plt.close()

    # Metrics
    rmse_prev = np.sqrt(mean_squared_error(y_prev, y_pred_prev))
    rmse_curr = np.sqrt(mean_squared_error(y_curr, y_pred_curr))
    mae_prev = mean_absolute_error(y_prev, y_pred_prev)
    mae_curr = mean_absolute_error(y_curr, y_pred_curr)

    drift_rmse = rmse_curr - rmse_prev

    with mlflow.start_run(run_name=f"model_drift_week_{current_week}", nested=True):

        mlflow.log_metric("rmse_prev_week", rmse_prev)
        mlflow.log_metric("rmse_curr_week", rmse_curr)
        mlflow.log_metric("mae_prev_week", mae_prev)
        mlflow.log_metric("mae_curr_week", mae_curr)

        mlflow.log_metric("model_drift_rmse_abs", drift_rmse)

        mlflow.log_metric("model_drift_ks_stat", ks_stat)
        mlflow.log_metric("model_drift_ks_p", ks_p)
        mlflow.log_metric("model_drift_wasserstein", wd)

        mlflow.log_artifact(drift_plot_path)

    return "model drift checked"