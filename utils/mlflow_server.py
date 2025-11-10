from data_loader import load_data
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pandas as pd
import numpy as np
import re

def parse_price(x):
    """Parse price string to float - MUST match training pipeline"""
    return float(str(x).replace(",", "").strip())

def hhmm_to_hour(series):
    """Convert HH:MM time to hour"""
    t = pd.to_datetime(series, format="%H:%M", errors="coerce")
    return t.dt.hour

def parse_duration_minutes(x):
    """Parse duration string to minutes"""
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
    """Parse stops string to number"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "non" in s:
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

def prepare_features(df):
    """Apply feature engineering to match training pipeline EXACTLY"""
    df = df.copy()
    
    # CRITICAL: Parse price first (matches training pipeline line 67)
    if 'price' in df.columns:
        df["price"] = df["price"].apply(parse_price)
    
    # Create engineered features
    df["dep_hour"] = hhmm_to_hour(df["dep_time"])
    df["arr_hour"] = hhmm_to_hour(df["arr_time"])
    df["duration_min"] = df["time_taken"].apply(parse_duration_minutes)
    df["stops_n"] = df["stop"].apply(parse_stops)
    
    # Define feature columns (matches training pipeline)
    feature_cols = [
        "airline", "ch_code", "from", "to", "Class", "dayofweek",
        "num_code", "dep_hour", "arr_hour", "duration_min", "stops_n"
    ]
    
    # Clean categorical columns (matches training pipeline lines 96-98)
    cat_cols = ["airline", "ch_code", "from", "to", "Class"]
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
    
    # Clean numerical columns (matches training pipeline lines 101-103)
    num_cols = ["dayofweek", "num_code", "dep_hour", "arr_hour", "duration_min", "stops_n"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df[feature_cols], df["price"]

def test_and_promote(curr_week, data_path='data/Flights.csv'):
    """
    Test and promote model based on RMSE comparison.
    Tests on week curr_week + 1 to simulate real drift conditions.
    - If no production model exists: promote new model
    - If production model exists: compare RMSE, lowest wins
    """
    client = mlflow.tracking.MlflowClient()
    
    # Load test data (week curr_week + 1 for drift simulation)
    test_data, _ = load_data(curr_week + 1, path=data_path)
    
    # Apply feature engineering AND get parsed price
    X_test, y_test = prepare_features(test_data)
    
    # Validate that y_test is numeric (catch issues early)
    if not pd.api.types.is_numeric_dtype(y_test):
        raise TypeError(f"y_test has non-numeric dtype: {y_test.dtype}")
    
    # Load contender model (new model)
    contender_name = f"flight_model_week_{curr_week}"
    contender_model = mlflow.sklearn.load_model(f"models:/{contender_name}/latest")
    
    # Find current production model
    production_model = None
    production_name = None
    
    for rm in client.search_registered_models():
        if "flight_model_week_" in rm.name:
            versions = client.search_model_versions(f"name='{rm.name}'")
            for v in versions:
                if v.current_stage == "Production":
                    production_model = mlflow.sklearn.load_model(f"models:/{rm.name}/Production")
                    production_name = rm.name
                    break
    
    # Evaluate contender
    y_pred_contender = contender_model.predict(X_test)
    rmse_contender = sqrt(mean_squared_error(y_test, y_pred_contender))
    mae_contender = mean_absolute_error(y_test, y_pred_contender)
    
    with mlflow.start_run(run_name=f"promotion_test_week_{curr_week}"):
        mlflow.log_param("test_week", curr_week + 1)
        mlflow.log_param("contender_model", contender_name)
        mlflow.log_metric("contender_rmse", rmse_contender)
        mlflow.log_metric("contender_mae", mae_contender)
        
        if production_model is None:
            # No production model → promote contender
            versions = client.search_model_versions(f"name='{contender_name}'")
            latest_version = max([int(v.version) for v in versions])
            
            client.transition_model_version_stage(
                name=contender_name,
                version=str(latest_version),
                stage="Production"
            )
            mlflow.log_param("promotion_decision", "promoted_first_model")
            print(f"✓ {contender_name} promoted (first model)")
            
        else:
            # Compare with production model
            y_pred_production = production_model.predict(X_test)
            rmse_production = sqrt(mean_squared_error(y_test, y_pred_production))
            mae_production = mean_absolute_error(y_test, y_pred_production)
            
            mlflow.log_param("production_model", production_name)
            mlflow.log_metric("production_rmse", rmse_production)
            mlflow.log_metric("production_mae", mae_production)
            mlflow.log_metric("rmse_improvement", rmse_production - rmse_contender)
            
            if rmse_contender < rmse_production:
                # Contender wins → promote
                # First demote current production
                prod_versions = client.search_model_versions(f"name='{production_name}'")
                for v in prod_versions:
                    if v.current_stage == "Production":
                        client.transition_model_version_stage(
                            name=production_name,
                            version=v.version,
                            stage="Archived"
                        )
                
                # Promote contender
                versions = client.search_model_versions(f"name='{contender_name}'")
                latest_version = max([int(v.version) for v in versions])
                
                client.transition_model_version_stage(
                    name=contender_name,
                    version=str(latest_version),
                    stage="Production"
                )
                mlflow.log_param("promotion_decision", "promoted_better_rmse")
                print(f"✓ {contender_name} promoted (RMSE: {rmse_contender:.2f} < {rmse_production:.2f})")
                
            else:
                # Production wins → keep it
                mlflow.log_param("promotion_decision", "rejected_worse_rmse")
                print(f"✗ {contender_name} not promoted (RMSE: {rmse_contender:.2f} >= {rmse_production:.2f})")