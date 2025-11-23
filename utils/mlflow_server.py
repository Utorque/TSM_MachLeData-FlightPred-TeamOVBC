from data_loader import load_data
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import pandas as pd
import numpy as np
import re

def test_and_promote(curr_week, data_path='data/Flights.csv'):
    test_data, _ = load_data(curr_week + 1, path=data_path)
    
    contender_model_name = f"flight_model_week_{curr_week}"
    
    # Load contender model
    contender_model = mlflow.sklearn.load_model(f"models:/{contender_model_name}/latest")
    
    # Get current production model (latest version in Production stage)
    client = mlflow.tracking.MlflowClient()
    production_models = [v for v in client.search_model_versions("") if v.current_stage == "Production"]
    current_production = max(production_models, key=lambda v: int(v.version)) if production_models else None
    
    # CRITICAL FIX: Prepare test data with exact columns and types
    feature_cols = [
        "airline", "ch_code", "from", "to", "Class", "dayofweek",
        "num_code", "dep_hour", "arr_hour", "time_taken_minutes", "stops_n"
    ]
    
    cat_cols = ["airline", "ch_code", "from", "to", "Class"]
    
    X_test = test_data[feature_cols].copy()
    y_test = test_data['price']
    
    # Convert categorical to string and handle NaN
    for col in cat_cols:
        X_test[col] = X_test[col].fillna("MISSING").astype(str)
    
    # Evaluate contender
    y_pred_contender = contender_model.predict(X_test)
    mse_contender = mean_squared_error(y_test, y_pred_contender)
    r2_contender = r2_score(y_test, y_pred_contender)
    
    with mlflow.start_run(run_name=f"test_week_{curr_week + 1}"):
        mlflow.log_metric("test_week", curr_week + 1)
        mlflow.log_metric("mse_contender", mse_contender)
        mlflow.log_metric("r2_contender", r2_contender)
        
        should_promote = False
        
        if current_production is None:
            should_promote = True
            mlflow.log_param("promotion_reason", "no_production_model")
        else:
            production_model = mlflow.sklearn.load_model(
                f"models:/{current_production.name}/{current_production.version}"
            )
            
            y_pred_production = production_model.predict(X_test)
            mse_production = mean_squared_error(y_test, y_pred_production)
            r2_production = r2_score(y_test, y_pred_production)
            
            mlflow.log_metric("mse_production", mse_production)
            mlflow.log_metric("r2_production", r2_production)
            mlflow.log_param("production_model_name", current_production.name)
            
            should_promote = mse_contender < mse_production
            mlflow.log_param("promotion_reason", "better_than_production" if should_promote else "worse_than_production")
        
        if should_promote:
            versions = client.search_model_versions(f"name='{contender_model_name}'")
            latest_version = max([int(v.version) for v in versions])
            
            client.transition_model_version_stage(
                name=contender_model_name,
                version=str(latest_version + 1),
                stage="Production"
            )
            mlflow.log_param("promoted", True)
        else:
            mlflow.log_param("promoted", False)
    
    return test_data
