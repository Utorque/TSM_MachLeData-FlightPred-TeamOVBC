import mlflow
import os
import shutil
import numpy as np
import pandas as pd
from glob import glob
from utils.data_drift_detection import rolling_drift
from utils.concept_drift_detection import compute_concept_drift

def clear_report_folder(folder):
    '''Clear report temp folder'''

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def compute_weighted_data_drift_score(data_drift_results):
    '''Compute data drift score using weighted feature'''

    # Define different weights for features
    feature_weights = {
        "price": 0.15,
        "time_taken_minutes": 0.1
        # else 0.05
    }

    weighted_score = 0
    total_weight = 0

    for _, row in data_drift_results.iterrows():
        feat = row["feature"]
        weight = feature_weights.get(feat, 0.05)

        is_numeric = not pd.isna(row.get("psi")) or not pd.isna(row.get("wasserstein"))

        if is_numeric:
            # Get PSI score for numerical features
            intensity = row["psi"]
            if pd.isna(intensity):
                intensity = row["wasserstein"]
        else:
            # Use p-value for Non-numerical features
            p_value = row.get("p_value", np.nan)
            if not pd.isna(p_value):
                intensity = 1 - min(1, p_value * 20)
            else:
                intensity = 1.0 if row.get("drift_detected", False) else 0.0

        if pd.isna(intensity):
            intensity = 0.0
        
        weighted_score += weight * intensity
        total_weight += weight

    return weighted_score / total_weight if total_weight > 0 else np.nan


def check_and_log_drift(train_df, current_week):
    '''Check and lo drift in mlflow'''

    with mlflow.start_run(run_name=f'drift_check_week_{current_week}', nested=True):

        # --- Data drift ---
        data_drift_results = rolling_drift(train_df, report_path='report/')

        result_path = f'report/data_drift_week_{current_week-1}_vs_{current_week}.csv'
        data_drift_results.to_csv(result_path, index=False)
        mlflow.log_artifact(result_path)

        # drift_ratio = data_drift_results['drift_detected'].mean()
        data_drift_ratio = compute_weighted_data_drift_score(data_drift_results)
        psi_mean = np.nanmean(data_drift_results['psi'])

        mlflow.log_metric('data_drift_ratio', data_drift_ratio)
        mlflow.log_metric('data_psi_mean', psi_mean)

        html_path = f'report/data_drift_week_{current_week-1}_vs_{current_week}.html'
        mlflow.log_artifact(html_path)

        distribution_graph_dir = f'report/distributions_week_{current_week-1}_vs_{current_week}/'
        png_files = glob(os.path.join(distribution_graph_dir, '**', '*.png'), recursive=True)
        for png_file in png_files:
            mlflow.log_artifact(png_file)

        # --- Concept drift ---
        drift_df = compute_concept_drift(train_df, report_path='report/')
        concept_drift_mean = drift_df["mean_corr_diff"].mean()
        concept_drift_ratio = drift_df["concept_drift_detected"].mean()
        concept_drift_detected = concept_drift_mean > 0.15 or concept_drift_ratio > 0.3

        drift_df.to_csv("report/concept_drift_results.csv", index=False)
        mlflow.log_artifact("report/concept_drift_results.csv")
        mlflow.log_artifact("report/corr_evolution_heatmap.png")

        mlflow.log_metric("concept_mean_corr_diff", concept_drift_mean)
        mlflow.log_metric("concept_drift_ratio", concept_drift_ratio)
        mlflow.log_param("concept_drift_detected", concept_drift_detected)

        # --- Retrain trigger ---
        retrain_threashold = 0.15

        combined_score = (0.6 * data_drift_ratio) + (0.4 * concept_drift_mean)
        retrain_needed = combined_score > retrain_threashold
        mlflow.log_param('retrain_triggered', retrain_needed)
        mlflow.log_param('current_week', current_week)

        print(f'Drift check done — data_drift_ratio={data_drift_ratio:.3f}, concept_drift_mean={concept_drift_mean:.3f} retrain={retrain_needed}')

        clear_report_folder('report/')

        return retrain_needed
