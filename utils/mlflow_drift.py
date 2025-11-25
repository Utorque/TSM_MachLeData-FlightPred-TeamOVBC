import mlflow
import os
import shutil
import numpy as np
import pandas as pd
from glob import glob
from utils.data_drift_detection import rolling_drift, expanding_drift
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
            print(f'Failed to delete {file_path}. Reason: {e}')

def compute_weighted_data_drift_score(data_drift_results):
    '''Compute data drift score using weighted feature'''

    # Define different weights for features
    feature_weights = {
        'price': 0.15,
        'time_taken_minutes': 0.10,
        'airline': 0.10,
        'Class': 0.02,
    }
    default_weight = 0.05

    # for categorical value
    significant_p_value = 0.05

    weighted_sum = 0
    total_weight = 0

    for _, row in data_drift_results.iterrows():
        feat = row['feature']
        weight = feature_weights.get(feat, default_weight)

        # Identify numeric vs categorical
        is_numeric = not pd.isna(row.get('psi')) or not pd.isna(row.get('wasserstein'))

        if is_numeric:
            # PSI preferred, fallback to Wasserstein
            intensity = row['psi']
            if pd.isna(intensity):
                intensity = row['wasserstein']

            # Cap extreme values for stability
            intensity = np.clip(intensity, 0, 1)
        else:
            # Categorical: derive intensity from p-value
            # Here H0 : The frequencies of the categories in the previous week and the current week are identical. There is NO category drift.
            # So for p-values smaller than significant_p_value, the hypothesis is rejected and we conclude that there is drift.
            p_value = row.get('p_value', np.nan)
            if not pd.isna(p_value):
                intensity = 1 - np.clip(p_value/significant_p_value, 0, 1)
            else:
                intensity = 1.0 if row.get('drift_detected', False) else 0.0
        
        weighted_sum += weight * intensity
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else np.nan


def check_and_log_drift(train_df, current_week, last_train_week, GLOBAL_DRIFT_THRESHOLD = 0.10, DATA_DRIFT_THRESHOLD = 0.15, CONCEPT_DRIFT_THRESHOLD = 0.15):
    '''Check and log drift in mlflow'''

    with mlflow.start_run(run_name=f'drift_check_week_{current_week}', nested=True):

        # --- Rolling Data drift ---
        data_drift_results = rolling_drift(train_df, report_path='report/')

        result_path = f'report/data_drift_rolling_week_{current_week-1}_vs_{current_week}.csv'
        data_drift_results.to_csv(result_path, index=False)
        mlflow.log_artifact(result_path, artifact_path='rolling')

        rolling_data_drift_ratio = compute_weighted_data_drift_score(data_drift_results)
        rolling_data_drift_detected = rolling_data_drift_ratio > DATA_DRIFT_THRESHOLD
        mlflow.log_metric('rolling_data_drift_ratio', rolling_data_drift_ratio)
        mlflow.log_param('rolling_data_drift_detected', rolling_data_drift_detected)

        html_path = f'report/data_drift_week_{current_week-1}_vs_{current_week}.html'
        mlflow.log_artifact(html_path, artifact_path='rolling')

        distribution_graph_dir = f'report/distributions_week_{current_week-1}_vs_{current_week}/'
        png_files = glob(os.path.join(distribution_graph_dir, '**', '*.png'), recursive=True)
        for png_file in png_files:
            mlflow.log_artifact(png_file, artifact_path='rolling')

        # --- Expanding Data drift ---
        expanding_drift_results = expanding_drift(
            train_df, 
            report_path='report/', 
            current_week=current_week,
            last_train_week=last_train_week
        )

        result_path = f'report/data_drift_expanding_week_{last_train_week}_to_{current_week-1}_vs_{current_week}.csv'
        expanding_drift_results.to_csv(result_path, index=False)
        mlflow.log_artifact(result_path, artifact_path='expanding')

        expanding_data_drift_ratio = compute_weighted_data_drift_score(expanding_drift_results)
        expanding_data_drift_detected = expanding_data_drift_ratio > DATA_DRIFT_THRESHOLD
        mlflow.log_metric('expanding_data_drift_ratio', expanding_data_drift_ratio)
        mlflow.log_param('expanding_data_drift_detected', expanding_data_drift_detected)

        html_path = f'report/expanding_drift_{last_train_week}_to_{current_week-1}_vs_{current_week}.html'
        mlflow.log_artifact(html_path, artifact_path='expanding')

        distribution_graph_dir = f'report/distributions_week_{last_train_week}_to_{current_week-1}_vs_{current_week}/'
        png_files = glob(os.path.join(distribution_graph_dir, '**', '*.png'), recursive=True)
        for png_file in png_files:
            mlflow.log_artifact(png_file, artifact_path='expanding')

        # --- Concept drift ---
        drift_df = compute_concept_drift(train_df, report_path='report/')

        concept_drift_mean = drift_df['mean_corr_diff'].mean()
        concept_drift_detected = concept_drift_mean > CONCEPT_DRIFT_THRESHOLD
        mlflow.log_metric('concept_mean_corr_diff', concept_drift_mean)
        mlflow.log_param('concept_drift_detected', concept_drift_detected)

        drift_df.to_csv('report/concept_drift_results.csv', index=False)
        mlflow.log_artifact('report/concept_drift_results.csv', artifact_path='concept')
        mlflow.log_artifact('report/corr_evolution_heatmap.png', artifact_path='concept')

        # --- Retrain trigger ---
        global_drift_score = (
            0.40 * rolling_data_drift_ratio
            + 0.30 * expanding_data_drift_ratio
            + 0.30 * concept_drift_mean
        )

        if rolling_data_drift_detected or expanding_data_drift_detected or concept_drift_detected:
            retrain_needed = True
        else:
            retrain_needed = global_drift_score > GLOBAL_DRIFT_THRESHOLD

        mlflow.log_metric('global_drift_score', global_drift_score)
        mlflow.log_param('retrain_triggered', retrain_needed)
        mlflow.log_param('current_week', current_week)

        print(
            'Drift check done:\n'
            f'  Rolling data drift ratio:   {rolling_data_drift_ratio:.3f}\n'
            f'  Expanding data drift ratio: {expanding_data_drift_ratio:.3f}\n'
            f'  Concept drift mean:         {concept_drift_mean:.3f}\n'
            f'  Global drift score:         {global_drift_score:.3f}\n'
            f'  Retrain needed:             {retrain_needed}'
        )

        clear_report_folder('report/')

        return retrain_needed
