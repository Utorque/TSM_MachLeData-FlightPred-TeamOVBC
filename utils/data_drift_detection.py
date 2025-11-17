import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from evidently import Report
from evidently.presets import DataDriftPreset

def calculate_psi(expected, actual, bins=10):
    '''Calculate PSI (Population Stability Index)'''

    if len(expected) < 5 or len(actual) < 5:
        return np.nan 

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi_value = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi_value

def detect_data_drift(reference_df, current_df):
    '''Detect data drift manually'''

    results = []

    numerical_features = ['num_code', 'price', 'time_taken_minutes']
    categorical_features = ['airline', 'ch_code', 'dep_time', 'from', 'stop', 'arr_time', 'to', 'Class', 'dayofweek']

    for col in categorical_features:
        ref_counts = reference_df[col].value_counts()
        cur_counts = current_df[col].value_counts()

        contingency = pd.DataFrame({
            'reference': ref_counts / ref_counts.sum(),
            'current': cur_counts / cur_counts.sum()
        }).fillna(0)

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        results.append({
            'feature': col,
            'test': 'Chi2',
            'stats': chi2,
            'wasserstein': None,
            'p_value': p_value,
            'psi': None,
            'drift_detected': p_value < 0.05
        })

    for col in numerical_features:
        ref, cur = reference_df[col].dropna(), current_df[col].dropna()

        ks, p_value = ks_2samp(ref, cur)
        psi_value = calculate_psi(ref, cur)
        wd = wasserstein_distance(ref, cur) / np.std(ref)

        results.append({
            'feature': col,
            'test': 'KS',
            'stats': ks,
            'wasserstein': wd,
            'p_value': p_value,
            'psi': psi_value,
            'drift_detected': wd > 0.1
        })

    return pd.DataFrame(results)

def visualize_distributions(reference_df, current_df, ref_w, cur_w, 
                            numerical_features=['price', 'num_code', 'time_taken_minutes'],
                            categorical_features=['airline', 'ch_code', 'from', 'to', 'Class', 'dayofweek'], 
                            path='report'):
    '''Print KDE (Kernel Density Estimation) graphs (for numerical value only)'''
    
    save_dir = f'{path}/distributions_week_{ref_w}_vs_{cur_w}/'
    os.makedirs(save_dir, exist_ok=True)

    # --- Numerical : KDE ---
    for col in numerical_features:
        plt.figure(figsize=(6,4))
        sns.kdeplot(reference_df[col], label=f'Week {ref_w}', fill=True, alpha=0.4)
        sns.kdeplot(current_df[col], label=f'Week {cur_w}', fill=True, alpha=0.4)
        plt.title(f'Distribution of {col} (Week {ref_w} vs {cur_w})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}{col}_kde.png')
        plt.close()

    # --- Categorical : Barplots ---
    for col in categorical_features:
        plt.figure(figsize=(6,4))
        ref_counts = reference_df[col].value_counts(normalize=True)
        cur_counts = current_df[col].value_counts(normalize=True)
        compare_df = pd.DataFrame({f'Week {ref_w}': ref_counts, f'Week {cur_w}': cur_counts}).fillna(0)
        compare_df.plot(kind='bar', alpha=0.7)
        plt.title(f'Distribution of {col} (Week {ref_w} vs {cur_w})')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(f'{save_dir}{col}_bar.png')
        plt.close()

def rolling_drift(data, report_path, current_week=None):
    '''
    Apply rolling aproach to detect data drift.

    Do not apply exapanding approach as it is not really appropriate (See exploration folder)
    '''

    # Prepare report path
    os.makedirs(report_path, exist_ok=True)

    # Get max week if not specified
    if current_week is None:
        current_week = data['week'].max()
    previous_week = current_week - 1

    reference_df = data[data['week'] == previous_week]
    current_df = data[data['week'] == current_week]

    # Manually detect data drift
    drift_results = detect_data_drift(reference_df, current_df)
    drift_results['approach'] = 'rolling'
    drift_results['ref_weeks'] = str(previous_week)
    drift_results['curr_week'] = current_week

    # Evidently data drift reports
    my_report = Report([DataDriftPreset()])
    my_eval = my_report.run(reference_data=reference_df, current_data=current_df)
    my_eval.save_html(f'{report_path}data_drift_week_{previous_week}_vs_{current_week}.html')

    visualize_distributions(reference_df, current_df, previous_week, current_week, path=report_path)

    return drift_results