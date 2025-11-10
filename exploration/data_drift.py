import sys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from evidently import Report
from evidently.presets import DataDriftPreset

def parse_convert_price(x):
    indian_ruppes_price = float(str(x).replace(",", "").strip())
    chf_price = indian_ruppes_price * 0.0091
    return chf_price

def calculate_psi(expected, actual, bins=10):
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

def visualize_distributions(reference_df, current_df, ref_w, cur_w, 
                            numerical_features=['price', 'num_code'], 
                            categorical_features=['airline', 'ch_code', 'from', 'to', 'Class', 'dayofweek'],
                            path='report'):
    
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
        compare_df = pd.DataFrame({'Reference': ref_counts, 'Current': cur_counts}).fillna(0)
        compare_df.plot(kind='bar', alpha=0.7)
        plt.title(f'Distribution of {col} (Week {ref_w} vs {cur_w})')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(f'{save_dir}{col}_bar.png')
        plt.close()

def detect_data_drift(reference_df, current_df):
    results = []

    numerical_features = ['num_code', 'price']
    categorical_features = ['airline', 'ch_code', 'dep_time', 'from', 'time_taken', 'stop', 'arr_time', 'to', 'Class', 'dayofweek']

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
            'drift_detected': p_value < 0.001
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
            #'drift_detected': p_value < 0.001
            'drift_detected': wd > 0.1
        })

    return pd.DataFrame(results)

def rolling_drift(data, report_path, print_evidently):

    if print_evidently:
        os.makedirs(report_path, exist_ok=True)

    weeks = sorted(data['week'].unique())

    all_results = []
    for i in range(len(weeks) - 1):
        ref_w, cur_w = weeks[i], weeks[i + 1]
        reference_df = data[data['week'] == ref_w]
        current_df = data[data['week'] == cur_w]

        drift_results = detect_data_drift(reference_df, current_df)
        drift_results['approach'] = 'rolling'
        drift_results['ref_weeks'] = str(ref_w)
        drift_results['curr_week'] = cur_w
        all_results.append(drift_results)

        if print_evidently:
            my_report = Report([
                DataDriftPreset()
            ])
            my_eval = my_report.run(
                reference_data=reference_df, 
                current_data=current_df
            )
            my_eval.save_html(f'{report_path}data_drift_week_{ref_w}_vs_{cur_w}.html')

        visualize_distributions(reference_df, current_df, ref_w, cur_w, path=report_path)

    return all_results

def expanding_drift(data, report_path, print_evidently):

    if print_evidently:
        os.makedirs(report_path, exist_ok=True)

    weeks = sorted(data['week'].unique())

    all_results = []
    for i in range(2, len(weeks)):
        ref_w, cur_w = weeks[:i], weeks[i]
        reference_df = data[data['week'].isin(ref_w)]
        current_df = data[data['week'] == cur_w]

        drift_results = detect_data_drift(reference_df, current_df)
        drift_results['approach'] = 'expanding'
        drift_results['ref_weeks'] = f'{min(ref_w)} to {max(ref_w)}'
        drift_results['curr_week'] = cur_w
        all_results.append(drift_results)

        if print_evidently:
            my_report = Report([
                DataDriftPreset()
            ])
            my_eval = my_report.run(
                reference_data=reference_df, 
                current_data=current_df
            )

            my_eval.save_html(f'{report_path}data_drift_week_{min(ref_w)}_to_{max(ref_w)}_vs_{cur_w}.html')

    return all_results

def drift_evolution_figure(combined_results, path):
    drift_summary = (
        combined_results.groupby(['approach', 'curr_week'])['drift_detected']
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 4))
    for approach in drift_summary['approach'].unique():
        subset = drift_summary[drift_summary['approach'] == approach]
        plt.plot(subset['curr_week'], subset['drift_detected'], marker='o', label=approach.capitalize())

    plt.title('Mean drift per week')
    plt.xlabel('Current week')
    plt.ylabel('Drift feature proportion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{path}/drift_evolution_rolling_vs_expanding.png')
    plt.close()

def drift_heatmap(combined_results, path):
    for approach in combined_results['approach'].unique():
        pivot = (
            combined_results[combined_results['approach'] == approach]
            .pivot_table(index='feature', columns='curr_week', values='drift_detected', aggfunc='mean')
            .fillna(0)
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Detected drift (1=True, 0=False)'})
        plt.title(f'Drift Heatmap ({approach} approach)')
        plt.xlabel('Current week')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f'{path}/heatmap_{approach}_drift.png')
        plt.close()

def psi_plot(combined_results, path):
    for approach in combined_results['approach'].unique():

        num_features = combined_results[(combined_results['approach'] == approach) & (combined_results['test'] == 'KS')]

        plt.figure(figsize=(6,3))
        plt.bar(num_features['feature'], num_features['psi'], color='skyblue')
        plt.axhline(0.1, color='orange', linestyle='--', label='Moderate drift')
        plt.axhline(0.25, color='red', linestyle='--', label='High drift')
        plt.title(f'PSI (Population Stability Index) for numerical variable ({approach} approach)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{path}/psi_plot_{approach}.png')
        plt.close()

if __name__=='__main__':
    mode = sys.argv[1]

    if mode != 'original' and mode != 'drifted':
        print('invalid args (original or drifted)')
        sys.exit(0)

    if mode == 'original':
        data = pd.read_csv('data/Flights.csv')
    else:
        data = pd.read_csv('data/Flights_drifted.csv')

    report_root_folder = f'report/data_drift_exploration/{mode}'

    data["price"] = data["price"].apply(parse_convert_price)

    rolling_report_path = f'{report_root_folder}/rolling/'
    expanding_report_path = f'{report_root_folder}/expanding/'

    rolling_results = rolling_drift(data, rolling_report_path, print_evidently=True)
    expanding_results = expanding_drift(data, expanding_report_path, print_evidently=True)

    combined_results = pd.concat(rolling_results + expanding_results, ignore_index=True)
    combined_results.to_csv(f'{report_root_folder}/data_drift_rolling_vs_expanding.csv', index=False)

    drift_evolution_figure(combined_results, report_root_folder)
    drift_heatmap(combined_results, report_root_folder)
    psi_plot(combined_results, report_root_folder)