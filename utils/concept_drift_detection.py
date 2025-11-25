import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def correlation_ratio(categories, values):   
    '''
    Compute correlation ration using one way ANOVA (Analysis of Variance) F-statistic.

    Determines how well a categorical variable explains a numerical variable
    '''

    groups = [values[categories == c] for c in np.unique(categories)]
    f, _ = f_oneway(*groups)
    k = len(groups)

    # Divide by f+(k-1) to get [0;1] values (normalize)
    return float(f / (f + (k - 1))) if np.isfinite(f) else np.nan
    
def plot_concept_drift(corr_by_week, output_dir):
    '''Plot correlation heatmap'''

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_by_week.T, cmap="coolwarm", center=0, annot=False)
    plt.title("Evolution of correlations between features and price")
    plt.xlabel("Week")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "corr_evolution_heatmap.png"))
    plt.close()

def compute_concept_drift(data, report_path, target="price"):
    '''Compute concept drift tests'''

    numerical_cols = data.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns
    weeks = sorted(data["week"].unique())

    categorical_cols = categorical_cols.drop("date")

    corr_summary = {}

    for week in weeks:
        df = data[data["week"] == week]
        corr_dict = {}

        # Numerical correlation
        for col in numerical_cols:
            if df[col].nunique() <= 1:
                corr_dict[col] = np.nan
            else:
                corr_dict[col] = df[col].corr(df[target])

        # Categorical correlation
        for col in categorical_cols:
            corr_dict[col] = correlation_ratio(df[col], df[target])

        corr_summary[week] = corr_dict

    corr_by_week = pd.DataFrame(corr_summary).T

    results = []

    for i in range(1, len(weeks)):
        w_prev, w_curr = weeks[i - 1], weeks[i]
        diff = (corr_by_week.loc[w_curr] - corr_by_week.loc[w_prev]).abs()
        mean_diff = diff.mean()

        results.append({
            "week_pair": f"{w_prev}-{w_curr}",
            "mean_corr_diff": mean_diff
        })

    plot_concept_drift(corr_by_week, report_path)

    return pd.DataFrame(results)