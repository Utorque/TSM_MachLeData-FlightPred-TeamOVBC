import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance

def load_predictions(path="exploration/", pattern=r"predictions_test_w(\d+).csv"):
    predictions = []
    for fname in os.listdir(path):
        match = re.match(pattern, fname)
        if match:
            week = int(match.group(1))
            df = pd.read_csv(os.path.join(path, fname))
            df["week"] = week
            df["error"] = df["y_true"] - df["y_pred"]
            predictions.append(df)
    return pd.concat(predictions, ignore_index=True).sort_values("week")

def compute_drift_metrics(df, col="error"):
    weeks = sorted(df["week"].unique())
    results = []
    for i in range(len(weeks) - 1):
        w1, w2 = weeks[i], weeks[i + 1]
        e1 = df[df["week"] == w1][col]
        e2 = df[df["week"] == w2][col]
        ks_stat, ks_p = ks_2samp(e1, e2)
        wd = wasserstein_distance(e1, e2) / np.std(e1)
        results.append({
            "week_pair": f"{w1}-{w2}",
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "wasserstein": wd,
            "drift_detected": wd > 0.1
        })
    return pd.DataFrame(results)

def plot_error_distributions(df, output_dir="report/concept_drift/"):
    os.makedirs(output_dir, exist_ok=True)
    weeks = sorted(df["week"].unique())
    for week in weeks:
        sns.kdeplot(df[df["week"] == week]["error"], label=f"Week {week}", fill=True, alpha=0.4)
    plt.title("Distribution of prediction errors per week")
    plt.xlabel("Prediction error (y_true - y_pred)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()

def plot_drift_evolution(drift_df, output_dir="report/concept_drift/"):
    plt.figure(figsize=(6,4))
    plt.plot(drift_df["week_pair"], drift_df["wasserstein"], marker="o", label="Wasserstein")
    plt.axhline(0.1, color="red", linestyle="--", label="Drift threshold")
    plt.title("Wasserstein distance between error distributions (per week)")
    plt.xlabel("Week pair")
    plt.ylabel("Wasserstein distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drift_evolution.png"))
    plt.close()

if __name__ == "__main__":
    output_dir = "report/concept_drift/"
    df = load_predictions()
    plot_error_distributions(df, output_dir=output_dir)
    drift_df = compute_drift_metrics(df)
    drift_df.to_csv(os.path.join(output_dir, "concept_drift_results.csv"), index=False)
    plot_drift_evolution(drift_df, output_dir=output_dir)
    print("Concept drift analysis completed.")
