import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_synthetic import run_name_for, LOG_DIR, SCRIPT_DIR

PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

def resolve_csv_path(tag):
    if tag:
        path = os.path.join(LOG_DIR, run_name_for(tag) + "_eval.csv")
        if not os.path.exists(path):
            raise SystemExit(f"No eval results at {path} -- run evaluate_synthetic.py --tag {tag} first.")
        return path

    candidates = sorted(glob.glob(os.path.join(LOG_DIR, "synthetic_ppo*_eval.csv")), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No eval results found in {LOG_DIR} -- run evaluate_synthetic.py first.")
    return candidates[0]

def plot_adaptivity(tag=None):
    csv_path = resolve_csv_path(tag)
    run_name = os.path.basename(csv_path).replace("_eval.csv", "")
    print(f"Plotting from: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # We want to plot:
    # 1. Chosen Parameter (e.g., normal_th) vs Noise and vs Inlier Ratio
    # 2. Metric (e.g., angle_error, inlier_recovery_rate) vs Noise and vs Inlier Ratio
    
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("RL Agent Adaptivity vs Fixed Baselines (as a function of Synthetic Noise)", fontsize=16)
    
    # 1. Epsilon
    sns.lineplot(data=df, x="noise_sigma", y="eps", hue="method", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Distance Threshold ($\epsilon$) vs Noise $\sigma$")
    axes[0, 0].set_ylabel("Epsilon (m)")
    
    # 2. Min Support
    sns.lineplot(data=df, x="noise_sigma", y="min_supp", hue="method", marker="o", ax=axes[0, 1])
    axes[0, 1].set_title("Minimum Support vs Noise $\sigma$")
    axes[0, 1].set_ylabel("Min Support (pts)")
    
    # 3. Normal Threshold
    sns.lineplot(data=df, x="noise_sigma", y="norm_th", hue="method", marker="o", ax=axes[0, 2])
    axes[0, 2].set_title("Normal Threshold vs Noise $\sigma$")
    axes[0, 2].set_ylabel("Normal Threshold")
    
    # 4. Angle Error
    sns.lineplot(data=df, x="noise_sigma", y="angle_error", hue="method", marker="o", ax=axes[1, 0])
    axes[1, 0].set_title("Performance: Angle Error")
    axes[1, 0].set_ylabel("Angle Error (degrees)")
    
    # 5. Inlier Recovery
    sns.lineplot(data=df, x="noise_sigma", y="inlier_recovery_rate", hue="method", marker="o", ax=axes[1, 1])
    axes[1, 1].set_title("Performance: Inlier Recovery Rate")
    axes[1, 1].set_ylabel("Recall")
    
    # 6. Total Score
    sns.lineplot(data=df, x="noise_sigma", y="score", hue="method", marker="o", ax=axes[1, 2])
    axes[1, 2].set_title("Overall Reward Score")
    axes[1, 2].set_ylabel("Score")
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_png = os.path.join(PLOTS_DIR, run_name + "_adaptivity.png")
    plt.savefig(out_png, dpi=300)
    print(f"Plots saved to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None,
                         help="Plot the eval results for this tag (e.g. 'v2'). Omit to auto-pick "
                              "the most recently generated eval CSV.")
    args = parser.parse_args()
    plot_adaptivity(tag=args.tag)
