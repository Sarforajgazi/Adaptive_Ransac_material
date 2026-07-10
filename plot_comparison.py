"""
plot_comparison.py

Generates static PNG charts comparing every trained RL agent (Day 9 `rl`,
plus any later `--tag`-ed retrain such as `rl_v2_reward`) against the three
fixed RANSAC parameter presets (Strict/Standard/Loose), for use in
reports/slides.

Reads the same source CSVs as compare_results.py / per_frame_comparison.py:
    logs/full_comparison_summary.csv
    logs/ALL_per_frame_comparison.csv

Usage:
    python plot_comparison.py
Outputs PNGs to plots/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE, "logs")
OUT_DIR = os.path.join(WORKSPACE, "plots")

BASELINE_ORDER = ["strict", "standard", "loose"]
BASELINE_LABEL = {"strict": "Strict", "standard": "Standard", "loose": "Loose"}
# Strict->Standard->Loose form a conservative->aggressive gray ramp;
# each RL variant gets its own accent hue from this cycle so multiple
# tagged retrains (e.g. rl, rl_v2_reward) stay visually distinct.
BASELINE_COLOR = {"strict": "#c3c2b7", "standard": "#8f8d84", "loose": "#5a584f"}
RL_ACCENT_CYCLE = ["#2a78d6", "#d6722a", "#2ea86b", "#a02ad6"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def build_mode_meta(modes_present):
    """
    Orders modes as baselines-first (strict, standard, loose) followed by
    every rl* mode found in the data (sorted so results are stable run to
    run), and assigns each a label + color. Rebuilt from what's actually in
    the CSVs each run, so a new --tag retrain shows up automatically without
    editing this file.
    """
    modes = [m for m in BASELINE_ORDER if m in modes_present]
    rl_modes = sorted(m for m in modes_present if m.startswith("rl"))
    modes += rl_modes

    label = dict(BASELINE_LABEL)
    color = dict(BASELINE_COLOR)
    for i, m in enumerate(rl_modes):
        label[m] = "RL (adaptive)" if m == "rl" else f"RL ({m[3:]})"
        color[m] = RL_ACCENT_CYCLE[i % len(RL_ACCENT_CYCLE)]
    return modes, label, color


def grouped_bar(df, modes, label, color, value_col, title, ylabel, out_name, pct=False):
    datasets = df["Dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.8 / len(modes)
    offset0 = -(len(modes) - 1) / 2

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, mode in enumerate(modes):
        sub = df[df["Mode"] == mode].set_index("Dataset").reindex(datasets)
        vals = sub[value_col].values * (100 if pct else 1)
        ax.bar(x + (offset0 + i) * width, vals, width, label=label[mode], color=color[mode],
               edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=False, ncol=len(modes), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def win_rate_chart(all_frames, rl_modes, label, color, out_name):
    """One grouped bar per dataset, one bar per RL variant present, showing
    that variant's outright per-frame win share against everything else."""
    rows = []
    for ds, g in all_frames.groupby("dataset"):
        share = g["winner"].value_counts(normalize=True) * 100
        row = {"dataset": ds}
        for m in rl_modes:
            row[m] = share.get(m, 0.0)
        rows.append(row)
    win_df = pd.DataFrame(rows).sort_values(rl_modes[0], ascending=True)

    datasets = win_df["dataset"].values
    y = np.arange(len(datasets))
    height = 0.8 / len(rl_modes)
    offset0 = -(len(rl_modes) - 1) / 2

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(rl_modes):
        ax.barh(y + (offset0 + i) * height, win_df[m].values, height, label=label[m], color=color[m])
    n_modes_total = len(rl_modes) + len(BASELINE_ORDER)
    tie_pct = 100 / n_modes_total
    ax.axvline(tie_pct, color="#c3c2b7", linewidth=1, linestyle="-")
    ax.text(tie_pct + 0.5, len(datasets) - 0.5, f"{tie_pct:.0f}% = tied with all baselines", fontsize=8, color="#898781")
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of frames where this mode had the single highest inlier ratio")
    ax.set_title("Per-frame win rate: RL variant(s) vs. Strict/Standard/Loose", fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=False, ncol=len(rl_modes), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def overall_summary(df, modes, label, color, out_name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("MeanInlier", "Mean inlier ratio", True, "higher is better"),
        ("BadFrames", "Bad-frame rate", None, "lower is better"),
        ("TotalRuntime", "Total runtime (s)", None, "lower is better"),
    ]
    for ax, (col, lbl, pct, sub) in zip(axes, metrics):
        agg = df.groupby("Mode").apply(
            lambda g: (g[col].sum() / g["Frames"].sum() * 100) if col == "BadFrames"
            else (g[col].mean() * 100 if pct else g[col].sum() if col == "TotalRuntime" else g[col].mean())
        ).reindex(modes)
        colors = [color[m] for m in modes]
        ax.bar([label[m] for m in modes], agg.values, color=colors)
        ax.set_title(f"{lbl}\n({sub})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Overall summary across all 10 environments", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(LOG_DIR, "full_comparison_summary.csv"))
    all_frames = pd.read_csv(os.path.join(LOG_DIR, "ALL_per_frame_comparison.csv"))

    modes, label, color = build_mode_meta(set(summary["Mode"].unique()))
    rl_modes = [m for m in modes if m.startswith("rl")]
    print(f"Modes found: {modes}")

    grouped_bar(summary, modes, label, color, "MeanInlier", "Mean inlier ratio, per environment",
                "Mean inlier ratio (%)", "inlier_ratio_comparison.png", pct=True)

    summary_bad = summary.copy()
    summary_bad["BadFramePct"] = summary_bad["BadFrames"] / summary_bad["Frames"] * 100
    grouped_bar(summary_bad, modes, label, color, "BadFramePct", "Bad-frame rate, per environment",
                "Bad frames (%)", "bad_frame_rate_comparison.png")

    if rl_modes:
        win_rate_chart(all_frames, rl_modes, label, color, "win_rate_comparison.png")
    overall_summary(summary, modes, label, color, "overall_summary.png")


if __name__ == "__main__":
    main()
