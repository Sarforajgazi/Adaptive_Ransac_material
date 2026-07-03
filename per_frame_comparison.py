"""
per_frame_comparison.py

Builds a true frame-by-frame comparison (not just aggregate averages like
compare_results.py) by joining Strict/Standard/Loose/RL results on their
shared frame_id column -- all four CSVs come from the exact same logging
code path in ransac_env.py, so the join is exact.

For each frame, shows what epsilon/min_support each mode used and what
inlier_ratio it got, plus which mode "won" that specific frame. Also
reports RL's per-frame win rate against each baseline -- a different
(and often more informative) statistic than comparing averages, since
a method can have a higher mean while still losing on most individual
frames (skewed by a few big wins), or vice versa.

Usage:
    python per_frame_comparison.py --env Office
    python per_frame_comparison.py --env all
"""

import os
import argparse
import pandas as pd

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE, "logs")

MODES = ["strict", "standard", "loose", "rl"]


def load_mode_csv(env, mode):
    path = os.path.join(LOG_DIR, f"{env}_{mode}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if mode == "rl":
        # Multi-step episodes can log more than one row per frame -- keep
        # only the terminal step, same convention as compare_results.py.
        df = df.sort_values(["frame_id", "steps_used"]).groupby("frame_id").tail(1)
    return df.drop_duplicates(subset="frame_id", keep="last")


def build_comparison(env):
    frames = {}
    for mode in MODES:
        df = load_mode_csv(env, mode)
        if df is None:
            print(f"  Skipping {mode} for {env} -- file not found")
            continue
        frames[mode] = df.set_index("frame_id")[["epsilon", "min_support", "inlier_ratio", "residual"]]

    if len(frames) < 2:
        print(f"  Not enough modes available for {env}, skipping.")
        return None

    merged = None
    for mode, df in frames.items():
        renamed = df.rename(columns={c: f"{c}_{mode}" for c in df.columns})
        merged = renamed if merged is None else merged.join(renamed, how="inner")

    inlier_cols = {mode: f"inlier_ratio_{mode}" for mode in frames if f"inlier_ratio_{mode}" in merged.columns}
    merged["winner"] = merged[list(inlier_cols.values())].idxmax(axis=1).map(
        {v: k for k, v in inlier_cols.items()}
    )
    return merged.reset_index()


def summarize(env, merged):
    print(f"\n=== {env}: per-frame comparison ({len(merged)} frames) ===")
    win_counts = merged["winner"].value_counts()
    for mode in MODES:
        if mode in win_counts.index:
            pct = win_counts[mode] / len(merged) * 100
            print(f"  {mode:10s} wins on {win_counts[mode]:5d} frames ({pct:.1f}%)")

    if "rl" in merged["winner"].unique() or "inlier_ratio_rl" in merged.columns:
        for baseline in ["strict", "standard", "loose"]:
            col_rl = "inlier_ratio_rl"
            col_base = f"inlier_ratio_{baseline}"
            if col_rl in merged.columns and col_base in merged.columns:
                rl_beats = (merged[col_rl] > merged[col_base]).sum()
                pct = rl_beats / len(merged) * 100
                print(f"  RL beats {baseline:10s} on {rl_beats:5d}/{len(merged)} frames ({pct:.1f}%) [per-frame head-to-head]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="all", help="Dataset name, or 'all'")
    parser.add_argument("--save-dir", type=str, default=LOG_DIR)
    args = parser.parse_args()

    if args.env == "all":
        envs = sorted(set(
            f.split("_rl.csv")[0] for f in os.listdir(LOG_DIR) if f.endswith("_rl.csv")
        ))
    else:
        envs = [args.env]

    all_merged = []
    for env in envs:
        print(f"\nBuilding per-frame comparison for {env}...")
        merged = build_comparison(env)
        if merged is None:
            continue
        summarize(env, merged)
        merged.insert(0, "dataset", env)
        all_merged.append(merged)

        out_path = os.path.join(args.save_dir, f"{env}_per_frame_comparison.csv")
        merged.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    if len(all_merged) > 1:
        combined = pd.concat(all_merged, ignore_index=True)
        print("\n=== Overall (all datasets combined) ===")
        summarize("ALL", combined)
        out_path = os.path.join(args.save_dir, "ALL_per_frame_comparison.csv")
        combined.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
