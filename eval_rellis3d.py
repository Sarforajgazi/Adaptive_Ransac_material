"""
eval_rellis3d.py

Real-world, real-ground-truth evaluation on RELLIS-3D (unmannedlab/RELLIS-3D),
run separately from the synthetic TartanAir pipeline (rl_evaluator.py,
baseline_evaluator.py) since the metric here is genuine IoU/precision/
recall/F1 against RELLIS-3D's point-wise semantic labels, not the
heuristic inlier_ratio used everywhere else.

Prerequisites:
    python download_rellis3d.py
    python rellis3d_convert.py

For each RELLIS-3D frame, runs the three fixed baseline configs
(BASELINE_MODES from baseline_evaluator.py) and the trained RL policy
(same model/VecNormalize loading as rl_evaluator.py) through the exact
same RansacEnv/Schnabel-RANSAC code path used everywhere else, then scores
the resulting ground-plane inlier mask against real ground truth:
nearest-neighbor-matches each (possibly voxel-downsampled) predicted point
back to the raw, unmodified RELLIS-3D point cloud to inherit its real
ground/not-ground label -- this sidesteps any dependency on RansacEnv's
internal downsampling being index-stable.

Usage:
    python eval_rellis3d.py                  # full run, all sequences
    python eval_rellis3d.py --limit 20        # first 20 frames per sequence
                                               # (smoke test)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import cKDTree
from stable_baselines3 import PPO

from ransac_env import RansacEnv, load_ply_xyz
from features.scene_features import compute_scene_features
from baseline_evaluator import BASELINE_MODES, build_action
from rl_evaluator import load_obs_normalizer, DEFAULT_MODEL_PATH, DEFAULT_VECNORMALIZE_PATH

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(WORKSPACE, "data", "RELLIS3D")
LOG_DIR = os.path.join(WORKSPACE, "logs")


def get_sequences():
    return sorted(d for d in glob.glob(os.path.join(DATA_ROOT, "RELLIS3D_*")) if os.path.isdir(d))


def get_frame_files(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir, "lidar", "*.ply")))


def load_raw_reference(ply_path):
    """
    Reads the frame's raw (undownsampled) xyz straight from disk -- NOT via
    ransac_env.load_ply_xyz, which voxel-downsamples -- plus its aligned
    ground-truth mask, and builds a KD-tree for nearest-neighbor lookup.
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    gt_path = ply_path[:-4] + "_gt.npy"
    gt = np.load(gt_path).astype(bool)
    tree = cKDTree(xyz)
    return tree, gt


def compute_metrics(pred_mask, gt_mask):
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return iou, precision, recall, f1


def score_frame(frame_path, env, info, reward, steps):
    tree, gt = load_raw_reference(frame_path)
    n = len(env.current_points)
    pred_mask = info.get("inlier_mask")
    if pred_mask is None:
        pred_mask = np.zeros(n, dtype=bool)
    _, nn_idx = tree.query(env.current_points, k=1)
    gt_for_frame = gt[nn_idx]
    iou, precision, recall, f1 = compute_metrics(pred_mask, gt_for_frame)
    return {
        "frame_id": os.path.basename(frame_path),
        "n_points": n,
        "steps_used": steps,
        "reward": float(reward),
        "inlier_ratio": float(env.inlier_ratio),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def eval_baseline_mode(mode, config, seq_name, seq_dir, files, limit=None):
    frames = files[:limit] if limit else files
    env = RansacEnv(data_dir=seq_dir, log_name=f"{seq_name}_{mode}.csv",
                     fixed_normal_thresh=config["normal_thresh"])
    action = build_action(config["epsilon"], config["min_support"])

    rows = []
    for f in frames:
        env.current_file = f
        env.current_points = load_ply_xyz(f)
        env.current_features = compute_scene_features(env.current_points)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(score_frame(f, env, info, reward, steps=1))
    return rows


def eval_rl_mode(model, normalize_obs, seq_name, seq_dir, files, limit=None):
    frames = files[:limit] if limit else files
    env = RansacEnv(data_dir=seq_dir, log_name=f"{seq_name}_rl.csv")

    rows = []
    for f in frames:
        env.step_count = 0
        env.inlier_ratio = 0.0
        env.prev_inlier_ratio = 0.0
        env.mean_residual = 0.0
        env.plane_normal = np.zeros(3, dtype=np.float32)
        env.prev_epsilon = 0.0
        env.prev_min_support = 0
        env.prev_normal_thresh = 0.0

        env.current_file = f
        env.current_points = load_ply_xyz(f)
        env.current_features = compute_scene_features(env.current_points)

        obs = env._get_obs()
        terminated = False
        reward, info = 0.0, {}
        while not terminated:
            action, _states = model.predict(normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        rows.append(score_frame(f, env, info, reward, steps=env.step_count))
    return rows


def write_mode_csv(mode, rows):
    if not rows:
        return
    out_path = os.path.join(LOG_DIR, f"RELLIS3D_{mode}_metrics.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Max frames per sequence (for a quick smoke test)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize", type=str, default=DEFAULT_VECNORMALIZE_PATH)
    args = parser.parse_args()

    seqs = get_sequences()
    if not seqs:
        print(f"No converted RELLIS-3D data found under {DATA_ROOT}. "
              f"Run download_rellis3d.py then rellis3d_convert.py first.")
        return

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)
    print(f"Loading observation normalizer: {args.vecnormalize}")
    normalize_obs = load_obs_normalizer(args.vecnormalize)

    modes = list(BASELINE_MODES.keys()) + ["rl"]
    all_rows = {mode: [] for mode in modes}

    for seq_dir in seqs:
        seq_name = os.path.basename(seq_dir)
        files = get_frame_files(seq_dir)
        if not files:
            continue
        print(f"\n=== {seq_name}: {len(files)} frames ===")

        for mode, config in BASELINE_MODES.items():
            rows = eval_baseline_mode(mode, config, seq_name, seq_dir, files, limit=args.limit)
            all_rows[mode].extend(rows)
            mean_iou = np.mean([r["iou"] for r in rows]) if rows else float("nan")
            print(f"  {mode:10s} mean IoU={mean_iou:.4f} ({len(rows)} frames)")

        rl_rows = eval_rl_mode(model, normalize_obs, seq_name, seq_dir, files, limit=args.limit)
        all_rows["rl"].extend(rl_rows)
        mean_iou = np.mean([r["iou"] for r in rl_rows]) if rl_rows else float("nan")
        print(f"  {'rl':10s} mean IoU={mean_iou:.4f} ({len(rl_rows)} frames)")

    print("\n=== Overall RELLIS-3D real-ground-truth comparison ===")
    summary_rows = []
    for mode in modes:
        rows = all_rows[mode]
        write_mode_csv(mode, rows)
        if not rows:
            continue
        mean_iou = np.mean([r["iou"] for r in rows])
        mean_precision = np.mean([r["precision"] for r in rows])
        mean_recall = np.mean([r["recall"] for r in rows])
        mean_f1 = np.mean([r["f1"] for r in rows])
        print(f"  {mode:10s} IoU={mean_iou:.4f}  Precision={mean_precision:.4f}  "
              f"Recall={mean_recall:.4f}  F1={mean_f1:.4f}  (n={len(rows)} frames)")
        summary_rows.append({
            "mode": mode, "iou": mean_iou, "precision": mean_precision,
            "recall": mean_recall, "f1": mean_f1, "n_frames": len(rows),
        })

    if summary_rows:
        summary_path = os.path.join(LOG_DIR, "RELLIS3D_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSaved overall summary: {summary_path}")


if __name__ == "__main__":
    main()
