"""
eval_real_lidar_frames.py

Real quantitative evaluation of the synthetic-trained model against real
per-frame LiDAR sweeps WITH ground truth (<frame>_gt_mask.npy, present only
for AbandonedFactory/CoalMine/Gascola/House/NordicHarbor/WesternDesertTown
under data/<scene>/Data_omni/P0000/lidar/).

Uses the same single-shot bypass as visualize_real_pointcloud.py (see that
file's docstring for why: step()'s shape selection needs a ground-truth
normal that doesn't exist for real data) and the same largest-support-
among-horizontal shape selection fix (SESSION_PROGRESS_LOG.md sec.27).

IMPORTANT: gt_mask is indexed against ransac_env.py's load_ply_xyz(),
which voxel-downsamples at 0.05m -- NOT the raw .ply point count. Loading
raw points here would silently misalign every index against the mask.

Usage:
    python eval_real_lidar_frames.py --tag v2 --frames_per_scene 4 --seed 0
"""
import os
import sys
import argparse
import pickle
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from ransac_env import load_ply_xyz
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GT_SCENES = ["AbandonedFactory", "CoalMine", "Gascola", "House", "NordicHarbor", "WesternDesertTown"]
HORIZONTAL_TOLERANCE_DEG = 30.0


def load_obs_normalizer(vecnormalize_path):
    if vecnormalize_path is None or not os.path.exists(vecnormalize_path):
        return lambda obs: obs
    with open(vecnormalize_path, "rb") as f:
        vec_normalize = pickle.load(f)
    obs_rms = vec_normalize.obs_rms
    clip_obs = vec_normalize.clip_obs
    epsilon = vec_normalize.epsilon
    return lambda obs: np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                                -clip_obs, clip_obs).astype(np.float32)


def choose_params(model, normalize_obs, points):
    scene_feat = compute_scene_features(points)
    feedback_feat = np.zeros(10, dtype=np.float32)
    obs = np.concatenate([scene_feat, feedback_feat]).astype(np.float32)
    action, _ = model.predict(normalize_obs(obs), deterministic=True)
    eps = EPS_LEVELS[int(action[0])]
    min_supp = max(1, int(round(MIN_SUPPORT_LEVELS[int(action[1])] * len(points))))
    norm_th = NORM_THRESH_LEVELS[int(action[2])]
    return eps, min_supp, norm_th


def select_shape(shapes, points):
    """Largest support among roughly-horizontal candidates -- see
    SESSION_PROGRESS_LOG.md sec.27 for why largest-support alone isn't enough."""
    candidates = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        plane_pts = points[mask]
        cov = np.cov(plane_pts.T)
        evals, evecs = np.linalg.eig(cov)
        normal = evecs[:, np.argmin(evals)]
        angle_from_vertical = np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
        candidates.append((shape, angle_from_vertical, int(np.sum(mask))))

    horizontal = [c for c in candidates if c[1] <= HORIZONTAL_TOLERANCE_DEG]
    pool = horizontal if horizontal else candidates
    best_shape, best_angle, best_support = max(pool, key=lambda c: c[2])
    return best_shape, best_angle, bool(horizontal)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--frames_per_scene", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(DATA_ROOT), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    rng = np.random.default_rng(args.seed)
    all_rows = []

    for scene in GT_SCENES:
        lidar_dir = os.path.join(DATA_ROOT, scene, "Data_omni", "P0000", "lidar")
        gt_files = sorted(glob.glob(os.path.join(lidar_dir, "*_gt_mask.npy")))
        if not gt_files:
            print(f"{scene}: no gt_mask files found, skipping")
            continue
        chosen = rng.choice(len(gt_files), size=min(args.frames_per_scene, len(gt_files)), replace=False)

        print(f"\n=== {scene} ({len(gt_files)} frames available, testing {len(chosen)}) ===")
        for idx in chosen:
            gt_path = gt_files[idx]
            ply_path = gt_path.replace("_gt_mask.npy", ".ply")
            frame_name = os.path.basename(ply_path)

            points = load_ply_xyz(ply_path)
            gt_mask = np.load(gt_path)
            if len(points) != len(gt_mask):
                print(f"  {frame_name}: SKIPPED -- point/mask length mismatch ({len(points)} vs {len(gt_mask)})")
                continue

            eps, min_supp, norm_th = choose_params(model, normalize_obs, points)
            try:
                shapes, _ = schnabel_ransac.detect(
                    points, shapes=["plane"], relative_epsilon=False,
                    epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
                    probability=0.001, normal_knn=20, max_shapes=10,
                )
            except Exception as e:
                print(f"  {frame_name}: RANSAC error: {e}")
                continue

            if not shapes:
                print(f"  {frame_name}: no planes found (eps={eps}, min_supp={min_supp}, norm_th={norm_th})")
                all_rows.append(dict(scene=scene, frame=frame_name, found=False, was_horizontal=False,
                                      iou=0.0, precision=0.0, recall=0.0, f1=0.0,
                                      n_points=len(points), gt_frac=float(gt_mask.mean())))
                continue

            best_shape, angle, was_horizontal = select_shape(shapes, points)
            pred_mask = best_shape["inlier_mask"]
            iou, precision, recall, f1 = compute_metrics(pred_mask, gt_mask)
            all_rows.append(dict(scene=scene, frame=frame_name, found=True, was_horizontal=was_horizontal,
                                  iou=iou, precision=precision, recall=recall, f1=f1,
                                  n_points=len(points), gt_frac=float(gt_mask.mean())))
            print(f"  {frame_name}: IoU={iou:.3f} precision={precision:.3f} recall={recall:.3f} "
                  f"f1={f1:.3f}  (angle={angle:.1f} deg, horizontal_candidate_available={was_horizontal}, "
                  f"gt covers {gt_mask.mean()*100:.1f}% of {len(points)} points)")

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    if not all_rows:
        print("No results.")
        return

    import csv
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"real_lidar_eval_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Raw results saved to {csv_path}")

    ious = [r["iou"] for r in all_rows]
    found_rate = np.mean([r["found"] for r in all_rows])
    print(f"\nTotal frames tested: {len(all_rows)}")
    print(f"Found-a-plane rate: {found_rate*100:.1f}%")
    print(f"Mean IoU (all frames, 0 if none found): {np.mean(ious):.4f}")
    found_rows = [r for r in all_rows if r["found"]]
    if found_rows:
        print(f"Mean IoU (frames where a plane was found): {np.mean([r['iou'] for r in found_rows]):.4f}")
        print(f"Mean precision: {np.mean([r['precision'] for r in found_rows]):.4f}")
        print(f"Mean recall: {np.mean([r['recall'] for r in found_rows]):.4f}")
        print(f"Mean F1: {np.mean([r['f1'] for r in found_rows]):.4f}")

    print("\nPer-scene breakdown:")
    for scene in GT_SCENES:
        scene_rows = [r for r in all_rows if r["scene"] == scene]
        if not scene_rows:
            continue
        scene_ious = [r["iou"] for r in scene_rows]
        print(f"  {scene:20s} n={len(scene_rows)}  mean_IoU={np.mean(scene_ious):.4f}  "
              f"found_rate={np.mean([r['found'] for r in scene_rows])*100:.0f}%")


if __name__ == "__main__":
    main()
