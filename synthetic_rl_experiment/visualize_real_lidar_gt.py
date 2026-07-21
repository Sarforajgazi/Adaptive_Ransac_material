"""
visualize_real_lidar_gt.py

Visual (not numeric) comparison of the synthetic-trained model's predicted
ground plane against REAL ground truth, on real per-frame LiDAR sweeps.
Ground truth (<frame>_gt_mask.npy) only exists for AbandonedFactory,
CoalMine, Gascola, House, NordicHarbor, WesternDesertTown under
data/<scene>/Data_omni/P0000/lidar/.

Point/mask alignment is NOT assumed -- confirmed empirically that some
scenes' gt_mask matches the RAW point count (e.g. CoalMine) and others
match ransac_env.py's voxel-downsampled (0.05m) point count (e.g. Gascola,
House). This tries raw first, falls back to downsampled, and skips the
frame with a clear message if neither matches rather than silently
comparing misaligned arrays.

Same single-shot bypass + largest-support-among-horizontal shape selection
as visualize_real_pointcloud.py / eval_real_lidar_frames.py.

Usage:
    python visualize_real_lidar_gt.py --tag v2 --frames_per_scene 2 --seed 0
"""
import os
import sys
import argparse
import pickle
import glob
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from ransac_env import load_ply_xyz
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "real_gt")
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


def load_points_matching_mask(ply_path, gt_mask):
    pcd = o3d.io.read_point_cloud(ply_path)
    raw = np.asarray(pcd.points).astype(np.float32)
    if len(raw) == len(gt_mask):
        return raw, "raw"
    downsampled = load_ply_xyz(ply_path)
    if len(downsampled) == len(gt_mask):
        return downsampled, "voxel_0.05"
    return None, None


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
    best_shape, best_angle, _ = max(pool, key=lambda c: c[2])
    return best_shape, best_angle


def save_screenshot(points, colors, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=960)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.5
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--frames_per_scene", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(DATA_ROOT), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    rng = np.random.default_rng(args.seed)

    for scene in GT_SCENES:
        lidar_dir = os.path.join(DATA_ROOT, scene, "Data_omni", "P0000", "lidar")
        gt_files = sorted(glob.glob(os.path.join(lidar_dir, "*_gt_mask.npy")))
        if not gt_files:
            continue
        chosen = rng.choice(len(gt_files), size=min(args.frames_per_scene, len(gt_files)), replace=False)

        print(f"\n=== {scene} ===")
        for idx in chosen:
            gt_path = gt_files[idx]
            ply_path = gt_path.replace("_gt_mask.npy", ".ply")
            frame_name = os.path.splitext(os.path.basename(ply_path))[0]
            gt_mask = np.load(gt_path).astype(bool)

            points, source = load_points_matching_mask(ply_path, gt_mask)
            if points is None:
                print(f"  {frame_name}: SKIPPED -- no point source matches gt_mask length ({len(gt_mask)})")
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

            colors = np.zeros((len(points), 3))
            colors[:] = [0.75, 0.75, 0.75]  # TN: gray
            colors[gt_mask] = [0.9, 0.6, 0.0]  # FN default (real ground, not yet predicted): orange

            if not shapes:
                print(f"  {frame_name} ({source}): no planes found -- showing ground truth only")
            else:
                best_shape, angle = select_shape(shapes, points)
                pred_mask = best_shape["inlier_mask"]
                colors[pred_mask & gt_mask] = [1.0, 1.0, 0.0]   # TP: yellow
                colors[pred_mask & ~gt_mask] = [1.0, 0.0, 0.0]  # FP: red
                # FN (gt & ~pred) stays orange from above; TN stays gray
                print(f"  {frame_name} ({source}): plane found, {angle:.1f} deg from vertical, "
                      f"{int(pred_mask.sum())} pts selected, gt covers {gt_mask.mean()*100:.1f}%")

            out_path = os.path.join(OUT_DIR, f"{scene}_{frame_name}.png")
            save_screenshot(points, colors, out_path)
            print(f"    saved {out_path}")

    print("\nLegend: yellow=correct (TP), orange=missed real ground (FN), "
          "red=wrongly selected (FP), gray=correctly not ground (TN)")


if __name__ == "__main__":
    main()
