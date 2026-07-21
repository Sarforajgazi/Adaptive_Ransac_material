"""
visualize_terrain_types_rl.py

Tests the synthetic-trained RL model against real per-frame LiDAR sweeps
from TartanGround scenes chosen specifically for terrain type, in response
to a request for beach/muddy-ground/water-body/park-like real data:
    SeasideTown  -- beach/coastal town
    GreatMarsh   -- marshland (muddy/wetland terrain)
    NordicHarbor -- harbor (water-adjacent)
    GothicIsland -- island (water-adjacent)
All 4 were already downloaded locally under data/<scene>/Data_omni/P0000/
lidar/ but never tested this session -- no new download needed.

Same single-shot bypass + largest-support-among-horizontal selection +
height colormap (these per-frame files have no RGB, confirmed in sec.
"Real-World Testing" earlier this session) as visualize_real_pointcloud.py
/ browse_real_lidar_frames.py. Organizes output the same way as the
off-road results (per-scene folder + results.csv + combined summary), per
the user's explicit ask for easy-to-find results.

Usage:
    python visualize_terrain_types_rl.py --frames_per_scene 3 --seed 0
"""
import os
import sys
import csv
import glob
import argparse
import pickle
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from ransac_env import load_ply_xyz
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "terrain_types")
HORIZONTAL_TOLERANCE_DEG = 30.0

SCENES = {
    "SeasideTown": "beach / coastal town",
    "GreatMarsh": "marshland (muddy/wetland terrain)",
    "NordicHarbor": "harbor (water-adjacent)",
    "GothicIsland": "island (water-adjacent)",
}

CSV_FIELDS = ["scene", "frame", "n_points", "found", "pred_pct", "angle", "eps", "min_supp", "norm_th"]


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
    return best_shape, best_angle, bool(horizontal)


def height_colormap(points):
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    normalized = (z - z_min) / (z_max - z_min + 1e-8)
    return cm.viridis(normalized)[:, :3]


def append_csv_row(csv_path, row):
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def process_frame(model, normalize_obs, scene, ply_path):
    frame_name = os.path.basename(ply_path)
    points = load_ply_xyz(ply_path)
    eps, min_supp, norm_th = choose_params(model, normalize_obs, points)

    try:
        shapes, _ = schnabel_ransac.detect(
            points, shapes=["plane"], relative_epsilon=False,
            epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
            probability=0.001, normal_knn=20, max_shapes=10,
        )
    except Exception as e:
        print(f"  RANSAC error: {e}")
        shapes = []

    colors = height_colormap(points)
    result = dict(scene=scene, frame=frame_name, n_points=len(points), found=False,
                  pred_pct=0.0, angle=None, eps=eps, min_supp=min_supp, norm_th=norm_th)
    if shapes:
        best_shape, angle, was_horizontal = select_shape(shapes, points)
        pred_mask = best_shape["inlier_mask"]
        colors[pred_mask] = [0.0, 1.0, 0.0]
        status = (f"FOUND: {int(pred_mask.sum())} pts ({pred_mask.mean()*100:.1f}%), "
                  f"{angle:.1f} deg from vertical, horizontal_candidate={was_horizontal}")
        result.update(found=True, pred_pct=pred_mask.mean() * 100, angle=angle)
    else:
        status = "NOT FOUND"
    print(f"  [{scene}] {frame_name} ({len(points)} pts) eps={eps} min_supp={min_supp} norm_th={norm_th} -> {status}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    return result, pcd, status


def show_interactive(pcd, scene, frame_name, status):
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"{scene}/{frame_name} -- color=height, green=predicted ground -- {status}",
        width=1400, height=900,
    )


def save_screenshot(pcd, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=960)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.5
    render_opt.light_on = True
    ctr = vis.get_view_control()
    ctr.rotate(200.0, 100.0)
    ctr.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--frames_per_scene", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interactive", action="store_true", help="open a window per frame (blocking, close to advance)")
    parser.add_argument("--skip_save", action="store_true", help="don't re-write screenshots/CSV rows -- use with --interactive to just re-view already-computed results")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    rng = np.random.default_rng(args.seed)
    all_results = []

    for scene, desc in SCENES.items():
        lidar_dir = os.path.join(DATA_ROOT, scene, "Data_omni", "P0000", "lidar")
        ply_files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
        if not ply_files:
            print(f"{scene}: no lidar files found, skipping")
            continue
        chosen = rng.choice(len(ply_files), size=min(args.frames_per_scene, len(ply_files)), replace=False)

        print(f"\n=== {scene} ({desc}) -- {len(ply_files)} frames available ===")
        for idx in chosen:
            ply_path = ply_files[idx]
            result, pcd, status = process_frame(model, normalize_obs, scene, ply_path)
            all_results.append(result)
            if not args.skip_save:
                frame_stem = os.path.splitext(os.path.basename(ply_path))[0]
                out_path = os.path.join(OUT_DIR, scene, f"{frame_stem}.png")
                save_screenshot(pcd, out_path)
                append_csv_row(os.path.join(OUT_DIR, scene, "results.csv"), result)
                append_csv_row(os.path.join(OUT_DIR, "ALL_SCENES_SUMMARY.csv"), result)
            if args.interactive:
                show_interactive(pcd, scene, os.path.basename(ply_path), status)

    print("\n" + "=" * 90)
    print(f"{'scene':>14} {'frame':>28} {'pts':>7} {'found':>7} {'pred%':>7} {'angle':>7}")
    for r in all_results:
        print(f"{r['scene']:>14} {r['frame']:>28} {r['n_points']:7d} {str(r['found']):>7} "
              f"{r['pred_pct']:7.1f} {('%.1f' % r['angle']) if r['angle'] is not None else '-':>7}")
    print(f"\nResults saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
