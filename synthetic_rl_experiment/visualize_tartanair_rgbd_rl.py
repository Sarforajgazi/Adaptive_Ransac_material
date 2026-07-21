"""
visualize_tartanair_rgbd_rl.py

Runs the synthetic-trained RL model on TartanGround's native image+depth
modality (not lidar) for the water/mud scenes already confirmed relevant
this session (SeasideTown, GreatMarsh, NordicHarbor, GothicIsland) --
downloaded via the project's tartanair library after patching
.venv/Lib/site-packages/tartanair/tartanair.py to defer the cupy-only
customizer import (see SESSION_PROGRESS_LOG.md), scoped to camera_name=
['lcam_front'] only (the unscoped download pulled 766MB+ of every camera
direction for one scene alone -- far more than needed).

Depth format confirmed via tartanair's own reader.py (read_depth ->
depth_rgba_float32): the "depth" PNG is RGBA-uint8 but the 4 bytes per
pixel are literally a little-endian float32 depth-in-meters value,
decoded via `cv2.imread(path, IMREAD_UNCHANGED).view('<f4').squeeze(-1)`.
Camera intrinsics confirmed via tartanair's customizer.py (hardcoded,
used internally for their own re-projection code): fx=fy=320,
cx=cy=319.5, matching the 640x640 image/depth resolution.

Same single-shot bypass + recentering + largest-support-among-horizontal
selection as visualize_diode_rl.py (the other RGB-D-to-point-cloud
script this session) -- including the same camera-frame -> z-up axis
swap, since TartanAir's NED-ish camera convention isn't z-up either.
No ground truth available for this modality, so visual-only (true RGB +
height-colored/predicted-ground), same standard as DIODE/Ridgecrest.

Usage:
    python visualize_tartanair_rgbd_rl.py --scene GothicIsland --sample 5 --save_screenshot
"""
import os
import sys
import csv
import glob
import argparse
import pickle
import random
import numpy as np
import cv2
import open3d as o3d
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "rgbd", "tartanair")
HORIZONTAL_TOLERANCE_DEG = 30.0
FX, FY, CX, CY = 320.0, 320.0, 319.5, 319.5  # tartanair/customizer.py, confirmed real not guessed

CSV_FIELDS = ["scene", "frame", "n_points", "found", "pred_pct", "angle_from_vertical_deg", "eps", "min_supp", "norm_th"]


def discover_frames(scene):
    img_dir = os.path.join(DATA_ROOT, scene, "Data_omni", "P0000", "image_lcam_front")
    depth_dir = os.path.join(DATA_ROOT, scene, "Data_omni", "P0000", "depth_lcam_front")
    frames = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]  # e.g. 000000_lcam_front
        depth_path = os.path.join(depth_dir, stem + "_depth.png")
        if os.path.exists(depth_path):
            frames.append((img_path, depth_path))
    return frames


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


def backproject(img_path, depth_path):
    rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.0
    depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4").squeeze(-1)

    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth
    x = (uu - CX) * z / FX
    y = (vv - CY) * z / FY
    # camera-frame (X right, Y down-image, Z forward) -> this session's z-up
    # convention, same swap as visualize_diode_rl.py
    points = np.stack([x, z, -y], axis=-1).reshape(-1, 3).astype(np.float32)
    colors = rgb.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (z.reshape(-1) > 0.05) & (z.reshape(-1) < 200)
    return points[valid], colors[valid]


def height_colormap(points):
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    normalized = (z - z_min) / (z_max - z_min + 1e-8)
    return cm.viridis(normalized)[:, :3]


def process_frame(model, normalize_obs, scene, img_path, depth_path):
    frame_name = os.path.basename(img_path)
    points_world, colors_rgb = backproject(img_path, depth_path)

    centroid = points_world.mean(axis=0)
    points = points_world - centroid

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

    result = dict(scene=scene, frame=frame_name, n_points=len(points), found=False,
                  pred_pct=0.0, angle=None, eps=eps, min_supp=min_supp, norm_th=norm_th)
    colors_height = height_colormap(points_world)
    if shapes:
        best_shape, angle, was_horizontal = select_shape(shapes, points)
        pred_mask = best_shape["inlier_mask"]
        colors_height[pred_mask] = [0.0, 1.0, 0.0]
        status = (f"FOUND: {int(pred_mask.sum())} pts ({pred_mask.mean()*100:.1f}%), "
                  f"{angle:.1f} deg from vertical, horizontal_candidate={was_horizontal}")
        result.update(found=True, pred_pct=pred_mask.mean() * 100, angle=angle)
    else:
        status = "NOT FOUND"
    print(f"  [{scene}] {frame_name} ({len(points)} pts) eps={eps} min_supp={min_supp} norm_th={norm_th} -> {status}")

    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points_world)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors_rgb)

    extent = points_world.max(axis=0) - points_world.min(axis=0)
    offset = np.array([extent[0] * 1.15, 0.0, 0.0], dtype=np.float32)
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_world + offset)
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_height)

    return result, pcd_rgb, pcd_pred, status


def show_interactive(pcd_rgb, pcd_pred, scene, frame_name, status):
    o3d.visualization.draw_geometries(
        [pcd_rgb, pcd_pred],
        window_name=f"{scene}/{frame_name} -- left: true RGB, right: height+predicted ground (green) -- {status}",
        width=1600, height=900,
    )


def save_screenshot(pcd_rgb, pcd_pred, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=900)
    vis.add_geometry(pcd_rgb)
    vis.add_geometry(pcd_pred)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.light_on = True
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 250.0)
    ctr.set_zoom(0.6)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def append_csv_row(csv_path, row):
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_screenshot", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    frames = discover_frames(args.scene)
    if not frames:
        print(f"No image/depth frames found for {args.scene} -- download image+depth modality first")
        sys.exit(1)
    print(f"{len(frames)} frames available for {args.scene}")

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    rng = random.Random(args.seed)
    chosen = rng.sample(frames, min(args.sample, len(frames)))

    all_results = []
    for img_path, depth_path in chosen:
        result, pcd_rgb, pcd_pred, status = process_frame(model, normalize_obs, args.scene, img_path, depth_path)
        all_results.append(result)
        if args.save_screenshot:
            out_path = os.path.join(OUT_DIR, args.scene, os.path.splitext(result["frame"])[0] + ".png")
            save_screenshot(pcd_rgb, pcd_pred, out_path)
            print(f"  saved {out_path}")
            append_csv_row(os.path.join(OUT_DIR, args.scene, "results.csv"), result)
            append_csv_row(os.path.join(OUT_DIR, "ALL_SCENES_SUMMARY.csv"), result)
        if args.interactive:
            show_interactive(pcd_rgb, pcd_pred, args.scene, result["frame"], status)

    print("\n" + "=" * 80)
    for r in all_results:
        print(f"{r['scene']:>14} {r['frame']:>28} {r['n_points']:8d} {str(r['found']):>7} "
              f"{r['pred_pct']:7.1f} {('%.1f' % r['angle']) if r['angle'] is not None else '-':>7}")


if __name__ == "__main__":
    main()
