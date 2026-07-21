"""
visualize_diode_rl.py

Converts DIODE (Dense Indoor/Outdoor DEpth) RGB-D crops to 3D point clouds
via back-projection, then runs the synthetic-trained RL model on them --
a third real data modality this session, after LiDAR tiles (new_data/,
off_road_data/): here the "point cloud" doesn't exist natively, it's
derived from a single RGB image + a dense depth map + camera intrinsics.

Source: http://diode-dataset.org, val.tar.gz (2.77GB, downloaded whole --
no per-scene download exists, unlike the OpenTopography LiDAR data).
Extracted only val/outdoor/ (3 scenes, 10 scans, 446 RGB-D crops; each
crop is one virtual-camera view reprojected from a laser-scanned panorama
at a given yaw/pitch, per the DIODE paper sec 3.2 -- crops are NOT
combined across yaw/pitch here since the inter-crop rotation isn't
published, only the flat intrinsics for one crop).

Camera intrinsics (from diode-devkit/intrinsics.txt, confirmed real, not
assumed): fx=886.81, fy=927.06, cx=512, cy=384, matching the 1024x768
RGB/depth resolution. Standard pinhole back-projection:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth[v, u]
Only pixels with depth_mask == 1 are kept (mask marks valid sensor
returns -- DIODE's depth has real gaps, e.g. sky, glare, out-of-range).

Same single-shot bypass + recentering + largest-support-among-horizontal
selection as the other real-data scripts this session. No ASPRS-style
ground truth exists here, so this is visual-only (height-colored terrain
+ green predicted-ground overlay), same as the Ridgecrest tiles.

Usage:
    python visualize_diode_rl.py --list                    # show available crops
    python visualize_diode_rl.py --crop scene_00022/scan_00193/00022_00193_outdoor_060_040
    python visualize_diode_rl.py --sample 6 --save_screenshot   # 6 random crops, headless
"""
import os
import sys
import csv
import glob
import argparse
import pickle
import random
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DIODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rgbd_data", "diode_raw", "val", "outdoor")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "rgbd", "diode")
HORIZONTAL_TOLERANCE_DEG = 30.0
FX, FY, CX, CY = 886.81, 927.06, 512.0, 384.0  # diode-devkit/intrinsics.txt

CSV_FIELDS = ["crop", "n_points", "found", "pred_pct", "angle_from_vertical_deg", "eps", "min_supp", "norm_th"]


def discover_crops():
    depth_files = sorted(glob.glob(os.path.join(DIODE_ROOT, "*", "*", "*_depth.npy")))
    return [f[:-len("_depth.npy")] for f in depth_files]  # strip suffix -> crop stem


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


def backproject(crop_stem):
    """RGB-D -> (points_xyz, colors_rgb01). Depth-camera convention: X right,
    Y down (image row direction), Z forward (depth) -- NOT the z-up
    convention the LiDAR scripts use. That's fine: compute_scene_features
    and schnabel_ransac don't assume a particular up-axis, and the "ground"
    plane here will simply have a normal roughly along Y instead of Z --
    select_shape's angle-from-vertical check uses axis [2] (Z), so for this
    camera-frame data we swap Y<->Z after back-projection to keep the same
    horizontal-plane convention as every other script this session."""
    depth = np.load(crop_stem + "_depth.npy").squeeze(-1)  # (768,1024)
    mask = np.load(crop_stem + "_depth_mask.npy").astype(bool)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    rgb = np.asarray(Image.open(crop_stem + ".png").convert("RGB")) / 255.0

    v, u = np.nonzero(mask)
    z = depth[v, u]
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    # swap to z-up: this session's "vertical" convention is axis 2 = up/down.
    # Camera Y (image-row direction) points down when looking roughly level,
    # so -Y behaves like "up" -- swap and negate so the ground plane comes
    # out near-horizontal in the same sense as the LiDAR data.
    points = np.stack([x, z, -y], axis=1).astype(np.float32)
    colors = rgb[v, u]
    return points, colors


def height_colormap(points):
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    normalized = (z - z_min) / (z_max - z_min + 1e-8)
    return cm.viridis(normalized)[:, :3]


def process_crop(model, normalize_obs, crop_stem):
    crop_name = os.path.relpath(crop_stem, DIODE_ROOT)
    print(f"Loading {crop_name} ...")
    points_world, colors_rgb = backproject(crop_stem)
    print(f"  {len(points_world)} valid points (of {768*1024} pixels, {len(points_world)/(768*1024)*100:.1f}% valid depth)")

    centroid = points_world.mean(axis=0)
    points = points_world - centroid
    print(f"  recentered on centroid {centroid}")

    eps, min_supp, norm_th = choose_params(model, normalize_obs, points)
    print(f"  RL chose: eps={eps} min_supp={min_supp} norm_th={norm_th}")

    try:
        shapes, _ = schnabel_ransac.detect(
            points, shapes=["plane"], relative_epsilon=False,
            epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
            probability=0.001, normal_knn=20, max_shapes=10,
        )
    except Exception as e:
        print(f"RANSAC error: {e}")
        shapes = []

    result = dict(crop=crop_name, n_points=len(points), found=False, pred_pct=0.0,
                  angle_from_vertical_deg=None, eps=eps, min_supp=min_supp, norm_th=norm_th)
    colors_height = height_colormap(points_world)
    if shapes:
        best_shape, angle, was_horizontal = select_shape(shapes, points)
        pred_mask = best_shape["inlier_mask"]
        colors_height[pred_mask] = [0.0, 1.0, 0.0]
        status = (f"FOUND: {int(pred_mask.sum())} pts ({pred_mask.mean()*100:.1f}%), "
                  f"{angle:.1f} deg from vertical, horizontal_candidate={was_horizontal}")
        result.update(found=True, pred_pct=pred_mask.mean() * 100, angle_from_vertical_deg=angle)
    else:
        status = "NOT FOUND"
    print(f"  {status}")
    result["status"] = status

    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points_world)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors_rgb)

    extent = points_world.max(axis=0) - points_world.min(axis=0)
    offset = np.array([extent[0] * 1.15, 0.0, 0.0], dtype=np.float32)
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_world + offset)
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_height)

    return result, pcd_rgb, pcd_pred, status


def show_interactive(pcd_rgb, pcd_pred, crop_name, status):
    o3d.visualization.draw_geometries(
        [pcd_rgb, pcd_pred],
        window_name=f"{crop_name} -- left: true RGB, right: height + predicted ground (green) -- {status}",
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
    ctr.rotate(0.0, 300.0)
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
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--list", action="store_true", help="list available crop names and exit")
    parser.add_argument("--crop", type=str, default=None, help="crop stem, e.g. scene_00022/scan_00193/00022_00193_outdoor_060_040")
    parser.add_argument("--sample", type=int, default=None, help="process N random crops")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_screenshot", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="with --sample: also open a window per crop")
    args = parser.parse_args()

    crops = discover_crops()
    if args.list:
        for c in crops:
            print(os.path.relpath(c, DIODE_ROOT))
        print(f"\n{len(crops)} crops total")
        return
    if not crops:
        print(f"No crops found under {DIODE_ROOT} -- extract val/outdoor/ from val.tar.gz first")
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    if args.sample is not None:
        rng = random.Random(args.seed)
        chosen = rng.sample(crops, min(args.sample, len(crops)))
        all_results = []
        for crop_stem in chosen:
            crop_name = os.path.relpath(crop_stem, DIODE_ROOT)
            print(f"\n=== {crop_name} ===")
            result, pcd_rgb, pcd_pred, status = process_crop(model, normalize_obs, crop_stem)
            all_results.append(result)
            out_path = os.path.join(OUT_DIR, crop_name.replace(os.sep, "_") + ".png")
            save_screenshot(pcd_rgb, pcd_pred, out_path)
            print(f"  saved {out_path}")
            append_csv_row(os.path.join(OUT_DIR, "results.csv"), {k: result.get(k, "") for k in CSV_FIELDS})
            if args.interactive:
                show_interactive(pcd_rgb, pcd_pred, crop_name, status)

        print("\n" + "=" * 80)
        print(f"{'crop':>55} {'pts':>8} {'found':>7} {'pred%':>7} {'angle':>7}")
        for r in all_results:
            angle_val = r['angle_from_vertical_deg']
            print(f"{r['crop']:>55} {r['n_points']:8d} {str(r['found']):>7} {r['pred_pct']:7.1f} "
                  f"{('%.1f' % angle_val) if angle_val is not None else '-':>7}")
        print(f"\nResults logged to {os.path.join(OUT_DIR, 'results.csv')}")
        return

    if args.crop is None:
        print("Specify --list, --crop NAME, or --sample N")
        sys.exit(1)
    crop_stem = os.path.join(DIODE_ROOT, args.crop)
    result, pcd_rgb, pcd_pred, status = process_crop(model, normalize_obs, crop_stem)
    if args.save_screenshot:
        out_path = os.path.join(OUT_DIR, args.crop.replace(os.sep, "_") + ".png")
        save_screenshot(pcd_rgb, pcd_pred, out_path)
        print(f"  saved {out_path}")
        append_csv_row(os.path.join(OUT_DIR, "results.csv"), {k: result.get(k, "") for k in CSV_FIELDS})
    show_interactive(pcd_rgb, pcd_pred, args.crop, status)


if __name__ == "__main__":
    main()
