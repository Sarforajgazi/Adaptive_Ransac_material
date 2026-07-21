"""
visualize_offroad_rl.py

Runs the synthetic-trained RL model on the off-road LiDAR tiles pulled from
OpenTopography (Post-Bobcat Fire survey of Sawpit Wash/Monrovia Canyon, CA --
steep, undeveloped mountain terrain, not urban roads like new_data/). See
convert_laz_to_ply.py and SESSION_PROGRESS_LOG.md for how these were
obtained and converted.

Unlike new_data/, these tiles carry REAL ground truth: the ASPRS
`classification` field (2 = Ground) comes from the survey's own LiDAR
processing, not something inferred here. So this script does a real
outcome-colored comparison (TP/FP/FN/TN), same coloring scheme as
visualize_real_lidar_gt.py, instead of just "predicted ground in green".

Same two fixes as visualize_new_data_rl.py, both needed again here:
- Recentering on centroid before compute_scene_features()/RANSAC (these are
  raw UTM coordinates, hundreds of thousands of meters from the origin).
- Single-shot bypass (model.predict() once on a fresh-reset-style
  observation, decode eps/min_support/normal_thresh, call
  schnabel_ransac.detect() directly) since there's no step()-compatible
  ground-truth normal for real data.

One real difference worth watching for: these tiles are airborne LiDAR at
~1-2 pts/m^2 (1.7M pts over a 1km^2 tile) -- close to the sparse end of the
density range that caused outright failures on merged TartanGround scenes
in visualize_real_pointcloud.py (SESSION_PROGRESS_LOG.md sec.27). Not
assumed to fail here, just flagged as the same known risk factor showing up
again in a new dataset.

Usage:
    python visualize_offroad_rl.py --file 0
    python visualize_offroad_rl.py --batch 0 2 --save_screenshot
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
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "off_road_data", "ply")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "off_road")
HORIZONTAL_TOLERANCE_DEG = 30.0

# Every file is grouped into a survey folder by filename prefix, so results
# land in report_screenshots/off_road/<survey>/ instead of one flat pile --
# see SESSION_PROGRESS_LOG.md sec.31 for why (user asked for results to be
# easy to find/understand after the fact).
SURVEY_INFO = {
    "CA17_": ("san_andreas_fault", "Lidar Survey over San Andreas Fault, CA (2017) -- fault-zone terrain"),
    "CA19_": ("ridgecrest", "Mobile Laser Scan over Ridgecrest, CA (2019) -- desert town streets"),
    "CO25_": ("rock_glaciers", "Mapping Rock Glaciers, San Juan Mountains, CO (2025) -- rocky alpine terrain"),
    "LA23_": ("deltaic_wetlands", "Lidar Survey of Deltaic Wetlands, LA (2023) -- muddy marsh/wetland terrain"),
    "CA23river_": ("sacramento_river", "Topo-Bathymetric Sacramento River, CA (2023) -- river/water body, has RGB"),
    "CA25_": ("coastal_dune_erosion", "Mapping Coastal Dune Erosion, CA (2025) -- beach/dune, water-adjacent"),
    "CA22_": ("dune_dune_interactions", "Quantitative Characterization of Dune-Dune Interactions, CA (2022) -- inland sand dunes"),
}
DEFAULT_SURVEY = ("sawpit_wash", "Post-Bobcat Fire lidar of Sawpit Wash, CA (2020) -- undeveloped canyon/wash")


def survey_for(basename):
    for prefix, info in SURVEY_INFO.items():
        if basename.startswith(prefix):
            return info
    return DEFAULT_SURVEY  # Sawpit Wash tiles were downloaded before this prefix convention existed


CSV_FIELDS = ["survey", "file", "color_mode", "n_points", "gt_pct", "found",
              "pred_pct", "angle_from_vertical_deg", "iou_vs_real_ground", "eps", "min_supp", "norm_th"]


def append_csv_row(csv_path, row):
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def result_to_csv_row(result):
    return dict(
        survey=result["survey"], file=result["file"], color_mode=result["color_mode"],
        n_points=result["n_points"], gt_pct=round(result["gt_pct"], 2), found=result["found"],
        pred_pct=round(result["pred_pct"], 2),
        angle_from_vertical_deg=round(result["angle"], 2) if result["angle"] is not None else "",
        iou_vs_real_ground=round(result["iou"], 4) if result["iou"] is not None else "",
        eps=result["eps"], min_supp=result["min_supp"], norm_th=result["norm_th"],
    )


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


def height_colormap(points):
    """Viridis, normalized per-tile by z min/max -- same convention as
    browse_real_lidar_frames.py. Makes the actual canyon/slope relief
    visible, unlike the flat gray/orange ground-truth coloring which only
    shows drainage patterns, not terrain shape."""
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    normalized = (z - z_min) / (z_max - z_min + 1e-8)
    return cm.viridis(normalized)[:, :3]


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


def load_tile(path, voxel):
    from plyfile import PlyData
    pd = PlyData.read(path)
    v = pd["vertex"].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    has_gt = "classification" in v.dtype.names
    gt_ground = (np.asarray(v["classification"]) == 2) if has_gt else np.zeros(len(points), dtype=bool)

    if voxel and voxel > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # carry gt through voxel downsampling by nearest-neighbor lookup post-hoc
        down = pcd.voxel_down_sample(voxel)
        down_pts = np.asarray(down.points).astype(np.float32)
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(down_pts, k=1)
        return down_pts, gt_ground[idx], has_gt
    return points, gt_ground, has_gt


def process_file(model, normalize_obs, path, voxel, color_mode="gt"):
    print(f"Loading {path} ...")
    points_world, gt_ground, has_gt = load_tile(path, voxel)
    if not has_gt and color_mode == "gt":
        print("  no ASPRS classification field in this file (e.g. Ridgecrest mobile-scan tiles never got classified) -- forcing color_mode=height")
        color_mode = "height"
    if has_gt:
        print(f"  {len(points_world)} points ({voxel}m voxel), {gt_ground.mean()*100:.1f}% real ground (ASPRS class 2)")
    else:
        print(f"  {len(points_world)} points ({voxel}m voxel), no ground-truth field available")

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

    colors = np.full((len(points), 3), [0.75, 0.75, 0.75])  # TN default: gray
    colors[gt_ground] = [0.9, 0.6, 0.0]  # FN default (real ground, not yet predicted): orange

    result = dict(file=os.path.basename(path), n_points=len(points),
                  gt_pct=gt_ground.mean() * 100, found=False, pred_pct=0.0,
                  angle=None, iou=None, eps=eps, min_supp=min_supp, norm_th=norm_th)
    pred_mask = None
    if shapes:
        best_shape, angle, was_horizontal = select_shape(shapes, points)
        pred_mask = best_shape["inlier_mask"]
        colors[pred_mask & gt_ground] = [1.0, 1.0, 0.0]   # TP: yellow
        colors[pred_mask & ~gt_ground] = [1.0, 0.0, 0.0]  # FP: red
        if has_gt:
            tp = int((pred_mask & gt_ground).sum())
            fp = int((pred_mask & ~gt_ground).sum())
            fn = int((~pred_mask & gt_ground).sum())
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            status = (f"FOUND: {int(pred_mask.sum())} pts ({pred_mask.mean()*100:.1f}%), "
                      f"{angle:.1f} deg from vertical, IoU vs real ground={iou:.3f}")
            result.update(found=True, pred_pct=pred_mask.mean() * 100, angle=angle, iou=iou)
        else:
            status = (f"FOUND: {int(pred_mask.sum())} pts ({pred_mask.mean()*100:.1f}%), "
                      f"{angle:.1f} deg from vertical (no ground-truth available to score against)")
            result.update(found=True, pred_pct=pred_mask.mean() * 100, angle=angle)
    else:
        status = "NOT FOUND"
    print(f"  {status}")
    result["status"] = status

    if color_mode == "height":
        colors = height_colormap(points_world)  # shows actual terrain relief instead of gt/pred outcome
        if pred_mask is not None:
            colors[pred_mask] = [0.0, 1.0, 0.0]  # green overlay: where the model's predicted plane is

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    return result, pcd, status


def show_interactive(pcd, basename, status, color_mode="gt"):
    legend = "yellow=TP orange=FN(missed real ground) red=FP gray=TN" if color_mode == "gt" else "color=height (viridis, low->high)"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{basename} -- {legend} -- {status}", width=1400, height=900)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.light_on = True  # shades points by their estimated normal -- makes the relief actually readable
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 400.0)  # tilt off top-down so the elevation relief (100+ m over 1km) is visible, not flattened
    vis.run()
    vis.destroy_window()


def save_screenshot(pcd, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1400, height=900)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.light_on = True
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 400.0)
    ctr.set_zoom(0.65)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--file", type=int, default=None)
    parser.add_argument("--batch", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--voxel", type=float, default=0.0, help="voxel downsample (0=off; these tiles are already sparse, ~1-2 pts/m^2)")
    parser.add_argument("--save_screenshot", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="with --batch: also open an interactive window per tile")
    parser.add_argument("--color_mode", choices=["gt", "height"], default="gt",
                         help="'gt' = TP/FP/FN/TN vs real ASPRS ground truth, 'height' = viridis by elevation (shows terrain relief)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.ply")))
    if not files:
        print(f"No .ply files in {DATA_DIR} -- run convert_laz_to_ply.py first")
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    if args.batch is not None:
        start, end = args.batch
        indices = list(range(start, min(end, len(files) - 1) + 1))
        all_results = []
        for i in indices:
            path = files[i]
            basename = os.path.basename(path)
            survey_folder, survey_desc = survey_for(basename)
            print(f"\n=== [{i}] {basename}  ({survey_desc}) ===")
            result, pcd, status = process_file(model, normalize_obs, path, args.voxel, args.color_mode)
            result["survey"] = survey_folder
            result["color_mode"] = args.color_mode
            all_results.append(result)

            survey_dir = os.path.join(OUT_DIR, survey_folder)
            out_path = os.path.join(survey_dir, f"{os.path.splitext(basename)[0]}_{args.color_mode}.png")
            save_screenshot(pcd, out_path)
            print(f"  saved {out_path}")

            csv_row = result_to_csv_row(result)
            append_csv_row(os.path.join(survey_dir, "results.csv"), csv_row)
            append_csv_row(os.path.join(OUT_DIR, "ALL_SURVEYS_SUMMARY.csv"), csv_row)

            if args.interactive:
                show_interactive(pcd, basename, status, args.color_mode)

        print("\n" + "=" * 90)
        print(f"{'idx':>3} {'survey':>18} {'file':>24} {'pts':>9} {'gt%':>6} {'found':>7} {'pred%':>7} {'angle':>7} {'IoU':>6}")
        for i, r in zip(indices, all_results):
            print(f"{i:3d} {r['survey']:>18} {r['file']:>24} {r['n_points']:9d} {r['gt_pct']:6.1f} "
                  f"{str(r['found']):>7} {r['pred_pct']:7.1f} "
                  f"{('%.1f' % r['angle']) if r['angle'] is not None else '-':>7} "
                  f"{('%.3f' % r['iou']) if r['iou'] is not None else '-':>6}")
        print(f"\nPer-survey results.csv written under {OUT_DIR}\\<survey>\\ , "
              f"combined log at {OUT_DIR}\\ALL_SURVEYS_SUMMARY.csv")
        return

    if args.file is None:
        print("Specify --file N or --batch START END")
        sys.exit(1)
    path = files[args.file]
    basename = os.path.basename(path)
    survey_folder, survey_desc = survey_for(basename)
    print(f"Survey: {survey_desc}")
    result, pcd, status = process_file(model, normalize_obs, path, args.voxel, args.color_mode)
    if args.save_screenshot:
        survey_dir = os.path.join(OUT_DIR, survey_folder)
        out_path = os.path.join(survey_dir, f"{os.path.splitext(basename)[0]}_{args.color_mode}.png")
        save_screenshot(pcd, out_path)
        print(f"  saved {out_path}")
        result["survey"] = survey_folder
        result["color_mode"] = args.color_mode
        csv_row = result_to_csv_row(result)
        append_csv_row(os.path.join(survey_dir, "results.csv"), csv_row)
        append_csv_row(os.path.join(OUT_DIR, "ALL_SURVEYS_SUMMARY.csv"), csv_row)
    show_interactive(pcd, basename, status, args.color_mode)


if __name__ == "__main__":
    main()
