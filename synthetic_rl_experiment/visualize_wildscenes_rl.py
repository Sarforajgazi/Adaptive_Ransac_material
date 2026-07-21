"""
visualize_wildscenes_rl.py

Runs the synthetic-trained RL model on WildScenes3d point cloud submaps
(real handheld-LiDAR SLAM submaps from two Australian forests -- Karawatha
"K-01"/"K-03", Venman "V-01"/"V-02"/"V-03") -- see download_wildscenes.py
for how these were pulled and converted, and SESSION_PROGRESS_LOG.md for
the .bin/.label format confirmation.

Real per-point semantic ground truth exists here (WildScenes devkit
METAINFO, 15 terrain/object classes), including genuine "mud" and "water"
classes -- the actual terrain types this data source was chosen for.
"Ground" for this script's scoring means natural traversable terrain:
dirt, grass, gravel, mud, other-terrain, rock. Water is tracked as its
own separate category (not counted as ground truth "ground") since a
water surface being picked up as a flat plane would be geometrically
correct but semantically a different thing worth seeing separately.
Everything else (bush, fence, log, other-object, sky, structure,
tree-foliage, tree-trunk) is background/non-ground.

Same single-shot bypass + recentering + largest-support-among-horizontal
selection as every other real-data script this session.

Usage:
    python visualize_wildscenes_rl.py --batch 0 19 --color_mode gt
    python visualize_wildscenes_rl.py --file 0 --interactive
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildscenes_data", "ply")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "wildscenes")
HORIZONTAL_TOLERANCE_DEG = 30.0

# cidx -> name, from wildscenes/tools/utils3d.py METAINFO (fetched from the
# official devkit, not guessed)
CLASS_NAMES = {0: "bush", 1: "dirt", 2: "fence", 3: "grass", 4: "gravel", 5: "log",
                6: "mud", 7: "other-object", 8: "other-terrain", 9: "rock", 10: "sky",
                11: "structure", 12: "tree-foliage", 13: "tree-trunk", 14: "water", 255: "unlabelled"}
GROUND_CLASSES = {1, 3, 4, 6, 8, 9}  # dirt, grass, gravel, mud, other-terrain, rock
WATER_CLASS = 14

CSV_FIELDS = ["seq", "frame", "n_points", "gt_ground_pct", "gt_water_pct", "found",
              "pred_pct", "angle_from_vertical_deg", "iou_vs_ground", "eps", "min_supp", "norm_th"]


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


def load_tile(path):
    from plyfile import PlyData
    pd = PlyData.read(path)
    v = pd["vertex"].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    cidx = np.asarray(v["classification"])
    gt_ground = np.isin(cidx, list(GROUND_CLASSES))
    gt_water = (cidx == WATER_CLASS)
    return points, gt_ground, gt_water


def process_file(model, normalize_obs, path, color_mode):
    seq = os.path.basename(os.path.dirname(path))
    frame = os.path.basename(path)
    points_world, gt_ground, gt_water = load_tile(path)

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

    # gt-mode coloring: TN=gray, FN=orange(ground), water=blue, TP=yellow, FP=red
    colors = np.full((len(points), 3), [0.75, 0.75, 0.75])
    colors[gt_ground] = [0.9, 0.6, 0.0]
    colors[gt_water] = [0.1, 0.4, 0.9]

    result = dict(seq=seq, frame=frame, n_points=len(points),
                  gt_ground_pct=gt_ground.mean() * 100, gt_water_pct=gt_water.mean() * 100,
                  found=False, pred_pct=0.0, angle=None, iou=None,
                  eps=eps, min_supp=min_supp, norm_th=norm_th)
    pred_mask = None
    if shapes:
        best_shape, angle, was_horizontal = select_shape(shapes, points)
        pred_mask = best_shape["inlier_mask"]
        colors[pred_mask & gt_ground] = [1.0, 1.0, 0.0]
        colors[pred_mask & ~gt_ground & ~gt_water] = [1.0, 0.0, 0.0]
        tp = int((pred_mask & gt_ground).sum())
        fp = int((pred_mask & ~gt_ground).sum())
        fn = int((~pred_mask & gt_ground).sum())
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        status = (f"FOUND: {int(pred_mask.sum())} pts ({pred_mask.mean()*100:.1f}%), "
                  f"{angle:.1f} deg from vertical, IoU vs ground-classes={iou:.3f}")
        result.update(found=True, pred_pct=pred_mask.mean() * 100, angle=angle, iou=iou)
    else:
        status = "NOT FOUND"
    print(f"  [{seq}/{frame}] {len(points)} pts, ground={gt_ground.mean()*100:.1f}% water={gt_water.mean()*100:.1f}% -> {status}")

    if color_mode == "height":
        colors = height_colormap(points_world)
        if pred_mask is not None:
            colors[pred_mask] = [0.0, 1.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    return result, pcd, status


def show_interactive(pcd, seq, frame, status, color_mode):
    legend = "yellow=TP orange=FN(ground) blue=water red=FP gray=other" if color_mode == "gt" else "color=height, green=predicted"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{seq}/{frame} -- {legend} -- {status}", width=1400, height=900)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.5
    render_opt.light_on = True
    ctr = vis.get_view_control()
    ctr.rotate(200.0, 150.0)
    vis.run()
    vis.destroy_window()


def save_screenshot(pcd, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1400, height=900)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.5
    render_opt.light_on = True
    ctr = vis.get_view_control()
    ctr.rotate(200.0, 150.0)
    ctr.set_zoom(0.7)
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
    parser.add_argument("--file", type=int, default=None)
    parser.add_argument("--batch", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--color_mode", choices=["gt", "height"], default="gt")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--skip_save", action="store_true", help="don't re-write screenshots/CSV rows -- use with --interactive to just re-view")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.ply")))
    if not files:
        print(f"No .ply files under {DATA_DIR} -- run download_wildscenes.py first")
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
            print(f"\n=== [{i}] {os.path.relpath(path, DATA_DIR)} ===")
            result, pcd, status = process_file(model, normalize_obs, path, args.color_mode)
            all_results.append(result)
            seq = result["seq"]
            if not args.skip_save:
                out_path = os.path.join(OUT_DIR, seq, os.path.splitext(result["frame"])[0] + f"_{args.color_mode}.png")
                save_screenshot(pcd, out_path)
                append_csv_row(os.path.join(OUT_DIR, seq, "results.csv"), result)
                append_csv_row(os.path.join(OUT_DIR, "ALL_SEQUENCES_SUMMARY.csv"), result)
            if args.interactive:
                show_interactive(pcd, seq, result["frame"], status, args.color_mode)

        print("\n" + "=" * 100)
        print(f"{'idx':>3} {'seq':>6} {'frame':>28} {'pts':>7} {'gnd%':>6} {'wtr%':>6} {'found':>7} {'pred%':>7} {'angle':>7} {'IoU':>6}")
        for i, r in zip(indices, all_results):
            print(f"{i:3d} {r['seq']:>6} {r['frame']:>28} {r['n_points']:7d} {r['gt_ground_pct']:6.1f} {r['gt_water_pct']:6.1f} "
                  f"{str(r['found']):>7} {r['pred_pct']:7.1f} "
                  f"{('%.1f' % r['angle']) if r['angle'] is not None else '-':>7} "
                  f"{('%.3f' % r['iou']) if r['iou'] is not None else '-':>6}")
        return

    if args.file is None:
        print("Specify --file N or --batch START END")
        sys.exit(1)
    path = files[args.file]
    result, pcd, status = process_file(model, normalize_obs, path, args.color_mode)
    show_interactive(pcd, result["seq"], result["frame"], status, args.color_mode)


if __name__ == "__main__":
    main()
