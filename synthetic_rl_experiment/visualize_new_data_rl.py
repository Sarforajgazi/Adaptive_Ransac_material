"""
visualize_new_data_rl.py

Runs the synthetic-trained RL model on the new_data/ tiles (real, dense,
true-RGB, semantically-labeled urban point clouds -- see
SESSION_PROGRESS_LOG.md for the field inspection) and shows the predicted
ground plane next to the true-color original.

Two things make this data different from everything tested before, both
handled here:

1. Scale: these tiles are ~150-200m across with millions of points, so we
   voxel-downsample before running RANSAC (full-res would be far too slow
   and far denser than anything the model saw in training).

2. Coordinate frame: this is the big one. compute_scene_features() computes
   "range" as norm(points) i.e. distance from the ORIGIN, assuming a
   sensor-centric frame like every dataset used so far (synthetic scenes and
   the per-frame LiDAR sweeps are both naturally centered near (0,0,0)).
   These tiles are in absolute survey coordinates (e.g. x=827-1008,
   y=3670-3831, z=112-129) -- centering nowhere near the origin. Feeding
   that straight into compute_scene_features would make scan_range_mean/std
   enormous (~3700 instead of a few meters), wildly out of the
   VecNormalize running-stats distribution the model was trained on, and
   likely to produce a degenerate action. So we recenter on the point
   cloud's own centroid before computing features AND before running RANSAC
   (recentering doesn't change any relative geometry, so plane detection
   is unaffected either way -- this is purely about making the observation
   match what the model actually learned from).

Same single-shot bypass + largest-support-among-horizontal shape selection
as visualize_real_pointcloud.py (see that file's docstring for the "why").

The tiles are split into new_data/static/ (background/scene points) and
new_data/dynamic/ (moving objects -- cars, pedestrians -- captured in that
tile's time window; often empty, 0 points, when nothing was moving). When a
dynamic file with the same name exists and is non-empty, its points are
concatenated onto the static cloud before feature computation/RANSAC (so the
model sees the same scene a downstream consumer would), and highlighted in
magenta in the true-color view so they're visually distinguishable from the
static background.

Usage:
    python visualize_new_data_rl.py --file 0                  # single tile, interactive window
    python visualize_new_data_rl.py --file 0 --voxel 0.15
    python visualize_new_data_rl.py --batch 5 16 --save_screenshot   # tiles 5..16 inclusive, headless, saves PNGs
"""
import os
import sys
import glob
import argparse
import pickle
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_data", "static")
DYNAMIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_data", "dynamic")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "new_data")
HORIZONTAL_TOLERANCE_DEG = 30.0
DYNAMIC_COLOR = [1.0, 0.0, 1.0]  # magenta -- marks moving-object points in the true-color view


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
    SESSION_PROGRESS_LOG.md sec.27."""
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


def process_file(model, normalize_obs, path, voxel, no_dynamic):
    basename = os.path.basename(path)
    print(f"Loading (static) {path} ...")
    pcd_full = o3d.io.read_point_cloud(path)
    print(f"  {len(pcd_full.points)} static points, has_colors={pcd_full.has_colors()}")
    pcd_full = pcd_full.voxel_down_sample(voxel)
    print(f"  downsampled to {len(pcd_full.points)} points (voxel={voxel})")

    n_dynamic = 0
    dynamic_path = os.path.join(DYNAMIC_DIR, basename)
    if no_dynamic:
        print("  --no_dynamic set, skipping dynamic file even if present")
    elif os.path.exists(dynamic_path):
        pcd_dyn = o3d.io.read_point_cloud(dynamic_path)
        n_dynamic = len(pcd_dyn.points)
        if n_dynamic > 0:
            print(f"  found matching dynamic file with {n_dynamic} points -- combining")
            pcd_full = pcd_full + pcd_dyn  # dynamic kept at full res -- usually small (thousands, not millions)
        else:
            print("  matching dynamic file exists but is empty (0 points) -- nothing to add")
    else:
        print(f"  no matching dynamic file for {basename}")

    points_world = np.asarray(pcd_full.points).astype(np.float32)
    colors = np.asarray(pcd_full.colors)
    if n_dynamic > 0:
        colors = colors.copy()
        colors[-n_dynamic:] = DYNAMIC_COLOR  # highlight the appended dynamic points
    centroid = points_world.mean(axis=0)
    points = points_world - centroid  # see module docstring: recenter to match the sensor-centric training frame
    print(f"  recentered on centroid {centroid} (this only affects features/RANSAC input, not the display)")

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

    colors_pred = np.full((len(points), 3), [0.75, 0.75, 0.75])
    result = dict(file=basename, n_points=len(points), n_dynamic=n_dynamic,
                  eps=eps, min_supp=min_supp, norm_th=norm_th,
                  found=False, pct=0.0, angle=None, horizontal_candidate=False)
    if shapes:
        best_shape, angle, was_horizontal = select_shape(shapes, points)
        pred_mask = best_shape["inlier_mask"]
        colors_pred[pred_mask] = [0.0, 0.85, 0.0]
        pct = pred_mask.mean() * 100
        status = (f"FOUND: {int(pred_mask.sum())} pts ({pct:.1f}%), "
                  f"{angle:.1f} deg from vertical, horizontal_candidate={was_horizontal}")
        result.update(found=True, pct=pct, angle=angle, horizontal_candidate=was_horizontal)
    else:
        status = "NOT FOUND"
    print(f"  {status}")
    result["status"] = status

    # Side-by-side: predicted copy offset along the cloud's widest horizontal axis.
    extent = points_world.max(axis=0) - points_world.min(axis=0)
    offset = np.zeros(3, dtype=np.float32)
    offset[0] = extent[0] * 1.1 if extent[0] >= extent[1] else 0.0
    offset[1] = extent[1] * 1.1 if extent[1] > extent[0] else 0.0

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points_world)
    pcd_original.colors = o3d.utility.Vector3dVector(colors)

    pcd_predicted = o3d.geometry.PointCloud()
    pcd_predicted.points = o3d.utility.Vector3dVector(points_world + offset)
    pcd_predicted.colors = o3d.utility.Vector3dVector(colors_pred)

    return result, pcd_original, pcd_predicted, status


def show_interactive(pcd_original, pcd_predicted, basename, status):
    o3d.visualization.draw_geometries(
        [pcd_original, pcd_predicted],
        window_name=f"{basename} -- left: true RGB, right: predicted ground (green) -- {status}",
        width=1600, height=900,
    )


def save_screenshot(pcd_original, pcd_predicted, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1600, height=900)
    vis.add_geometry(pcd_original)
    vis.add_geometry(pcd_predicted)
    render_opt = vis.get_render_option()
    render_opt.point_size = 1.5
    ctr = vis.get_view_control()
    ctr.set_zoom(0.55)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--file", type=int, default=None, help="index into sorted new_data/static/*.ply, single-tile interactive mode")
    parser.add_argument("--batch", type=int, nargs=2, default=None, metavar=("START", "END"),
                         help="process file indices START..END inclusive, headless")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                         help="process this explicit (possibly non-contiguous) list of file indices")
    parser.add_argument("--voxel", type=float, default=0.15, help="voxel size for downsampling before RANSAC")
    parser.add_argument("--no_dynamic", action="store_true", help="skip combining with the matching dynamic/ file, static only")
    parser.add_argument("--save_screenshot", action="store_true", help="save a PNG instead of (or in addition to, for --file) opening a window")
    parser.add_argument("--interactive", action="store_true",
                         help="with --batch/--indices: also open an interactive window per tile (blocks until closed) in addition to saving a screenshot")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.ply")))

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    if args.batch is not None or args.indices is not None:
        if args.indices is not None:
            indices = args.indices
        else:
            start, end = args.batch
            indices = list(range(start, min(end, len(files) - 1) + 1))
        all_results = []
        for i in indices:
            path = files[i]
            print(f"\n=== [{i}] {os.path.basename(path)} ===")
            result, pcd_original, pcd_predicted, status = process_file(model, normalize_obs, path, args.voxel, args.no_dynamic)
            all_results.append(result)
            out_path = os.path.join(OUT_DIR, f"tile{i:02d}_{os.path.splitext(os.path.basename(path))[0]}.png")
            save_screenshot(pcd_original, pcd_predicted, out_path)
            print(f"  saved {out_path}")
            if args.interactive:
                show_interactive(pcd_original, pcd_predicted, os.path.basename(path), status)

        print("\n" + "=" * 90)
        print(f"{'idx':>3} {'file':>28} {'pts':>9} {'dyn':>6} {'eps':>6} {'found':>7} {'pct':>6} {'angle':>7}")
        for i, r in zip(indices, all_results):
            print(f"{i:3d} {r['file']:>28} {r['n_points']:9d} {r['n_dynamic']:6d} {r['eps']:6.2f} "
                  f"{str(r['found']):>7} {r['pct']:6.1f} {('%.1f' % r['angle']) if r['angle'] is not None else '-':>7}")
        return

    if args.file is None:
        print("Specify --file N (single tile) or --batch START END (headless range)")
        sys.exit(1)
    if args.file >= len(files):
        print(f"--file {args.file} out of range (only {len(files)} files)")
        sys.exit(1)
    path = files[args.file]
    result, pcd_original, pcd_predicted, status = process_file(model, normalize_obs, path, args.voxel, args.no_dynamic)
    if args.save_screenshot:
        out_path = os.path.join(OUT_DIR, f"tile{args.file:02d}_{os.path.splitext(os.path.basename(path))[0]}.png")
        save_screenshot(pcd_original, pcd_predicted, out_path)
        print(f"  saved {out_path}")
    else:
        show_interactive(pcd_original, pcd_predicted, os.path.basename(path), status)


if __name__ == "__main__":
    main()
