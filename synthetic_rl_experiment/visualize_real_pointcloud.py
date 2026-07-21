"""
visualize_real_pointcloud.py

Single-shot real-point-cloud inference for the synthetic-trained model
(synthetic_ppo_v2 by default). Answers: "can the model trained purely on
synthetic scenes pick sensible RANSAC parameters for a REAL point cloud,
and does the resulting fit look like the actual ground?"

Necessarily single-shot, not the full multi-step refinement loop --
SyntheticRansacEnv.step()'s shape-selection compares RANSAC candidates
against a known ground-truth normal, which doesn't exist for real,
unlabeled data (see SESSION_PROGRESS_LOG.md sec.6/sec.25). This script
bypasses that: builds the observation the model would see on a fresh
reset() (scene features + all-zero feedback, matching step_count=0), gets
one action from the model, decodes eps/min_support/normal_thresh (min_support
is fraction-based, materialized against this cloud's REAL point count -- see
sec.7.1), and runs schnabel_ransac.detect() directly. Since there's no
ground truth to pick "the correct" shape by angle, this selects the
LARGEST-support shape -- the deployment-appropriate heuristic flagged as
still-needed in sec.16 item 3.

No ground truth exists for TartanGround scenes (unlike RELLIS-3D), so this
can only show "what the model picked as ground," not TP/FP/FN correctness.

Usage:
    python visualize_real_pointcloud.py --ply path/to/scene.ply --tag v2
    python visualize_real_pointcloud.py --ply path/to/scene.ply --tag v2 --save_screenshot out.png
"""
import os
import sys
import argparse
import pickle
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "schnabel_cython"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schnabel_ransac
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS


def load_obs_normalizer(vecnormalize_path):
    if vecnormalize_path is None or not os.path.exists(vecnormalize_path):
        print("No VecNormalize stats found -- using raw observations.")
        return lambda obs: obs
    with open(vecnormalize_path, "rb") as f:
        vec_normalize = pickle.load(f)
    obs_rms = vec_normalize.obs_rms
    clip_obs = vec_normalize.clip_obs
    epsilon = vec_normalize.epsilon
    return lambda obs: np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                                -clip_obs, clip_obs).astype(np.float32)


def choose_params_single_shot(model_path, vecnormalize_path, points):
    """
    Builds the observation for a fresh (step_count=0) episode on this real
    cloud and asks the model for one action -- see module docstring for why
    this can't be the normal multi-step env.step() loop.
    """
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    scene_feat = compute_scene_features(points)
    feedback_feat = np.zeros(10, dtype=np.float32)  # matches reset()'s defaults exactly
    obs = np.concatenate([scene_feat, feedback_feat]).astype(np.float32)

    action, _ = model.predict(normalize_obs(obs), deterministic=True)
    eps = EPS_LEVELS[int(action[0])]
    min_supp_frac = MIN_SUPPORT_LEVELS[int(action[1])]
    min_supp = max(1, int(round(min_supp_frac * len(points))))
    norm_th = NORM_THRESH_LEVELS[int(action[2])]
    return eps, min_supp, min_supp_frac, norm_th


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to a real .ply point cloud.")
    parser.add_argument("--tag", type=str, default="v2",
                         help="Model tag (default 'v2' -- models/synthetic_ppo_v2.zip).")
    parser.add_argument("--save_screenshot", type=str, default=None,
                         help="Render off-screen and save a PNG instead of opening an interactive window.")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(model_path):
        raise SystemExit(f"No model at {model_path}")

    print(f"Loading real point cloud: {args.ply}")
    pcd_raw = o3d.io.read_point_cloud(args.ply)
    points = np.asarray(pcd_raw.points).astype(np.float32)
    print(f"Loaded {len(points)} points, bbox extent = {points.max(axis=0) - points.min(axis=0)}")

    print(f"Using model: {model_path}")
    eps, min_supp, min_supp_frac, norm_th = choose_params_single_shot(model_path, vecnormalize_path, points)
    print(f"Model chose (single-shot): eps={eps}m, min_support={min_supp} "
          f"({min_supp_frac*100:.3f}% of {len(points)} points), normal_thresh={norm_th}")

    print("Running RANSAC with the chosen parameters...")
    shapes, _ = schnabel_ransac.detect(
        points,
        shapes=["plane"],
        relative_epsilon=False,
        epsilon=eps,
        normal_thresh=norm_th,
        min_support=min_supp,
        probability=0.001,
        normal_knn=20,
        max_shapes=10,
    )

    if not shapes:
        print("RANSAC found no planes with these parameters.")
        return

    # No ground truth on real data, so shape selection can't use step()'s
    # angle-to-ground-truth comparison. Largest-support alone isn't enough
    # either -- confirmed on real data (SESSION_PROGRESS_LOG.md sec.27:
    # Restaurant/AbandonedFactory) that it sometimes picks a large flat
    # WALL over a smaller-but-real floor, if the wall happens to have more
    # visible coplanar points in that sweep. Fix: rank by support only
    # AMONG candidates that are plausibly horizontal (normal within
    # HORIZONTAL_TOLERANCE_DEG of vertical); fall back to largest-overall,
    # with an explicit warning, only if literally nothing qualifies.
    HORIZONTAL_TOLERANCE_DEG = 30.0

    candidates = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        plane_pts = points[mask]
        cov = np.cov(plane_pts.T)
        evals, evecs = np.linalg.eig(cov)
        normal = evecs[:, np.argmin(evals)]
        angle_from_vertical = np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
        support = int(np.sum(mask))
        candidates.append((shape, normal, angle_from_vertical, support))

    horizontal_candidates = [c for c in candidates if c[2] <= HORIZONTAL_TOLERANCE_DEG]
    if horizontal_candidates:
        best_shape, n_fitted, best_angle, n_selected = max(horizontal_candidates, key=lambda c: c[3])
        print(f"RANSAC found {len(shapes)} candidate plane(s), {len(horizontal_candidates)} "
              f"roughly horizontal (<= {HORIZONTAL_TOLERANCE_DEG:.0f} deg from vertical).")
        print(f"Selected the largest of those: {n_selected} points "
              f"({n_selected/len(points)*100:.1f}% of cloud), {best_angle:.1f} deg from vertical.")
    else:
        best_shape, n_fitted, best_angle, n_selected = max(candidates, key=lambda c: c[3])
        print(f"RANSAC found {len(shapes)} candidate plane(s), NONE within "
              f"{HORIZONTAL_TOLERANCE_DEG:.0f} deg of horizontal.")
        print(f"WARNING: falling back to the largest candidate overall -- {best_angle:.1f} deg "
              f"from vertical, likely NOT the ground (probably a wall).")

    pred_mask = best_shape["inlier_mask"]
    mean_pt = np.mean(points[pred_mask], axis=0)
    print(f"Fitted plane normal: {n_fitted}, mean point: {mean_pt}")

    # --- Visualization: no ground truth on real data, so this shows what
    # the model picked as ground vs. everything else, not TP/FP/FN.
    colors = np.full((len(points), 3), [0.7, 0.7, 0.7])  # gray = not selected
    colors[pred_mask] = [0.0, 0.85, 0.0]  # green = selected as ground plane
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\nVisualizing...")
    print(f"- Green points: selected as the ground plane by the model's chosen parameters")
    print(f"- Gray points: everything else (real scene structure, not classified)")

    if args.save_screenshot:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_screenshot)) or ".", exist_ok=True)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=960)
        vis.add_geometry(pcd)
        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        ctr = vis.get_view_control()
        ctr.set_zoom(0.6)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save_screenshot, do_render=True)
        vis.destroy_window()
        print(f"Screenshot saved to {args.save_screenshot}")
    else:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
