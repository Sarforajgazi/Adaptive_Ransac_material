import os
import sys
import glob
import argparse
import numpy as np
import open3d as o3d
from stable_baselines3 import PPO

from ransac_env import load_ply_xyz, find_ground_plane, EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS
from features.scene_features import compute_scene_features
from rl_evaluator import load_obs_normalizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKSPACE, "models")
DATA_ROOT = os.path.join(WORKSPACE, "data")
# The 2026-07-03 retrain (ent_coef + VecNormalize fix -- see DAILY_LOG.md Day 8).
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_ransac_final.zip")
DEFAULT_VECNORMALIZE_PATH = os.path.join(MODEL_DIR, "ppo_ransac_final_vecnormalize.pkl")


def resolve_frame(dataset, frame):
    dataset_dir = os.path.join(DATA_ROOT, dataset, "Data_omni", "P0000", "lidar")
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.ply")))
    if not files:
        raise FileNotFoundError(f"No .ply files found under {dataset_dir}")

    if frame is None:
        return files[0]

    # Allow exact filename or a substring match (e.g. "000032")
    matches = [f for f in files if frame in os.path.basename(f)]
    if not matches:
        raise FileNotFoundError(f"No frame matching '{frame}' found in {dataset_dir}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Visualize the RL agent's ground segmentation on a single frame")
    parser.add_argument("--dataset", type=str, default="Office", help="Dataset folder name under data/, e.g. Supermarket")
    parser.add_argument("--frame", type=str, default=None, help="Frame filename or substring, e.g. 000032. Defaults to the first frame.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to a trained model .zip")
    parser.add_argument("--vecnormalize", type=str, default=DEFAULT_VECNORMALIZE_PATH,
                         help="Path to the matching VecNormalize .pkl (must correspond to --model)")
    args = parser.parse_args()

    print("=" * 60)
    print("Visualizing PPO Inference")
    print("=" * 60)

    filepath = resolve_frame(args.dataset, args.frame)
    filename = os.path.basename(filepath)
    print(f"Dataset: {args.dataset}  |  Frame: {filename}")

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    print(f"Loading observation normalizer: {args.vecnormalize}")
    normalize_obs = load_obs_normalizer(args.vecnormalize)

    points = load_ply_xyz(filepath)
    features = compute_scene_features(points)

    # Build the observation exactly like RansacEnv._get_obs() at step 0
    # (no prior action/result yet, so feedback features are all zero)
    feedback_feat = np.zeros(10, dtype=np.float32)
    obs = np.concatenate([features, feedback_feat])

    action, _ = model.predict(normalize_obs(obs), deterministic=True)
    eps = EPS_LEVELS[int(action[0])]
    min_supp = MIN_SUPPORT_LEVELS[int(action[1])]
    norm_th = NORM_THRESH_LEVELS[int(action[2])]

    print(f"Agent chose: Epsilon={eps}m, MinSupport={min_supp}, NormalThresh={norm_th}")

    try:
        shapes, _ = schnabel_ransac.detect(
            points,
            shapes=["plane"],
            relative_epsilon=False,
            epsilon=eps,
            normal_thresh=norm_th,
            min_support=min_supp,
            probability=0.001,
            normal_knn=20,
            max_shapes=20,
        )
    except Exception as e:
        print(f"RANSAC crashed with chosen parameters: {e}")
        return

    ground_shape, avg_z, z_align, residual = find_ground_plane(shapes, points, z_mode="z_down")

    colors = np.ones((len(points), 3)) * 0.5
    for shape in shapes:
        colors[shape["inlier_mask"]] = [0.0, 0.4, 1.0]  # Blue for other shapes

    print(f"Total shapes detected: {len(shapes)}")
    if ground_shape is None:
        print("No ground plane was found with these parameters (this frame would count as a 'bad frame').")
    else:
        ground_mask = ground_shape["inlier_mask"]
        colors[ground_mask] = [0.0, 1.0, 0.0]  # Bright green for ground
        print(f"Found ground with {np.sum(ground_mask)} points ({(np.sum(ground_mask) / len(points)) * 100:.1f}%), residual={residual:.4f}")

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points)
    pcd_original.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)) * 0.7)

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    bbox = pcd_original.get_axis_aligned_bounding_box()
    offset = bbox.get_extent()[0] * 1.2 + 2.0
    pcd_colored.translate([offset, 0, 0])

    print("\n" + "=" * 60)
    print("Opening Interactive 3D Viewer...")
    print("LEFT: Original Point Cloud | RIGHT: Ground (Green) / Other Shapes (Blue) / Unclassified (Grey)")
    print("Controls: Click & Drag to rotate, Scroll to zoom, 'P' to screenshot, close window to exit")
    print("=" * 60)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    o3d.visualization.draw_geometries(
        [pcd_original, pcd_colored, axes],
        window_name=f"{args.dataset}/{filename} - Original (Left) vs RL Segmentation (Right)",
        width=1600, height=800
    )


if __name__ == "__main__":
    main()
