import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


GROUND_NAMES = {
    "ground", "floor", "grass", "road", "sand", "dirt", "rock",
    "terrain", "pavement", "asphalt", "concretefloor", "floordirt"
}

TRANSFORM_PRESETS = {
    "raw": (np.array([0, 1, 2]), np.array([1, 1, 1])),
    "swapxy_negz": (np.array([1, 0, 2]), np.array([1, 1, -1])),
    "swapxy": (np.array([1, 0, 2]), np.array([1, 1, 1])),
    "negz": (np.array([0, 1, 2]), np.array([1, 1, -1])),
    "xzy": (np.array([0, 2, 1]), np.array([1, 1, 1])),
    "yzx": (np.array([1, 2, 0]), np.array([1, 1, 1])),
    "zxy": (np.array([2, 0, 1]), np.array([1, 1, 1])),
    "raw_negx": (np.array([0, 1, 2]), np.array([-1, 1, 1])),
    "raw_negy": (np.array([0, 1, 2]), np.array([1, -1, 1])),
}


def get_ground_colors(env_name, data_dir="data"):
    label_map_path = os.path.join(data_dir, env_name, "seg_label_map.json")
    if not os.path.exists(label_map_path):
        if env_name == "House":
            return np.array([[132 / 255.0, 237 / 255.0, 195 / 255.0]])
        return None

    with open(label_map_path, "r") as f:
        name_map = json.load(f)["name_map"]

    ground_ids = []
    for name, id_ in name_map.items():
        if any(token in name.lower() for token in GROUND_NAMES):
            ground_ids.append(id_)

    with open(".venv/Lib/site-packages/tartanair/seg_rgbs.txt", "r") as f:
        lines = f.readlines()

    ground_colors_float = []
    for gid in ground_ids:
        b, g, r = [int(x) for x in lines[gid].strip().split(",")]
        ground_colors_float.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(ground_colors_float)


def apply_transform(points, transform_name):
    axes, signs = TRANSFORM_PRESETS[transform_name]
    return points[:, axes] * signs


def score_transform(points_local, pose_row, kdtree, global_colors, ground_colors,
                    transform_name, radius, color_tol):
    t = pose_row[:3]
    q = pose_row[3:]
    rot = R.from_quat(q).as_matrix()

    transformed = apply_transform(points_local, transform_name)
    pts_global = (rot @ transformed.T).T + t

    distances, indices = kdtree.query(pts_global, k=1, distance_upper_bound=radius)
    valid_mask = distances != np.inf
    match_rate = float(np.mean(valid_mask))
    mean_dist = float(np.mean(distances[valid_mask])) if np.any(valid_mask) else None

    ground_fraction = 0.0
    if ground_colors is not None and len(ground_colors) > 0 and np.any(valid_mask):
        nn_colors = global_colors[indices[valid_mask]]
        is_ground = np.zeros(len(nn_colors), dtype=bool)
        for color in ground_colors:
            is_ground |= np.linalg.norm(nn_colors - color, axis=1) < color_tol

        gt_labels = np.zeros(len(points_local), dtype=bool)
        gt_labels[valid_mask] = is_ground
        ground_fraction = float(np.mean(gt_labels))

    return {
        "transform": transform_name,
        "match_rate": match_rate,
        "mean_dist": mean_dist,
        "ground_fraction": ground_fraction,
        "score": (ground_fraction, match_rate),
    }


def debug_alignment(env_name="Office", frame_idx=0, radius=0.5, map_downsample=0.02,
                    color_tol=0.05, top_k=5):
    data_dir = "data"
    print(f"--- Debugging Alignment for {env_name} Frame {frame_idx} ---")

    lidar_file = os.path.join(
        data_dir, env_name, "Data_omni", "P0000", "lidar",
        f"{frame_idx:06d}_lcam_front_lidar.ply"
    )
    if not os.path.exists(lidar_file):
        print(f"Error: {lidar_file} not found")
        return

    frame_pcd = o3d.io.read_point_cloud(lidar_file)
    pts_local = np.asarray(frame_pcd.points)
    print(f"LiDAR points: {len(pts_local)}")
    print(f"Local Min: {pts_local.min(axis=0)}")
    print(f"Local Max: {pts_local.max(axis=0)}")

    pose_file = os.path.join(data_dir, env_name, "Data_omni", "P0000", "pose_lcam_front.txt")
    poses = np.loadtxt(pose_file)
    pose = poses[frame_idx]

    pcd_path = os.path.join(data_dir, env_name, f"{env_name}_sem.pcd")
    print(f"Loading {pcd_path}...")
    global_pcd = o3d.io.read_point_cloud(pcd_path)
    original_points = len(global_pcd.points)
    if 0.0 < map_downsample < 1.0:
        global_pcd = global_pcd.random_down_sample(map_downsample)
    global_points = np.asarray(global_pcd.points)
    global_colors = np.asarray(global_pcd.colors)
    print(f"Global points: {len(global_points)} (from {original_points})")

    kdtree = cKDTree(global_points)
    ground_colors = get_ground_colors(env_name, data_dir)
    if ground_colors is None or len(ground_colors) == 0:
        print("No ground colors found for this environment.")
        ground_colors = None

    results = []
    for transform_name in TRANSFORM_PRESETS:
        results.append(
            score_transform(
                pts_local, pose, kdtree, global_colors, ground_colors,
                transform_name, radius, color_tol
            )
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    print(f"Top {min(top_k, len(results))} transforms:")
    for row in results[:top_k]:
        mean_dist = "n/a" if row["mean_dist"] is None else f"{row['mean_dist']:.4f}m"
        print(
            f"  {row['transform']:12s} "
            f"match={row['match_rate'] * 100:6.2f}%  "
            f"ground={row['ground_fraction'] * 100:6.2f}%  "
            f"dist={mean_dist}"
        )

    best = results[0]
    print(f"\nBest transform guess: {best['transform']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="House")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--map_downsample", type=float, default=0.02,
                        help="Random-downsample ratio for the semantic map")
    parser.add_argument("--color_tol", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    debug_alignment(
        env_name=args.env,
        frame_idx=args.frame,
        radius=args.radius,
        map_downsample=args.map_downsample,
        color_tol=args.color_tol,
        top_k=args.top_k,
    )
